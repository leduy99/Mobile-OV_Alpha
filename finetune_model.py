"""
OmniVideo Finetuning Script

A professional training script for the OmniVideo mixed condition model supporting
multiple tasks (T2I, I2I, T2V) with DeepSpeed optimization and comprehensive logging.

Author: OmniVideo Team
License: Apache 2.0
"""

import argparse
import json
import logging
import math
import os
import pickle as pkl
import random
import sys
import types
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import deepspeed
import torch
import torch.distributed as dist
import yaml
from easydict import EasyDict
from torch.optim import AdamW
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

# Project imports
from nets.omni.datasets.omnivideo_dataset_patched import create_omnivideo_dataloader
from nets.omni.modules.omni_video_model import OmniVideoMixedConditionModel
from nets.omni.modules.mobile_ov_model import MobileOVModel
from nets.omni.modules.mobile_ov_model_sana import MobileOVModelSANA
from nets.omni.modules.schedulers.flow_match import FlowMatchScheduler
from nets.third_party.wan.utils.utils import str2bool

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
SPECIAL_TOKENS_SUBDIR = "special_tokens"
UNCOND_CONTEXT_SUBDIR = "unconditioned_context"

def _init_logging(rank: int, args: Optional[Any] = None) -> None:
    """
    Initialize distributed logging configuration.
    
    Args:
        rank: Process rank (only rank 0 logs INFO level)
        args: Optional arguments (reserved for future use)
    """
    if rank == 0:
        # Use format without rank field to avoid KeyError in other modules
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] [Rank 0] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
            force=True
        )
        # Add rank attribute to all log records for modules that need it
        class RankFilter(logging.Filter):
            def __init__(self, rank):
                super().__init__()
                self.rank = rank
            def filter(self, record):
                record.rank = getattr(record, 'rank', self.rank)
                return True
        logging.getLogger().addFilter(RankFilter(rank))
    else:
        logging.basicConfig(level=logging.ERROR, force=True)

def str2tuple(v: str) -> Tuple[int, ...]:
    """
    Convert string representation to tuple of integers.
    
    Args:
        v: String representation (e.g., '1,2,2' or '(1,2,2)')
        
    Returns:
        Tuple of integers
        
    Examples:
        >>> str2tuple('1,2,2')
        (1, 2, 2)
        >>> str2tuple('(1,2,2)')
        (1, 2, 2)
    """
    v = v.strip()
    if v.startswith('(') and v.endswith(')'):
        v = v[1:-1]
    return tuple(int(x.strip()) for x in v.split(','))

def load_and_merge_config(yaml_path: str, cmd_args: Optional[argparse.Namespace] = None) -> EasyDict:
    """
    Load YAML configuration and merge with command line arguments.
    
    Args:
        yaml_path: Path to the YAML configuration file
        cmd_args: Command line arguments to override config values
        
    Returns:
        Merged configuration as EasyDict
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    # Check if file exists
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    
    # Add the parent directory to sys.path
    parent_dir = os.path.dirname(os.path.dirname(yaml_path))  # Go up two levels
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    try:
        # Load YAML file
        with open(yaml_path, 'r') as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing YAML file: {e}")
        
        # Convert to EasyDict
        def dict_to_easydict(d):
            if isinstance(d, dict):
                return EasyDict({k: dict_to_easydict(v) for k, v in d.items()})
            elif isinstance(d, list):
                return [dict_to_easydict(item) if isinstance(item, (dict, list)) else item for item in d]
            elif isinstance(d, str):
                # Try to convert string to number
                try:
                    # Handle scientific notation (e.g., 1e-4)
                    if 'e' in d.lower():
                        return float(d)
                    # Handle regular floats
                    elif '.' in d:
                        return float(d)
                    # Handle integers
                    elif d.isdigit():
                        return int(d)
                    # Handle negative integers
                    elif d.startswith('-') and d[1:].isdigit():
                        return int(d)
                    # Handle negative floats
                    elif d.startswith('-') and '.' in d[1:]:
                        return float(d)
                    # Handle negative scientific notation
                    elif d.startswith('-') and 'e' in d[1:].lower():
                        return float(d)
                except ValueError:
                    pass
            return d
        
        config = dict_to_easydict(config_dict)
        
        # If no command line arguments provided, return just the YAML config
        if cmd_args is None:
            return config
        
        # Only merge basic command line arguments
        basic_args = ['output_dir', 'resume_from', 'ckpt_dir']
        for arg_name in basic_args:
            arg_value = getattr(cmd_args, arg_name, None)
            if arg_value is not None:
                # Ensure the attribute exists in the config
                if not hasattr(config, arg_name):
                    setattr(config, arg_name, None)
                # Set the value
                setattr(config, arg_name, arg_value)
                print(f"Overriding config value for {arg_name} with command-line value: {arg_value}")
            else:
                # If the argument is not provided, ensure it exists in config with None
                if not hasattr(config, arg_name):
                    setattr(config, arg_name, None)
        
        return config
    finally:
        # Clean up sys.path
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)

def save_config_to_yaml(args: EasyDict, output_path: str) -> None:
    """
    Save configuration to YAML file for reproducibility.
    
    Args:
        args: Configuration object to save
        output_path: Path where to save the YAML file
    """
    def easydict_to_dict(ed: Union[EasyDict, Any]) -> Any:
        """Recursively convert EasyDict to regular dict."""
        if isinstance(ed, EasyDict):
            return {key: easydict_to_dict(value) for key, value in ed.items()}
        elif isinstance(ed, (list, tuple)):
            return [easydict_to_dict(item) for item in ed]
        elif isinstance(ed, dict):
            return {k: easydict_to_dict(v) for k, v in ed.items()}
        elif isinstance(ed, (int, float, str, bool, type(None))):
            return ed
        else:
            return str(ed)  # Convert other types to string
    
    try:
        config_dict = easydict_to_dict(args)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, 
                     default_flow_style=False, 
                     sort_keys=False,
                     allow_unicode=True,
                     indent=2)
        
        logging.info(f"Training configuration saved to {output_path}")
    except Exception as e:
        logging.warning(f"Failed to save configuration: {e}")

def _parse_args() -> EasyDict:
    """
    Parse command line arguments and merge with YAML configuration.
    
    Returns:
        Merged configuration as EasyDict
        
    Raises:
        ValueError: If required parameters are missing
    """
    parser = argparse.ArgumentParser(
        description="Train OmniVideoMixedConditionModel with multiple tasks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints and logs. Overrides config if set."
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from. Overrides config if set."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Directory containing base WAN model checkpoints. Overrides config if set."
    )

    cmd_args = parser.parse_args()
    
    # Load and merge configurations
    args = load_and_merge_config(cmd_args.config, cmd_args)
    
    # Ensure required parameters exist
    if not hasattr(args, 'output_dir') or args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"output/training_{timestamp}"
    
    if not hasattr(args, 'ckpt_dir') or args.ckpt_dir is None:
        raise ValueError("ckpt_dir must be specified either in config or via command line.")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get the original YAML filename and create the output path
    original_yaml_name = os.path.basename(cmd_args.config)
    args_file = os.path.join(args.output_dir, original_yaml_name)
    
    # Save the configuration
    save_config_to_yaml(args, args_file)
    
    return args

def load_uncond_feature(
    unconditioned_context_path: str, 
    precision_dtype: torch.dtype, 
    device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Load unconditioned context features for classifier-free guidance.
    
    Args:
        unconditioned_context_path: Path to pickle file containing unconditioned context
        precision_dtype: Target data type for tensor conversion
        device: Target device for tensors
        
    Returns:
        Dictionary with 'uncond_context' and 'uncond_ar_vision' keys, or None if loading fails
    """
    try:
        with open(unconditioned_context_path, 'rb') as f:
            pstate = pkl.load(f)
        
        # Validate required keys
        required_keys = ['text_emb', 'vlm_last_hidden_states']
        for key in required_keys:
            if key not in pstate:
                raise KeyError(f"Required key '{key}' not found in {unconditioned_context_path}")
        
        # Process text embeddings
        unconditioned_t5 = pstate['text_emb'][0].to(precision_dtype).to(device)
        if unconditioned_t5.dim() < 2:
            unconditioned_t5 = unconditioned_t5.unsqueeze(0)
        
        # Process vision embeddings
        uncond_ar_vision = pstate['vlm_last_hidden_states'].to(precision_dtype).to(device)
        
        unconditioned_context = {
            'uncond_context': unconditioned_t5, 
            'uncond_ar_vision': uncond_ar_vision
        }
        
        logging.info(
            f"Loaded unconditioned context - T5: {unconditioned_t5.shape}, "
            f"Vision: {uncond_ar_vision.shape}"
        )
        
        return unconditioned_context
        
    except Exception as e:
        logging.error(f"Failed to load unconditioned context from {unconditioned_context_path}: {e}")
        return None


def create_dataloaders(
    args: EasyDict, 
    rank: int, 
    world_size: int
) -> Tuple[Dict[str, Any], Dict[str, int], int]:
    """
    Create dataloaders for all configured tasks.
    
    Args:
        args: Configuration containing dataloader specifications
        rank: Process rank for distributed training
        world_size: Total number of processes
        
    Returns:
        Tuple of (dataloaders dict, dataset sizes dict, total batch size)
        
    Raises:
        ValueError: If no dataloaders are configured
    """
    if not hasattr(args.training, 'dataloaders') or not args.training.dataloaders:
        raise ValueError("Configuration must contain 'dataloaders' for multi-task training")

    dataloaders = {}
    dataset_sizes = {}
    total_batch_size = 0

    for task_name, task_config in args.training.dataloaders.items():
        data_file = task_config.get('data_file')
        if not data_file or not os.path.exists(data_file):
            logging.warning(f"Data file not found for task '{task_name}': {data_file}")
            continue
            
        batch_size = task_config.get('batch_size', args.training.hyperparameters.batch_size)
        num_workers = task_config.get('num_workers', 4)
        shuffle = task_config.get('shuffle', True)
        
        logging.info(f"Creating dataloader for task '{task_name}' - file: {data_file}")
        
        try:
            # Detect dataset type:
            # - OpenVid-1M: CSV file with columns ['video', 'caption', ...]
            # - OmniVideo: text file listing pickle paths
            video_dir = task_config.get('video_dir')
            preprocessed_dir = task_config.get('preprocessed_dir')
            use_preprocessed = task_config.get('use_preprocessed', True)
            max_samples = task_config.get('max_samples', None)

            use_openvid = False
            
            # Check if explicitly using OmniVideo dataset format
            use_omnivideo = task_config.get('use_omnivideo_dataset', False)
            if use_omnivideo:
                use_openvid = False
            # Heuristic 1: explicit OpenVid config keys
            elif video_dir or preprocessed_dir:
                use_openvid = True

            # Heuristic 2: file extension (.csv) indicates OpenVid metadata
            if data_file.lower().endswith(".csv"):
                use_openvid = True

            if use_openvid:
                from nets.omni.datasets.openvid_dataset import create_openvid_dataloader
                # If video_dir is not provided or doesn't exist, use a dummy directory
                # (videos are optional when using preprocessed features)
                if not video_dir or not os.path.exists(video_dir):
                    video_dir = task_config.get('video_dir', 'data/openvid_videos_dummy')

                logging.info(
                    f"Using OpenVid-1M dataset loader for task '{task_name}' "
                    f"(csv={data_file}, video_dir={video_dir}, preprocessed_dir={preprocessed_dir})"
                )

                dataloader = create_openvid_dataloader(
                    csv_path=data_file,
                    video_dir=video_dir,
                    preprocessed_dir=preprocessed_dir,
                    use_preprocessed=use_preprocessed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    distributed=dist.is_initialized(),
                    rank=rank,
                    world_size=world_size,
                    max_samples=max_samples,
                )
            else:
                # Fallback: standard OmniVideo dataset loader (expects text file of pickle paths)
                # Check if test mode is enabled (limit dataset size)
                test_max_samples = None
                if hasattr(args.training, 'test_mode') and getattr(args.training.test_mode, 'enabled', False):
                    test_max_samples = getattr(args.training.test_mode, 'max_samples', None)
                    if rank == 0:
                        logging.info(f"TEST MODE: Limiting OmniVideo dataset to {test_max_samples} samples")
                
                dataloader = create_omnivideo_dataloader(
                    file_path=data_file,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=shuffle,
                    distributed=dist.is_initialized(),
                    rank=rank, 
                    world_size=world_size,
                    max_samples=test_max_samples if test_max_samples is not None else max_samples,
                )
            
            dataloaders[task_name] = dataloader
            dataset_sizes[task_name] = len(dataloader)
            total_batch_size += batch_size
            
            logging.info(
                f"Created dataloader for '{task_name}': {len(dataloader)} batches, "
                f"batch_size={batch_size}"
            )
            
        except Exception as e:
            logging.error(f"Failed to create dataloader for task '{task_name}': {e}")
            continue
    
    if not dataloaders:
        raise ValueError("No valid dataloaders were created")
    
    return dataloaders, dataset_sizes, total_batch_size

def process_batch(
    batch: Dict[str, Any], 
    task_name: str, 
    device: torch.device, 
    args: Optional[EasyDict] = None, 
    rank: int = 0, 
    epoch: int = 0, 
    step_idx: int = 0
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Process a batch for a specific task, handling different data formats.
    
    Args:
        batch: Raw batch data from dataloader
        task_name: Name of the task (t2i, i2i, t2v, etc.)
        device: Target device for tensors
        args: Configuration object
        rank: Process rank for logging
        epoch: Current epoch for logging
        step_idx: Current step for logging
        
    Returns:
        Tuple of processed tensors: (videos, context, aligned_emb, ar_vision_input, visual_emb, ref_images, prompts, adapter_output_gt)
    """
    # Extract aligned embeddings (SIGLIP features)
    aligned_emb = None
    for key in ['siglip2_img_pooled_output_tgt', 'aligned_emb', 'siglip2_feature']:
        if key in batch:
            aligned_emb = batch[key]
            break

    # Extract AR vision input
    ar_vision_input = batch.get('vlm_last_hidden_states', None)
    if ar_vision_input is not None:
        ar_vision_input = [it.to(device) for it in ar_vision_input]
    
    # Extract adapter_output_gt for distillation loss
    adapter_output_gt = batch.get('adapter_output_gt', None)
    if adapter_output_gt is not None:
        # Convert to list if single tensor, ensure on device
        if isinstance(adapter_output_gt, torch.Tensor):
            adapter_output_gt = [adapter_output_gt.to(device)]
        elif isinstance(adapter_output_gt, list):
            adapter_output_gt = [it.to(device) for it in adapter_output_gt]
    
    # Extract text context and prompts
    context = batch.get('text_emb', None)
    prompts = batch.get('prompt', ['N/A'])
    
    # Extract video latents
    tgt_videos = batch.get('latent_feature_tgt', None)
    src_videos = batch.get('latent_feature', None)
    ref_images = batch.get('ref_images', None)
    
    # Move to device if not None
    if tgt_videos is not None:
        tgt_videos = tgt_videos.to(device)
    if src_videos is not None:
        src_videos = src_videos.to(device)
    if ref_images is not None:
        ref_images = ref_images.to(device)
    
    # Task-specific logic for determining input/output videos
    videos = src_videos if tgt_videos is None else tgt_videos
    visual_emb = src_videos if tgt_videos is not None else None
    
    # Handle i2v tasks - use first frame as reference if not provided
    if ref_images is None and 'i2v' in task_name and videos is not None:
        ref_images = videos[:, :, 0:1]  # [B, C, 1, H, W]
    
    # Validate we have valid video data
    if videos is None:
        logging.warning(f"Task {task_name} batch has no video data, skipping")
        return None, None, None, None, None, None, None
    
    # Log batch info on first step of first epoch (rank 0 only)
    if step_idx == 0 and epoch == 0 and rank == 0:
        logging.info(f"Task '{task_name}' batch info:")
        logging.info(f"  Batch size: {len(prompts)}")
        logging.info(f"  Sample prompt: {prompts[0]}")
        logging.info(f"  Video shape: {videos.shape}")
        logging.info(f"  Context: {len(context) if context else 'None'}")
        logging.info(f"  Aligned emb: {aligned_emb.shape if aligned_emb is not None else 'None'}")
    
    # Move remaining data to device and handle context replacement
    videos = videos.to(device)
    
    # Check if we should replace context with adapter output
    if args and hasattr(args.training, 'model_settings'):
        replace_context = getattr(args.training.model_settings, 'replace_context_with_adapter', False)
        if replace_context:
            context = None
        
        # Option 3 Fix: Disable T5 context when using SmolVLM2 to avoid double-conditioning
        disable_t5_context = getattr(args.training.model_settings, 'disable_t5_context', False)
        use_precomputed_features = getattr(args.training.model_settings, 'use_precomputed_features', True)
        if disable_t5_context and not use_precomputed_features:
            # When using SmolVLM2 on-the-fly encoding, skip T5 context to avoid double-conditioning
            context = None
            if rank == 0 and step_idx == 0 and epoch == 0:
                logging.info(f"T5 context disabled (disable_t5_context=True, use_precomputed_features=False)")
    
    # Move context to device
    if context is not None:
        context = [it.to(device) for it in context]
    
    # Move aligned embeddings to device
    if aligned_emb is not None:
        if isinstance(aligned_emb, list):
            aligned_emb = [it.to(device) for it in aligned_emb]
        else:
            aligned_emb = aligned_emb.to(device)
    
    return videos, context, aligned_emb, ar_vision_input, visual_emb, ref_images, prompts, adapter_output_gt

def train(args: EasyDict) -> None:
    """
    Main training function for multi-task OmniVideo model.
    
    Args:
        args: Configuration object containing all training parameters
    """
    # Initialize distributed training if needed
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    # Setup device
    device = torch.device(f"cuda:{local_rank}")
    print(f'Using device {device}', flush=True)
    
    # Set environment variables for DeepSpeed if not already set
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(local_rank)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = str(rank)
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(world_size)
        
    # Initialize distributed environment
    # Skip MPI discovery for single GPU training
    try:
        deepspeed.init_distributed()
    except (RuntimeError, ImportError) as e:
        if "MPI" in str(e) or "mpi4py" in str(e).lower():
            # Single GPU training - initialize PyTorch distributed manually if needed
            if not dist.is_initialized():
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(12355)
                dist.init_process_group(backend='nccl', rank=0, world_size=1)
                logging.info("Initialized single GPU distributed environment (bypassed MPI)")
        else:
            raise
    
    # Log environment information
    logging.info("=== Training Environment ===")
    logging.info(f"Rank: {rank}/{world_size}, Local Rank: {local_rank}")
    logging.info(f"Device: {device}")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU count: {torch.cuda.device_count()}")
    
    # Extract model configuration
    num_train_timesteps = args.model.num_train_timesteps
    param_dtype = getattr(torch, args.model.param_dtype)
    patch_size = args.model.transformer.get('patch_size', (1, 2, 2))
    
    # Determine precision type
    # FIX: Default to bfloat16 to match OmniVideo for faster training/inference
    precision_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32
    }
    precision_dtype = precision_map.get(
        args.training.precision.mixed_precision, 
        torch.bfloat16  # FIX: Changed default from float32 to bfloat16 (matches OmniVideo)
    )
    
    logging.info(f"Training precision: {precision_dtype}")
    logging.info(f"Parameter dtype: {param_dtype}")
    logging.info(f"Patch size: {patch_size}")
    
    # Determine model type from config
    model_type = getattr(args.training.model_settings, 'model_type', 'omnivideo').lower()
    use_precomputed_features = getattr(args.training.model_settings, 'use_precomputed_features', True)
    
    if model_type == 'mobile_ov':
        logging.info("Using MobileOVModel (SmolVLM2 + WAN)")
        smolvlm2_ckpt_path = getattr(args.training.model_settings, 'smolvlm2_ckpt_path', None)
        if smolvlm2_ckpt_path is None:
            raise ValueError("smolvlm2_ckpt_path must be provided in config when using MobileOVModel")
        
        # Get WAN checkpoint directory from config or use default
        wan_ckpt_dir = getattr(args.training.model_settings, 'wan_ckpt_dir', 'omni_ckpts/wan/wanxiang1_3b')
        if not os.path.exists(wan_ckpt_dir):
            raise ValueError(f"WAN checkpoint directory not found: {wan_ckpt_dir}")
        
        model = MobileOVModel.from_pretrained(
            wan_ckpt_dir=wan_ckpt_dir,
            adapter_ckpt_dir=None,
            smolvlm2_ckpt_path=smolvlm2_ckpt_path,
            adapter_in_channels=args.model.adapter.in_channels,
            adapter_out_channels=args.model.adapter.out_channels,
            adapter_query_length=args.model.adapter.query_length,
            precision_dtype=precision_dtype,
            device_id=local_rank,
            rank=rank,
            dit_fsdp=args.training.model_settings.dit_fsdp,
            use_usp=args.training.model_settings.use_usp,
            use_visual_context_adapter=args.training.model_settings.train_visual_context_adapter,
            visual_context_adapter_patch_size=args.model.visual_context_adapter.visual_context_adapter_patch_size,
            max_context_len=args.training.model_settings.max_context_len,
            use_precomputed_features=use_precomputed_features,
            disable_t5_context=getattr(args.training.model_settings, 'disable_t5_context', True),  # Default True to avoid double-conditioning
            use_smol_vh=getattr(args.training.model_settings, 'use_smol_vh', True),  # Use VisionHead-style resampler (recommended)
            smol_vh_num_queries=getattr(args.training.model_settings, 'smol_vh_num_queries', 1),  # Q=1 for T2V bring-up
        )
    elif model_type == 'mobile_ov_sana':
        logging.info("Using MobileOVModelSANA (SmolVLM2 + SANA)")
        smolvlm2_ckpt_path = getattr(args.training.model_settings, 'smolvlm2_ckpt_path', None)
        if smolvlm2_ckpt_path is None:
            raise ValueError("smolvlm2_ckpt_path must be provided in config when using MobileOVModelSANA")

        sana_ckpt_dir = getattr(
            args.training.model_settings,
            'sana_ckpt_dir',
            'omni_ckpts/sana_video_2b_480p',
        )
        sana_config_path = getattr(
            args.training.model_settings,
            'sana_config_path',
            'configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml',
        )
        if not os.path.exists(sana_ckpt_dir):
            raise ValueError(f"SANA checkpoint directory not found: {sana_ckpt_dir}")
        if not os.path.exists(sana_config_path):
            raise ValueError(f"SANA config not found: {sana_config_path}")

        model = MobileOVModelSANA.from_pretrained(
            sana_ckpt_dir=sana_ckpt_dir,
            sana_config_path=sana_config_path,
            adapter_ckpt_dir=None,
            smolvlm2_ckpt_path=smolvlm2_ckpt_path,
            adapter_in_channels=args.model.adapter.in_channels,
            adapter_out_channels=args.model.adapter.out_channels,
            adapter_query_length=args.model.adapter.query_length,
            precision_dtype=precision_dtype,
            device_id=local_rank,
            rank=rank,
            dit_fsdp=args.training.model_settings.dit_fsdp,
            use_usp=args.training.model_settings.use_usp,
            use_visual_context_adapter=args.training.model_settings.train_visual_context_adapter,
            visual_context_adapter_patch_size=args.model.visual_context_adapter.visual_context_adapter_patch_size,
            max_context_len=args.training.model_settings.max_context_len,
            use_precomputed_features=use_precomputed_features,
            disable_t5_context=getattr(args.training.model_settings, 'disable_t5_context', True),
            use_smol_vh=getattr(args.training.model_settings, 'use_smol_vh', True),
            smol_vh_num_queries=getattr(args.training.model_settings, 'smol_vh_num_queries', 1),
        )
    else:
        logging.info("Using OmniVideoMixedConditionModel (original)")
        # Initialize OmniVideoMixedConditionModel with updated parameters
        model = OmniVideoMixedConditionModel.from_pretrained(
            wan_ckpt_dir=args.ckpt_dir,
            adapter_ckpt_dir=None,
            vision_head_ckpt_dir=None, 
            learnable_query_length=args.model.get('ar_vision_head', {}).get('learnable_query_length', 4),
            adapter_in_channels=args.model.adapter.in_channels,
            adapter_out_channels=args.model.adapter.out_channels,
            adapter_query_length=args.model.adapter.query_length,
            device_id=local_rank,
            rank=rank,
            dit_fsdp=args.training.model_settings.dit_fsdp,
            use_usp=args.training.model_settings.use_usp,
            use_visual_context_adapter=args.training.model_settings.train_visual_context_adapter,
            visual_context_adapter_patch_size=args.model.visual_context_adapter.visual_context_adapter_patch_size,  # Updated to use new config path
            max_context_len=args.training.model_settings.max_context_len,
        )
    
    # Set training mode for model components
    if args.training.model_settings.train_wan_model:
        model.wan_model.train().requires_grad_(True)
        logging.info("WanModel parameters are unfrozen and will be trained")
    else:
        # CRITICAL: Even if we don't train WAN model, we MUST set requires_grad=True
        # to allow gradients to flow through it for loss computation
        # We will exclude WAN parameters from optimizer later
        model.wan_model.eval().requires_grad_(True)
        logging.info("WanModel parameters are frozen (eval mode) but requires_grad=True for gradient flow")
    
    if args.training.model_settings.train_adapter:
        model.adapter.train().requires_grad_(True)
        logging.info("Adapter parameters are unfrozen and will be trained")
    else:
        model.adapter.eval().requires_grad_(False)
        logging.info("Adapter parameters are frozen and will not be trained")

    if args.training.model_settings.train_visual_context_adapter and model.visual_context_adapter is not None:
        model.visual_context_adapter.train().requires_grad_(True)
        logging.info("Visual Context Adapter parameters are unfrozen and will be trained")
    elif model.visual_context_adapter is not None:
        model.visual_context_adapter.eval().requires_grad_(False)
        logging.info("Visual Context Adapter parameters are frozen and will not be trained")
    
    # Handle AR Vision Head (only for OmniVideoMixedConditionModel)
    if hasattr(model, 'ar_vision_head') and model.ar_vision_head is not None:
        train_ar_vision_head = getattr(args.training.model_settings, 'train_ar_vision_head', False)
        if train_ar_vision_head: 
            model.ar_vision_head.train().requires_grad_(True)
            logging.info("AR Vision Head parameters are unfrozen and will be trained")
        else:
            model.ar_vision_head.eval().requires_grad_(False)
            logging.info("AR Vision Head parameters are frozen and will not be trained")
    
    # Handle SmolVLM2 (only for MobileOVModel)
    if hasattr(model, 'smolvlm2_model') and model.smolvlm2_model is not None:
        train_smolvlm2 = getattr(args.training.model_settings, 'train_smolvlm2', False)
        if train_smolvlm2:
            model.smolvlm2_model.train().requires_grad_(True)
            logging.info("SmolVLM2 parameters are unfrozen and will be trained")
        else:
            # CRITICAL: Even if we don't train SmolVLM2, we MUST set requires_grad=True
            # to allow gradients to flow through it to downstream modules (vision_head, adapter)
            # We will exclude SmolVLM2 parameters from optimizer later
            model.smolvlm2_model.eval().requires_grad_(True)
            logging.info("SmolVLM2 parameters are frozen (eval mode) but requires_grad=True for gradient flow")
    
    # Handle SmolVLM2 projection (legacy, only if not using VisionHead)
    if hasattr(model, 'smolvlm2_projection') and model.smolvlm2_projection is not None:
        train_smolvlm2_proj = getattr(args.training.model_settings, 'train_smolvlm2_projection', False)
        if train_smolvlm2_proj:
            model.smolvlm2_projection.train().requires_grad_(True)
            logging.info("SmolVLM2 projection parameters are unfrozen and will be trained")
        else:
            model.smolvlm2_projection.eval().requires_grad_(False)
            logging.info("SmolVLM2 projection parameters are frozen and will not be trained")
    
    # Handle SmolVLM2VisionHead (new, recommended)
    if hasattr(model, 'smolvlm2_vision_head') and model.smolvlm2_vision_head is not None:
        train_smolvlm2_vh = getattr(args.training.model_settings, 'train_smolvlm2_vision_head', True)
        if train_smolvlm2_vh:
            model.smolvlm2_vision_head.train().requires_grad_(True)
            vh_params_count = sum(p.numel() for p in model.smolvlm2_vision_head.parameters())
            logging.info(f"✅ SmolVLM2VisionHead parameters are unfrozen and will be trained ({vh_params_count:,} params)")
        else:
            model.smolvlm2_vision_head.eval().requires_grad_(False)
            logging.info("SmolVLM2VisionHead parameters are frozen and will not be trained")
    else:
        if rank == 0:
            logging.error(f"❌ CRITICAL: Vision head does not exist or is None!")
            logging.error(f"   Model type: {type(model)}")
            logging.error(f"   Has attr: {hasattr(model, 'smolvlm2_vision_head')}")
            if hasattr(model, 'smolvlm2_vision_head'):
                logging.error(f"   Is None: {model.smolvlm2_vision_head is None}")
            if hasattr(model, 'use_smol_vh'):
                logging.error(f"   use_smol_vh: {model.use_smol_vh}")
            raise RuntimeError("Vision head is required but not found in model!")

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        total_params = sum(p.numel() for p in trainable_params)
        logging.info(f"Total trainable parameters: {total_params:,}")
        # adapter parameters
        adapter_params = [p for p in model.adapter.parameters()]
        total_adapter_params = sum(p.numel() for p in adapter_params)
        logging.info(f"Total adapter parameters: {total_adapter_params:,}")
    
    # Separate parameter groups for different learning rates
    # This is especially important for MobileOVModel where:
    # - Projection layer is new and needs higher LR to learn mapping
    # - Adapter is pretrained and should use lower LR to avoid forgetting
    param_groups = []
    base_lr = float(args.training.hyperparameters.learning_rate)
    
    # Check if we have MobileOVModel with VisionHead (recommended) or projection layer (legacy)
    if hasattr(model, 'smolvlm2_vision_head') and model.smolvlm2_vision_head is not None:
        train_vh = getattr(args.training.model_settings, 'train_smolvlm2_vision_head', True)  # FIX: Default True to match line 747
        if train_vh:
            vh_params = [p for p in model.smolvlm2_vision_head.parameters() if p.requires_grad]
        else:
            vh_params = []
        if vh_params:
            # VisionHead: higher LR (2x) since it's learning new mapping
            vh_lr = base_lr * 2.0
            param_groups.append({
                'params': vh_params,
                'lr': vh_lr,
                'name': 'smolvlm2_vision_head'
            })
            if rank == 0:
                vh_params_count = sum(p.numel() for p in vh_params)
                logging.info(f"SmolVLM2VisionHead: {vh_params_count:,} params, LR={vh_lr:.6f}")
    elif hasattr(model, 'smolvlm2_projection') and model.smolvlm2_projection is not None:
        projection_params = [p for p in model.smolvlm2_projection.parameters() if p.requires_grad]
        if projection_params:
            # Projection layer: higher LR (2x) since it's learning new mapping
            projection_lr = base_lr * 2.0
            param_groups.append({
                'params': projection_params,
                'lr': projection_lr,
                'name': 'smolvlm2_projection'
            })
            if rank == 0:
                projection_params_count = sum(p.numel() for p in projection_params)
                logging.info(f"Projection layer: {projection_params_count:,} params, LR={projection_lr:.6f}")
    
    # Adapter parameters
    adapter_trainable_params = [p for p in model.adapter.parameters() if p.requires_grad]
    if adapter_trainable_params:
        # Adapter: lower LR (0.5x) since it's pretrained
        adapter_lr = base_lr * 0.5
        param_groups.append({
            'params': adapter_trainable_params,
            'lr': adapter_lr,
            'name': 'adapter'
        })
        if rank == 0:
            adapter_trainable_count = sum(p.numel() for p in adapter_trainable_params)
            logging.info(f"Adapter: {adapter_trainable_count:,} trainable params, LR={adapter_lr:.6f}")
    
    # Visual context adapter
    if hasattr(model, 'visual_context_adapter') and model.visual_context_adapter is not None:
        vca_params = [p for p in model.visual_context_adapter.parameters() if p.requires_grad]
        if vca_params:
            vca_lr = base_lr * 0.5  # Similar to adapter
            param_groups.append({
                'params': vca_params,
                'lr': vca_lr,
                'name': 'visual_context_adapter'
            })
            if rank == 0:
                vca_count = sum(p.numel() for p in vca_params)
                logging.info(f"Visual Context Adapter: {vca_count:,} params, LR={vca_lr:.6f}")
    
    # Other trainable parameters (WAN, SmolVLM2 if unfrozen, etc.)
    other_params = []
    vh_param_ids = set()
    if hasattr(model, 'smolvlm2_vision_head') and model.smolvlm2_vision_head is not None:
        vh_param_ids = {id(p) for p in model.smolvlm2_vision_head.parameters()}
    projection_param_ids = set()
    if hasattr(model, 'smolvlm2_projection') and model.smolvlm2_projection is not None:
        projection_param_ids = {id(p) for p in model.smolvlm2_projection.parameters()}
    adapter_param_ids = {id(p) for p in model.adapter.parameters()}
    vca_param_ids = set()
    if hasattr(model, 'visual_context_adapter') and model.visual_context_adapter is not None:
        vca_param_ids = {id(p) for p in model.visual_context_adapter.parameters()}
    
    # CRITICAL: Exclude SmolVLM2 and WAN parameters from optimizer if not training them
    # But keep requires_grad=True to allow gradient flow
    smolvlm2_param_ids = set()
    if hasattr(model, 'smolvlm2_model') and model.smolvlm2_model is not None:
        train_smolvlm2 = getattr(args.training.model_settings, 'train_smolvlm2', False)
        if not train_smolvlm2:
            # Exclude SmolVLM2 parameters from optimizer (but keep requires_grad=True for gradient flow)
            smolvlm2_param_ids = {id(p) for p in model.smolvlm2_model.parameters()}
    
    # CRITICAL: Exclude WAN parameters from optimizer if not training them
    # But keep requires_grad=True to allow gradient flow
    wan_param_ids = set()
    train_wan = getattr(args.training.model_settings, 'train_wan_model', False)
    if not train_wan:
        # Exclude WAN parameters from optimizer (but keep requires_grad=True for gradient flow)
        wan_param_ids = {id(p) for p in model.wan_model.parameters()}
    
    # Collect other trainable parameters (exclude those already in param_groups)
    for p in trainable_params:
        if id(p) not in vh_param_ids and id(p) not in projection_param_ids and id(p) not in adapter_param_ids and id(p) not in vca_param_ids and id(p) not in smolvlm2_param_ids and id(p) not in wan_param_ids:
            other_params.append(p)
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'name': 'other'
        })
        if rank == 0:
            other_count = sum(p.numel() for p in other_params)
            logging.info(f"Other trainable params: {other_count:,}, LR={base_lr:.6f}")
    
    # If no param groups created (shouldn't happen), fallback to all params
    if not param_groups:
        param_groups = [{'params': trainable_params, 'lr': base_lr, 'name': 'all'}]
        if rank == 0:
            logging.warning("No param groups created, using all params with base LR")

    # Load special token embeddings
    special_tokens = None
    if args.training.special_tokens.enabled:
        special_tokens_path = os.path.join(args.ckpt_dir, SPECIAL_TOKENS_SUBDIR, 'tokens.pkl')
        if os.path.exists(special_tokens_path):
            try:
                with open(special_tokens_path, 'rb') as f:
                    special_tokens = pkl.load(f)
                assert isinstance(special_tokens, dict), "Special tokens must be a dictionary"
                
                for key, value in special_tokens.items():
                    # CRITICAL: Detach special tokens to prevent gradient flow
                    # Special tokens are constants and should not be part of computation graph
                    special_tokens[key] = value.to(precision_dtype).to(device).detach()
                
                logging.info(f"Loaded special token embeddings: {list(special_tokens.keys())}")
            except Exception as e:
                logging.warning(f"Failed to load special tokens: {e}")
    
    # Load or create unconditioned context for classifier-free guidance
    unconditioned_context = None
    if args.training.classifier_free.ratio > 0:
        # For MobileOVModel with SmolVLM2, create unconditioned context from SmolVLM2
        if model_type in ('mobile_ov', 'mobile_ov_sana') and not use_precomputed_features:
            logging.info("Creating unconditioned context from SmolVLM2 for MobileOVModel...")
            # CRITICAL: Convert projection layer to float32 BEFORE moving model to device
            # This prevents model.to(device) from converting it to bfloat16
            if model.smolvlm2_projection is not None:
                model.smolvlm2_projection = model.smolvlm2_projection.to(dtype=param_dtype)
            # Ensure model is on the correct device before creating unconditioned context
            model.to(device)
            model.eval()  # Set to eval mode for creating unconditioned context
            with torch.no_grad():
                # Encode null prompt with SmolVLM2
                # Use minimal prompt instead of empty string to avoid tokenization issues
                null_prompt = "a generic scene"  # Minimal prompt for unconditional generation
                smolvlm2_hidden = model.encode_prompts_with_smolvlm2([null_prompt], device)  # [1, L, D_smolvlm2]
                
                # Ensure hidden states are on device
                if smolvlm2_hidden.device != device:
                    smolvlm2_hidden = smolvlm2_hidden.to(device)
                
                # CRITICAL: Ensure input and projection weights have matching dtype
                # HACK: Create temporary projection layer in float32 to avoid dtype mismatch
                # This is needed because the actual projection layer might be in bfloat16
                # due to DeepSpeed or other optimizations
                if model.smolvlm2_projection is not None:
                    # Convert input to float32
                    smolvlm2_hidden = smolvlm2_hidden.to(dtype=param_dtype)
                    # Get original projection layer state and convert all tensors to float32
                    orig_proj_state = model.smolvlm2_projection.state_dict()
                    # Convert state_dict values to float32
                    converted_state = {}
                    for key, value in orig_proj_state.items():
                        if isinstance(value, torch.Tensor):
                            converted_state[key] = value.to(dtype=param_dtype)
                        else:
                            converted_state[key] = value
                    # Create temporary projection layer in float32
                    temp_proj = torch.nn.Linear(
                        model.smolvlm2_projection.in_features,
                        model.smolvlm2_projection.out_features,
                        bias=model.smolvlm2_projection.bias is not None
                    ).to(device=device, dtype=param_dtype)
                    # Load converted state_dict (all tensors are now float32)
                    temp_proj.load_state_dict(converted_state, strict=False)
                    # Use temporary projection layer (disable autocast to be safe)
                    with torch.cuda.amp.autocast(enabled=False):
                        projected_hidden = temp_proj(smolvlm2_hidden)  # [1, L, adapter_in_channels]
                    # Clean up
                    del temp_proj, converted_state, orig_proj_state
                else:
                    projected_hidden = smolvlm2_hidden.to(dtype=param_dtype)
                
                # Ensure projected hidden is on device
                if projected_hidden.device != device:
                    projected_hidden = projected_hidden.to(device)
                
                # Pass through adapter to create unconditioned adapter output
                # Adapter output shape: [1, 256, 4096]
                # CRITICAL: Adapter weights might be in bfloat16, but we need float32
                # Convert adapter weights to float32 temporarily (or convert input to match adapter dtype)
                # Since param_dtype is float32, we'll convert adapter weights to float32
                adapter_weight_dtype = next(model.adapter.parameters()).dtype
                if rank == 0:
                    logging.info(f"Adapter weight dtype: {adapter_weight_dtype}, Input dtype: {projected_hidden.dtype}")
                
                # If adapter weights are not in param_dtype, we need to handle it
                # Option 1: Convert adapter weights to float32 (might be expensive for large adapter)
                # Option 2: Convert input to match adapter dtype (simpler)
                # We'll use Option 2: convert input to match adapter dtype
                if adapter_weight_dtype != param_dtype:
                    if rank == 0:
                        logging.info(f"Converting input from {projected_hidden.dtype} to {adapter_weight_dtype} to match adapter weights")
                    projected_hidden = projected_hidden.to(dtype=adapter_weight_dtype)
                
                # Disable autocast for adapter to avoid dtype issues
                with torch.cuda.amp.autocast(enabled=False):
                    uncond_adapter_output = model.adapter(projected_hidden)  # [1, L, 1152] -> [1, 256, 4096]
                
                # Convert output back to param_dtype for consistency
                if uncond_adapter_output.dtype != param_dtype:
                    uncond_adapter_output = uncond_adapter_output.to(dtype=param_dtype)
                
                # Ensure adapter output is on device
                if uncond_adapter_output.device != device:
                    uncond_adapter_output = uncond_adapter_output.to(device)
                
                # CRITICAL: Create a completely new tensor (not a view or reference)
                # This ensures the tensor has no connection to the computation graph
                # and can be used in forward pass without grad_fn issues
                # Method: Create new tensor with same shape/dtype, then copy values
                uncond_adapter_output_new = torch.empty_like(uncond_adapter_output)
                uncond_adapter_output_new.copy_(uncond_adapter_output.detach(), non_blocking=False)
                uncond_adapter_output_new.requires_grad_(False)
                
                # Create unconditioned context dict matching MobileOVModel's expected format
                unconditioned_context = {
                    'uncond_ar_vision': uncond_adapter_output_new,  # Adapter output for MobileOVModel (new tensor)
                    'uncond_context': None  # No T5 context when using SmolVLM2
                }
                
                if rank == 0:
                    logging.info(f"Created unconditioned context from SmolVLM2:")
                    logging.info(f"  uncond_ar_vision shape: {uncond_adapter_output.shape}, device: {uncond_adapter_output.device}")
                    logging.info(f"  uncond_context: None (T5 disabled)")
            
            model.train()  # Set back to training mode
        else:
            # For OmniVideoMixedConditionModel or MobileOVModel with precomputed features, load from file
            unconditioned_context_path = os.path.join(args.ckpt_dir, UNCOND_CONTEXT_SUBDIR, 'context.pkl')
        if os.path.exists(unconditioned_context_path):
            unconditioned_context = load_uncond_feature(
                unconditioned_context_path, precision_dtype, device
            )
            if unconditioned_context is None:
                args.training.classifier_free.ratio = 0.0
                logging.warning("Disabling classifier-free guidance due to loading failure")
        else:
            args.training.classifier_free.ratio = 0.0
            logging.warning(f"Unconditioned context not found: {unconditioned_context_path}")
    
    if args.training.classifier_free.ratio > 0:
        logging.info(f"Classifier-free guidance enabled with ratio: {args.training.classifier_free.ratio}")
        if rank == 0 and unconditioned_context is not None:
            if isinstance(unconditioned_context, dict):
                logging.info(f"Unconditioned context keys: {list(unconditioned_context.keys())}")
    else:
        logging.info("Classifier-free guidance disabled")

    # Enable gradient checkpointing to save memory (especially important when WAN/SmolVLM2 have requires_grad=True)
    if getattr(args.training.optimization, 'gradient_checkpointing', False):
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            if rank == 0:
                logging.info("Gradient checkpointing enabled to save memory")

    # Initialize Flow Match scheduler for training
    flow_scheduler = FlowMatchScheduler(
        num_train_timesteps=args.model.num_train_timesteps,
        num_inference_steps=args.model.num_train_timesteps,
        shift=args.training.model_settings.flow_shift,
        sigma_min=0.0,
        extra_one_step=True,
        is_training=True)
    
    model.to(device)

    sp_size = 1
    
    # Create dataloaders for all tasks
    dataloaders, dataset_sizes, total_batch_size = create_dataloaders(args, rank, world_size)
    if not dataloaders:
        raise ValueError("No dataloaders were created. Check dataloaders configuration.")
    
    # Extract weights for each task (now used for loss weighting, not sampling)
    task_weights = {task: config.get('weight', 1.0) 
                   for task, config in args.training.dataloaders.items()}
    
    # Calculate steps per epoch
    steps_per_epoch = min(dataset_sizes.values())
    total_training_steps = args.training.hyperparameters.num_epochs * steps_per_epoch
    
    logging.info("=== Training Schedule ===")
    logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info(f"Total epochs: {args.training.hyperparameters.num_epochs}")
    logging.info(f"Total training steps: {total_training_steps}")
    logging.info(f"Gradient accumulation steps: {args.training.hyperparameters.gradient_accumulation_steps}")
    
    # Initialize optimizer with separate param groups
    if param_groups:
        optimizer = AdamW(
            param_groups,
            weight_decay=float(args.training.hyperparameters.weight_decay)
        )
        if rank == 0:
            lr_str = ", ".join([f"{g['name']}: {g['lr']:.6f}" for g in param_groups])
            logging.info(f"Optimizer: AdamW with separate param groups ({lr_str}), "
                        f"weight_decay={args.training.hyperparameters.weight_decay}")
    else:
        # Fallback (shouldn't happen)
        optimizer = AdamW(
            trainable_params, 
            lr=float(args.training.hyperparameters.learning_rate),
            weight_decay=float(args.training.hyperparameters.weight_decay)
        )
        logging.info(f"Optimizer: AdamW (lr={args.training.hyperparameters.learning_rate}, "
                    f"weight_decay={args.training.hyperparameters.weight_decay})")

    # Initialize learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.training.hyperparameters.num_warmup_steps,
        num_training_steps=total_training_steps
    )
    
    logging.info(f"Scheduler: Cosine with warmup ({args.training.hyperparameters.num_warmup_steps} warmup steps)")
    
    # Setup for DeepSpeed
    if args.training.deepspeed.config_path is not None and os.path.exists(args.training.deepspeed.config_path):
        with open(args.training.deepspeed.config_path, 'r') as f:
            ds_config = json.load(f)
        logging.info(f"Loaded DeepSpeed config from {args.training.deepspeed.config_path}")
        # Update the DeepSpeed config with the total batch size
        ds_config['train_micro_batch_size_per_gpu'] = total_batch_size
        ds_config['train_batch_size'] = total_batch_size * dist.get_world_size() * args.training.hyperparameters.gradient_accumulation_steps if dist.is_initialized() else total_batch_size * args.training.hyperparameters.gradient_accumulation_steps
        
        if precision_dtype == torch.bfloat16:
            if 'bf16' not in ds_config:
                ds_config['bf16'] = {'enabled': True}
            else:
                ds_config['bf16']['enabled'] = True 
        elif precision_dtype == torch.float16:
            if 'fp16' not in ds_config:
                ds_config['fp16'] = {'enabled': True}
            else:
                ds_config['fp16']['enabled'] = True
        
    # Save the generated config for reference
    if rank == 0:
        config_path = os.path.join(args.output_dir, "ds_config.json")
        with open(config_path, 'w') as f:
            json.dump(ds_config, f, indent=4)
        logging.info(f"Generated DeepSpeed config saved to {config_path}")
    
    if args.resume_from is not None and args.resume_from.endswith('.pt'):
        logging.info("Loading from a .pt file, so we use pytorch.")
        state_dict = torch.load(args.resume_from, map_location='cpu')
        if 'module' in state_dict:
            state_dict = state_dict['module']
        model.load_state_dict(state_dict)
        logging.info(f"Loaded checkpoint from {args.resume_from}")

    # Initialize DeepSpeed
    # CRITICAL: Pass model_parameters explicitly to ensure all trainable params are included
    # DeepSpeed needs to know which parameters to track for checkpointing
    model_parameters = list(model.parameters())
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        model_parameters=model_parameters,  # Explicitly pass all model parameters
        config=ds_config
    )
            
    start_epoch = 0
    global_step = 0
    
    # Track best loss for saving best checkpoint
    best_loss = float('inf')
    save_latest = getattr(args.training.logging, 'save_latest', True)
    save_best = getattr(args.training.logging, 'save_best', True)
    
    # Initialize TensorBoard writer
    writer = None
    if rank == 0 and args.training.logging.use_tensorboard:
        if TENSORBOARD_AVAILABLE and SummaryWriter is not None:
            writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard"))
            logging.info(f"TensorBoard logging enabled: {args.output_dir}/tensorboard")
        else:
            logging.warning("TensorBoard requested but not available. Disabling tensorboard logging.")
            args.training.logging.use_tensorboard = False
    
    # Start training
    logging.info("Starting mixed tasks training with all tasks in each iteration...")
    for epoch in range(start_epoch, args.training.hyperparameters.num_epochs):
        # Set epoch for all samplers
        for task_name, dataloader in dataloaders.items():
            if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
                dataloader.sampler.set_epoch(epoch)
                logging.info(f"Set epoch {epoch} for dataloader {task_name}")
                
        # Create iterators for all dataloaders
        iterators = {task_name: iter(dataloader) for task_name, dataloader in dataloaders.items()}
        
        # Create a progress bar based on steps_per_epoch
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{args.training.hyperparameters.num_epochs}") if rank == 0 else range(steps_per_epoch)
        
        # Track losses per task for logging
        task_losses = {task_name: [] for task_name in dataloaders.keys()}
        epoch_losses = []  # Track all losses in epoch for average calculation
        
        for step_idx in pbar:
            # Process a batch from each task and accumulate losses
            all_task_losses = {}
            weighted_loss = 0.0
            valid_task_count = 0
            
            for task_name, iterator in iterators.items():
                # Get a batch for the current task
                batch = next(iterator)
                
                # Process the batch to get necessary inputs
                batch_result = process_batch(
                    batch, task_name, device, args, rank, epoch, step_idx
                )
                videos, context, aligned_emb, ar_vision_input, visual_emb, ref_images, prompts, adapter_output_gt = batch_result
                
                # Skip if video data is invalid
                if videos is None:
                    continue
                
                batch_size, _, frames, height, width = videos.shape
                seq_len = math.ceil((height * width) / 
                                (patch_size[1] * patch_size[2]) * 
                                frames / sp_size) * sp_size
                
                # FIX: Use bfloat16 for training to match OmniVideo and improve performance
                with torch.autocast(
                    device_type='cuda', 
                    dtype=precision_dtype, 
                    enabled=args.training.precision.mixed_precision != "fp32"
                ):
                    # Uniformly sample timesteps
                    timestep = torch.randint(0, flow_scheduler.num_train_timesteps, (batch_size,))
                    t = flow_scheduler.timesteps[timestep].to(dtype=param_dtype, device=device)            

                    # Generate noise
                    noise = torch.randn_like(videos)

                    # Add noise to target video using flow matching scheduler
                    noisy_video = flow_scheduler.add_noise(videos, noise, t)

                    # Add visual embeddings as input if specified
                    if args.training.model_settings.use_visual_as_input and visual_emb is not None:
                        noisy_video = noisy_video + visual_emb

                    # Get training target (velocity field)
                    target = flow_scheduler.training_target(videos, noise, t)

                    # Get training weights for current timesteps
                    weights = flow_scheduler.training_weight(t)

                    # Full clue training (no gamma gating)
                    # Always use full clue (gamma=1.0) for best results
                    # Gamma parameter is kept for backward compatibility but not used (full clue always)
                    gamma_value = 1.0  # Always use full clue
                    
                    # Forward pass (supports both OmniVideoMixedConditionModel and MobileOVModel)
                    forward_kwargs = {
                        'x': noisy_video,
                        't': t,
                        'context': context,
                        'aligned_emb': aligned_emb,
                        'ar_vision_input': ar_vision_input,
                        'visual_emb': visual_emb,
                        'ref_images': ref_images,
                        'seq_len': seq_len,
                        'special_token_dict': special_tokens,
                        'classifier_free_ratio': args.training.classifier_free.ratio,
                        'unconditioned_context': unconditioned_context,
                        'condition_mode': args.training.model_settings.condition_mode,
                    }
                    
                    # Add MobileOVModel-specific arguments (only for MobileOVModel)
                    # Check by model instance type, not config (more reliable)
                    is_mobile_ov = isinstance(model, (MobileOVModel, MobileOVModelSANA)) or (
                        hasattr(model, 'module') and isinstance(model.module, (MobileOVModel, MobileOVModelSANA))
                    )
                    use_distill = adapter_output_gt is not None and len(adapter_output_gt) > 0 and adapter_output_gt[0] is not None
                    
                    if is_mobile_ov:
                        # Add prompts for MobileOVModel (if not using pre-computed features)
                        if not use_precomputed_features:
                            forward_kwargs['prompts'] = prompts
                        
                        # Add gamma for MobileOVModel (if supported) - always 1.0 (full clue)
                        actual_model = model.module if hasattr(model, 'module') else model
                        if hasattr(actual_model, 'forward') and 'gamma' in actual_model.forward.__code__.co_varnames:
                            forward_kwargs['gamma'] = gamma_value  # Always 1.0 (full clue)
                            if hasattr(actual_model, 'use_smol_vh') and actual_model.use_smol_vh and global_step % 100 == 0:
                                logging.info(f"Step {global_step}: Using full clue (gamma=1.0, no gating)")
                        
                        # Add return_adapter_output for distillation loss
                        if use_distill:
                            forward_kwargs['return_adapter_output'] = True
                    
                    # Forward pass
                    forward_result = model_engine(**forward_kwargs)
                    
                    # Handle return value (can be velocity_pred or (velocity_pred, adapter_output))
                    if isinstance(forward_result, tuple) and len(forward_result) == 2:
                        velocity_pred, adapter_output = forward_result
                    else:
                        velocity_pred = forward_result
                        adapter_output = None
                    
                    if isinstance(velocity_pred, list):
                        velocity_pred = torch.stack(velocity_pred, dim=0)

                    # Flow matching loss: weighted MSE between predicted and target velocity
                    if weights.ndim > 0:  # If weights are per-sample
                        weights = weights.view(-1, 1, 1, 1, 1).to(device)
                        task_loss = torch.mean(weights * (velocity_pred - target) ** 2)
                    else:  # If weights are scalar
                        task_loss = torch.nn.functional.mse_loss(velocity_pred, target)
                    
                    # Distillation loss (if adapter_output_gt is available)
                    distill_loss = None
                    if use_distill and adapter_output is not None:
                        # Get distillation loss weight from config
                        distill_weight = getattr(args.training.hyperparameters, 'distill_loss_weight', 1.0)
                        
                        # Convert adapter_output_gt to batch tensor
                        if isinstance(adapter_output_gt, list):
                            # Stack list of [64, 4096] tensors to [B, 64, 4096]
                            adapter_output_gt_batch = torch.stack(adapter_output_gt, dim=0).to(device)
                        else:
                            adapter_output_gt_batch = adapter_output_gt.to(device)
                        
                        # Ensure shapes match
                        if adapter_output.shape != adapter_output_gt_batch.shape:
                            # Truncate or pad to match
                            min_tokens = min(adapter_output.shape[1], adapter_output_gt_batch.shape[1])
                            adapter_output = adapter_output[:, :min_tokens, :]
                            adapter_output_gt_batch = adapter_output_gt_batch[:, :min_tokens, :]
                        
                        # Calculate MSE loss between adapter outputs
                        distill_loss = torch.nn.functional.mse_loss(adapter_output, adapter_output_gt_batch)
                        
                        if global_step % args.training.logging.log_interval == 0:
                            logging.info(f"Step {global_step}: Distill loss: {distill_loss.item():.6f}, Weight: {distill_weight}")
                    
                    # Combine losses
                    if distill_loss is not None:
                        distill_weight = getattr(args.training.hyperparameters, 'distill_loss_weight', 1.0)
                        total_loss = task_loss + distill_weight * distill_loss
                    else:
                        total_loss = task_loss
                    
                    # Apply task weight to the loss
                    task_weight = task_weights.get(task_name, 1.0)
                    weighted_task_loss = total_loss * task_weight
                    model_engine.backward(weighted_task_loss)
                    
                    # Accumulate losses
                    all_task_losses[task_name] = task_loss.item()
                    if distill_loss is not None:
                        all_task_losses[f"{task_name}_distill"] = distill_loss.item()
                    weighted_loss += weighted_task_loss
                    valid_task_count += 1
                    
                    # Track loss per task for monitoring
                    task_losses[task_name].append(task_loss.item())
                    if distill_loss is not None:
                        if f"{task_name}_distill" not in task_losses:
                            task_losses[f"{task_name}_distill"] = []
                        task_losses[f"{task_name}_distill"].append(distill_loss.item())
            
            # Skip optimization if no valid tasks were processed
            if valid_task_count == 0:
                logging.warning(f"Rank {rank}: No valid tasks processed in step {step_idx}, skipping.")
                continue
                
            # Backward and optimize with DeepSpeed
            model_engine.step()
            
            # Update learning rate scheduler
            if global_step % args.training.hyperparameters.gradient_accumulation_steps == 0:
                scheduler.step()
            
            global_step += 1
                        
            # Compute and synchronize average loss across all ranks every log_interval steps
            if step_idx % args.training.logging.log_interval == 0:
                # Compute average loss for each task
                avg_task_losses = {}
                for task, losses in task_losses.items():
                    if losses:  # If we have losses for this task
                        avg_task_losses[task] = sum(losses) / len(losses)
                
                # Compute average overall loss on current device
                local_avg_loss = weighted_loss.detach() / valid_task_count
                
                # Synchronize loss across all processes
                if dist.is_initialized():
                    dist.all_reduce(local_avg_loss, op=dist.ReduceOp.SUM)
                    local_avg_loss = local_avg_loss / dist.get_world_size()
                
                # Only print log on main process
                if rank == 0:
                    task_loss_str = ", ".join([f"{t}: {l:.4f}" for t, l in avg_task_losses.items()])
                    logging.info(f"Epoch {epoch}, Step {step_idx}, "
                                f"Avg Loss: {local_avg_loss.item():.4f}, Task Losses: {task_loss_str}, "
                                f"LR: {scheduler.get_last_lr()[0]:.6f}")
                    
                    # Reset task losses after logging
                    task_losses = {task_name: [] for task_name in dataloaders.keys()}
            
            # Update progress bar
            if isinstance(pbar, tqdm):
                individual_losses = {task: f"{loss:.4f}" for task, loss in all_task_losses.items()}
                pbar.set_postfix({"loss": weighted_loss.item() / valid_task_count, **individual_losses, "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
            
            # Log to tensorboard
            if rank == 0 and args.training.logging.use_tensorboard and writer is not None:
                for task_name, loss in all_task_losses.items():
                    writer.add_scalar(f"Loss/{task_name}", loss, global_step)
                writer.add_scalar("Loss/overall", weighted_loss.item() / valid_task_count, global_step)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

            # Track loss for epoch average
            if valid_task_count > 0:
                step_loss = weighted_loss.item() / valid_task_count
                epoch_losses.append(step_loss)
            
            # Calculate current average loss for checkpoint saving
            current_avg_loss = (weighted_loss.item() / valid_task_count) if valid_task_count > 0 else float('inf')
            
            # Save checkpoint using DeepSpeed
            should_save_interval = (step_idx + 1) % args.training.logging.save_interval == 0
            is_end_of_epoch = (step_idx + 1) >= steps_per_epoch
            
            # Calculate epoch average loss at end of epoch
            if is_end_of_epoch and epoch_losses:
                epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            else:
                epoch_avg_loss = current_avg_loss
            
            should_save_latest = save_latest and is_end_of_epoch
            should_save_best = save_best and is_end_of_epoch and epoch_avg_loss < best_loss
            
            if should_save_interval or should_save_latest or should_save_best:
                # Determine checkpoint type and directory
                if should_save_best:
                    checkpoint_dir = os.path.join(args.output_dir, "checkpoint_best")
                    checkpoint_tag = "best"
                    best_loss = epoch_avg_loss
                    if rank == 0:
                        logging.info(f"🎯 New best epoch loss: {best_loss:.4f}, saving best checkpoint")
                elif should_save_latest:
                    checkpoint_dir = os.path.join(args.output_dir, "checkpoint_latest")
                    checkpoint_tag = "latest"
                    if rank == 0:
                        logging.info(f"💾 Saving latest checkpoint at end of epoch {epoch}")
                else:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}_step_{step_idx}")
                    checkpoint_tag = f"epoch_{epoch}_step_{step_idx}"
                
                # Save model and optimizer state using DeepSpeed's save_checkpoint method
                # Use epoch_avg_loss for end-of-epoch checkpoints, current_avg_loss for interval checkpoints
                checkpoint_loss = epoch_avg_loss if is_end_of_epoch else current_avg_loss
                client_state = {
                    'epoch': epoch,
                    'step': step_idx,
                    'loss': checkpoint_loss
                }
                # all processes must call this
                try:
                    model_engine.save_checkpoint(
                        save_dir=checkpoint_dir,
                        client_state=client_state,
                        tag=checkpoint_tag
                    )
                    if rank == 0:
                        logging.info(f"DeepSpeed checkpoint saved to {checkpoint_dir} (tag: {checkpoint_tag})")
                        # CRITICAL: Also save vision_head separately if it exists
                        # DeepSpeed with ZeRO may not save all parameters in checkpoint
                        # Access model via model_engine.module after DeepSpeed wrapping
                        # DeepSpeed always wraps model in .module attribute
                        actual_model = model_engine.module
                        
                        # Log model structure for debugging
                        if global_step % 1000 == 0:  # Log every 1000 steps to avoid spam
                            logging.info(f"[DEBUG] Model type: {type(actual_model)}")
                            logging.info(f"[DEBUG] Has module attr: {hasattr(model_engine, 'module')}")
                            logging.info(f"[DEBUG] Has vision_head attr: {hasattr(actual_model, 'smolvlm2_vision_head')}")
                            if hasattr(actual_model, 'smolvlm2_vision_head'):
                                logging.info(f"[DEBUG] Vision head is None: {actual_model.smolvlm2_vision_head is None}")
                        
                        if hasattr(actual_model, 'smolvlm2_vision_head') and actual_model.smolvlm2_vision_head is not None:
                            try:
                                # Use checkpoint_tag for subdirectory
                                # DeepSpeed saves to checkpoint_dir/checkpoint_tag/, so we save vision head there too
                                vh_save_dir = os.path.join(checkpoint_dir, checkpoint_tag, "smolvlm2_vision_head")
                                
                                # Ensure parent directory exists and is a directory
                                parent_dir = os.path.join(checkpoint_dir, checkpoint_tag)
                                if not os.path.exists(parent_dir):
                                    os.makedirs(parent_dir, exist_ok=True)
                                elif not os.path.isdir(parent_dir):
                                    logging.error(f"❌ Parent directory exists but is not a directory: {parent_dir}")
                                    raise OSError(f"Path exists but is not a directory: {parent_dir}")
                                
                                os.makedirs(vh_save_dir, exist_ok=True)
                                
                                # Get state dict
                                vh_state_dict = actual_model.smolvlm2_vision_head.state_dict()
                                
                                # Verify state dict is not empty
                                if len(vh_state_dict) == 0:
                                    logging.error(f"❌ Vision head state dict is EMPTY!")
                                else:
                                    logging.info(f"[DEBUG] Vision head state dict has {len(vh_state_dict)} keys")
                                
                                # Save with error handling
                                vh_save_path = os.path.join(vh_save_dir, "pytorch_model.bin")
                                
                                # Double-check the path is valid before saving
                                if os.path.exists(vh_save_path) and not os.path.isfile(vh_save_path):
                                    logging.error(f"❌ Save path exists but is not a file: {vh_save_path}")
                                    raise OSError(f"Path exists but is not a file: {vh_save_path}")
                                
                                torch.save(vh_state_dict, vh_save_path)
                                
                                # Verify file was saved
                                if os.path.exists(vh_save_path):
                                    file_size = os.path.getsize(vh_save_path)
                                    logging.info(f"✅ Saved SmolVLM2VisionHead separately to {vh_save_path} (size: {file_size:,} bytes)")
                                    
                                    # Verify can load it back
                                    try:
                                        test_load = torch.load(vh_save_path, map_location='cpu')
                                        if len(test_load) == len(vh_state_dict):
                                            logging.info(f"✅ Verified: Vision head checkpoint can be loaded ({(len(test_load))} keys)")
                                        else:
                                            logging.warning(f"⚠️  Warning: Loaded checkpoint has {len(test_load)} keys, expected {len(vh_state_dict)}")
                                    except Exception as verify_e:
                                        logging.error(f"❌ Failed to verify vision head checkpoint: {verify_e}")
                                else:
                                    logging.error(f"❌ Vision head checkpoint file was not created: {vh_save_path}")
                                    
                            except (RuntimeError, OSError, Exception) as e:
                                logging.error(f"❌ Failed to save SmolVLM2VisionHead: {e}")
                                import traceback
                                logging.error(f"Traceback: {traceback.format_exc()}")
                                logging.warning("⚠️  Continuing training without saving vision_head checkpoint")
                        else:
                            logging.warning(f"⚠️  Vision head not found or is None. Has attr: {hasattr(actual_model, 'smolvlm2_vision_head')}, Is None: {hasattr(actual_model, 'smolvlm2_vision_head') and actual_model.smolvlm2_vision_head is None if hasattr(actual_model, 'smolvlm2_vision_head') else 'N/A'}")
                except (RuntimeError, OSError) as e:
                    if rank == 0:
                        logging.error(f"❌ Failed to save checkpoint at epoch {epoch}, step {step_idx}: {e}")
                        import traceback
                        logging.error(f"Traceback: {traceback.format_exc()}")
            
            # Reset epoch losses after saving checkpoints at end of epoch
            if is_end_of_epoch:
                if rank == 0 and epoch_losses:
                    epoch_avg = sum(epoch_losses) / len(epoch_losses)
                    logging.info(f"📊 Epoch {epoch} completed - Average loss: {epoch_avg:.4f}")
                epoch_losses = []  # Reset for next epoch
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    rank = int(os.getenv("RANK", 0))
    _init_logging(rank)
    args = _parse_args()
    
    # Train model
    train(args)
