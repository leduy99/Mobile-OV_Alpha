import os
import pickle
import logging
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Set, Union

logger = logging.getLogger(__name__)


class OpenVidDataset(Dataset):
    """
    Dataset for loading OpenVid-1M video-text pairs.
    
    Can work in two modes:
    1. Pre-processed mode: Load from pickle files (faster, requires pre-processing)
    2. Raw mode: Load videos and encode on-the-fly (slower, but no pre-processing needed)
    """
    
    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        preprocessed_dir: Optional[str] = None,
        use_preprocessed: bool = True,
        max_samples: Optional[int] = None,
        modality_filter: Optional[Union[str, List[str], Set[str]]] = None,
    ):
        """
        Initialize OpenVid-1M dataset.
        
        Args:
            csv_path: Path to OpenVid-1M.csv file
            video_dir: Directory containing video files
            preprocessed_dir: Directory containing pre-processed pickle files (if use_preprocessed=True)
            use_preprocessed: If True, load from pickle files. If False, load videos directly.
            max_samples: Maximum number of samples to load (for testing)
            modality_filter: Optional modality filter ("video"/"image" or list of values).
                Useful for building separate dataloaders for joint iv training.
        """
        self.video_dir = video_dir
        self.preprocessed_dir = preprocessed_dir
        self.use_preprocessed = use_preprocessed
        self.modality_filter: Optional[Set[str]] = None
        if modality_filter is not None:
            if isinstance(modality_filter, str):
                values = [modality_filter]
            else:
                values = list(modality_filter)
            normalized = {str(v).strip().lower() for v in values if str(v).strip()}
            self.modality_filter = normalized if normalized else None
        
        # Load CSV file
        logger.info(f"Loading OpenVid-1M CSV from {csv_path}")
        try:
            # Try reading with different quoting options
            try:
                df = pd.read_csv(csv_path, quoting=1, escapechar='\\')  # QUOTE_ALL with escape char
            except:
                try:
                    df = pd.read_csv(csv_path, quoting=1)  # QUOTE_ALL
                except:
                    df = pd.read_csv(csv_path)  # Default
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
        
        # Supported schemas:
        # 1) Legacy OpenVid schema: [video, caption, ...]
        # 2) Unified/mixed schema: [caption, preprocessed_path, ...] or [caption, sample_idx, ...]
        has_caption_col = "caption" in df.columns
        has_video_col = "video" in df.columns
        has_preprocessed_path_col = "preprocessed_path" in df.columns
        has_sample_idx_col = "sample_idx" in df.columns
        if not has_caption_col:
            raise ValueError("CSV file must contain 'caption' column")
        self.direct_preprocessed_mode = bool(
            use_preprocessed and (has_preprocessed_path_col or has_sample_idx_col)
        )
        if not has_video_col and not self.direct_preprocessed_mode:
            raise ValueError(
                "CSV missing required columns for legacy mode. Need either "
                "['video','caption'] or ['caption','preprocessed_path'] / ['caption','sample_idx']"
            )
        
        # Pre-scan preprocessed files to avoid per-row os.path.exists calls.
        preprocessed_index = None
        if use_preprocessed and preprocessed_dir and (not self.direct_preprocessed_mode):
            try:
                files = [f for f in os.listdir(preprocessed_dir) if f.endswith(".pkl")]
                preprocessed_index = set()
                for f in files:
                    stem = os.path.splitext(f)[0]
                    preprocessed_index.add(stem)
                    if stem.endswith("_features"):
                        preprocessed_index.add(stem[: -len("_features")])
                logger.info(f"Found {len(preprocessed_index)} preprocessed samples in {preprocessed_dir}")
            except Exception as e:
                logger.warning(f"Failed to scan preprocessed dir {preprocessed_dir}: {e}")
                preprocessed_index = None

        # Filter valid entries
        self.data = []
        for idx, row in enumerate(df.itertuples(index=False), start=0):
            if idx % 100000 == 0 and idx > 0:
                logger.info(f"Scanned {idx} rows, kept {len(self.data)} samples so far")
            if max_samples and len(self.data) >= max_samples:
                break
            
            # Validate row data
            caption_val = getattr(row, "caption", None)
            if pd.isna(caption_val):
                logger.warning(f"Skipping row {idx}: missing caption")
                continue
            caption = str(caption_val).strip()
            row_modality = None

            # New schema path: preprocessed pickle path is directly provided or derivable from sample_idx.
            if self.direct_preprocessed_mode:
                preprocessed_path = None
                row_preprocessed = getattr(row, "preprocessed_path", None)
                if row_preprocessed is not None and not pd.isna(row_preprocessed):
                    preprocessed_path = str(row_preprocessed).strip()
                if (not preprocessed_path) and preprocessed_dir and hasattr(row, "sample_idx"):
                    try:
                        sample_idx = int(getattr(row, "sample_idx"))
                        preprocessed_path = os.path.join(preprocessed_dir, f"sample_{sample_idx:08d}.pkl")
                    except Exception:
                        preprocessed_path = None
                if not preprocessed_path:
                    continue
                if not os.path.isabs(preprocessed_path) and preprocessed_dir:
                    preprocessed_path = os.path.join(preprocessed_dir, preprocessed_path)
                if not os.path.exists(preprocessed_path):
                    continue

                video_val = getattr(row, "video", None)
                video_name = ""
                if video_val is not None and not pd.isna(video_val):
                    video_name = str(video_val).strip()
                if not video_name:
                    for candidate_col in ("video_path", "media_path", "image_path", "source_id"):
                        candidate_val = getattr(row, candidate_col, None)
                        if candidate_val is not None and not pd.isna(candidate_val):
                            video_name = os.path.basename(str(candidate_val).strip())
                            break
                if not video_name:
                    video_name = os.path.basename(preprocessed_path)
                row_modality = str(getattr(row, "modality", "") or "").strip().lower()
                if not row_modality:
                    row_modality = "image" if video_name.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                    ) else "video"
                if self.modality_filter is not None and row_modality not in self.modality_filter:
                    continue

                video_path = None
                for candidate_col in ("video_path", "media_path", "image_path"):
                    candidate_val = getattr(row, candidate_col, None)
                    if candidate_val is not None and not pd.isna(candidate_val):
                        candidate_path = str(candidate_val).strip()
                        if candidate_path:
                            video_path = candidate_path
                            break
                if video_path is None and video_name:
                    candidate_path = os.path.join(video_dir, video_name)
                    if os.path.exists(candidate_path):
                        video_path = candidate_path

                sample_idx_val = getattr(row, "sample_idx", None)
                if sample_idx_val is None or pd.isna(sample_idx_val):
                    sample_idx_val = len(self.data)

                self.data.append(
                    {
                        "video_path": video_path,
                        "video_name": video_name,
                        "caption": caption,
                        "preprocessed_path": preprocessed_path,
                        "sample_idx": int(sample_idx_val),
                        "modality": row_modality,
                    }
                )
                continue

            # Legacy schema path: expect a 'video' column.
            video_val = getattr(row, "video", None)
            if pd.isna(video_val):
                logger.warning(f"Skipping row {idx}: missing video")
                continue
            video_name = str(video_val).strip()
            row_modality = str(getattr(row, "modality", "") or "").strip().lower()
            if not row_modality:
                row_modality = "image" if video_name.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".webp", ".bmp")
                ) else "video"
            if self.modality_filter is not None and row_modality not in self.modality_filter:
                continue
            
            # Skip header row if accidentally included
            if video_name.lower() == 'video' or video_name == 'video,caption,aesthetic score,motion score,temporal consistency score,camera motion,frame,fps,seconds':
                logger.warning(f"Skipping row {idx}: appears to be header row")
                continue
            
            # Validate video name format (should be a filename, not entire CSV row)
            # Video names should be short filenames, not long text
            if len(video_name) > 255:
                logger.warning(f"Skipping row {idx}: video name too long ({len(video_name)} chars): {video_name[:50]}...")
                continue
            
            # Video name should look like a filename (has extension or is short)
            if ',' in video_name and len(video_name) > 100:
                logger.warning(f"Skipping row {idx}: video name contains comma and is too long, likely malformed CSV row")
                continue
            
            # If using preprocessed mode, check for preprocessed file first
            if use_preprocessed and preprocessed_dir:
                video_basename = os.path.splitext(video_name)[0]
                candidate_names = [
                    f"{video_basename}_features.pkl",
                    f"{video_name}_features.pkl",
                    f"{video_basename}.pkl",
                    f"{video_name}.pkl",
                ]
                # If we have a pre-scanned index, use it.
                if preprocessed_index is not None:
                    if video_basename not in preprocessed_index and video_name not in preprocessed_index:
                        continue
                    preprocessed_path = None
                    for candidate_name in candidate_names:
                        candidate_path = os.path.join(preprocessed_dir, candidate_name)
                        if os.path.exists(candidate_path):
                            preprocessed_path = candidate_path
                            break
                    if preprocessed_path is None:
                        continue
                else:
                    # Fallback to filesystem checks.
                    preprocessed_path = None
                    for candidate_name in candidate_names:
                        candidate_path = os.path.join(preprocessed_dir, candidate_name)
                        if os.path.exists(candidate_path):
                            preprocessed_path = candidate_path
                            break
                    if preprocessed_path is None:
                        continue
                
                # Preprocessed file exists, add to dataset
                # Video path is optional in preprocessed mode
                video_path = os.path.join(video_dir, video_name)
                if not os.path.exists(video_path):
                    video_path = os.path.join(video_dir, f"{video_name}.mp4")
                    if not os.path.exists(video_path):
                        video_path = None  # Video not needed if preprocessed exists
                
                self.data.append({
                    'video_path': video_path,
                    'video_name': video_name,
                    'caption': caption,
                    'preprocessed_path': preprocessed_path,
                    'sample_idx': int(idx),
                    'modality': row_modality,
                })
            else:
                # Raw mode: need video file
                video_path = os.path.join(video_dir, video_name)
                if not os.path.exists(video_path):
                    video_path = os.path.join(video_dir, f"{video_name}.mp4")
                    if not os.path.exists(video_path):
                        continue
                
                self.data.append({
                    'video_path': video_path,
                    'video_name': video_name,
                    'caption': caption,
                    'preprocessed_path': None,
                    'sample_idx': int(idx),
                    'modality': row_modality,
                })
        
        if self.modality_filter is None:
            logger.info(f"Loaded {len(self.data)} valid samples from OpenVid-style CSV")
        else:
            logger.info(
                "Loaded %d valid samples from OpenVid-style CSV (modality_filter=%s)",
                len(self.data),
                sorted(self.modality_filter),
            )
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Load and return a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: Dictionary containing all required inputs for the model
        """
        if len(self.data) == 0:
            logger.error("Dataset is empty!")
            return {
                'prompt': 'Dataset is empty',
                'latent_feature': torch.zeros(1, 16, 21, 32, 32),
            }
        
        real_idx = idx % len(self.data)  # Ensure valid index
        try_idx = 0
        max_retries = min(20, len(self.data))  # Don't retry more than dataset size
        
        while try_idx < max_retries:
            try:
                item = self.data[real_idx]
                
                if self.use_preprocessed and item.get('preprocessed_path') and os.path.exists(item['preprocessed_path']):
                    # Load from preprocessed pickle file
                    with open(item['preprocessed_path'], 'rb') as f:
                        data = pickle.load(f)
                    
                    # Ensure required keys
                    if 'prompt' not in data:
                        data['prompt'] = item.get('caption', '')
                    
                    # Convert to tensors if needed
                    for key in data:
                        if hasattr(data[key], 'shape') and not isinstance(data[key], torch.Tensor):
                            data[key] = torch.tensor(data[key])
                    if "sample_idx" not in data:
                        data["sample_idx"] = int(item.get("sample_idx", real_idx))
                    else:
                        try:
                            data["sample_idx"] = int(data["sample_idx"])
                        except Exception:
                            data["sample_idx"] = int(item.get("sample_idx", real_idx))
                    if "modality" not in data:
                        data["modality"] = item.get("modality", "video")
                    
                    return data
                else:
                    # Raw mode: return video path and caption (will need on-the-fly encoding)
                    # Note: This mode requires additional processing in training loop
                    return {
                        'video_path': item.get('video_path'),
                        'prompt': item.get('caption', ''),
                        'video_name': item.get('video_name', ''),
                        'sample_idx': int(real_idx),
                        'modality': item.get('modality', 'video'),
                    }
                    
            except Exception as e:
                video_name = item.get('video_name', 'unknown') if 'item' in locals() else 'unknown'
                logger.warning(f"Error loading sample {real_idx} (video: {video_name[:50]}...): {str(e)[:200]}")
                try_idx += 1
                if try_idx < max_retries:
                    # Try next index instead of random to avoid infinite loops
                    real_idx = (real_idx + 1) % len(self.data)
                    logger.info(f"Retrying with next sample (try {try_idx + 1}/{max_retries})")
        
        # Fallback: return dummy data (should never reach here if dataset is valid)
        logger.error(f"Failed to load any sample after {try_idx} tries. Dataset size: {len(self.data)}")
        return {
            'prompt': 'Error loading sample',
            'latent_feature': torch.zeros(1, 16, 21, 32, 32),  # Dummy shape
            'sample_idx': int(real_idx),
        }


def openvid_collate_fn(batch):
    """
    Custom collate function for OpenVidDataset.
    Similar to omnivideo_collate_fn but handles OpenVid-1M format.
    """
    # Filter out None values
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        raise ValueError("Batch is empty after filtering None values")
    
    elem = batch[0]
    if elem is None:
        raise ValueError("First element in batch is None")
    
    result = {}
    
    for key in elem:
        if key == 'prompt':
            result[key] = [d[key] for d in batch]
        elif key == 'video_path' or key == 'video_name':
            # Keep as list
            result[key] = [d[key] for d in batch]
        elif key == "sample_idx":
            result[key] = torch.tensor([int(d[key]) for d in batch], dtype=torch.long)
        elif key == 'text_emb' or key == 'vlm_last_hidden_states':
            # Keep variable-length text embeddings as a list to avoid stacking errors.
            result[key] = [d[key] for d in batch]
        elif isinstance(elem[key], torch.Tensor):
            tensors = [d[key] for d in batch]
            shapes = [t.shape for t in tensors]
            if len(set(shapes)) == 1:
                result[key] = torch.stack(tensors, dim=0)
            else:
                try:
                    result[key] = torch.stack(tensors, dim=0)
                except RuntimeError as e:
                    if "stack expects each tensor to be equal size" in str(e):
                        # Use first tensor's shape as target and match each dimension.
                        target_shape = shapes[0]
                        corrected_tensors = []
                        for tensor in tensors:
                            if tensor.shape == target_shape:
                                corrected_tensors.append(tensor)
                                continue
                            fixed = tensor
                            for dim, target_dim in enumerate(target_shape):
                                cur_dim = fixed.shape[dim]
                                if cur_dim > target_dim:
                                    fixed = fixed.narrow(dim, 0, target_dim)
                                elif cur_dim < target_dim:
                                    if cur_dim <= 0:
                                        raise RuntimeError(
                                            f"Cannot pad empty tensor for key={key}, shape={tuple(tensor.shape)}"
                                        )
                                    pad_num = target_dim - cur_dim
                                    tail = fixed.select(dim, cur_dim - 1).unsqueeze(dim)
                                    repeat_shape = [1] * fixed.dim()
                                    repeat_shape[dim] = pad_num
                                    tail = tail.repeat(*repeat_shape)
                                    fixed = torch.cat([fixed, tail], dim=dim)
                            corrected_tensors.append(fixed)
                        result[key] = torch.stack(corrected_tensors, dim=0)
                    else:
                        raise e
        else:
            result[key] = [d[key] for d in batch]
    
    return result


def create_openvid_dataloader(
    csv_path: str,
    video_dir: str,
    preprocessed_dir: Optional[str] = None,
    use_preprocessed: bool = True,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    max_samples: Optional[int] = None,
):
    """
    Create DataLoader for OpenVid-1M dataset.
    
    Args:
        csv_path: Path to OpenVid-1M.csv
        video_dir: Directory containing video files
        preprocessed_dir: Directory with preprocessed pickle files
        use_preprocessed: Whether to use preprocessed features
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        distributed: Whether using distributed training
        rank: Process rank
        world_size: Total number of processes
        max_samples: Maximum samples to load (for testing)
        
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = OpenVidDataset(
        csv_path=csv_path,
        video_dir=video_dir,
        preprocessed_dir=preprocessed_dir,
        use_preprocessed=use_preprocessed,
        max_samples=max_samples,
    )
    
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=openvid_collate_fn
    )
    
    return dataloader
