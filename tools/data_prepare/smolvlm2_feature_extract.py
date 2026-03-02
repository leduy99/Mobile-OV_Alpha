import argparse
import os
import sys
import glob
import pickle
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Add project root to path
# File is at: tools/data_prepare/smolvlm2_feature_extract.py
# Need to go up 3 levels to reach project root
script_dir = os.path.dirname(os.path.abspath(__file__))  # tools/data_prepare/
tools_dir = os.path.dirname(script_dir)  # tools/
project_root = os.path.dirname(tools_dir)  # project root
sys.path.insert(0, project_root)

from nets.smolvlm2 import load_smolvlm2_from_ckpt, SmolVLMModel


"""
SmolVLM2-500M Feature Extraction Script (Experiment 1)
=====================================================

Mục tiêu:
    - Thay thế nguồn 'vlm_last_hidden_states' hiện tại (Vila / LLaVA)
      bằng embedding từ SmolVLM2-500M (pure PyTorch, không cần transformers).
    - Giữ nguyên toàn bộ pipeline training / inference Omni-Video:
        * Dataset vẫn đọc các file .pkl đã chứa 'vlm_last_hidden_states'
        * Model OmniVideoMixedConditionModel, OmniVideoX2XUnified không cần sửa.

Ý tưởng:
    - Input: 1 file text (--data-file) chứa list đường dẫn tới các .pkl base feature
      (giống như pipeline VAE / AR hiện tại).
      Mỗi .pkl gốc cần có tối thiểu các key:
        - 'video_path'
        - 'prompt'
        - 'latent_feature'
        - 'text_emb'
        - 'frame_num'
    - Script này:
        - Load từng .pkl
        - Gọi SmolVLM2-500M để tính hidden states cho prompt tương ứng
        - Lưu ra .pkl mới trong --result-folder với format:
              {
                "video_path_tgt": ...,
                "prompt": ...,
                "latent_feature_tgt": ...,
                "t5_emb": ...,
                "frame_num": ...,
                "vlm_last_hidden_states": <tensor [1, L, D] từ SmolVLM2>
              }
      Format này tương thích trực tiếp với OmniVideoDataset.

Phụ thuộc:
    - KHÔNG cần transformers! Chỉ cần checkpoint đã convert (dùng convert_smolvlm2_weight.py)

Ví dụ chạy:
    # Bước 1: Convert weight (chạy 1 lần, cần conda env có transformers)
    conda activate smolvlm2
    python tools/convert_weights/convert_smolvlm2_weight.py \
        --model-id HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --output-path omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt
    
    # Bước 2: Extract features (không cần transformers)
    conda activate omnivideo
    CUDA_VISIBLE_DEVICES=0 \
    python tools/data_prepare/smolvlm2_feature_extract.py \
        --data-file path/to/base_feature_list.txt \
        --result-folder path/to/smolvlm2_feats \
        --ckpt-path omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt
"""


class BaseFeatureDataset(Dataset):
    """
    Dataset đơn giản đọc các file .pkl đã có latent + text_emb + metadata.
    Dataset này tương tự như VideoDataset trong ar_feature_extract.py
    nhưng không phụ thuộc vào LLaVA.
    """

    def __init__(self, data_file: str, result_folder: str):
        result_files = list(glob.glob(os.path.join(result_folder, "*.pkl")))
        processed_results = {
            os.path.splitext(os.path.basename(f))[0]: 1 for f in result_files
        }

        if not os.path.exists(result_folder):
            os.makedirs(result_folder, exist_ok=True)

        # file text: mỗi dòng là 1 đường dẫn tới file .pkl base feature
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"Data file not found: {data_file}\n"
                "Please provide a text file where each line is a path to a .pkl feature file.\n"
                "Example format:\n"
                "  /path/to/feature1.pkl\n"
                "  /path/to/feature2.pkl\n"
                "  /path/to/feature3.pkl"
            )
        all_items = [line.strip() for line in open(data_file, "r") if line.strip()]

        self.items: List[str] = []
        for pkl_path in tqdm(all_items, desc="Scanning base feature files"):
            item_filename = os.path.splitext(os.path.basename(pkl_path))[0]
            if item_filename not in processed_results:
                self.items.append(pkl_path)

        print(f"Total {len(all_items)} base samples, left {len(self.items)} to process")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pkl_path = self.items[index]
        try:
            item = pickle.load(open(pkl_path, mode="rb"))
            item_id = os.path.splitext(os.path.basename(pkl_path))[0]
            item["id"] = item_id
            item["_pkl_path"] = pkl_path
        except Exception as e:
            # Nếu file bị lỗi, thử file kế tiếp
            print(f"[WARN] Failed to load {pkl_path}: {e}")
            return self.__getitem__((index + 1) % len(self))

        # Kiểm tra các key cơ bản
        required_keys = ["video_path", "prompt", "latent_feature", "text_emb", "frame_num"]
        for k in required_keys:
            if k not in item:
                raise KeyError(f"Required key '{k}' not found in base feature file {pkl_path}")

        return item


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate đơn giản vì batch_size mặc định sẽ là 1.
    Viết theo kiểu tổng quát để sau này có thể mở rộng.
    """
    return {
        "ids": [s["id"] for s in samples],
        "pkl_paths": [s["_pkl_path"] for s in samples],
        "prompts": [s["prompt"] for s in samples],
        "video_paths": [s["video_path"] for s in samples],
        "latent_features": [s["latent_feature"] for s in samples],
        "text_embs": [s["text_emb"] for s in samples],
        "frame_nums": [s["frame_num"] for s in samples],
    }


def build_smolvlm2(ckpt_path: str, device: torch.device) -> SmolVLMModel:
    """
    Khởi tạo SmolVLM2-500M từ converted checkpoint (không cần transformers).

    Args:
        ckpt_path: Path to converted checkpoint file (.pt)
        device: Device to load model on
        
    Returns:
        Loaded SmolVLM2 model
    """
    print(f"[INFO] Loading SmolVLM2 model from checkpoint: {ckpt_path}")
    model = load_smolvlm2_from_ckpt(ckpt_path, device=device)
    print(f"[INFO] Model loaded successfully on {device}")
    return model


def encode_with_smolvlm2(
    model: SmolVLMModel,
    prompts: List[str],
    device: torch.device,
) -> torch.Tensor:
    """
    Encode list prompt bằng SmolVLM2, trả về last_hidden_state.

    Ở đây tạm thời chỉ dùng text (không dùng video frames) tương tự như
    script ar_feature_extract.py (images=None).
    Nếu sau này bạn muốn dùng video frames, có thể mở rộng hàm này.
    """
    # Get tokenizer from model
    tokenizer = model.get_tokenizer()
    if tokenizer is None:
        raise RuntimeError("Tokenizer not available in model checkpoint")
    
    # Tokenize prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # Reasonable max length
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    
    # Extract last hidden state
    if hasattr(outputs, "last_hidden_state"):
        last_hidden = outputs.last_hidden_state  # [B, L, D]
    else:
        raise RuntimeError("Model output does not have last_hidden_state")
    
    return last_hidden


def main(args: argparse.Namespace) -> None:
    if args.pdb_debug:
        import pdb

        pdb.set_trace()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Dataset
    dataset = BaseFeatureDataset(args.data_file, args.result_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # 2. Load SmolVLM2 from converted checkpoint
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt_path}\n"
            "Please run convert_smolvlm2_weight.py first to create the checkpoint."
        )
    model = build_smolvlm2(args.ckpt_path, device)

    os.makedirs(args.result_folder, exist_ok=True)

    # 3. Loop
    for batch in tqdm(dataloader, desc="Extracting SmolVLM2 features"):
        # Với thiết kế hiện tại, batch_size thường là 1, nhưng code này hỗ trợ >1
        ids = batch["ids"]
        prompts = batch["prompts"]
        pkl_paths = batch["pkl_paths"]

        vlm_last_hidden = encode_with_smolvlm2(
            model=model,
            prompts=prompts,
            device=device,
        )  # [B, L, D]

        # Tách từng sample trong batch để save riêng file .pkl
        for i, sample_id in enumerate(ids):
            base_pkl = pkl_paths[i]
            video_path = batch["video_paths"][i]
            latent_feature = batch["latent_features"][i]
            text_emb = batch["text_embs"][i]
            frame_num = batch["frame_nums"][i]
            prompt = prompts[i]

            # [1, L, D] để tương thích kỳ vọng OmniVideo (giống code comment)
            sample_hidden = vlm_last_hidden[i : i + 1].detach().cpu()

            saved_feat = {
                "video_path_tgt": video_path,
                "prompt": prompt,
                "latent_feature_tgt": latent_feature,
                "t5_emb": text_emb,
                "frame_num": frame_num,
                "vlm_last_hidden_states": sample_hidden,
            }

            out_path = os.path.join(args.result_folder, f"{sample_id}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(saved_feat, f)

            if args.verbose:
                print(
                    f"[INFO] Saved SmolVLM2 features for '{base_pkl}' "
                    f"-> '{out_path}', hidden shape={tuple(sample_hidden.shape)}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract SmolVLM2-500M features as 'vlm_last_hidden_states' for Omni-Video (Experiment 1)."
    )
    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to text file; each line is a path to a base .pkl feature file.",
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        required=True,
        help="Folder to save new .pkl files containing SmolVLM2 vlm_last_hidden_states.",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to converted SmolVLM2 checkpoint file (.pt). "
             "Create this using tools/convert_weights/convert_smolvlm2_weight.py",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for prompts. 1 is safest if you worry about VRAM.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pdb-debug",
        action="store_true",
        help="Enable pdb debugging and set num_workers=0.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed info per-sample.",
    )

    _args = parser.parse_args()
    if _args.pdb_debug:
        _args.num_workers = 0

    main(_args)


