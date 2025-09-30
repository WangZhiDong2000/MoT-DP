import os
import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.vlm_backbone import VLMDriveBackbone
import zarr



class vlm_feature_extractor:
    def __init__(self):
        self.vlm_device = 'cuda'
        self.vlm_backbone = VLMDriveBackbone(
            model_type='qwen',
            checkpoint_path='Qwen/Qwen2.5-VL-3B-Instruct',
            device='cuda'
        )
        self.dataset_path = '/home/wang/projects/diffusion_policy_z/data/pusht/pusht/pusht_cchi_v7_replay.zarr'
        self.output_dir = '/home/wang/projects/diffusion_policy_z/data/pusht/pusht/vlm_features'
        os.makedirs(self.output_dir, exist_ok=True)

        # read from zarr dataset
        dataset_root = zarr.open(self.dataset_path, 'r')
        self._image_data = dataset_root['data']['img'][:]
        self._image_data = np.moveaxis(self._image_data, -1, 1)

    def extract_and_save_features(self, batch_size=64):
        from tqdm import tqdm
        num_images = self._image_data.shape[0]
        # 先用第一帧推理一次，获得特征 shape
        first_feature = self._generate_vlm_features_batch(self._image_data[:1])[0]
        seq_len, hidden_size = first_feature.shape
        # 创建 zarr 文件保存所有特征
        zarr_path = os.path.join(self.output_dir, 'vlm_features.zarr')
        vlm_zarr = zarr.open(zarr_path, mode='w')
        features_arr = vlm_zarr.create(
            'features',
            shape=(num_images, seq_len, hidden_size),
            dtype='float32',
            chunks=(batch_size, seq_len, hidden_size),
            overwrite=True
        )

        for start_idx in tqdm(range(0, num_images, batch_size), desc="Extracting VLM features (batch)"):
            end_idx = min(start_idx + batch_size, num_images)
            batch_images = self._image_data[start_idx:end_idx]
            batch_features = self._generate_vlm_features_batch(batch_images)
            features_arr[start_idx:end_idx, :, :] = batch_features

        self.remove_vlm_model_from_gpu()

    def _generate_vlm_features_batch(self, batch_images):
        test_text = "What actions should be taken based on this scene?"
        messages = []
        for img in batch_images:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": test_text}
                ]
            })
        text_inputs = [self.vlm_backbone.tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=True) for msg in messages]
        inputs = self.vlm_backbone.tokenizer(
            text=text_inputs,
            images=list(batch_images),
            return_tensors="pt",
            padding=True
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.vlm_backbone.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)
            batch_features = hidden_states.float().cpu().numpy()
        return batch_features

    def remove_vlm_model_from_gpu(self):
        self.vlm_backbone.model = self.vlm_backbone.model.cpu()
        del self.vlm_backbone.model
        self.vlm_backbone.model = None
        torch.cuda.empty_cache()
        print("✓ VLM model moved to CPU and GPU memory cleared")


if __name__ == "__main__":
    extractor = vlm_feature_extractor()
    extractor.extract_and_save_features(batch_size=32)