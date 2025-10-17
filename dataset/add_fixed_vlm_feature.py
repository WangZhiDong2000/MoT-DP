
import os
import numpy as np
import torch
import zarr


class vlm_feature_extractor:
    def __init__(self):
        self.dataset_path = '/home/wang/projects/diffusion_policy_z/data/pusht/pusht/pusht_cchi_v7_replay.zarr'
        self.output_dir = '/home/wang/projects/diffusion_policy_z/data/pusht/pusht/vlm_features'
        self.fixed_vlm_template_path = '/home/wang/projects/diffusion_policy_z/fixed_vlm_template.pt'
        os.makedirs(self.output_dir, exist_ok=True)

        # read from zarr dataset
        dataset_root = zarr.open(self.dataset_path, 'r')
        self._image_data = dataset_root['data']['img'][:]
        self._image_data = np.moveaxis(self._image_data, -1, 1)

    def extract_and_save_features(self, chunk_size=1024):
        from tqdm import tqdm
        print(f"Loading fixed VLM features from {self.fixed_vlm_template_path}")
        fixed_features = torch.load(self.fixed_vlm_template_path)
        # fixed_features shape: (seq_len, hidden_size) or (1, seq_len, hidden_size)
        if fixed_features.ndim == 2:
            fixed_features = fixed_features[None, ...]  # (1, seq_len, hidden_size)

        num_images = self._image_data.shape[0]
        seq_len, hidden_size = fixed_features.shape[1:]
        zarr_path = os.path.join(self.output_dir, 'vlm_features.zarr')
        vlm_zarr = zarr.open(zarr_path, mode='w')
        features_arr = vlm_zarr.create(
            'features',
            shape=(num_images, seq_len, hidden_size),
            dtype='float32',
            chunks=(min(chunk_size, num_images), seq_len, hidden_size),
            overwrite=True
        )
        for start in tqdm(range(0, num_images, chunk_size), desc="Writing features"):
            end = min(start + chunk_size, num_images)
            features_arr[start:end, :, :] = np.tile(fixed_features, (end - start, 1, 1)).astype('float32')
        print(f"Saved fixed VLM features to {zarr_path}")




if __name__ == "__main__":
    extractor = vlm_feature_extractor()
    extractor.extract_and_save_features()