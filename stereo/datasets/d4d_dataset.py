import os
import numpy as np
import torch.utils.data as torch_data
from PIL import Image
from .dataset_template import DatasetTemplate


class D4DDataset(DatasetTemplate):
    """D4D (Dresden) surgical stereo dataset.

    Each line in the split txt file has 4 space-separated values:
        left_path right_path depth_npy_path fx*baseline

    Depth is stored as .npy (float, meters). Converted to disparity via:
        disparity = fx_baseline / depth
    """

    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.max_disp = self.data_info.get('MAX_DISP', 256)
        self.min_depth = self.data_info.get('MIN_DEPTH', 0.01)
        self.max_depth = self.data_info.get('MAX_DEPTH', 0.5)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        left_rel, right_rel, depth_rel, fx_baseline_str = item
        fx_baseline = float(fx_baseline_str)

        left_path = os.path.join(self.root, left_rel)
        right_path = os.path.join(self.root, right_rel)
        depth_path = os.path.join(self.root, depth_rel)

        left_img = np.array(Image.open(left_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_path).convert('RGB'), dtype=np.float32)

        depth = np.load(depth_path).astype(np.float32)

        # Convert depth to disparity, mask invalid regions
        valid_depth = (depth > self.min_depth) & (depth < self.max_depth)
        disp = np.zeros_like(depth)
        disp[valid_depth] = fx_baseline / depth[valid_depth]

        # Clip to max disparity
        disp[disp > self.max_disp] = 0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp,
        }

        sample = self.transform(sample)
        sample['valid'] = sample['disp'] > 0
        sample['index'] = idx
        sample['name'] = left_path

        return sample
