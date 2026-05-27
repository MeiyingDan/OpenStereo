import os
import cv2
import numpy as np
import torch.utils.data as torch_data
from PIL import Image
from .dataset_template import DatasetTemplate


class ScaredDataset(DatasetTemplate):
    """SCARED endoscopic stereo dataset.

    Each line in the split txt file has 3 space-separated relative paths:
        left_rectified_path right_rectified_path disparity_path

    Disparity is stored as uint16 PNG: real_disp = pixel_value / SCALE_FACTOR.
    Default SCALE_FACTOR is 128.0 (set in your YAML via data_info.SCALE_FACTOR).
    """

    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.scale_factor = self.data_info.get('SCALE_FACTOR', 128.0)
        self.return_right_disp = self.data_info.get('RETURN_RIGHT_DISP', False)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_img_path, right_img_path, disp_img_path = full_paths

        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)

        disp_raw = cv2.imread(disp_img_path, cv2.IMREAD_UNCHANGED)
        disp_img = disp_raw.astype(np.float32) / self.scale_factor
        disp_img[disp_img == np.inf] = 0
        disp_img[np.isnan(disp_img)] = 0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }

        sample = self.transform(sample)
        sample['valid'] = sample['disp'] > 0
        sample['index'] = idx
        sample['name'] = left_img_path

        return sample
