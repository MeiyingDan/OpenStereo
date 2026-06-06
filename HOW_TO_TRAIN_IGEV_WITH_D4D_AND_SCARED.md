# D4D + SCARED Combined Training Pipeline

## Overview

This setup trains the IGEV stereo matching model using two surgical stereo datasets together:

- **SCARED** — endoscopic stereo images with disparity maps (PNG)
- **D4D (Dresden)** — surgical stereo images with depth maps (numpy .npy)

Both datasets are merged into one big dataset during training. The model sees mixed samples from both in every epoch.

---

## How It Works

```
YAML config (igev_scared_d4d.yaml)
        │
        ▼
build_dataloader() reads DATA_INFOS list
        │
        ├── ScaredDataset → loads left, right, disparity (from PNG)
        ├── D4DDataset    → loads left, right, depth (from .npy) → converts to disparity
        │
        ▼
ConcatDataset merges both into one dataset
        │
        ▼
RandomSampler shuffles all samples together
        │
        ▼
IGEV model trains on mixed batches
```

---

## File Roles

| File | Purpose |
|------|---------|
| `cfgs/igev/igev_scared_d4d.yaml` | Training config: defines both datasets, augmentation, model, optimizer |
| `stereo/datasets/d4d_dataset.py` | Dataset class that loads D4D images and converts depth to disparity |
| `stereo/datasets/scared_dataset.py` | Dataset class that loads SCARED images and reads disparity from PNG |
| `stereo/datasets/__init__.py` | Registers D4DDataset so the framework can find it by name |
| `data/D4D/generate_splits.py` | Script to generate train/val/test split txt files from raw D4D data |

---

## D4D Dataset: Depth to Disparity Conversion

SCARED provides disparity directly. D4D provides depth in meters. IGEV predicts disparity, so D4D depth must be converted:

```
disparity = fx * baseline / depth
```

- `fx` = focal length in pixels (from camera parameters)
- `baseline` = distance between left and right cameras in meters
- `depth` = distance from camera to surface in meters

Invalid pixels (depth too close, too far, or disparity too large) are set to 0 and excluded from the loss.

---

## Output Format (Both Datasets)

Both dataset classes return the same dictionary:

```python
{
    'left':  [H, W, 3],   # left RGB image
    'right': [H, W, 3],   # right RGB image
    'disp':  [H, W],      # ground truth disparity in pixels
    'valid': [H, W],      # boolean mask, True where disparity is valid
}
```

Because the output format is identical, the model and loss function do not need to know which dataset a sample came from.

---

## Data Augmentation (Shared)

Both datasets go through the same augmentation pipeline during training:

1. **StereoColorJitter** — randomly change brightness, contrast, saturation, hue
2. **RandomErase** — randomly erase patches (simulates occlusion)
3. **RandomCrop [320, 640]** — crop to fixed size (required for batching)
4. **TransposeImage** — HWC to CHW format
5. **ToTensor** — numpy array to PyTorch tensor

---

## Split Strategy for D4D

The `generate_splits.py` script uses leave-one-specimen-out:

- **Train**: specimen 1, 2, 3, 4 (`d4d_train.txt`, every 3rd frame, ~13k)
- **Val**: first clip of each training specimen
- **Test**: specimen 5 (entire specimen held out)

Each line in the split txt file:
```
left_path right_path depth_path fx_baseline
```

---

## Steps to Run

1. Install the d4d package:
   ```bash
   git clone https://github.com/reubendocea/d4d
   cd d4d && pip install -e .
   ```

2. Generate split files:
   ```bash
   python data/D4D/generate_splits.py --d4d_root /path/to/your/D4D/data
   ```

3. Check `DATA_PATH` in `cfgs/data_basic.py`

4. Start training:
   ```bash
   python tools/train.py --cfg_file cfgs/igev/igev_scared_d4d.yaml
   ```

5. Evaluate:
   ```bash
   CKPT=output/MultiDataset/IGEV/igev_scared_d4d/default/ckpt/checkpoint_epoch_XX.pth

   python tools/eval.py --cfg_file cfgs/igev/igev_scared_d4d.yaml --eval_data_cfg_file cfgs/igev/scared_eval_igev.yaml --pretrained_model $CKPT
   python tools/eval.py --cfg_file cfgs/igev/igev_scared_d4d.yaml --eval_data_cfg_file cfgs/igev/d4d_eval_igev.yaml --pretrained_model $CKPT
   ```
