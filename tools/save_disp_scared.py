"""
Save predicted disparity maps for SCARED dataset evaluation.

Usage:
    cd /home/meiying/Meiying_Masterarbeit/OpenStereo
    python tools/save_disp_scared.py \
        --cfg_file cfgs/igev/igev_scared.yaml \
        --eval_data_cfg_file cfgs/igev/scared_eval_igev.yaml \
        --pretrained_model output/ScaredDataset/IGEV/igev_scared/default/ckpt/checkpoint_epoch_39.pth \
        --output_dir output/ScaredDataset/IGEV/disp_predictions

Output:
    - <output_dir>/<dataset_X>/<keyframe_Y>/data/disparity/<frame>.npy  (float32 disparity in pixels)
    - <output_dir>/<dataset_X>/<keyframe_Y>/data/disparity/<frame>.png  (uint16 PNG, value = disp * 128)
"""

import sys
import os
import argparse
import csv
import json
import time
import numpy as np
import torch
import datetime
import cv2
from pathlib import Path
from easydict import EasyDict

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from cfgs.data_basic import DATA_PATH_DICT

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_config():
    parser = argparse.ArgumentParser(description='Save disparity predictions for SCARED')
    parser.add_argument('--cfg_file', type=str, required=True)
    parser.add_argument('--eval_data_cfg_file', type=str, required=True)
    parser.add_argument('--pretrained_model', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--scale_factor', type=float, default=128.0)
    parser.add_argument('--no_npy', action='store_true', help='skip saving .npy files')
    parser.add_argument('--no_png', action='store_true', help='skip saving .png files')
    parser.add_argument('--skip_save', action='store_true', help='run inference only (no .npy/.png output)')
    args = parser.parse_args()

    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)
    cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model

    eval_data_yaml_config = common_utils.config_loader(args.eval_data_cfg_file)
    eval_data_cfgs = EasyDict(eval_data_yaml_config)
    cfgs.DATA_CONFIG = eval_data_cfgs.DATA_CONFIG
    cfgs.EVALUATOR = eval_data_cfgs.EVALUATOR

    for each in cfgs.DATA_CONFIG.DATA_INFOS:
        dataset_name = each.DATASET
        each.DATA_PATH = DATA_PATH_DICT[dataset_name]

    args.run_mode = 'eval'
    args.dist_mode = False
    args.workers = 0
    args.pin_memory = False
    args.exp_group_path = os.path.join(cfgs.DATA_CONFIG.DATA_INFOS[0].DATASET, cfgs.MODEL.NAME)

    return args, cfgs


def _pad_to_int(pad_value):
    if torch.is_tensor(pad_value):
        return int(pad_value.reshape(-1)[0].item())
    if isinstance(pad_value, (list, tuple)):
        return _pad_to_int(pad_value[0])
    return int(pad_value)


def unpad_disp_pred(disp_pred, pad):
    if pad is None:
        return disp_pred
    pad_top = _pad_to_int(pad[0])
    pad_right = _pad_to_int(pad[1])
    if pad_top > 0:
        disp_pred = disp_pred[:, pad_top:, :]
    if pad_right > 0:
        disp_pred = disp_pred[:, :, :-pad_right]
    return disp_pred


def save_disp_png(path, disp_np, scale_factor):
    out = np.nan_to_num(disp_np.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.maximum(out, 0.0) * scale_factor
    out = out.clip(0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), out)


@torch.no_grad()
def main():
    args, cfgs = parse_config()
    local_rank = 0
    global_rank = 0
    torch.cuda.set_device(local_rank)
    common_utils.set_random_seed(seed=0)

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'save_disp_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=local_rank)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model
    model.eval()

    eval_loader = trainer.eval_loader
    logger.info(f'Total samples: {len(trainer.eval_set)}')
    logger.info(f'Saving to: {args.output_dir}')

    skip_save = args.skip_save or (args.no_npy and args.no_png)
    runtime_rows = []
    total_infer_sec = 0.0
    wall_start = time.perf_counter()

    for i, data in enumerate(eval_loader):
        for k, v in data.items():
            data[k] = v.to(local_rank) if torch.is_tensor(v) else v

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
            model_pred = model(data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_sec = time.perf_counter() - t0

        disp_pred = model_pred['disp_pred'].squeeze(1)  # [B, H, W]
        disp_pred = unpad_disp_pred(disp_pred, data.get('pad'))

        batch_size = disp_pred.shape[0]
        for b in range(batch_size):
            disp_np = disp_pred[b].cpu().numpy().astype(np.float32)

            # Build output path from original filename
            name = data['name'][b] if isinstance(data['name'], (list, tuple)) else data['name']
            parts = Path(name).parts
            dataset_part = None
            keyframe_part = None
            frame_name = Path(name).stem
            for p in parts:
                if p.startswith('dataset_') or p.startswith('test_dataset_'):
                    dataset_part = p
                if p.startswith('keyframe_'):
                    keyframe_part = p

            per_frame_sec = infer_sec / batch_size
            total_infer_sec += per_frame_sec
            runtime_rows.append({
                'dataset': dataset_part or 'unknown',
                'keyframe': keyframe_part or 'unknown',
                'frame': f'{frame_name}.png',
                'infer_sec': per_frame_sec,
            })

            if skip_save:
                continue

            if dataset_part and keyframe_part:
                sub_dir = os.path.join(args.output_dir, dataset_part, keyframe_part, 'disparity')
            else:
                sub_dir = args.output_dir
            os.makedirs(sub_dir, exist_ok=True)
            base_path = os.path.join(sub_dir, frame_name)

            if not args.no_npy:
                np.save(base_path + '.npy', disp_np)

            if not args.no_png:
                save_disp_png(base_path + '.png', disp_np, args.scale_factor)

        if i % 100 == 0:
            logger.info(f'Processed {i}/{len(eval_loader)} batches')

    runtime_csv = os.path.join(args.output_dir, 'runtime_inference.csv')
    with open(runtime_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'keyframe', 'frame', 'infer_sec'])
        writer.writeheader()
        writer.writerows(runtime_rows)

    wall_sec = time.perf_counter() - wall_start
    summary = {
        'total_infer_frames': len(runtime_rows),
        'total_infer_sec': float(total_infer_sec),
        'avg_infer_sec_per_frame': float(total_infer_sec / len(runtime_rows)) if runtime_rows else None,
        'overall_wall_sec': float(wall_sec),
    }
    runtime_summary = os.path.join(args.output_dir, 'runtime_summary.json')
    with open(runtime_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f'Runtime summary: {summary}')
    if skip_save:
        logger.info(f'Done! Timed {len(runtime_rows)} frames (no disparity files saved)')
    else:
        logger.info(f'Done! Saved {len(trainer.eval_set)} disparity maps to {args.output_dir}')


if __name__ == '__main__':
    main()
