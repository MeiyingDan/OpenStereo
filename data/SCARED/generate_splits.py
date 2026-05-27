"""Generate train/val/test split txt files for SCARED dataset in OpenStereo format.

Split strategy (following scared_split_v1.json from the CREStereo fork):
  - Train datasets: 1, 2, 3, 6, 7 (exclude 4, 5 due to calibration issues)
  - Val keyframes:  dataset_1/keyframe_1, dataset_2/keyframe_1,
                    dataset_3/keyframe_1, dataset_6/keyframe_1, dataset_7/keyframe_3
  - Test datasets:  8, 9

Only frames listed in valid.csv (GT coverage >= 10%) are included.
Each output line: left_path right_path disp_path (space-separated, relative to SCARED root)
"""

import os
import numpy as np

SCARED_ROOT = "/home/meiying/Meiying_Masterarbeit/SCARED/SCARED_DATASET_processed"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DATASET_IDS = [1, 2, 3, 6, 7]  # exclude 4, 5 (calibration issues)
TEST_DATASET_IDS = [8, 9]
VAL_KEYFRAMES = {
    "dataset_1/keyframe_1",   # 197 frames
    "dataset_2/keyframe_1",   #  88 frames
    "dataset_3/keyframe_1",   # 329 frames
    "dataset_6/keyframe_1",   # 637 frames
    "dataset_7/keyframe_3",   # 408 frames
}                             # total ~1659 val, ~13055 train

def load_valid_ids(ds_dir, kf_name):
    """Load valid frame IDs from valid.csv; return None if missing."""
    csv_path = os.path.join(ds_dir, kf_name, "valid.csv")
    if not os.path.exists(csv_path):
        return None
    arr = np.loadtxt(csv_path, dtype=int)
    return set(arr.reshape(-1).tolist())  # handle single-frame case


def collect_samples(ds_id, kf_name):
    """Collect (left, right, disp) triplets for one keyframe."""
    ds_dir = os.path.join(SCARED_ROOT, f"dataset_{ds_id}")
    base = os.path.join(ds_dir, kf_name, "data")
    left_dir = os.path.join(base, "left_rectified")
    right_dir = os.path.join(base, "right_rectified")
    disp_dir = os.path.join(base, "disparity")

    if not all(os.path.isdir(d) for d in [left_dir, right_dir, disp_dir]):
        return []

    valid_ids = load_valid_ids(ds_dir, kf_name)

    samples = []
    for fname in sorted(os.listdir(left_dir)):
        if not fname.endswith(".png"):
            continue
        fid = int(fname.replace(".png", ""))
        if valid_ids is not None and fid not in valid_ids:
            continue  # skip low-coverage frames
        if os.path.exists(os.path.join(right_dir, fname)) and \
           os.path.exists(os.path.join(disp_dir, fname)):
            rel = f"dataset_{ds_id}/{kf_name}/data"
            samples.append(f"{rel}/left_rectified/{fname} "
                           f"{rel}/right_rectified/{fname} "
                           f"{rel}/disparity/{fname}")
    return samples


def main():
    splits = {"scared_train.txt": [], "scared_val.txt": [], "scared_test.txt": []}

    # train + val from datasets 1, 2, 3, 6, 7
    for ds_id in TRAIN_DATASET_IDS:
        ds_dir = os.path.join(SCARED_ROOT, f"dataset_{ds_id}")
        if not os.path.isdir(ds_dir):
            print(f"Warning: {ds_dir} not found, skipping")
            continue
        for kf_name in sorted(os.listdir(ds_dir)):
            if not kf_name.startswith("keyframe_"):
                continue
            samples = collect_samples(ds_id, kf_name)
            key = f"dataset_{ds_id}/{kf_name}"
            target = "scared_val.txt" if key in VAL_KEYFRAMES else "scared_train.txt"
            splits[target].extend(samples)

    # test from datasets 8, 9
    for ds_id in TEST_DATASET_IDS:
        ds_dir = os.path.join(SCARED_ROOT, f"dataset_{ds_id}")
        if not os.path.isdir(ds_dir):
            print(f"Warning: {ds_dir} not found, skipping")
            continue
        for kf_name in sorted(os.listdir(ds_dir)):
            if not kf_name.startswith("keyframe_"):
                continue
            splits["scared_test.txt"].extend(collect_samples(ds_id, kf_name))

    for name, lines in splits.items():
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {path}  ({len(lines)} samples)")


if __name__ == "__main__":
    main()
