"""Generate train/val/test split txt files for D4D dataset in OpenStereo format.

Split strategy (5-fold leave-one-specimen-out, default: specimen_5 for test):
  - Train: specimen_1, specimen_2, specimen_3, specimen_4
  - Val:   first clip of each training specimen
  - Test:  specimen_5

Each output line: left_path right_path depth_path fx_baseline
(space-separated, relative to D4D root)

Requires: pip install -e . from the d4d repo (https://github.com/reubendocea/d4d)
"""

import os
import sys
import argparse

try:
    from d4d import D4D
except ImportError:
    sys.exit("Error: d4d package not installed. Run: git clone https://github.com/reubendocea/d4d && cd d4d && pip install -e .")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_clip_samples(clip, specimen_name, session_name, clip_name):
    """Generate sample lines for one clip."""
    params = clip.endoscope_params
    fx = params['fx']
    baseline = params['baseline']
    fx_baseline = fx * baseline

    left_paths = clip.left_img_paths
    right_paths = clip.right_img_paths
    depth_paths = clip.stereo_depth_paths

    n = min(len(left_paths), len(right_paths), len(depth_paths))
    samples = []
    for i in range(n):
        left_rel = os.path.relpath(left_paths[i], d4d_root)
        right_rel = os.path.relpath(right_paths[i], d4d_root)
        depth_rel = os.path.relpath(depth_paths[i], d4d_root)
        samples.append(f"{left_rel} {right_rel} {depth_rel} {fx_baseline:.6f}")
    return samples


def main():
    global d4d_root

    parser = argparse.ArgumentParser(description="Generate D4D splits for OpenStereo")
    parser.add_argument("--d4d_root", type=str, required=True,
                        help="Path to D4D dataset root directory")
    parser.add_argument("--test_specimen", type=int, default=5,
                        help="Specimen ID to use for testing (default: 5)")
    parser.add_argument("--val_clips_per_specimen", type=int, default=1,
                        help="Number of clips per specimen to hold out for validation")
    args = parser.parse_args()

    d4d_root = args.d4d_root
    dataset = D4D(d4d_root)

    train_samples = []
    val_samples = []
    test_samples = []

    for specimen in dataset:
        specimen_name = os.path.basename(specimen.path)
        # Determine specimen ID from folder name
        specimen_id = None
        for part in specimen_name.split('_'):
            if part.isdigit():
                specimen_id = int(part)
                break
        if specimen_id is None:
            # Try to extract from path
            try:
                specimen_id = int(specimen_name.replace('specimen_', ''))
            except ValueError:
                print(f"Warning: cannot determine ID for {specimen_name}, skipping")
                continue

        is_test = (specimen_id == args.test_specimen)
        val_clip_count = 0

        for session in specimen:
            session_name = os.path.basename(session.path)

            for clip_idx, clip in enumerate(session):
                clip_name = f"Clip_{clip_idx}"
                samples = get_clip_samples(clip, specimen_name, session_name, clip_name)

                if not samples:
                    continue

                if is_test:
                    test_samples.extend(samples)
                elif val_clip_count < args.val_clips_per_specimen:
                    val_samples.extend(samples)
                    val_clip_count += 1
                else:
                    train_samples.extend(samples)

    # Write split files
    for name, lines in [("d4d_train.txt", train_samples),
                        ("d4d_val.txt", val_samples),
                        ("d4d_test.txt", test_samples)]:
        path = os.path.join(OUTPUT_DIR, name)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {path}  ({len(lines)} samples)")

    print(f"\nTotal: {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test")


if __name__ == "__main__":
    main()
