from argparse import ArgumentParser
from PIL import Image
from pathlib import Path
from typing import List
import shutil
import numpy as np


def evaluate_mask(image: Image.Image, min_valid_pct: float) -> bool:
    """Returns True if the mask has at least min_valid_pct% of non-black pixels."""
    arr = np.array(image)
    valid_pixels = np.count_nonzero(arr)
    total_pixels = arr.size
    return (valid_pixels / total_pixels) >= min_valid_pct


def extract_masks(masks_dir: Path, min_valid_pct: float) -> List[Path]:
    """Returns paths of masks that pass the minimum valid pixel threshold."""
    valid_masks = []
    for mask_path in sorted(masks_dir.iterdir()):
        if not mask_path.is_file():
            continue
        try:
            img = Image.open(mask_path)
            if evaluate_mask(img, min_valid_pct):
                valid_masks.append(mask_path)
        except Exception as e:
            print(f"[WARN] Could not read mask {mask_path}: {e}")
    return valid_masks


def copy_pair(frame_path: Path, mask_path: Path, output_dir: Path, sample_name: str) -> None:
    """Copies a (frame, mask) pair preserving the sample/frames|masks structure."""
    dest_frames = output_dir / sample_name / "frames"
    dest_masks  = output_dir / sample_name / "masks"
    dest_frames.mkdir(parents=True, exist_ok=True)
    dest_masks.mkdir(parents=True, exist_ok=True)

    shutil.copy2(frame_path, dest_frames / frame_path.name)
    shutil.copy2(mask_path,  dest_masks  / mask_path.name)


if __name__ == "__main__":
    parser = ArgumentParser(
        "extract_masks.py",
        description=(
            "Loads every mask image of the directory and extracts the pair "
            "(frame, mask) if the mask contains enough valid (non-black) pixels."
        ),
    )
    parser.add_argument("dirs", nargs="+", help="Input directories to process.")
    parser.add_argument(
        "--is_parent",
        action="store_true",
        help="If set, each dir is a parent containing multiple sample subdirs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Destination directory where filtered pairs will be copied.",
    )
    parser.add_argument(
        "--min_valid_pct",
        type=float,
        default=0.01,
        help="Minimum fraction of non-black pixels for a mask to be kept (default: 0.01 = 1%%).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    # Collect all sample directories to process
    sample_dirs: List[Path] = []
    for d in args.dirs:
        path = Path(d).resolve()
        if args.is_parent:
            sample_dirs.extend(p for p in sorted(path.iterdir()) if p.is_dir())
        else:
            sample_dirs.append(path)

    total_kept = 0
    total_seen = 0
    skipped: List[tuple[str, str]] = []  # (sample_name, reason)

    for sample_dir in sample_dirs:
        masks_dir  = sample_dir / "masks"
        frames_dir = sample_dir / "frames"

        if not masks_dir.exists():
            print(f"[SKIP] No masks/ dir found in {sample_dir}")
            skipped.append((sample_dir.name, "missing masks/"))
            continue
        if not frames_dir.exists():
            print(f"[SKIP] No frames/ dir found in {sample_dir}")
            skipped.append((sample_dir.name, "missing frames/"))
            continue

        valid_masks = extract_masks(masks_dir, args.min_valid_pct)
        total_seen += sum(1 for _ in masks_dir.iterdir() if _.is_file())

        missing_frames = []
        for mask_path in valid_masks:
            # Match frame by stem (same filename, any extension)
            matches = list(frames_dir.glob(f"{mask_path.stem}.*"))
            if not matches:
                missing_frames.append(mask_path.name)
                continue
            frame_path = matches[0]
            copy_pair(frame_path, mask_path, output_dir, sample_dir.name)
            total_kept += 1

        if missing_frames:
            print(f"[WARN] {sample_dir.name}: {len(missing_frames)} mask(s) had no matching frame — {missing_frames[:5]}{'...' if len(missing_frames) > 5 else ''}")

        print(f"[OK]   {sample_dir.name}: {len(valid_masks)} / {sum(1 for _ in masks_dir.iterdir() if _.is_file())} masks kept")

    print(f"\nDone. {total_kept} pairs copied to {output_dir}  ({total_seen} masks evaluated)")
    if skipped:
        print(f"Skipped {len(skipped)} sample(s):")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")