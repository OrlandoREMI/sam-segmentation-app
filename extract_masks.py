from argparse import ArgumentParser
from PIL import Image
from pathlib import Path
from typing import List


def evaluate_mask(image: Image.Image) -> bool:
    return

def extract_masks(path: Path) -> List[Path]:
    return


if __name__ == "__main__":
    parser = ArgumentParser("extract_masks.py", description="Loads every mask \
                            image of the directory and extracts the pair \
                            (frame, mask) if mask is non empty.")
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--is_parent", action="store_true")
    args = parser.parse_args()
    

    for d in args.dirs:
        path = Path(d).resolve()
        if args.is_parent:
            dirs = path.glob("*")
        else:
            masks_dir = path.joinpath("masks")
            masks = extract_masks(masks_dir)