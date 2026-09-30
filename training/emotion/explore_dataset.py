"""Dataset sanity check: image count per emotion, plus one sample of each.

    python explore_dataset.py --data dataset/train
"""

from __future__ import annotations

import argparse
import os

from common import check_class_names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dataset/train")
    ap.add_argument("--no-plot", action="store_true", help="counts only, no sample images window")
    args = ap.parse_args(argv)

    emotions = sorted(e for e in os.listdir(args.data) if os.path.isdir(os.path.join(args.data, e)))
    check_class_names(emotions)
    print("Images per emotion:")
    for emotion in emotions:
        print(f"  {emotion:10} {len(os.listdir(os.path.join(args.data, emotion))):6}")
    if args.no_plot:
        return 0

    import cv2
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    for i, emotion in enumerate(emotions):
        folder = os.path.join(args.data, emotion)
        image = cv2.imread(os.path.join(folder, sorted(os.listdir(folder))[0]))
        plt.subplot(1, len(emotions), i + 1)
        if image.ndim == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap="gray")
        plt.title(f"{emotion}\n{image.shape[1]}x{image.shape[0]}")
        plt.axis("off")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
