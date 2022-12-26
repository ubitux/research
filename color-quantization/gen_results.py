import argparse
import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from pal import ALGOS, COLORSPACES, ImageData, MedianCut

# Each field is a CSV column
_FIELDNAMES = ["ipath", "opath", "palpath", "best_mse"] + [
    f"mse_{algo}" for algo in ALGOS
]


def _quantize(args, path: Path):
    best_mse = None
    row: dict[str, Any] = dict(ipath=path)

    print(f"building {path} stats")
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    imd = ImageData(im, path, Counter(im.getdata()))

    for algo in ALGOS:
        result = MedianCut(
            args.colorspace, algo, args.max_colors, args.refine_max_count
        )(imd)

        # Unique identifier for this analysis
        uid = f"{args.colorspace}-{algo}-{args.max_colors}-{args.refine_max_count}-{path.stem}"

        # Write palette in a file
        palpath = f"pal-{uid}.png"
        print(f"saving palette to {args.outdir / palpath}")
        result.palette.save(args.outdir / palpath)

        # Write output image in a file
        opath = f"out-{uid}.png"
        print(f"saving output image to {args.outdir / opath}")
        result.output.save(args.outdir / opath)

        # Register palette and output image if it's the best MSE
        row[f"mse_{algo}"] = result.mse
        if best_mse is None or result.mse < best_mse:
            best_mse = result.mse
            row["opath"] = opath
            row["palpath"] = palpath
            row["best_mse"] = algo

    return row


def _main():
    parser = argparse.ArgumentParser(
        "Quantize, analyze and generate results of all specified images"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Destination output dir",
    )
    parser.add_argument(
        "--colorspace",
        choices=COLORSPACES,
        default="lab",
        help="Colorspace to work with",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=255,
        help="Maximum number of colors in the palette",
    )
    parser.add_argument(
        "--refine-max-count",
        type=int,
        default=0,
        help="Maximum number of K-means refinement passes",
    )
    parser.add_argument("files", nargs="+", help="All the images to analyze")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    with open(args.outdir / "data.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for path in args.files:
            writer.writerow(_quantize(args, Path(path)))


if __name__ == "__main__":
    _main()
