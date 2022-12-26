import argparse
import csv
import os
from itertools import repeat
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

from pal import ALGOS, COLORSPACES, ImageData, MedianCut

# Each field is a CSV column
_FIELDNAMES = ["ipath", "opath", "palpath", "best_mse"] + [
    f"mse_{algo}" for algo in ALGOS
]


def _quantize(args, path: Path):
    best_mse = None
    row: dict[str, Any] = dict(ipath=path)

    imd = ImageData.from_path(path)

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


def _quantize_mt(mt_args):
    args, path = mt_args
    return _quantize(args, Path(path))


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
    parser.add_argument(
        "--nb-threads",
        type=int,
        default=cpu_count(),
        help="Number of parallel threads",
    )
    parser.add_argument("files", nargs="+", help="All the images to analyze")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    quant_args = zip(repeat(args), args.files)
    with Pool(args.nb_threads) as p:
        rows = p.map(_quantize_mt, quant_args)

    with open(args.outdir / "data.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["ipath"]))


if __name__ == "__main__":
    _main()
