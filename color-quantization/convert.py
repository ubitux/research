import argparse
from pathlib import Path

from pal import ALGOS, COLORSPACES, ImageData, MedianCut


def _main():
    parser = argparse.ArgumentParser("Quantize a given image")
    parser.add_argument(
        "--algo",
        choices=ALGOS,
        default="xer2_max_er2absW",
        help="Median cut method to use",
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
        "--palette",
        type=Path,
        help="Palette output",
    )
    parser.add_argument("input", type=Path, help="The image to convert")
    parser.add_argument("output", type=Path, help="The output image")
    args = parser.parse_args()

    mc = MedianCut(args.colorspace, args.algo, args.max_colors, args.refine_max_count)
    result = mc(ImageData.from_path(args.input))
    if args.palette:
        result.palette.save(args.palette)
    result.output.save(args.output)


if __name__ == "__main__":
    _main()
