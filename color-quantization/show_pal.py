import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from pal import ImageData, MedianCut, Palette


@dataclass
class _Spec:
    field: str
    title: str
    axis_labels: tuple[str, str, str]


def _main(files: list[Path], as_image: bool, show_2d: bool, specs: dict[str, _Spec]):
    plt.style.use("dark_background")
    fig = plt.figure()

    nrows, ncols = len(files), len(specs)
    if as_image:
        mc = MedianCut()
        ncols += 1
    else:
        mc = None

    if show_2d:
        ncols += 1

    for i, path in enumerate(files):
        base_idx = i * ncols + 1

        if mc is None:
            pal = Palette.from_path(path)
        else:
            imd = ImageData.from_path(path)
            box = mc.encapsulate_all_colors(imd)
            boxes = mc.median_cut(box)
            pal = Palette.from_boxes(boxes)

            ax = fig.add_subplot(nrows, ncols, base_idx)
            ax.imshow(imd.img)
            base_idx += 1

        if show_2d:
            ax = fig.add_subplot(nrows, ncols, base_idx)
            ax.imshow(pal.as_image2d())
            base_idx += 1

        for j, (colorspace, spec) in enumerate(specs.items()):
            ax = fig.add_subplot(nrows, ncols, base_idx + j, projection="3d")
            field = spec.field
            labels = spec.axis_labels

            ax.set_title(spec.title)
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
            if colorspace == "oklab-normed":
                ax.set_xlim([0, 1])
                ax.set_ylim([-0.5, 0.5])
                ax.set_zlim([-0.5, 0.5])
            for color in pal.colors:
                ax.plot(*getattr(color, field), "o", color=f"#{color.srgb_rgb:06x}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Display a palette")
    parser.add_argument(
        "--show-srgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show palette as sRGB",
    )
    parser.add_argument(
        "--show-2d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show 2D palette image",
    )
    parser.add_argument(
        "--show-linear",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show palette as sRGB linear",
    )
    parser.add_argument(
        "--show-oklab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show palette as OkLab",
    )
    parser.add_argument(
        "--show-oklab-normed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show palette as OkLab with normalized axis",
    )
    parser.add_argument(
        "--as-image",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Consider the input files as images (a palette will be computed)",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input files: 16x16 palette images or input images (if --as-image is specified)",
    )
    args = parser.parse_args()

    specs = {}
    if args.show_srgb:
        specs["srgb"] = _Spec("srgb", "sRGB", ("R'", "G'", "B'"))
    if args.show_linear:
        specs["linear"] = _Spec("linear", "sRGB linear", tuple("RGB"))
    if args.show_oklab:
        specs["oklab"] = _Spec("lab", "OkLab (axis scaled)", tuple("Lab"))
    if args.show_oklab_normed:
        specs["oklab-normed"] = _Spec("lab", "OkLab (axis normalized)", tuple("Lab"))

    if not specs:
        print("Palette needs at least one representation")
        sys.exit(1)

    _main(args.files, args.as_image, args.show_2d, specs)
