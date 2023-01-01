import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from pal import Palette


@dataclass
class _Spec:
    field: str
    title: str
    axis_labels: tuple[str, str, str]


def _main(files: list[Path], specs: dict[str, _Spec]):
    plt.style.use("dark_background")
    fig = plt.figure()

    nrows, ncols = len(files), len(specs)

    for i, path in enumerate(files):
        pal = Palette.from_path(path)

        base_idx = i * ncols + 1
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

    plt.tight_layout()
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
    parser.add_argument("files", nargs="+", type=Path, help="16x16 palette files")
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

    _main(args.files, specs)
