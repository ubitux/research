import sys
from pathlib import Path

import matplotlib.pyplot as plt

from pal import Palette

_PAL_DISPLAY = [
    ("srgb", "sRGB", ("R'", "G'", "B'")),
    ("linear", "sRGB linear", tuple("RGB")),
    ("lab", "OkLab", tuple("Lab")),
]


def _main():
    plt.style.use("dark_background")
    fig = plt.figure()

    files = list(map(Path, sys.argv[1:]))
    nrows, ncols = len(files), len(_PAL_DISPLAY)

    for i, path in enumerate(files):
        pal = Palette.from_path(path)

        base_idx = i * ncols + 1
        axes = [
            fig.add_subplot(nrows, ncols, base_idx + j, projection="3d")
            for j in range(len(_PAL_DISPLAY))
        ]
        for ax, (field, space, labels) in zip(axes, _PAL_DISPLAY):
            ax.set_title(f"{path.stem} ({space})")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_zlabel(labels[2])
            for color in pal.colors:
                ax.plot(*getattr(color, field), "o", color=f"#{color.srgb_rgb:06x}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _main()
