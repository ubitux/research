import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pal import ImageData, MedianCut, Palette


def mix(a, b, x):
    return (1 - x) * a + b * x


def linear(a, b, x):
    return (x - a) / (b - a)


def clamp(x, a, b):
    return min(max(x, a), b)


def _main():
    parser = argparse.ArgumentParser("Debug median cuts")
    parser.add_argument(
        "--max-colors",
        type=int,
        default=8,
        help="Maximum number of colors in the palette",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.5,
        help="How long a box is displayed",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=60,
        help="Video frame per seconds",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=180,
        help="Total rotation",
    )
    parser.add_argument("input", type=Path, help="The image to convert")
    parser.add_argument("output", type=Path, help="The destination video")
    args = parser.parse_args()

    # We don't want to show all the points to avoid lags, but enough to make it
    # represent the colors in the input colors
    mc = MedianCut(max_colors=1000)
    imd = ImageData.from_path(args.input)
    box = mc.encapsulate_all_colors(imd)
    boxes = [b for b in mc.median_cut(box)]
    display_pal = Palette.from_boxes(boxes)

    # Build up an iterative Median Cut
    mc = MedianCut(max_colors=args.max_colors, compute_mse=False)
    box = mc.encapsulate_all_colors(imd)
    all_icolors = box.colors
    box_gen = mc.median_cut(box)
    boxes = []

    cfg = {
        "font.family": "monospace",
        "font.size": 6,
    }
    plt.style.use("dark_background")
    with plt.rc_context(cfg):

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)

            nb_frames = round(args.duration * args.max_colors * args.framerate)

            prev_nb_colors = None
            fig, ax3d = None, None

            for frame in range(nb_frames):
                ratio = linear(0, nb_frames, frame)
                angle = mix(0, args.rotation, ratio)
                nb_colors = clamp(
                    round(mix(1, args.max_colors + 1, ratio)), 1, args.max_colors
                )

                if prev_nb_colors != nb_colors:
                    print(f"get next box K={nb_colors}")
                    boxes.append(next(box_gen))
                    res = 4  # arbitrary resolution
                    fig = plt.figure(figsize=(2 * res, 1 * res))
                    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
                    ax3d.set_xlabel("L")
                    ax3d.set_ylabel("a")
                    ax3d.set_zlabel("b")

                    colors_hex = [f"#{c.srgb_rgb:06x}" for c in display_pal.colors]
                    L, a, b = zip(*[c.lab for c in display_pal.colors])
                    ax3d.scatter3D(L, a, b, c=colors_hex)

                    for box in boxes:
                        color = box.get_average_color()
                        hexcolor = f"#{color.srgb_rgb:06x}"

                        ax3d.add_collection3d(
                            Poly3DCollection(
                                _get_cube_coords(box),
                                facecolors=hexcolor,
                                linewidths=1,
                                edgecolors=hexcolor,
                                alpha=0.1,
                            )
                        )

                    pal = Palette.from_boxes(boxes)
                    full_map = mc.build_colormap(all_icolors, pal)
                    output, _ = mc.quantize_image(imd, full_map)

                    ax_img = fig.add_subplot(1, 2, 2)
                    ax_img.imshow(output)
                    ax_img.set_title(
                        f"Quantized to {nb_colors} color{'s' if nb_colors > 1 else ''}"
                    )

                    prev_nb_colors = nb_colors

                assert fig is not None
                assert ax3d is not None
                ax3d.view_init(azim=angle)

                img_path = tmpdir / f"{frame:08d}.png"
                print(f"saving file {img_path}")
                fig.savefig(img_path, dpi=144)

            print(f"encoding {args.output}")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-nostats",
                    "-r",
                    str(args.framerate),
                    "-f",
                    "image2",
                    "-i",
                    tmpdir / "%08d.png",
                    "-y",
                    "-pix_fmt",
                    "yuv420p",
                    args.output,
                ]
            )


def _get_cube_coords(box):
    px, py, pz = (
        min(c.xyz[0] for c in box.colors),
        min(c.xyz[1] for c in box.colors),
        min(c.xyz[2] for c in box.colors),
    )
    qx, qy, qz = (
        max(c.xyz[0] for c in box.colors),
        max(c.xyz[1] for c in box.colors),
        max(c.xyz[2] for c in box.colors),
    )
    p = [
        (px, py, pz),
        (qx, py, pz),
        (qx, qy, pz),
        (px, qy, pz),
        (px, py, qz),
        (qx, py, qz),
        (qx, qy, qz),
        (px, qy, qz),
    ]
    return [
        [p[0], p[1], p[2], p[3]],  # bottom
        [p[0], p[1], p[5], p[4]],  # right
        [p[3], p[2], p[6], p[7]],  # left
        [p[1], p[5], p[6], p[2]],  # back
        [p[0], p[4], p[7], p[3]],  # front
        [p[4], p[5], p[6], p[7]],  # top
    ]


if __name__ == "__main__":
    _main()
