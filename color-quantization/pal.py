import operator
from collections import Counter
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

from PIL import Image

ivec3 = tuple[int, int, int]
vec3 = tuple[float, float, float]

COLORSPACES = ["lab", "linear", "srgb"]

# <axis-selection>_<operator>_<measurement>
ALGOS = [
    "xer2_max_er2absW",
    "xer2_max_er2noW",
    "xer2_max_er2normW",
    "xer2_max_len",
    "xer2_mul_er2absW",
    "xer2_mul_er2noW",
    "xer2_mul_er2normW",
    "xer2_mul_len",
    "xer2_sum_er2absW",
    "xer2_sum_er2noW",
    "xer2_sum_er2normW",
    "xer2_sum_len",
    "xlen_max_er2absW",
    "xlen_max_er2noW",
    "xlen_max_er2normW",
    "xlen_max_len",
    "xlen_mul_er2absW",
    "xlen_mul_er2noW",
    "xlen_mul_er2normW",
    "xlen_mul_len",
    "xlen_sum_er2absW",
    "xlen_sum_er2noW",
    "xlen_sum_er2normW",
    "xlen_sum_len",
]


@dataclass(slots=True, unsafe_hash=True)
class Color:
    srgb: ivec3 = field(hash=True)
    colorspace: str = field(hash=True)
    count: int = 1

    lab: vec3 = field(init=False)
    linear: vec3 = field(init=False)
    srgbf: vec3 = field(init=False)
    xyz: vec3 = field(init=False)

    @property
    def srgb_rgb(self) -> int:
        r, g, b = self.srgb
        return r << 16 | g << 8 | b

    @property
    def srgb_bgr(self) -> int:
        """Needed because PIL is retarded, RGB means 0xBBGGRR"""
        r, g, b = self.srgb
        return b << 16 | g << 8 | r

    @classmethod
    def _srgb2linear(cls, x: float) -> float:
        """sRGB to Linear (EOTF)"""
        return x / 12.92 if x < 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    def __post_init__(self):
        # normalized sRGB
        r8, g8, b8 = self.srgb
        self.srgbf = (r8 / 255, g8 / 255, b8 / 255)

        # linear sRGB
        r = self._srgb2linear(self.srgbf[0])
        g = self._srgb2linear(self.srgbf[1])
        b = self._srgb2linear(self.srgbf[2])
        self.linear = (r, g, b)

        # OkLab
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = l ** (1 / 3)
        m_ = m ** (1 / 3)
        s_ = s ** (1 / 3)
        self.lab = (
            0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
            1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
            0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
        )

        # Select preferred colorspace
        fields = dict(lab=self.lab, linear=self.linear, srgb=self.srgbf)
        self.xyz = fields[self.colorspace]


@dataclass(slots=True)
class Box:
    """A box is a cuboid containing a set of Color"""

    colors: list[Color]
    colorspace: str
    algo: str

    sort_order: ivec3 = field(init=False)
    cut_score: float = field(init=False)
    weight: int = field(init=False)
    sorted_by: ivec3 | None = field(init=False)
    average: vec3 | None = field(init=False)

    def __post_init__(self):
        self.update_stats()

    def _get_err(self, measure: str) -> vec3:
        """
        Compute the squared error of each colors. Depending on the measure, we
        decide to honor their weight or normalize them.
        """
        self.update_average()
        r = self.average
        assert r is not None
        if measure == "er2noW":
            return (
                sum((c.xyz[0] - r[0]) ** 2 for c in self.colors),
                sum((c.xyz[1] - r[1]) ** 2 for c in self.colors),
                sum((c.xyz[2] - r[2]) ** 2 for c in self.colors),
            )
        err = (
            sum(c.count * (c.xyz[0] - r[0]) ** 2 for c in self.colors),
            sum(c.count * (c.xyz[1] - r[1]) ** 2 for c in self.colors),
            sum(c.count * (c.xyz[2] - r[2]) ** 2 for c in self.colors),
        )
        if measure == "er2normW":
            err = (err[0] / self.weight, err[1] / self.weight, err[2] / self.weight)
        return err

    def _get_ranges(self) -> vec3:
        """Compute the ranges (or lengths) of each axis of the box"""
        return (
            max(c.xyz[0] for c in self.colors) - min(c.xyz[0] for c in self.colors),
            max(c.xyz[1] for c in self.colors) - min(c.xyz[1] for c in self.colors),
            max(c.xyz[2] for c in self.colors) - min(c.xyz[2] for c in self.colors),
        )

    def update_stats(self):
        """
        Update the necessary statistics about the box, except the average
        (unless required by another metrics).
        """
        self.weight = sum(c.count for c in self.colors)
        self.average = None

        cut, axis_op, measure = self.algo.split("_")
        axis_op = {"sum": sum, "mul": prod, "max": max}[axis_op]

        er2 = self._get_err(measure) if "er2" in self.algo else None
        ranges = self._get_ranges() if "len" in self.algo else None

        # Compute cut_score according to the selected algo. This score is the
        # factor responsible for choosing the next box to split
        if measure.startswith("er2"):
            assert er2 is not None
            self.cut_score = axis_op(er2)
        elif measure == "len":
            assert ranges is not None
            self.cut_score = axis_op(ranges)
        else:
            assert False

        # Compute the sort_order according to the selected algo. This order
        # defines which axis is going to be cut.
        if cut == "xer2":
            assert er2 is not None
            self.sort_order = self.sort_columns(*er2)
        elif cut == "xlen":
            assert ranges is not None
            self.sort_order = self.sort_columns(*ranges)
        else:
            assert False

        self.sorted_by = None

    def update_average(self):
        """
        Compute the average of all the colors, usually honoring their weight.
        """
        if self.algo.endswith("noW"):
            self.average = (
                sum(c.xyz[0] for c in self.colors) / len(self.colors),
                sum(c.xyz[1] for c in self.colors) / len(self.colors),
                sum(c.xyz[2] for c in self.colors) / len(self.colors),
            )
        else:
            self.average = (
                sum(c.count * c.xyz[0] for c in self.colors) / self.weight,
                sum(c.count * c.xyz[1] for c in self.colors) / self.weight,
                sum(c.count * c.xyz[2] for c in self.colors) / self.weight,
            )

    @classmethod
    def sort_columns(cls, x: float, y: float, z: float) -> ivec3:
        """
        Get the sorting order of x, y and z, prioritizing them in this order
        in case of equality. Technically only the major column matters, but we
        make it more deterministic by specifying the full order.
        """
        if x >= y:
            if y >= z:
                return (0, 1, 2)
            if x >= z:
                return (0, 2, 1)
            return (2, 0, 1)
        if x >= z:
            return (1, 0, 2)
        if y >= z:
            return (1, 2, 0)
        return (2, 1, 0)

    @classmethod
    def _linear2srgb(cls, x: float) -> float:
        """Linear to sRGB (OETF)"""
        return x * 12.92 if x < 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

    @classmethod
    def _oklab2linear(cls, L: float, a: float, b: float):
        """Convert OkLab to Linear"""
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l = l_**3
        m = m_**3
        s = s_**3
        return (
            +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        )

    def get_average_color(self) -> Color:
        """Build the average color of the box and wrap it in a Color object"""
        self.update_average()
        assert self.average is not None

        if self.colorspace == "lab":
            linear = self._oklab2linear(*self.average)
            triplet = tuple(self._linear2srgb(linear[i]) for i in range(3))
        elif self.colorspace == "linear":
            triplet = tuple(self._linear2srgb(self.average[i]) for i in range(3))
        elif self.colorspace == "srgb":
            triplet = tuple(self.average[i] for i in range(3))
        else:
            assert False

        return Color(
            srgb=(
                max(min(round(triplet[0] * 0xFF), 0xFF), 0x00),
                max(min(round(triplet[1] * 0xFF), 0xFF), 0x00),
                max(min(round(triplet[2] * 0xFF), 0xFF), 0x00),
            ),
            colorspace=self.colorspace,
        )


@dataclass
class Palette:
    """A 16×16 set of Color"""

    colors: list[Color]

    def save(self, path: Path):
        im = Image.new(mode="RGB", size=(16, 16))
        colors = sorted(self.colors, key=lambda c: operator.itemgetter(1, 2, 0)(c.lab))
        im.putdata([c.srgb_bgr for c in colors])
        im.save(path)

    @classmethod
    def from_path(cls, path: Path):
        im = Image.open(path)
        assert im.size == (16, 16)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")

        # Ignore transparent colors
        if im.mode == "RGBA":
            px = [c[:3] for c in im.getdata() if c[3] == 0xFF]
        else:
            px = im.getdata()

        colors = [Color(c, colorspace="lab") for c in sorted(px)]
        return cls(colors)


@dataclass
class Result:
    """Result of the analysis"""

    colorspace: str
    algo: str
    max_colors: int
    refine_max_count: int
    output: Image.Image
    palette: Palette
    mse: float


@dataclass
class ImageData:
    img: Image.Image
    path: Path
    stats: Counter

    @classmethod
    def from_path(cls, path: Path):
        im = Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        print(f"building {path} stats")
        return cls(im, path, Counter(im.getdata()))


@dataclass
class MedianCut:
    """
    Heckbert's Median-Cut algorithm with various box selection and cut customizations
    """

    colorspace: str = "lab"
    algo: str = "xer2_max_er2absW"
    max_colors: int = 256
    refine_max_count: int = 0

    def __call__(self, imd: ImageData) -> Result:
        print(f"building initial box in {self.colorspace}")
        all_icolors = [
            Color(srgb[:3], self.colorspace, count=count)
            for srgb, count in imd.stats.most_common()
        ]
        box = Box(colors=all_icolors, colorspace=self.colorspace, algo=self.algo)

        print(f"start cutting initial box of {len(all_icolors)} different colors")
        boxes = self._median_cut(box)

        honor_weights = self.algo.endswith("noW")
        for i in range(1, self.refine_max_count + 1):
            print(f"running kmeans refinement {i}/{self.refine_max_count}")
            boxes, nb_changed = self._kmeans_iteration(boxes, honor_weights)
            if nb_changed == 0:
                print(f"reached best state at iteration {i}, stopping")
                break

        print(f"averaging the {len(boxes)} boxes")
        colors = [box.get_average_color() for box in boxes]
        pal = Palette(colors)

        print(f"creating colormap minimizing ΔE in {self.colorspace}")
        pal_set = set(colors)
        full_map = {
            # find closest color in palette: the one minimizing the distance squared
            c.srgb: min(
                ((self._distsq(pal_c.xyz, c.xyz), pal_c) for pal_c in pal_set),
                key=operator.itemgetter(0),
            )[1]
            for c in all_icolors
        }

        print(f"quantize image of {imd.stats.total()} colors")
        idata = imd.img.getdata()
        ocolors = [full_map[c] for c in idata]
        assert len(set(ocolors)) <= self.max_colors
        output = Image.new(mode="RGB", size=imd.img.size)
        output.putdata([c.srgb_bgr for c in ocolors])

        print("calculating MSE (in lab space) of the final image")
        icolors = [Color(srgb[:3], self.colorspace) for srgb in idata]
        assert len(icolors) == len(ocolors) == imd.stats.total()
        mse = sum(self._distsq(a.lab, b.lab) for a, b in zip(icolors, ocolors)) / len(
            icolors
        )
        print(f"MSE={mse}")

        return Result(
            self.colorspace,
            self.algo,
            self.max_colors,
            self.refine_max_count,
            output,
            pal,
            mse,
        )

    @classmethod
    def _distsq(cls, p0: vec3, p1: vec3) -> float:
        """Distance squared of 2 vectors or points"""
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        return (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2

    def _get_next_box_to_split(self, boxes: list[Box]) -> Box | None:
        """
        Select the next box to split by picking the one with the maximum
        cut_score.
        """
        if len(boxes) == self.max_colors:
            return None
        best = None
        max_score = -1
        for box in boxes:
            if len(box.colors) < 2:
                continue
            if box.cut_score > max_score:
                best = box
                max_score = box.cut_score
        return best

    def _median_cut(self, box: Box) -> list[Box]:
        """
        The core of the algorithm: split boxes until we reach our maximum
        number of colors
        """

        boxes = [box]
        while True:
            if len(box.colors) < 2:
                break

            # sort along sort_order axis if needed
            if box.sorted_by is None or box.sorted_by != box.sort_order:
                box.colors.sort(
                    key=lambda c: operator.itemgetter(*box.sort_order)(c.xyz)
                )
                box.sorted_by = box.sort_order

            # identify cut point
            if self.algo.endswith("noW"):
                cut = (len(box.colors) + 1) >> 1
            else:
                median = (box.weight + 1) >> 1
                w = 0
                cut = 0
                for i, color in enumerate(box.colors[:-2]):
                    w += color.count
                    cut = i
                    if w > median:
                        break
                cut += 1

            # split box
            sorted_by = box.sorted_by
            new_box = Box(
                colors=box.colors[:cut], colorspace=self.colorspace, algo=self.algo
            )
            new_box.sorted_by = sorted_by
            box.colors = box.colors[cut:]
            box.update_stats()
            box.sorted_by = sorted_by
            assert len(box.colors) >= 1
            assert len(new_box.colors) >= 1
            boxes.append(new_box)

            # fetch the next best candidate for splitting
            next_box = self._get_next_box_to_split(boxes)
            if next_box is None:
                break
            box = next_box

        return boxes

    def _kmeans_iteration(self, boxes: list[Box], honor_weights: bool) -> tuple[list[Box], int]:
        """
        Extremelly naive K-means iteration that can be used for refinement.
        Still experimental.
        TODO: optimize
        """
        colors = {}
        for box in boxes:
            box.update_average()
            colors.update({color: box for color in box.colors})
            box.colors = []

        box_change_count = 0
        for color, prv_box in colors.items():
            new_box = self._get_closest_box(boxes, color, honor_weights)
            new_box.colors.append(color)
            if prv_box != new_box:
                box_change_count += 1

        print(f"{box_change_count}/{len(colors)} colors changed boxes")

        for box in boxes:
            if not box.colors:
                print(f"box with {box.average=} has no color")
                box.colors.append(Color(srgb=(0, 0, 0), colorspace=box.colorspace))
            box.update_stats()
            box.update_average()

        return boxes, box_change_count

    def _get_closest_box(
        self, boxes: list[Box], color: Color, honor_weights: bool
    ) -> Box:
        """Find the closest box color actually belongs to"""
        best_d = float("inf")
        best = None
        for box in boxes:
            assert box.average is not None
            d = self._distsq(box.average, color.xyz)
            if honor_weights:
                d *= color.count
            if d < best_d:
                best = box
                best_d = d
        assert best is not None
        return best
