"""Draw an academic multi-panel Kimi KDA bottleneck figure.

All marks are deterministically derived from archived Nsight Systems, Nsight
Compute, and SASS CSV exports.  The script uses only the Python standard
library and emits an editable SVG; no generative image model is involved.
"""

from __future__ import annotations

import ast
import csv
from html import escape
from math import sqrt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2000, 1500

# Okabe–Ito-inspired, print-friendly palette.
INK = "#1a1a1a"
MUTED = "#5d6470"
GRID = "#d8dce2"
LIGHT = "#f4f6f8"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GOLD = "#E69F00"
GRAY = "#8b95a5"

LABELS = ["official_h12", "v16_h12", "official_h74", "v64_h74"]
DISPLAY = {
    "official_h12": "H12 · V128",
    "v16_h12": "H12 · V16",
    "official_h74": "H74 · V128",
    "v64_h74": "H74 · V64",
}


def load_ncu(path: Path) -> dict[str, dict[str, float | tuple[int, int, int]]]:
    data: dict[str, dict[str, float | tuple[int, int, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record = data.setdefault(row["label"], {})
            record[row["metric"]] = float(row["value"])
            record["grid"] = ast.literal_eval(row["grid_size"])
            record["block"] = ast.literal_eval(row["block_size"])
    return data


def load_timeline(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for name in ("official_v128", "auto_valueslice"):
        selected = [row for row in rows if row["range_name"] == name]
        final_iteration = max(int(row["iteration"]) for row in selected)
        result[name] = [row for row in selected if int(row["iteration"]) == final_iteration]
    return result


def load_ranges(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["range_name"]: float(row["projected_duration_us"])
            for row in csv.DictReader(handle)
        }


def load_sass(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["opcode"]: int(row["count"])
            for row in csv.DictReader(handle)
            if row["scope"] == "recurrence_family_total"
        }


def cta_coverage(record: dict[str, float | tuple[int, int, int]]) -> tuple[int, float]:
    grid = record["grid"]
    assert isinstance(grid, tuple)
    ctas = grid[0] * grid[1] * grid[2]
    return ctas, min(ctas, 148) / 148 * 100


class SVG:
    def __init__(self) -> None:
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            "<defs>",
            f'<marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{BLUE}"/></marker>',
            f'<marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{ORANGE}"/></marker>',
            "</defs>",
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        ]

    def rect(self, x: float, y: float, w: float, h: float, fill: str = "none",
             stroke: str = "none", sw: float = 1, rx: float = 0) -> None:
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = INK,
             sw: float = 1, dash: str | None = None, marker: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        extra += f' marker-end="url(#{marker})"' if marker else ""
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def circle(self, x: float, y: float, radius: float, fill: str, stroke: str = "#ffffff",
               sw: float = 2, opacity: float = 1) -> None:
        self.items.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def path(self, d: str, stroke: str, sw: float = 2, fill: str = "none",
             dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
        )

    def text(self, x: float, y: float, value: str, size: int = 16, fill: str = INK,
             weight: int = 400, anchor: str = "start", italic: bool = False,
             rotate: float | None = None) -> None:
        style = "italic" if italic else "normal"
        transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica Neue,Arial,sans-serif" font-size="{size}" font-weight="{weight}" font-style="{style}" fill="{fill}" text-anchor="{anchor}"{transform}>{escape(value)}</text>'
        )

    def panel(self, x: float, y: float, w: float, h: float, letter: str, title: str) -> None:
        self.text(x, y + 22, f"({letter})", 20, INK, 700)
        self.text(x + 48, y + 22, title, 20, INK, 650)
        self.line(x, y + 38, x + w, y + 38, INK, 1.2)

    def chart_frame(self, x: float, y: float, w: float, h: float) -> None:
        self.rect(x, y, w, h, "none", INK, 1)

    def finish(self) -> str:
        return "\n".join(self.items + ["</svg>"])


def linear(value: float, domain: tuple[float, float], output: tuple[float, float]) -> float:
    lo, hi = domain
    out_lo, out_hi = output
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def axes(svg: SVG, x: float, y: float, w: float, h: float,
         x_ticks: list[float], x_domain: tuple[float, float], x_label: str,
         y_ticks: list[float], y_domain: tuple[float, float], y_label: str,
         x_suffix: str = "", y_suffix: str = "") -> None:
    svg.chart_frame(x, y, w, h)
    for value in x_ticks:
        px = linear(value, x_domain, (x, x + w))
        svg.line(px, y, px, y + h, GRID, 0.8)
        svg.text(px, y + h + 23, f"{value:g}{x_suffix}", 13, MUTED, anchor="middle")
    for value in y_ticks:
        py = linear(value, y_domain, (y + h, y))
        svg.line(x, py, x + w, py, GRID, 0.8)
        svg.text(x - 10, py + 5, f"{value:g}{y_suffix}", 13, MUTED, anchor="end")
    svg.text(x + w / 2, y + h + 50, x_label, 14, INK, 500, anchor="middle")
    svg.text(x - 58, y + h / 2, y_label, 14, INK, 500, anchor="middle", rotate=-90)


def panel_timeline(svg: SVG, timeline: dict[str, list[dict[str, str]]], ranges: dict[str, float]) -> None:
    x, y, w, h = 70, 125, 900, 360
    svg.panel(x, y, w, h, "a", "GPU execution timeline: last captured iteration")
    px, py, pw, ph = x + 155, y + 78, 695, 165
    axes(svg, px, py, pw, ph, list(range(0, 701, 100)), (0, 700), "Time from first GPU kernel (µs)", [], (0, 1), "")
    colors = {"copy": GRAY, "prepare": GOLD, "recurrence": BLUE}
    lanes = [("official_v128", "Official V128", py + 30), ("auto_valueslice", "Auto → V16", py + 100)]
    recurrence_ends = {}
    for name, label, yy in lanes:
        rows = timeline[name]
        origin = min(float(row["range_relative_start_us"]) for row in rows)
        svg.text(px - 18, yy + 22, label, 14, INK, 600, anchor="end")
        for row in rows:
            start = float(row["range_relative_start_us"]) - origin
            duration = float(row["duration_us"])
            left = linear(start, (0, 700), (px, px + pw))
            width = max(4, duration / 700 * pw)
            svg.rect(left, yy, width, 30, colors[row["kernel_class"]], rx=2)
            if row["kernel_class"] == "recurrence":
                recurrence_ends[name] = (left + width, duration)
                svg.text(left + width / 2, yy + 21, f"K2 {duration:.1f} µs", 13, "#ffffff", 700, "middle")
    auto_end, auto_duration = recurrence_ends["auto_valueslice"]
    official_end, official_duration = recurrence_ends["official_v128"]
    svg.line(auto_end, py + 138, official_end, py + 138, BLUE, 1.5)
    svg.line(auto_end, py + 132, auto_end, py + 144, BLUE, 1.5)
    svg.line(official_end, py + 132, official_end, py + 144, BLUE, 1.5)
    svg.text((auto_end + official_end) / 2, py + 158, f"ΔK2 = −{official_duration-auto_duration:.1f} µs", 13, BLUE, 700, "middle")

    legend_y = y + 340
    for index, (name, color) in enumerate((("copy", GRAY), ("prepare", GOLD), ("K2 recurrence", BLUE))):
        xx = x + 170 + index * 145
        svg.rect(xx, legend_y - 12, 18, 12, color)
        svg.text(xx + 25, legend_y, name, 13, MUTED)
    official_span = ranges["official_v128"]
    auto_span = ranges["auto_valueslice"]
    reduction = (official_span - auto_span) / official_span * 100
    svg.text(x + 575, legend_y, f"5-run NVTX: {official_span/1000:.3f} → {auto_span/1000:.3f} ms  (−{reduction:.1f}%)", 14, INK, 650)


def regression(points: list[tuple[float, float]]) -> tuple[float, float]:
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    slope = numerator / denominator
    return slope, mean_y - slope * mean_x


def panel_coverage_latency(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 1040, 125, 890, 360
    svg.panel(x, y, w, h, "b", "Latency versus whole-GPU SM coverage")
    px, py, pw, ph = x + 95, y + 70, 740, 220
    axes(svg, px, py, pw, ph, [0, 25, 50, 75, 100], (0, 100), "Ideal SM coverage ceiling (%)",
         [450, 500, 550, 600, 650], (430, 680), "NCU K2 duration (µs)")
    values = {}
    for label in LABELS:
        _, cover = cta_coverage(ncu[label])
        values[label] = (cover, float(ncu[label]["Duration"]))

    slope, intercept = regression(list(values.values()))
    y0, y1 = intercept, slope * 100 + intercept
    svg.line(px, linear(y0, (430, 680), (py + ph, py)), px + pw,
             linear(y1, (430, 680), (py + ph, py)), MUTED, 1.4, "7 5")
    svg.text(px + 12, py + 19, "OLS guide (n=4; descriptive)", 12, MUTED, italic=True)

    pairs = [
        ("official_h12", "v16_h12", BLUE, "arrow-blue", "H12"),
        ("official_h74", "v64_h74", ORANGE, "arrow-orange", "H74"),
    ]
    for before, after, color, marker, group in pairs:
        bx, by = values[before]
        ax, ay = values[after]
        bx_p = linear(bx, (0, 100), (px, px + pw))
        by_p = linear(by, (430, 680), (py + ph, py))
        ax_p = linear(ax, (0, 100), (px, px + pw))
        ay_p = linear(ay, (430, 680), (py + ph, py))
        svg.line(bx_p, by_p, ax_p, ay_p, color, 2.2, marker=marker)
        svg.circle(bx_p, by_p, 7, "#ffffff", color, 2.5)
        svg.circle(ax_p, ay_p, 8, color, "#ffffff", 2)
        svg.text(bx_p + 10, by_p - 12, f"{group} V128", 13, color, 650)
        svg.text(ax_p - 8, ay_p - 13, f"{group} sliced", 13, color, 650, "end")
        gain = (by - ay) / by * 100
        svg.text((bx_p + ax_p) / 2, (by_p + ay_p) / 2 - 10, f"−{gain:.1f}%", 13, color, 700, "middle")
    svg.text(x + w - 8, y + h - 8, "Arrows denote paired measurements, not a causal fit.", 12, MUTED, italic=True, anchor="end")


def panel_throughput(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 70, 535, 900, 360
    svg.panel(x, y, w, h, "c", "Peak-throughput utilization remains low")
    px, py, pw, ph = x + 145, y + 68, 705, 232
    svg.chart_frame(px, py, pw, ph)
    for tick in (0, 25, 50, 75, 100):
        xx = linear(tick, (0, 100), (px, px + pw))
        svg.line(xx, py, xx, py + ph, GRID, 0.8, "3 4" if tick in (50, 100) else None)
        svg.text(xx, py + ph + 22, f"{tick}%", 13, MUTED, anchor="middle")
    svg.text(px + pw / 2, py + ph + 48, "Achieved throughput relative to hardware peak", 14, INK, 500, "middle")
    svg.text(px + pw - 4, py - 10, "100% = profiler hardware roof", 12, MUTED, italic=True, anchor="end")

    row_gap = 53
    for index, label in enumerate(LABELS):
        yy = py + 18 + index * row_gap
        svg.text(px - 14, yy + 15, DISPLAY[label], 13, INK, 600, anchor="end")
        for offset, metric, color in ((0, "Compute (SM) Throughput", BLUE), (20, "DRAM Throughput", ORANGE)):
            value = float(ncu[label][metric])
            width = value / 100 * pw
            svg.rect(px, yy + offset, width, 13, color)
            svg.circle(px + width, yy + offset + 6.5, 4.5, color, "#ffffff", 1)
            svg.text(px + max(width + 10, 18), yy + offset + 11, f"{value:.2f}%", 12, color, 650)
    svg.rect(x + 630, y + 15, 16, 11, BLUE)
    svg.text(x + 653, y + 26, "SM", 12, MUTED)
    svg.rect(x + 700, y + 15, 16, 11, ORANGE)
    svg.text(x + 723, y + 26, "DRAM", 12, MUTED)
    maximum = max(float(ncu[label]["Compute (SM) Throughput"]) for label in LABELS)
    svg.text(x + w - 5, y + h - 7, f"Largest observed SM utilization = {maximum:.2f}%", 12, MUTED, italic=True, anchor="end")


def panel_slopegraphs(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 1040, 535, 890, 360
    svg.panel(x, y, w, h, "d", "Paired diagnostic counters")
    facets = [
        ("L2 hit rate", "L2 Hit Rate", (45, 90), [50, 70, 90]),
        ("No eligible warp", "No Eligible", (60, 90), [60, 75, 90]),
        ("Achieved occupancy", "Achieved Occupancy", (0, 12), [0, 6, 12]),
    ]
    facet_w = 245
    for index, (title, metric, domain, ticks) in enumerate(facets):
        fx = x + 45 + index * 285
        fy, fh = y + 80, 205
        svg.text(fx + facet_w / 2, y + 62, title, 14, INK, 650, anchor="middle")
        svg.chart_frame(fx, fy, facet_w, fh)
        for tick in ticks:
            yy = linear(tick, domain, (fy + fh, fy))
            svg.line(fx, yy, fx + facet_w, yy, GRID, 0.8)
            svg.text(fx - 7, yy + 4, f"{tick:g}%", 11, MUTED, anchor="end")
        left, right = fx + 52, fx + facet_w - 52
        svg.text(left, fy + fh + 22, "V128", 12, MUTED, anchor="middle")
        svg.text(right, fy + fh + 22, "sliced", 12, MUTED, anchor="middle")
        for before, after, color, group in (
            ("official_h12", "v16_h12", BLUE, "H12"),
            ("official_h74", "v64_h74", ORANGE, "H74"),
        ):
            v1 = float(ncu[before][metric])
            v2 = float(ncu[after][metric])
            y1 = linear(v1, domain, (fy + fh, fy))
            y2 = linear(v2, domain, (fy + fh, fy))
            svg.line(left, y1, right, y2, color, 2)
            svg.circle(left, y1, 5, "#ffffff", color, 2)
            svg.circle(right, y2, 5.5, color, "#ffffff", 1.5)
            baseline_label_y = y1 - 8 if group == "H12" else y1 + 16
            svg.text(left - 7, baseline_label_y, f"{v1:.1f}", 11, color, 650, "end")
            svg.text(right + 7, y2 + 4, f"{v2:.1f} {group}", 11, color, 650)
    svg.line(x + 620, y + 24, x + 645, y + 24, BLUE, 2)
    svg.text(x + 652, y + 28, "H12", 12, MUTED)
    svg.line(x + 710, y + 24, x + 735, y + 24, ORANGE, 2)
    svg.text(x + 742, y + 28, "H74", 12, MUTED)
    svg.text(x + w - 4, y + h - 7, "Whole-GPU coverage and per-active-SM occupancy measure different layers.", 12, MUTED, italic=True, anchor="end")


def panel_resources(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 70, 945, 900, 335
    svg.panel(x, y, w, h, "e", "Launch-resource map")
    px, py, pw, ph = x + 105, y + 70, 740, 190
    axes(svg, px, py, pw, ph, [40, 60, 80, 100], (40, 105), "Dynamic shared memory / CTA (KiB)",
         [50, 60, 70, 80], (48, 80), "Registers / thread")
    unique = [
        ("official_h12", "V128 (H12/H74)", BLUE, (12, -15)),
        ("v16_h12", "V16 · H12", GREEN, (12, 20)),
        ("v64_h74", "V64 · H74", ORANGE, (12, -12)),
    ]
    points = {}
    for label, display, color, offset in unique:
        smem = float(ncu[label]["Dynamic Shared Memory Per Block"])
        registers = float(ncu[label]["Registers Per Thread"])
        block = ncu[label]["block"]
        assert isinstance(block, tuple)
        threads = block[0] * block[1] * block[2]
        xx = linear(smem, (40, 105), (px, px + pw))
        yy = linear(registers, (48, 80), (py + ph, py))
        radius = 9 + 9 * sqrt(threads / 192)
        points[label] = (xx, yy)
        svg.circle(xx, yy, radius, color, "#ffffff", 2.5, 0.9)
        svg.text(xx + offset[0], yy + offset[1], display, 13, color, 700)
    base_x, base_y = points["official_h12"]
    for label, color in (("v16_h12", GREEN), ("v64_h74", ORANGE)):
        xx, yy = points[label]
        svg.line(base_x - 10, base_y + 10, xx + 10, yy - 7, color, 1.6, "6 5")
    svg.text(px + 10, py + 18, "Bubble area ∝ threads / CTA", 12, MUTED, italic=True)
    v16_smem_drop = (float(ncu["official_h12"]["Dynamic Shared Memory Per Block"]) - float(ncu["v16_h12"]["Dynamic Shared Memory Per Block"])) / float(ncu["official_h12"]["Dynamic Shared Memory Per Block"]) * 100
    v16_reg_drop = (float(ncu["official_h12"]["Registers Per Thread"]) - float(ncu["v16_h12"]["Registers Per Thread"])) / float(ncu["official_h12"]["Registers Per Thread"]) * 100
    svg.text(x + w - 5, y + h - 7, f"V128 → V16: shared memory −{v16_smem_drop:.1f}%; registers −{v16_reg_drop:.1f}%", 12, MUTED, italic=True, anchor="end")


def blend(low: tuple[int, int, int], high: tuple[int, int, int], fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    values = [round(a + (b - a) * fraction) for a, b in zip(low, high)]
    return "#" + "".join(f"{value:02x}" for value in values)


def panel_heatmap(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 1040, 945, 890, 335
    svg.panel(x, y, w, h, "f", "Evidence matrix (raw percentages)")
    columns = [
        ("CTA / SM", "coverage", None),
        ("SM", "Compute (SM) Throughput", None),
        ("DRAM", "DRAM Throughput", None),
        ("L2 hit", "L2 Hit Rate", None),
        ("No eligible", "No Eligible", None),
        ("Occupancy", "Achieved Occupancy", None),
    ]
    left, top = x + 145, y + 82
    cell_w, cell_h = 112, 43
    for col, (title, _, _) in enumerate(columns):
        svg.text(left + col * cell_w + cell_w / 2, top - 13, title, 12, INK, 600, "middle")
    for row, label in enumerate(LABELS):
        yy = top + row * cell_h
        svg.text(left - 12, yy + 27, DISPLAY[label], 13, INK, 600, "end")
        _, cover = cta_coverage(ncu[label])
        values = [
            cover,
            float(ncu[label]["Compute (SM) Throughput"]),
            float(ncu[label]["DRAM Throughput"]),
            float(ncu[label]["L2 Hit Rate"]),
            float(ncu[label]["No Eligible"]),
            float(ncu[label]["Achieved Occupancy"]),
        ]
        for col, value in enumerate(values):
            xx = left + col * cell_w
            fill = blend((247, 251, 255), (8, 81, 156), value / 100)
            svg.rect(xx, yy, cell_w - 4, cell_h - 4, fill, "#ffffff", 1)
            text_color = "#ffffff" if value >= 58 else INK
            svg.text(xx + (cell_w - 4) / 2, yy + 26, f"{value:.1f}%", 13, text_color, 650, "middle")
    legend_x, legend_y = left, top + 4 * cell_h + 25
    for index in range(11):
        svg.rect(legend_x + index * 18, legend_y, 18, 10, blend((247, 251, 255), (8, 81, 156), index / 10))
    svg.text(legend_x, legend_y + 28, "0%", 11, MUTED)
    svg.text(legend_x + 198, legend_y + 28, "100%", 11, MUTED, anchor="end")
    svg.text(x + w - 5, y + h - 7, "Cell color encodes magnitude, not desirability.", 12, MUTED, italic=True, anchor="end")


def footer(svg: SVG, sass: dict[str, int]) -> None:
    y = 1330
    svg.line(70, y, 1930, y, INK, 1.4)
    svg.text(70, y + 34, "Primary inference", 16, INK, 700)
    svg.text(220, y + 34, "Value slicing shortens K2 by expanding whole-GPU CTA coverage; it does not require higher per-SM occupancy.", 16, INK, 500)
    svg.text(70, y + 64, "Boundary evidence", 16, INK, 700)
    svg.text(220, y + 64, "SM and DRAM remain far below peak while scheduler no-eligible cycles stay high → issue/latency + CTA-distribution bound.", 16, INK, 500)
    svg.text(70, y + 102, "Figure 1 | B300, CUDA 13.0, CC 10.3, 148 SM; T=4096, D=128, BF16. Nsys 2025.3.2; NCU 2025.3.1.", 13, MUTED)
    svg.text(70, y + 126, f"SASS: recurrence HMMA={sass['HMMA']:,}, TCGEN={sass['TCGEN']}, UTCMMA={sass['UTCMMA']}. NCU durations are compared only within the same profiler methodology.", 13, MUTED)
    svg.text(1930, y + 126, "Measured data · deterministic SVG", 13, MUTED, 600, anchor="end")


def main() -> None:
    ncu = load_ncu(ROOT / "data/k2_ncu_metrics.csv")
    timeline = load_timeline(ROOT / "data/k2_nsys_timeline.csv")
    ranges = load_ranges(ROOT / "data/k2_nsys_ranges.csv")
    sass = load_sass(ROOT / "data/sass_opcode_summary.csv")
    svg = SVG()
    svg.text(70, 48, "Kimi KDA K2 on NVIDIA B300: profiler-guided bottleneck diagnosis", 30, INK, 700)
    svg.text(70, 82, "Paired ValueSlice measurements with execution timeline, coverage–latency trend, resource ceilings, and scheduler diagnostics", 16, MUTED)
    svg.text(1930, 48, "T=4096 · D=128 · BF16", 15, MUTED, 600, anchor="end")
    panel_timeline(svg, timeline, ranges)
    panel_coverage_latency(svg, ncu)
    panel_throughput(svg, ncu)
    panel_slopegraphs(svg, ncu)
    panel_resources(svg, ncu)
    panel_heatmap(svg, ncu)
    footer(svg, sass)
    output = ROOT / "figures/kimi_kda_b300_bottleneck.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg.finish(), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
