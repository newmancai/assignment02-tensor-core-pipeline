"""Create the Kimi KDA B300 bottleneck figure from archived profiler data.

This is a deterministic scientific plotting script.  It reads the compact CSV
exports produced by Nsight Systems and Nsight Compute and emits an editable SVG.
No generative image model is involved.
"""

from __future__ import annotations

import ast
import csv
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1800, 1280
BG = "#f6f8fb"
INK = "#14213d"
MUTED = "#5c667a"
GRID = "#dce3ed"
OFFICIAL = "#e07a2f"
SLICED = "#157a8a"
COMPUTE = "#4967c4"
DRAM = "#9a6ac1"
L2 = "#2f8f67"
STALL = "#c74f50"
OCC = "#60758f"


def load_ncu(path: Path) -> dict[str, dict[str, float | tuple[int, int, int]]]:
    result: dict[str, dict[str, float | tuple[int, int, int]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = row["label"]
            record = result.setdefault(label, {})
            record[row["metric"]] = float(row["value"])
            record["grid"] = ast.literal_eval(row["grid_size"])
            record["block"] = ast.literal_eval(row["block_size"])
    return result


def load_timeline(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for name in ("official_v128", "auto_valueslice"):
        selected = [row for row in rows if row["range_name"] == name]
        final_iteration = max(int(row["iteration"]) for row in selected)
        grouped[name] = [
            row for row in selected if int(row["iteration"]) == final_iteration
        ]
    return grouped


def load_ranges(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["range_name"]: {
                "projected_us": float(row["projected_duration_us"]),
                "original_us": float(row["original_duration_us"]),
            }
            for row in csv.DictReader(handle)
        }


def load_sass(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["opcode"]: int(row["count"])
            for row in csv.DictReader(handle)
            if row["scope"] == "recurrence_family_total"
        }


class SVG:
    def __init__(self) -> None:
        self.items = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
            "<defs><filter id=\"shadow\" x=\"-5%\" y=\"-5%\" width=\"110%\" height=\"115%\"><feDropShadow dx=\"0\" dy=\"2\" stdDeviation=\"3\" flood-color=\"#17324d\" flood-opacity=\"0.12\"/></filter></defs>",
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}"/>',
        ]

    def rect(self, x: float, y: float, w: float, h: float, fill: str, rx: int = 0,
             stroke: str = "none", sw: float = 1, opacity: float = 1) -> None:
        self.items.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = GRID,
             sw: float = 1, dash: str | None = None) -> None:
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"{dashed}/>'
        )

    def text(self, x: float, y: float, value: str, size: int = 20, fill: str = INK,
             weight: int = 400, anchor: str = "start", family: str = "Inter,Arial,sans-serif") -> None:
        self.items.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def panel(self, x: float, y: float, w: float, h: float, title: str) -> None:
        self.items.append(f'<g filter="url(#shadow)">')
        self.rect(x, y, w, h, "#ffffff", 18, "#e3e8f0")
        self.items.append("</g>")
        self.text(x + 24, y + 36, title, 23, INK, 700)

    def finish(self) -> str:
        return "\n".join(self.items + ["</svg>"])


def panel_a(svg: SVG, timeline: dict[str, list[dict[str, str]]], ranges: dict[str, dict[str, float]]) -> None:
    x, y, w, h = 40, 118, 1720, 262
    svg.panel(x, y, w, h, "A. Nsight Systems — last captured GPU iteration (T=4096, H=12, BF16)")
    axis_x, axis_w = x + 245, 1120
    axis_y = y + 210
    maximum = 700.0
    for tick in range(0, 701, 100):
        px = axis_x + axis_w * tick / maximum
        svg.line(px, y + 70, px, axis_y, GRID, 1)
        svg.text(px, axis_y + 25, str(tick), 14, MUTED, anchor="middle")
    svg.text(axis_x + axis_w / 2, axis_y + 47, "GPU time from first kernel (µs)", 15, MUTED, anchor="middle")

    colors = {"copy": "#9aa7b8", "prepare": "#e8b04d", "recurrence": "#4967c4"}
    labels = [("official_v128", "Official V128", y + 90), ("auto_valueslice", "Auto → V16", y + 152)]
    for name, label, row_y in labels:
        rows = timeline[name]
        origin = min(float(row["range_relative_start_us"]) for row in rows)
        svg.text(x + 24, row_y + 24, label, 18, INK, 650)
        for row in rows:
            klass = row["kernel_class"]
            start = float(row["range_relative_start_us"]) - origin
            duration = float(row["duration_us"])
            px = axis_x + axis_w * start / maximum
            pw = max(4, axis_w * duration / maximum)
            svg.rect(px, row_y, pw, 34, colors.get(klass, MUTED), 5)
            if klass == "recurrence":
                svg.text(px + pw / 2, row_y + 23, f'K2 {duration:.1f} µs', 15, "#ffffff", 700, "middle")

    legend_x = x + 1415
    for index, (label, color) in enumerate((("copy", colors["copy"]), ("prepare", colors["prepare"]), ("K2 recurrence", colors["recurrence"]))):
        yy = y + 72 + index * 28
        svg.rect(legend_x, yy - 14, 20, 14, color, 3)
        svg.text(legend_x + 30, yy, label, 14, MUTED)
    official = ranges["official_v128"]["projected_us"]
    automatic = ranges["auto_valueslice"]["projected_us"]
    delta = (official - automatic) / official * 100
    svg.text(legend_x, y + 177, "5-iteration projected NVTX span", 14, MUTED)
    svg.text(legend_x, y + 202, f"{official/1000:.3f} → {automatic/1000:.3f} ms", 20, INK, 700)
    svg.text(legend_x, y + 228, f"−{delta:.1f}%", 22, SLICED, 800)


def panel_b(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 40, 402, 840, 365
    svg.panel(x, y, w, h, "B. Value slicing trades per-CTA size for whole-GPU coverage")
    chart_x, chart_y, chart_w, chart_h = x + 74, y + 78, 720, 205
    maximum = 700.0
    for tick in (0, 200, 400, 600):
        py = chart_y + chart_h - chart_h * tick / maximum
        svg.line(chart_x, py, chart_x + chart_w, py, GRID)
        svg.text(chart_x - 10, py + 5, str(tick), 13, MUTED, anchor="end")
    svg.text(chart_x, y + 65, "NCU duration (µs)", 13, MUTED)

    labels = ["official_h12", "v16_h12", "official_h74", "v64_h74"]
    short = ["H12\nV128", "H12\nV16", "H74\nV128", "H74\nV64"]
    centers = [chart_x + 90, chart_x + 245, chart_x + 480, chart_x + 635]
    for index, (label, center) in enumerate(zip(labels, centers)):
        record = ncu[label]
        duration = float(record["Duration"])
        grid = record["grid"]
        assert isinstance(grid, tuple)
        ctas = grid[0] * grid[1] * grid[2]
        coverage = min(ctas, 148) / 148 * 100
        bh = chart_h * duration / maximum
        color = OFFICIAL if "official" in label else SLICED
        svg.rect(center - 35, chart_y + chart_h - bh, 70, bh, color, 7)
        svg.text(center, chart_y + chart_h - bh - 9, f"{duration:.0f}", 15, INK, 700, "middle")
        line1, line2 = short[index].split("\n")
        svg.text(center, chart_y + chart_h + 24, line1, 14, INK, 650, "middle")
        svg.text(center, chart_y + chart_h + 42, line2, 14, color, 700, "middle")
        svg.text(center, chart_y + chart_h + 66, f"{ctas} CTA · {coverage:.1f}% SM", 13, MUTED, 400, "middle")
    svg.line(chart_x + 363, chart_y - 8, chart_x + 363, chart_y + chart_h + 73, "#c7cfdb", 1, "5 5")
    h12_gain = (float(ncu["official_h12"]["Duration"]) - float(ncu["v16_h12"]["Duration"])) / float(ncu["official_h12"]["Duration"]) * 100
    h74_gain = (float(ncu["official_h74"]["Duration"]) - float(ncu["v64_h74"]["Duration"])) / float(ncu["official_h74"]["Duration"]) * 100
    svg.text(chart_x + 270, y + 65, f"H12 gain −{h12_gain:.1f}%", 14, SLICED, 750, "middle")
    svg.text(chart_x + 625, y + 65, f"H74 gain −{h74_gain:.1f}%", 14, SLICED, 750, "middle")


def panel_c(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 920, 402, 840, 365
    svg.panel(x, y, w, h, "C. Peak resources remain unsaturated after slicing")
    chart_x, chart_y, chart_w, chart_h = x + 70, y + 78, 730, 215
    maximum = 30.0
    for tick in (0, 10, 20, 30):
        py = chart_y + chart_h - chart_h * tick / maximum
        svg.line(chart_x, py, chart_x + chart_w, py, GRID)
        svg.text(chart_x - 9, py + 5, f"{tick}%", 13, MUTED, anchor="end")
    labels = ["official_h12", "v16_h12", "official_h74", "v64_h74"]
    names = ["H12 V128", "H12 V16", "H74 V128", "H74 V64"]
    for index, (label, name) in enumerate(zip(labels, names)):
        center = chart_x + 90 + index * 175
        for offset, metric, color in ((-27, "Compute (SM) Throughput", COMPUTE), (27, "DRAM Throughput", DRAM)):
            value = float(ncu[label][metric])
            bh = chart_h * value / maximum
            svg.rect(center + offset - 20, chart_y + chart_h - bh, 40, bh, color, 5)
            svg.text(center + offset, chart_y + chart_h - bh - 7, f"{value:.1f}", 13, color, 700, "middle")
        svg.text(center, chart_y + chart_h + 25, name, 14, INK, 600, "middle")
    svg.rect(x + 514, y + 32, 18, 12, COMPUTE, 2)
    svg.text(x + 540, y + 43, "SM", 13, MUTED)
    svg.rect(x + 585, y + 32, 18, 12, DRAM, 2)
    svg.text(x + 611, y + 43, "DRAM", 13, MUTED)
    svg.text(x + 425, y + h - 17, "Fastest case still uses only 26.4% SM and 11.3% DRAM throughput", 15, MUTED, 600, "middle")


def mini_metric(svg: SVG, x: float, y: float, w: float, title: str, metric: str,
                color: str, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    svg.text(x, y, title, 17, INK, 700)
    labels = ["official_h12", "v16_h12", "official_h74", "v64_h74"]
    names = ["H12 V128", "H12 V16", "H74 V128", "H74 V64"]
    for index, (label, name) in enumerate(zip(labels, names)):
        yy = y + 30 + index * 40
        value = float(ncu[label][metric])
        svg.text(x, yy + 15, name, 13, MUTED)
        svg.rect(x + 88, yy, w - 145, 17, "#edf1f6", 8)
        svg.rect(x + 88, yy, (w - 145) * value / 100, 17, color, 8)
        svg.text(x + w - 48, yy + 15, f"{value:.1f}%", 13, INK, 650, "end")


def panel_d(svg: SVG, ncu: dict[str, dict[str, float | tuple[int, int, int]]]) -> None:
    x, y, w, h = 40, 790, 1720, 286
    svg.panel(x, y, w, h, "D. Scheduler and cache evidence — global coverage ≠ per-SM occupancy")
    mini_metric(svg, x + 30, y + 65, 525, "L2 hit rate", "L2 Hit Rate", L2, ncu)
    svg.line(x + 565, y + 62, x + 565, y + h - 30, GRID)
    mini_metric(svg, x + 595, y + 65, 525, "No eligible warp", "No Eligible", STALL, ncu)
    svg.line(x + 1130, y + 62, x + 1130, y + h - 30, GRID)
    mini_metric(svg, x + 1160, y + 65, 525, "Achieved occupancy", "Achieved Occupancy", OCC, ncu)
    svg.text(x + 860, y + h - 18, "V16 is faster although achieved occupancy falls: it activates 96 CTAs across the GPU instead of only 12.", 15, SLICED, 750, "middle")


def conclusion(svg: SVG, sass: dict[str, int]) -> None:
    x, y, w, h = 40, 1098, 1720, 132
    svg.rect(x, y, w, h, "#14213d", 18)
    svg.text(x + 26, y + 36, "Bottleneck diagnosis", 21, "#ffffff", 800)
    svg.text(x + 26, y + 69, "K2 dominates the trace; slicing raises CTA/SM coverage; compute and HBM stay far below peak; scheduler idle time remains high.", 18, "#e9eef7", 500)
    svg.text(x + 26, y + 102, "⇒ Primarily issue/latency + CTA-distribution bound — not a traditional Tensor Core compute roof or HBM bandwidth roof.", 19, "#73d5c4", 800)
    svg.line(x + 1240, y + 20, x + 1240, y + h - 20, "#41506a")
    svg.text(x + 1265, y + 42, "Code-path evidence", 16, "#ffffff", 700)
    svg.text(x + 1265, y + 70, f"recurrence SASS: {sass['HMMA']:,} HMMA", 15, "#e9eef7", 600)
    svg.text(x + 1265, y + 97, f"TCGEN/UTCMMA: {sass['TCGEN']}/{sass['UTCMMA']} · grid 12 → 96 CTA", 15, "#e9eef7", 600)


def main() -> None:
    ncu = load_ncu(ROOT / "data/k2_ncu_metrics.csv")
    timeline = load_timeline(ROOT / "data/k2_nsys_timeline.csv")
    ranges = load_ranges(ROOT / "data/k2_nsys_ranges.csv")
    sass = load_sass(ROOT / "data/sass_opcode_summary.csv")
    svg = SVG()
    svg.text(40, 48, "Kimi KDA K2 on NVIDIA B300: where the time goes", 32, INK, 800)
    svg.text(40, 82, "Real Nsight Systems + Nsight Compute evidence · CUDA 13.0 · CC 10.3 · 148 SM · measured 2026-09-01", 17, MUTED, 500)
    svg.text(1760, 48, "T=4096 · D=128", 17, MUTED, 650, "end")
    panel_a(svg, timeline, ranges)
    panel_b(svg, ncu)
    panel_c(svg, ncu)
    panel_d(svg, ncu)
    conclusion(svg, sass)
    svg.text(40, 1262, "Reproducible source: k2_nsys_timeline.csv · k2_nsys_ranges.csv · k2_ncu_metrics.csv · make_bottleneck_figure.py", 13, MUTED)
    svg.text(1760, 1262, "Nsight Systems 2025.3.2 · Nsight Compute 2025.3.1", 13, MUTED, anchor="end")
    output = ROOT / "figures/kimi_kda_b300_bottleneck.svg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg.finish(), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
