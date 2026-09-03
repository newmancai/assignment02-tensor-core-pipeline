#!/usr/bin/env python3
"""Build the final C1 report figures from the checked-in raw measurements.

The script intentionally has no plotting-framework dependency.  It uses Pillow
for high-resolution PNG output and emits the corresponding SVG primitives from
the same drawing calls, so both formats are generated from exactly the same
numbers and geometry.

Run with the workspace Python (it includes Pillow)::

    python3 make_final_figures.py

All measured values are parsed from data/raw.  The only analytical constants
are documented where used (for example, the exact Phase-6 atom share 128/416 in
the Amdahl sensitivity model).
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - actionable message for users
    raise SystemExit(
        "Pillow is required. Run with the bundled workspace Python documented "
        "in this file's module docstring."
    ) from exc


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "figures"
SUMMARY_CSV = ROOT / "data" / "summary_metrics.csv"
SUMMARY_JSON = ROOT / "data" / "summary_metrics.json"

W, H = 1600, 1000
SCALE = 2

INK = "#183044"
MUTED = "#657786"
GRID = "#DCE4E8"
LIGHT = "#F4F7F8"
WHITE = "#FFFFFF"
NAVY = "#15557A"
TEAL = "#128C8C"
ORANGE = "#F28E2B"
RED = "#D4504C"
GREEN = "#4B956A"
PURPLE = "#7467A8"
CYAN = "#63B7C6"

FONT_REGULAR = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
FONT_BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def _mix(a: str, b: str, t: float) -> str:
    ar, ag, ab = _hex(a)
    br, bg, bb = _hex(b)
    t = min(1.0, max(0.0, t))
    return "#%02X%02X%02X" % (
        round(ar + (br - ar) * t),
        round(ag + (bg - ag) * t),
        round(ab + (bb - ab) * t),
    )


class Canvas:
    """Small dual PNG/SVG drawing surface with top-left coordinates."""

    def __init__(self, width: int = W, height: int = H):
        self.width, self.height = width, height
        self.image = Image.new("RGB", (width * SCALE, height * SCALE), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">',
            f'<rect width="{width}" height="{height}" fill="{WHITE}"/>',
        ]
        self._font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}

    def font(self, size: int, bold: bool = False):
        key = (size, bold)
        if key not in self._font_cache:
            path = FONT_BOLD if bold else FONT_REGULAR
            self._font_cache[key] = ImageFont.truetype(str(path), size * SCALE)
        return self._font_cache[key]

    def line(self, points, color=INK, width=2, dash: tuple[int, int] | None = None):
        pts = [(round(x * SCALE), round(y * SCALE)) for x, y in points]
        if dash is None:
            self.draw.line(pts, fill=color, width=max(1, round(width * SCALE)), joint="curve")
        else:
            for p1, p2 in zip(points[:-1], points[1:]):
                x1, y1 = p1
                x2, y2 = p2
                dist = math.hypot(x2 - x1, y2 - y1)
                if not dist:
                    continue
                pos = 0.0
                on = True
                while pos < dist:
                    seg = dash[0 if on else 1]
                    end = min(dist, pos + seg)
                    if on:
                        a = pos / dist
                        b = end / dist
                        self.line(
                            [(x1 + (x2 - x1) * a, y1 + (y2 - y1) * a),
                             (x1 + (x2 - x1) * b, y1 + (y2 - y1) * b)],
                            color, width,
                        )
                    pos = end
                    on = not on
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        dash_attr = f' stroke-dasharray="{dash[0]},{dash[1]}"' if dash else ""
        self.svg.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round"{dash_attr}/>'
        )

    def rect(self, x, y, w, h, fill=WHITE, stroke=None, width=1, radius=0):
        box = tuple(round(v * SCALE) for v in (x, y, x + w, y + h))
        if radius:
            self.draw.rounded_rectangle(
                box, radius=round(radius * SCALE), fill=fill, outline=stroke,
                width=max(1, round(width * SCALE)),
            )
            self.svg.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                f'fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>'
            )
        else:
            self.draw.rectangle(
                box, fill=fill, outline=stroke, width=max(1, round(width * SCALE))
            )
            self.svg.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                f'fill="{fill}" stroke="{stroke or "none"}" stroke-width="{width}"/>'
            )

    def circle(self, x, y, r, fill=WHITE, stroke=INK, width=2):
        box = tuple(round(v * SCALE) for v in (x - r, y - r, x + r, y + r))
        self.draw.ellipse(box, fill=fill, outline=stroke, width=max(1, round(width * SCALE)))
        self.svg.append(
            f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"/>'
        )

    def polygon(self, points, fill, stroke=None, width=1):
        pts = [(round(x * SCALE), round(y * SCALE)) for x, y in points]
        self.draw.polygon(pts, fill=fill)
        if stroke:
            self.draw.line(pts + [pts[0]], fill=stroke, width=max(1, round(width * SCALE)))
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.svg.append(
            f'<polygon points="{path}" fill="{fill}" stroke="{stroke or "none"}" '
            f'stroke-width="{width}"/>'
        )

    def text(self, x, y, value, size=24, color=INK, bold=False, anchor="la", rotate=0):
        value = str(value)
        font = self.font(size, bold)
        # PIL anchor names match our simple SVG mapping: l/m/r + a/m/d.
        if rotate:
            bbox = font.getbbox(value)
            tw, th = bbox[2] - bbox[0] + 12 * SCALE, bbox[3] - bbox[1] + 12 * SCALE
            layer = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
            ld = ImageDraw.Draw(layer)
            ld.text((6 * SCALE, 6 * SCALE - bbox[1]), value, font=font, fill=color)
            layer = layer.rotate(-rotate, expand=True, resample=Image.Resampling.BICUBIC)
            self.image.paste(layer, (round(x * SCALE - layer.width / 2), round(y * SCALE - layer.height / 2)), layer)
            svg_anchor = "middle"
            self.svg.append(
                f'<text x="{x}" y="{y}" transform="rotate({rotate} {x} {y})" '
                f'text-anchor="{svg_anchor}" dominant-baseline="middle" '
                f'font-family="Arial, sans-serif" font-size="{size}" '
                f'font-weight="{"700" if bold else "400"}" fill="{color}">'
                f'{html.escape(value)}</text>'
            )
            return
        self.draw.text((round(x * SCALE), round(y * SCALE)), value, font=font, fill=color, anchor=anchor)
        ha = {"l": "start", "m": "middle", "r": "end"}.get(anchor[0], "start")
        va = {"a": "hanging", "m": "middle", "d": "auto"}.get(anchor[1], "hanging")
        self.svg.append(
            f'<text x="{x}" y="{y}" text-anchor="{ha}" dominant-baseline="{va}" '
            f'font-family="Arial, sans-serif" font-size="{size}" '
            f'font-weight="{"700" if bold else "400"}" fill="{color}">'
            f'{html.escape(value)}</text>'
        )

    def save(self, stem: str):
        OUT.mkdir(parents=True, exist_ok=True)
        png = OUT / f"{stem}.png"
        svg = OUT / f"{stem}.svg"
        self.image.save(png, optimize=True, dpi=(200, 200))
        svg.write_text("\n".join(self.svg + ["</svg>"]) + "\n", encoding="utf-8")


def title(c: Canvas, main: str, sub: str):
    c.text(72, 55, main, 38, bold=True)
    c.text(72, 107, sub, 21, color=MUTED)
    c.line([(72, 145), (1528, 145)], GRID, 2)


def source(c: Canvas, text_value: str):
    c.line([(72, 935), (1528, 935)], GRID, 1)
    c.text(72, 951, text_value, 17, color=MUTED)


def panel(c: Canvas, x, y, w, h, heading: str):
    c.rect(x, y, w, h, fill=WHITE, stroke=GRID, width=1, radius=12)
    c.text(x + 24, y + 20, heading, 24, bold=True)


def nice_ticks(vmax: float, count: int = 5):
    raw = vmax / count
    power = 10 ** math.floor(math.log10(raw)) if raw else 1
    frac = raw / power
    step = (1 if frac <= 1 else 2 if frac <= 2 else 2.5 if frac <= 2.5 else 5 if frac <= 5 else 10) * power
    top = math.ceil(vmax / step) * step
    return [i * step for i in range(round(top / step) + 1)], top


def parse_official():
    path = RAW / "01_official_benchmark_17926.log"
    rows = []
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        m = re.match(r"(varlen )?shape=\[8192,(\d+),128\](?: seq_lens=\[(.*)\])?", line)
        if m:
            heads = int(m.group(2))
            seq = m.group(3)
            case = "Fixed" if not m.group(1) else ("8 x 1024" if seq and seq.count("1024") == 8 else "Ragged 6")
            current = {"heads": heads, "case": case}
            continue
        if current is None:
            continue
        fm = re.match(r"flash_kda \(bf16 state\)\s*: mean=([0-9.]+) ms", line)
        cm = re.match(r"chunk_kda\s*: mean=([0-9.]+) ms", line)
        if fm:
            current["flash_ms"] = float(fm.group(1))
        elif cm:
            current["chunk_ms"] = float(cm.group(1))
            current["speedup"] = current["chunk_ms"] / current["flash_ms"]
            rows.append(current)
            current = None
    if len(rows) != 6:
        raise ValueError(f"expected six official benchmark rows, got {len(rows)}")
    return rows


def read_csv(name: str, comments: bool = False):
    path = RAW / name
    with path.open(newline="", encoding="utf-8") as fh:
        if comments:
            lines = [line for line in fh if not line.startswith("#")]
            return list(csv.DictReader(lines))
        return list(csv.DictReader(fh))


def figure_official(rows, summary):
    c = Canvas()
    title(c, "Official FlashKDA is already a strong B300 baseline", "Forward, T=8192, D=128  |  lower latency is better")
    colors = {"flash": NAVY, "chunk": ORANGE}
    for idx, heads in enumerate((96, 64)):
        x0 = 72 + idx * 748
        panel(c, x0, 177, 708, 710, f"H={heads}")
        data = [r for r in rows if r["heads"] == heads]
        y0, ph = 810, 545
        ticks, ymax = nice_ticks(max(r["chunk_ms"] for r in data) * 1.06, 5)
        for t in ticks:
            y = y0 - ph * t / ymax
            c.line([(x0 + 82, y), (x0 + 674, y)], GRID, 1)
            c.text(x0 + 68, y, f"{t:.1f}", 17, color=MUTED, anchor="rm")
        c.text(x0 + 23, 530, "Latency (ms)", 18, color=MUTED, rotate=-90)
        group_w = 176
        for j, r in enumerate(data):
            gx = x0 + 120 + j * group_w
            for k, key in enumerate(("flash_ms", "chunk_ms")):
                value = r[key]
                bx = gx + k * 54
                bh = ph * value / ymax
                color = colors["flash" if key == "flash_ms" else "chunk"]
                c.rect(bx, y0 - bh, 42, bh, fill=color, stroke=None)
                c.text(bx + 21, y0 - bh - 12, f"{value:.3f}", 15, color=INK, bold=True, anchor="md")
            c.text(gx + 48, y0 + 18, r["case"], 17, anchor="ma")
            c.text(gx + 48, y0 - ph - 27, f"{r['speedup']:.2f}x", 19, color=GREEN, bold=True, anchor="mm")
            summary.append(metric(f"official_h{heads}_{slug(r['case'])}_flash", r["flash_ms"], "ms", "01_official_benchmark_17926.log"))
            summary.append(metric(f"official_h{heads}_{slug(r['case'])}_chunk", r["chunk_ms"], "ms", "01_official_benchmark_17926.log"))
            summary.append(metric(f"official_h{heads}_{slug(r['case'])}_speedup_vs_chunk", r["speedup"], "x", "derived from Job 17926 means"))
        c.rect(x0 + 170, 846, 18, 18, fill=NAVY)
        c.text(x0 + 196, 845, "FlashKDA", 17)
        c.rect(x0 + 340, 846, 18, 18, fill=ORANGE)
        c.text(x0 + 366, 845, "FLA chunk KDA", 17)
    source(c, "Source: Slurm Job 17926; official commit 1ce47ea; warmup=30, iters=200, repeats=5.")
    c.save("fig01_official_vs_fla")


def figure_chunk(rows, summary):
    c = Canvas()
    title(c, "Why CHUNK=16 is a numerical and compute boundary", "Mechanical C=32/64 enlargement breaks the current unscaled path before workspace becomes dominant")
    chunks = [int(r["chunk"]) for r in rows]
    configs = [
        (72, 180, 455, 700, "FTZ / overflow fraction", [float(r["decay_zero_fraction"]) * 100 for r in rows], "% per channel", 80, RED),
        (572, 180, 455, 700, "Naive Neumann cost / sequence", [float(r["neumann_flops_per_sequence_vs_c16"]) for r in rows], "x vs C16", 30, ORANGE),
        (1072, 180, 455, 700, "Workspace / head", [float(r["workspace_mib_per_head"]) for r in rows], "MiB", 9, TEAL),
    ]
    for x0, y0p, pw, phh, heading, vals, unit, ymax, color in configs:
        panel(c, x0, y0p, pw, phh, heading)
        plot_x, plot_y, plot_w, plot_h = x0 + 74, y0p + 95, 340, 475
        base = plot_y + plot_h
        for t in range(0, 6):
            val = ymax * t / 5
            y = base - plot_h * t / 5
            c.line([(plot_x, y), (plot_x + plot_w, y)], GRID, 1)
            c.text(plot_x - 12, y, f"{val:g}", 16, color=MUTED, anchor="rm")
        for j, (ch, val) in enumerate(zip(chunks, vals)):
            bx = plot_x + 34 + j * 105
            bh = plot_h * val / ymax
            bar_color = NAVY if ch == 16 else color
            c.rect(bx, base - bh, 62, bh, fill=bar_color)
            c.text(bx + 31, base - bh - 10, f"{val:.2f}" if val < 10 else f"{val:.1f}", 18, bold=True, anchor="md")
            c.text(bx + 31, base + 16, f"{ch}", 18, anchor="ma")
        c.text(plot_x + plot_w / 2, base + 49, "CHUNK", 16, color=MUTED, anchor="ma")
        c.text(plot_x, y0p + 635, unit, 17, color=MUTED)
    c.rect(106, 822, 20, 20, fill=NAVY)
    c.text(138, 822, "Current safe boundary", 18)
    c.rect(360, 822, 20, 20, fill=RED)
    c.text(392, 822, "First FTZ/overflow at token 18 for C=32 and C=64", 18, color=RED, bold=True)
    for r in rows:
        ch = int(r["chunk"])
        summary.extend([
            metric(f"chunk{ch}_ftz_fraction", float(r["decay_zero_fraction"]), "fraction", "04_chunk_analysis_17935.csv"),
            metric(f"chunk{ch}_neumann_sequence_cost_vs_c16", float(r["neumann_flops_per_sequence_vs_c16"]), "x", "04_chunk_analysis_17935.csv"),
            metric(f"chunk{ch}_workspace_per_head", float(r["workspace_mib_per_head"]), "MiB", "04_chunk_analysis_17935.csv"),
        ])
    source(c, "Source: Job 17935 analytical sweep, T=8192, H=12, D=128, lower_bound=-5. FLA block probe is not used as a FlashKDA timing claim.")
    c.save("fig02_chunk_constraints")


def figure_tcgen(rows, summary):
    c = Canvas()
    title(c, "tcgen05 direct-swap gate fails at K3's V=128", "Measured MMA/TCGEN speedup; above 1.0 favors tcgen05, below 1.0 favors mma.sync")
    for pidx, level in enumerate(("L0", "L1")):
        x0 = 72 + pidx * 748
        heading = "L0: preferred on-chip layouts" if level == "L0" else "L1: + state/gate + conservative U reformat"
        panel(c, x0, 177, 708, 710, heading)
        px, py, pw, ph = x0 + 84, 260, 575, 500
        ymin, ymax = 0.2, 1.6
        for t in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
            y = py + ph * (ymax - t) / (ymax - ymin)
            c.line([(px, y), (px + pw, y)], RED if t == 1 else GRID, 3 if t == 1 else 1, (9, 7) if t == 1 else None)
            c.text(px - 13, y, f"{t:.2f}", 16, color=RED if t == 1 else MUTED, bold=t == 1, anchor="rm")
        xs = {16: px + 35, 32: px + 205, 64: px + 375, 128: px + 545}
        series = [
            (12, 1, NAVY, (4, 5), "grid 12, one-shot"),
            (148, 1, CYAN, (4, 5), "grid 148, one-shot"),
            (12, 64, ORANGE, None, "grid 12, amortized x64"),
            (148, 64, PURPLE, None, "grid 148, amortized x64"),
        ]
        for grid, inner, color, dash, label in series:
            selected = sorted(
                (r for r in rows if r["level"] == level and int(r["grid"]) == grid and int(r["inner"]) == inner),
                key=lambda r: int(r["V"]),
            )
            points = []
            for r in selected:
                v, speed = int(r["V"]), float(r["speedup_median"])
                y = py + ph * (ymax - speed) / (ymax - ymin)
                points.append((xs[v], y))
            c.line(points, color, 3, dash)
            for r, (x, y) in zip(selected, points):
                v = int(r["V"])
                c.circle(x, y, 9 if v == 128 else 5, WHITE, RED if v == 128 else color, 4 if v == 128 else 2)
                if v == 128 and inner == 64:
                    # Only annotate the decision-critical amortized rows.  All
                    # one-shot values remain visible as outlined V128 markers.
                    dy = -18 if grid == 12 else 24
                    c.text(x - 3, y + dy, f"{float(r['speedup_median']):.3f}x", 14, color=RED, bold=True, anchor="mm")
                    summary.append(metric(f"tcgen_{level.lower()}_v128_grid{grid}_inner{inner}_speedup", float(r["speedup_median"]), "x MMA/TCGEN", "03_tcgen05_probe_17937.csv"))
        for v, x in xs.items():
            c.text(x, py + ph + 20, f"V={v}", 17, bold=v == 128, color=RED if v == 128 else INK, anchor="ma")
        c.text(px + pw / 2, py + ph + 51, "Value dimension", 18, color=MUTED, anchor="ma")
        for li, (_, _, color, dash, label) in enumerate(series):
            lx = x0 + 83 + (li % 2) * 285
            ly = 838 + (li // 2) * 27
            c.line([(lx, ly + 8), (lx + 35, ly + 8)], color, 3, dash)
            c.text(lx + 44, ly, label, 15)
    source(c, "Source: Job 17937 Phase-6 microbenchmark; warmup=30, iters=200, repeats=5. L0/L1 are probes, not full-K2 timings.")
    c.save("fig03_tcgen05_speedup")


def aggregate_dispatch(rows):
    groups = defaultdict(list)
    meta = {}
    for r in rows:
        key = (r["case"], r["mode"])
        groups[key].append(float(r["median_ms"]))
        meta[r["case"]] = (int(r["nseq"]), int(r["total_tokens"]))
    return {(case, mode): median(vals) for (case, mode), vals in groups.items()}, meta


def heat_color(value: float):
    # Diverging scale centered on zero: red is slower, green is faster.
    if value >= 0:
        return _mix(WHITE, GREEN, min(value / 30.0, 1.0))
    return _mix(WHITE, RED, min(-value / 110.0, 1.0))


def figure_dispatch(rows, summary):
    agg, meta = aggregate_dispatch(rows)
    c = Canvas()
    title(c, "ValueSlice needs sequence-distribution-aware dispatch", "Median latency reduction relative to V128; positive is faster, negative is slower")
    cases = ["fixed_1x8192", "packed_1x8192", "packed_ragged6", "packed_8x1024", "packed_32x256"]
    labels = ["Fixed 1x8192", "Packed 1x8192", "Ragged 6", "8x1024", "32x256"]
    columns = [("v16", "V16"), ("v32", "V32"), ("v64", "V64"), ("compat_v128", "V128"), ("auto", "Auto")]
    panel(c, 72, 177, 1456, 710, "Workload distribution x ValueSlice")
    left, top, cw, ch = 350, 285, 214, 93
    for j, (_, label) in enumerate(columns):
        c.text(left + j * cw + cw / 2, top - 40, label, 20, bold=True, anchor="mm")
    for i, (case, label) in enumerate(zip(cases, labels)):
        nseq, total = meta[case]
        c.text(left - 28, top + i * ch + ch / 2 - 9, label, 21, bold=True, anchor="rm")
        c.text(left - 28, top + i * ch + ch / 2 + 18, f"N={nseq}, T={total}", 15, color=MUTED, anchor="rm")
        base = agg[(case, "compat_v128")]
        for j, (mode, _) in enumerate(columns):
            val = agg[(case, mode)]
            reduction = 100.0 * (base - val) / base
            fill = heat_color(reduction)
            x, y = left + j * cw, top + i * ch
            c.rect(x, y, cw - 7, ch - 7, fill=fill, stroke=WHITE, width=2, radius=5)
            c.text(x + (cw - 7) / 2, y + 34, f"{reduction:+.1f}%", 25, bold=True, anchor="mm", color=WHITE if abs(reduction) > 22 else INK)
            c.text(x + (cw - 7) / 2, y + 65, f"{val:.4f} ms", 14, anchor="mm", color=WHITE if abs(reduction) > 45 else MUTED)
            summary.extend([
                metric(f"dispatch_{case}_{mode}_median", val, "ms", "05_dispatch_upgrade_17947.csv; median of repeat medians"),
                metric(f"dispatch_{case}_{mode}_latency_reduction_vs_v128", reduction, "%", "derived from Job 17947 medians"),
            ])
    c.text(350, 790, "Green = faster than V128", 18, color=GREEN, bold=True)
    c.text(650, 790, "White = parity", 18, color=MUTED, bold=True)
    c.text(850, 790, "Red = slower than V128", 18, color=RED, bold=True)
    c.text(350, 833, "Auto safely selects V16 only for single-sequence long prefill; multi-sequence varlen falls back to V128.", 19, color=INK)
    source(c, "Source: Job 17947, B300/SM103, H=12, D=128; each cell is the median across 3 repeat medians.")
    c.save("fig04_valueslice_distribution_heatmap")


def figure_correctness(rows, summary):
    c = Canvas()
    title(c, "Observed relative RMSE remains below 1%", "Worst observed FlashKDA error by workload and tensor; ValueSlice itself is bitwise-equal to V128")
    allowed = {"accuracy_vs_naive", "accuracy_vs_chunk"}
    case_order = [
        "fixed_short_stateful", "ragged_short_stateful", "long_8192_random", "long_8192_memory",
        "k3_fixed_8192", "k3_ragged6", "k3_packed_8x1024",
    ]
    case_labels = ["Short fixed", "Short ragged", "Long random", "Long memory", "K3 fixed", "K3 ragged6", "K3 8x1024"]
    maxima = {}
    for case in case_order:
        for tensor in ("output", "final_state"):
            vals = [float(r["rel_rmse"]) * 100 for r in rows if r["case"] == case and r["tensor"] == tensor and r["comparison_kind"] in allowed]
            maxima[(case, tensor)] = max(vals) if vals else None
    panel(c, 72, 177, 1070, 710, "Maximum across specified FP32 references")
    px, py, pw, ph = 180, 280, 900, 500
    ymax = 1.05
    for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = py + ph * (ymax - t) / ymax
        c.line([(px, y), (px + pw, y)], RED if t == 1 else GRID, 3 if t == 1 else 1, (10, 7) if t == 1 else None)
        c.text(px - 15, y, f"{t:.1f}%", 16, color=RED if t == 1 else MUTED, anchor="rm")
    gw = pw / len(case_order)
    for i, (case, label) in enumerate(zip(case_order, case_labels)):
        for j, (tensor, color) in enumerate((("output", NAVY), ("final_state", TEAL))):
            val = maxima[(case, tensor)]
            if val is None:
                continue
            bx = px + i * gw + 25 + j * 42
            bh = ph * val / ymax
            c.rect(bx, py + ph - bh, 34, bh, fill=color)
            c.text(bx + 17, py + ph - bh - 8, f"{val:.2f}", 13, bold=True, anchor="md")
            summary.append(metric(f"correctness_{case}_{tensor}_worst_rel_rmse", val, "%", "03_reference_correctness_17934.csv; max across requested references"))
        c.text(px + i * gw + gw / 2, py + ph + 34, label, 15, anchor="mm", rotate=-35)
    c.rect(760, 226, 18, 18, fill=NAVY)
    c.text(788, 224, "Output", 17)
    c.rect(885, 226, 18, 18, fill=TEAL)
    c.text(913, 224, "Final state", 17)
    bitwise = [r for r in rows if r["comparison_kind"] == "bitwise_valueslice"]
    finite = [r for r in rows if r["finite"] == "True"]
    panel(c, 1180, 177, 348, 710, "Audit summary")
    c.text(1354, 315, f"{len(finite)}/{len(rows)}", 50, color=GREEN, bold=True, anchor="mm")
    c.text(1354, 365, "finite comparisons", 18, color=MUTED, anchor="mm")
    c.text(1354, 505, f"{sum(r['bitwise_equal'] == 'True' for r in bitwise)}/{len(bitwise)}", 50, color=NAVY, bold=True, anchor="mm")
    c.text(1354, 555, "ValueSlice bitwise equal", 18, color=MUTED, anchor="mm")
    worst = max(float(r["rel_rmse"]) for r in rows if r["comparison_kind"] in allowed) * 100
    c.text(1354, 695, f"{worst:.3f}%", 50, color=ORANGE, bold=True, anchor="mm")
    c.text(1354, 745, "worst relative RMSE", 18, color=MUTED, anchor="mm")
    summary.extend([
        metric("correctness_finite_rows", len(finite), "rows", "03_reference_correctness_17934.csv"),
        metric("correctness_rows_total", len(rows), "rows", "03_reference_correctness_17934.csv"),
        metric("valueslice_bitwise_rows_passed", sum(r["bitwise_equal"] == "True" for r in bitwise), "rows", "03_reference_correctness_17934.csv"),
        metric("valueslice_bitwise_rows_total", len(bitwise), "rows", "03_reference_correctness_17934.csv"),
        metric("correctness_worst_rel_rmse", worst, "%", "03_reference_correctness_17934.csv"),
    ])
    source(c, "Source: Job 17934; FLA 0.5.2; naive.py SHA 60a32285..., chunk.py SHA a15aa6ac...; FLA_FLASH_KDA=0.")
    c.save("fig05_correctness_rmse")


def figure_ncu(summary_rows, metric_rows, summary):
    by_label = {r["label"]: r for r in summary_rows}
    tensor_elapsed = {
        r["label"]: float(r["value"])
        for r in metric_rows
        if r["metric"] == "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"
    }
    official = by_label["official_v128"]
    sliced = by_label["valueslice_v16"]

    def duration_us(row):
        value = float(row["duration_value"])
        return value * 1000 if row["duration_unit"] == "ms" else value

    duration = [duration_us(official), duration_us(sliced)]
    ctas = [int(official["cta_count"]), int(sliced["cta_count"])]
    util = {
        "SM throughput": [float(official["compute_sm_value"]), float(sliced["compute_sm_value"])],
        "DRAM throughput": [float(official["dram_value"]), float(sliced["dram_value"])],
        "Tensor elapsed": [tensor_elapsed["official_v128"], tensor_elapsed["valueslice_v16"]],
    }
    c = Canvas()
    title(c, "ValueSlice exposes parallelism; B300 remains far from saturation", "Targeted NCU on K3 TP8 shape: T=8192, H=12, D=128")
    panel(c, 72, 177, 610, 710, "Kernel duration and grid coverage")
    panel(c, 724, 177, 804, 710, "Elapsed utilization (% of peak)")
    colors = [NAVY, GREEN]
    names = ["Official V128", "ValueSlice V16"]
    for i, (name, color) in enumerate(zip(names, colors)):
        x = 125 + i * 270
        # Duration bar; 1400 us fixed scale makes both raw profiler values legible.
        bh = 350 * duration[i] / 1400
        c.rect(x, 635 - bh, 120, bh, fill=color)
        c.text(x + 60, 635 - bh - 15, f"{duration[i]:,.2f} us", 23, color=color, bold=True, anchor="md")
        c.text(x + 60, 660, name, 20, bold=True, anchor="ma")
        c.text(x + 60, 725, f"{ctas[i]} CTAs", 31, color=color, bold=True, anchor="mm")
        c.text(x + 60, 764, f"{ctas[i] / 148:.1%} of 148-SM count", 17, color=MUTED, anchor="mm")
    reduction = 100 * (duration[0] - duration[1]) / duration[0]
    c.text(377, 263, f"-{reduction:.1f}%", 30, color=GREEN, bold=True, anchor="mm")
    c.text(377, 301, "NCU kernel duration", 17, color=MUTED, anchor="mm")

    px, py, pw, ph = 815, 290, 630, 430
    ymax = 8.0
    for t in [0, 2, 4, 6, 8]:
        y = py + ph * (ymax - t) / ymax
        c.line([(px, y), (px + pw, y)], GRID, 1)
        c.text(px - 14, y, f"{t}%", 17, color=MUTED, anchor="rm")
    gw = pw / 3
    for j, (label, vals) in enumerate(util.items()):
        for i, color in enumerate(colors):
            bx = px + j * gw + 46 + i * 64
            bh = ph * vals[i] / ymax
            c.rect(bx, py + ph - bh, 50, bh, fill=color)
            c.text(bx + 25, py + ph - bh - 10, f"{vals[i]:.2f}%", 17, color=color, bold=True, anchor="md")
        c.text(px + j * gw + gw / 2, py + ph + 34, label, 17, anchor="mm")
    c.rect(937, 810, 18, 18, fill=NAVY)
    c.text(966, 808, "Official V128", 17)
    c.rect(1162, 810, 18, 18, fill=GREEN)
    c.text(1191, 808, "ValueSlice V16", 17)
    for label, row, idx in (("official_v128", official, 0), ("valueslice_v16", sliced, 1)):
        summary.extend([
            metric(f"ncu_{label}_duration", duration[idx], "us", "05_targeted_ncu_summary_17965.csv"),
            metric(f"ncu_{label}_cta_count", ctas[idx], "CTA", "05_targeted_ncu_summary_17965.csv"),
            metric(f"ncu_{label}_sm_throughput", float(row["compute_sm_value"]), "%", "05_targeted_ncu_summary_17965.csv"),
            metric(f"ncu_{label}_dram_throughput", float(row["dram_value"]), "%", "05_targeted_ncu_summary_17965.csv"),
            metric(f"ncu_{label}_tensor_elapsed", tensor_elapsed[label], "%", "05_targeted_ncu_metrics_17965.csv"),
        ])
    summary.append(metric("ncu_valueslice_duration_reduction", reduction, "%", "derived from Job 17965 NCU durations"))
    source(c, "Source: Job 17965 NCU summaries. Tensor uses pct_of_peak_sustained_elapsed (not the active-cycle denominator).")
    c.save("fig07_ncu_bottleneck")


def amdahl(p: float, s: float) -> float:
    return 1.0 / ((1.0 - p) + p / s)


def figure_amdahl(tcgen_rows, summary):
    c = Canvas()
    phase_share = 128 / 416  # Phase-6 HMMA atoms / all four K2 phases, exact source-code count.
    title(c, "Amdahl sensitivity: a Phase-6 win has a bounded full-K2 impact", "Analytical model only; measured V128 direct-swap points are slower than 1.0x")
    panel(c, 72, 177, 1456, 710, "Full-K2 speedup = 1 / ((1 - phase share) + phase share / Phase-6 speedup)")
    px, py, pw, ph = 180, 285, 1150, 465
    xmin, xmax, ymin, ymax = 0.25, 4.0, 0.55, 1.9
    for t in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        y = py + ph * (ymax - t) / (ymax - ymin)
        c.line([(px, y), (px + pw, y)], RED if t == 1 else GRID, 3 if t == 1 else 1, (10, 7) if t == 1 else None)
        c.text(px - 16, y, f"{t:.1f}x", 17, color=RED if t == 1 else MUTED, anchor="rm")
    for t in [0.25, 0.5, 1, 1.5, 2, 3, 4]:
        x = px + pw * (t - xmin) / (xmax - xmin)
        c.line([(x, py), (x, py + ph)], GRID, 1)
        c.text(x, py + ph + 22, f"{t:g}x", 17, anchor="ma")
    curves = [(0.10, CYAN, "10% phase share"), (phase_share, NAVY, "30.77% measured atom share"), (0.50, ORANGE, "50% sensitivity")]
    for p, color, label in curves:
        pts = []
        for i in range(240):
            s = xmin + (xmax - xmin) * i / 239
            x = px + pw * (s - xmin) / (xmax - xmin)
            yv = amdahl(p, s)
            y = py + ph * (ymax - yv) / (ymax - ymin)
            pts.append((x, y))
        c.line(pts, color, 4)
    measured = [r for r in tcgen_rows if r["level"] == "L0" and int(r["V"]) == 128 and int(r["inner"]) == 64]
    for r in measured:
        s = float(r["speedup_median"])
        full = amdahl(phase_share, s)
        x = px + pw * (s - xmin) / (xmax - xmin)
        y = py + ph * (ymax - full) / (ymax - ymin)
        c.circle(x, y, 12, RED, WHITE, 3)
        # Separate the nearly coincident grid=12/148 callouts with leaders.
        label_y = y - 54 if int(r["grid"]) == 12 else y + 54
        label_x = x + 85
        c.line([(x + 10, y), (label_x - 10, label_y)], RED, 1)
        c.text(label_x, label_y, f"grid {r['grid']}: {s:.3f}x -> {full:.3f}x model", 17, color=RED, bold=True, anchor="lm")
        summary.append(metric(f"amdahl_v128_grid{r['grid']}_modeled_full_k2_speedup", full, "x", "derived from Job 17937 L0 inner64 and phase share 128/416"))
    upper = 1 / (1 - phase_share)
    c.line([(px + pw * (1 - xmin) / (xmax - xmin), py), (px + pw * (1 - xmin) / (xmax - xmin), py + ph)], RED, 2, (6, 6))
    c.text(1363, 323, "Phase-6-only ceiling", 22, color=NAVY, bold=True, anchor="mm")
    c.text(1363, 353, f"{upper:.3f}x", 22, color=NAVY, bold=True, anchor="mm")
    c.rect(250, 806, 24, 6, fill=CYAN)
    c.text(287, 794, "10%", 18)
    c.rect(400, 806, 24, 6, fill=NAVY)
    c.text(437, 794, "30.77% (=128/416 atoms)", 18)
    c.rect(735, 806, 24, 6, fill=ORANGE)
    c.text(772, 794, "50% sensitivity", 18)
    c.text(180, 850, "x-axis: isolated Phase-6 speedup", 18, color=MUTED)
    summary.extend([
        metric("phase6_static_hmma_atom_share", phase_share, "fraction", "source-code static count: 128 Phase-6 atoms / 416 total K2 atoms per chunk/head"),
        metric("phase6_only_infinite_speedup_ceiling", upper, "x", "Amdahl model using 128/416 atom share"),
    ])
    source(c, "Model source: exact static HMMA atom count (Phase 6 = 128, K2 total = 416). Markers use Job 17937 V128 L0 inner=64 measurements.")
    c.save("fig06_amdahl_sensitivity")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def metric(key: str, value, unit: str, source_name: str, notes: str = ""):
    return {"metric": key, "value": value, "unit": unit, "source": source_name, "notes": notes}


def write_summary(rows):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "value", "unit", "source", "notes"]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_outputs():
    expected = [
        "fig01_official_vs_fla", "fig02_chunk_constraints", "fig03_tcgen05_speedup",
        "fig04_valueslice_distribution_heatmap", "fig05_correctness_rmse", "fig06_amdahl_sensitivity",
        "fig07_ncu_bottleneck",
    ]
    checks = []
    for stem in expected:
        png, svg = OUT / f"{stem}.png", OUT / f"{stem}.svg"
        if not png.is_file() or not svg.is_file():
            raise FileNotFoundError(f"missing output for {stem}")
        with Image.open(png) as im:
            if im.size != (W * SCALE, H * SCALE):
                raise ValueError(f"unexpected PNG size for {stem}: {im.size}")
            checks.append({"file": png.name, "width": im.width, "height": im.height, "bytes": png.stat().st_size})
        svg_text = svg.read_text(encoding="utf-8")
        if not svg_text.startswith("<svg") or not svg_text.rstrip().endswith("</svg>"):
            raise ValueError(f"invalid SVG envelope for {stem}")
        checks.append({"file": svg.name, "width": W, "height": H, "bytes": svg.stat().st_size})
    return checks


def main():
    summary = []
    official = parse_official()
    chunk = read_csv("04_chunk_analysis_17935.csv")
    tcgen = read_csv("03_tcgen05_probe_17937.csv", comments=True)
    dispatch = read_csv("05_dispatch_upgrade_17947.csv")
    correctness = read_csv("03_reference_correctness_17934.csv")
    ncu_summary = read_csv("05_targeted_ncu_summary_17965.csv")
    ncu_metrics = read_csv("05_targeted_ncu_metrics_17965.csv")

    figure_official(official, summary)
    figure_chunk(chunk, summary)
    figure_tcgen(tcgen, summary)
    figure_dispatch(dispatch, summary)
    figure_correctness(correctness, summary)
    figure_amdahl(tcgen, summary)
    figure_ncu(ncu_summary, ncu_metrics, summary)
    write_summary(summary)
    checks = verify_outputs()
    print(json.dumps({"figures": checks, "summary_rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
