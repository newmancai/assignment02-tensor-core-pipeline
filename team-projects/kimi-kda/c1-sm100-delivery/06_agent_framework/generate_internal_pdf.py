#!/usr/bin/env python3
"""Generate the internal FlashKDA typed-IR stage delivery PDF."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
OUTPUT = ROOT / "output" / "pdf" / "flashkda_typed_ir_stage2_internal.pdf"

NAVY = HexColor("#071526")
INK = HexColor("#172033")
MUTED = HexColor("#5D6878")
CYAN = HexColor("#00AFC8")
BLUE = HexColor("#2563EB")
ORANGE = HexColor("#F59E0B")
GREEN = HexColor("#16A36A")
RED = HexColor("#D94B4B")
PALE = HexColor("#F3F7FA")
LINE = HexColor("#D8E1E8")
WHITE = colors.white

pdfmetrics.registerFont(
    TTFont("CJK", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
)
pdfmetrics.registerFontFamily(
    "CJK", normal="CJK", bold="CJK", italic="CJK", boldItalic="CJK"
)


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float, height: float = 70 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        nodes = [
            ("SOURCE", "真实 kernel\n+ source lines", CYAN),
            ("TYPED IR", "warp roles\nbarrier/resources", BLUE),
            ("VERIFIER", "proof obligations\nnamed rules", GREEN),
            ("SEARCH", "semantic delta\nnot raw PTX", ORANGE),
            ("CALIBRATION", "B300 evidence\nconfidence", CYAN),
            ("TACTIC", "activate / retain\n/ learn", BLUE),
        ]
        gap = 4 * mm
        box_w = (self.width - gap * 2) / 3
        box_h = 25 * mm
        coords = []
        for index in range(6):
            row, col = divmod(index, 3)
            x = col * (box_w + gap)
            y = self.height - (row + 1) * box_h - row * 10 * mm
            coords.append((x, y))
        for index, ((label, detail, accent), (x, y)) in enumerate(zip(nodes, coords)):
            c.setFillColor(PALE)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, box_w, box_h, 3 * mm, fill=1, stroke=1)
            c.setFillColor(accent)
            c.rect(x, y, 2.4 * mm, box_h, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 8)
            c.drawString(x + 6 * mm, y + 16 * mm, label)
            c.setFont("CJK", 8)
            for line_index, line in enumerate(detail.splitlines()):
                c.drawString(x + 6 * mm, y + (10 - 4 * line_index) * mm, line)
            if index not in (2, 5):
                nx, ny = coords[index + 1]
                c.setStrokeColor(MUTED)
                c.setFillColor(MUTED)
                c.line(x + box_w, y + box_h / 2, nx - 1.5 * mm, ny + box_h / 2)
                c.line(nx - 1.5 * mm, ny + box_h / 2, nx - 4 * mm, ny + box_h / 2 + 1.5 * mm)
                c.line(nx - 1.5 * mm, ny + box_h / 2, nx - 4 * mm, ny + box_h / 2 - 1.5 * mm)
        # Learning loop from tactic back to verifier/IR.
        x0, y0 = coords[5]
        x1, y1 = coords[1]
        c.setStrokeColor(ORANGE)
        c.setLineWidth(1.2)
        c.line(x0 + box_w / 2, y0, x0 + box_w / 2, 2 * mm)
        c.line(x0 + box_w / 2, 2 * mm, x1 + box_w / 2, 2 * mm)
        c.line(x1 + box_w / 2, 2 * mm, x1 + box_w / 2, y1 - 1.5 * mm)
        c.setFont("CJK", 7)
        c.setFillColor(ORANGE)
        c.drawCentredString(self.width / 2, 4 * mm, "失败沉淀为 rule / primitive；收益沉淀为 calibration / tactic")


class ProtocolDiagram(Flowable):
    def __init__(self, width: float, height: float = 65 * mm):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        roles = [
            ("aux_mma", "W10-11", CYAN),
            ("compute", "W0-3", BLUE),
            ("mma", "W9", ORANGE),
            ("prep", "W12-31", GREEN),
        ]
        lane_x = [22 * mm, 66 * mm, 110 * mm, 154 * mm]
        top = self.height - 10 * mm
        for (role, warps, color), x in zip(roles, lane_x):
            c.setFillColor(color)
            c.roundRect(x - 16 * mm, top, 32 * mm, 10 * mm, 2 * mm, fill=1, stroke=0)
            c.setFillColor(WHITE)
            c.setFont("Helvetica-Bold", 8)
            c.drawCentredString(x, top + 6 * mm, role)
            c.setFont("Helvetica", 7)
            c.drawCentredString(x, top + 2.5 * mm, warps)
            c.setStrokeColor(LINE)
            c.line(x, top, x, 5 * mm)
        events = [
            (0, 1, top - 12 * mm, "qk_full · elect_one · count=1"),
            (0, 2, top - 20 * mm, "fanout consumer"),
            (2, 1, top - 31 * mm, "final_ready · elect_commit"),
            (1, 3, top - 43 * mm, "smem_free · per-warp · count=4"),
        ]
        for src, dst, y, label in events:
            x1, x2 = lane_x[src], lane_x[dst]
            c.setStrokeColor(INK)
            c.setFillColor(INK)
            c.line(x1, y, x2 - 2 * mm if x2 > x1 else x2 + 2 * mm, y)
            direction = 1 if x2 > x1 else -1
            c.line(x2 - 2 * mm * direction, y, x2 - 5 * mm * direction, y + 1.5 * mm)
            c.line(x2 - 2 * mm * direction, y, x2 - 5 * mm * direction, y - 1.5 * mm)
            c.setFont("CJK", 7)
            c.setFillColor(MUTED)
            c.drawCentredString((x1 + x2) / 2, y + 2 * mm, label)


class SpeedupChart(Flowable):
    def __init__(self, width: float, cases: list[dict], height: float = 77 * mm):
        super().__init__()
        self.width = width
        self.height = height
        self.cases = cases

    def draw(self):
        c = self.canv
        labels = [
            "H96 fixed", "H96 mixed", "H96 uniform",
            "H64 fixed", "H64 mixed", "H64 uniform",
        ]
        left = 34 * mm
        chart_w = self.width - left - 12 * mm
        max_value = 1.35
        row_h = 10.5 * mm
        for tick in (1.0, 1.1, 1.2, 1.3):
            x = left + chart_w * tick / max_value
            c.setStrokeColor(LINE if tick != 1.0 else RED)
            c.line(x, 4 * mm, x, self.height - 5 * mm)
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 7)
            c.drawCentredString(x, 1 * mm, f"{tick:.1f}x")
        for index, (label, item) in enumerate(zip(labels, self.cases)):
            y = self.height - (index + 1) * row_h
            speedup = item["speedup"]
            eligible = item["activation_tactic"].startswith("activate")
            c.setFont("Helvetica", 7.5)
            c.setFillColor(INK)
            c.drawRightString(left - 3 * mm, y + 2.2 * mm, label)
            c.setFillColor(GREEN if eligible else MUTED)
            c.roundRect(left, y, chart_w * speedup / max_value, 6 * mm, 1.5 * mm, fill=1, stroke=0)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 7.5)
            c.drawString(left + chart_w * speedup / max_value + 2 * mm, y + 2 * mm, f"{speedup:.3f}x")


def styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="CJK", fontSize=25, leading=34, textColor=WHITE, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", fontName="CJK", fontSize=11, leading=18, textColor=HexColor("#C8D9E7")),
        "h1": ParagraphStyle("h1", fontName="CJK", fontSize=18, leading=25, textColor=NAVY, spaceAfter=5 * mm),
        "h2": ParagraphStyle("h2", fontName="CJK", fontSize=11.5, leading=17, textColor=BLUE, spaceBefore=3 * mm, spaceAfter=2 * mm),
        "body": ParagraphStyle("body", fontName="CJK", fontSize=9.4, leading=15.2, textColor=INK, spaceAfter=2.5 * mm),
        "small": ParagraphStyle("small", fontName="CJK", fontSize=7.5, leading=11.5, textColor=MUTED),
        "cell": ParagraphStyle("cell", fontName="CJK", fontSize=7.6, leading=10.2, textColor=INK),
        "quote": ParagraphStyle("quote", fontName="CJK", fontSize=13, leading=21, textColor=NAVY, leftIndent=8 * mm, rightIndent=8 * mm, borderColor=CYAN, borderWidth=0, borderPadding=5 * mm, backColor=PALE, spaceAfter=5 * mm),
        "metric": ParagraphStyle("metric", fontName="Helvetica-Bold", fontSize=19, leading=22, textColor=BLUE, alignment=TA_CENTER),
        "metric_label": ParagraphStyle("metric_label", fontName="CJK", fontSize=7.8, leading=11, textColor=MUTED, alignment=TA_CENTER),
        "cover_label": ParagraphStyle("cover_label", fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=CYAN, spaceAfter=8 * mm),
    }


def page_decor(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 9 * mm, "INTERNAL · FlashKDA / B300 / Typed IR")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"{doc.page}")
    canvas.restoreState()


def cover_decor(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(CYAN)
    canvas.rect(0, A4[1] - 9 * mm, A4[0], 9 * mm, fill=1, stroke=0)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(HexColor("#B8C8D6"))
    canvas.drawString(18 * mm, 14 * mm, "CONFIDENTIAL · INTERNAL DELIVERY · 2026-09-08")
    canvas.restoreState()


def metric_table(items, style):
    cells = []
    for value, label in items:
        cells.append([Paragraph(value, style["metric"]), Paragraph(label, style["metric_label"])])
    table = Table(cells, colWidths=[(A4[0] - 36 * mm) / len(cells)] * len(cells))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
    ]))
    return table


def build():
    report = json.loads((EVIDENCE / "b300_stage2_calibration.json").read_text())
    cases = report["calibration"]["cases"]
    compute = report["compute_accounting"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=19 * mm,
        title="FlashKDA on SM100/SM103: Source-Grounded Typed IR Stage 2",
        author="AI Infra Internal",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame], onPage=cover_decor),
        PageTemplate(id="body", frames=[frame], onPage=page_decor),
    ])
    s = styles()
    story = []

    story += [Spacer(1, 36 * mm), Paragraph("AI INFRA / KERNEL SYSTEMS", s["cover_label"]),
              Paragraph("FlashKDA on B300<br/>从专家 kernel 到可搜索、可证明、可部署的语义控制面", s["title"]),
              Spacer(1, 9 * mm), Paragraph("内部阶段交付 · Stage 2", s["subtitle"]),
              Spacer(1, 52 * mm), Paragraph("核心判断", ParagraphStyle("cover_h", parent=s["subtitle"], textColor=CYAN, fontSize=9)),
              Paragraph("真正值得构建的不是另一个 CUDA 自动调参器，而是一层能理解硬件协议、约束搜索边界、积累失败知识，并决定何时把候选接入生产的 AI-native kernel control plane。", ParagraphStyle("cover_q", parent=s["subtitle"], fontSize=13, leading=22, textColor=WHITE)),
              PageBreak()]
    doc.handle_nextPageTemplate("body")

    story += [Paragraph("01 · 结论先行", s["h1"]),
              Paragraph("这条支线已经从“typed IR 是否值得”推进到一个可量化结论：真实 FlashKDA 源码可以被提升为带物理语义与源码 provenance 的 IR；候选 corpus 可以被还原为 typed semantic delta；正确性、性能置信与部署状态可以在同一证据链中表达。", s["body"]),
              metric_table([("29", "唯一物理语义候选"), ("6/6", "B300 correctness"), ("1.153×", "方向性几何平均提升"), ("5/6", "置信下界 > 1")], s),
              Spacer(1, 6 * mm),
              Paragraph("我们得到的不是“Agent 找到几个参数”，而是一个更重要的系统边界：<font color='#2563EB'>候选生成成功不等于生产优化完成</font>。当前 public auto 在六个形状上仍全部解析为 CAKE 基线。因而 deployment tactic 必须成为 AI infra 的一等对象。", s["quote"]),
              Paragraph("阶段状态", s["h2"]),
              Paragraph("已完成：source-to-IR、协议证明、typed corpus、B300 calibration、资源核算。未完成：将 verifier-clean winner 作为显式 public backend 接入，并用完全相同的 public API scope 复测。因此本报告把 1.153×称为方向性 kernel evidence，而非生产 speedup。", s["body"]),
              PageBreak()]

    story += [Paragraph("02 · 我们要超越的不是一个数字，而是一种工作方式", s["h1"]),
              Paragraph("CAKE 证明了 Agent 可以在 hardware-explicit typed IR 上发现专家级物理 schedule，并在其 FlashKDA 实验中报告 2.05×几何平均收益。这个结果确认了“搜索结构”比“搜索 launch 参数”更有价值。我们的推进点不是复刻 CAKE，而是把 typed IR 从 synthesis language 扩展为 production-kernel control plane。", s["body"]),
              Table([
                  [Paragraph("CAKE 的强项", s["h2"]), Paragraph("我们的推进", s["h2"])],
                  [Paragraph("从 typed IR 搜索新的 kernel 结构", s["body"]), Paragraph("从真实专家源码反向恢复 IR，并保留源码行号", s["body"])],
                  [Paragraph("候选经 test gate 进入测量", s["body"]), Paragraph("候选同时携带 verifier proof、calibration 与 deployment state", s["body"])],
                  [Paragraph("KDA 中以官方实现作为黑盒 timing baseline", s["body"]), Paragraph("把 barrier/elect-commit/work-body sync 提升为可分析事件", s["body"])],
                  [Paragraph("失败促进语言演化", s["body"]), Paragraph("失败区分为 rule、primitive、calibration、tactic 四种知识", s["body"])],
              ], colWidths=[doc.width/2]*2, style=TableStyle([
                  ("BACKGROUND", (0,0), (-1,0), NAVY), ("TEXTCOLOR", (0,0), (-1,0), WHITE),
                  ("GRID", (0,0), (-1,-1), 0.5, LINE), ("VALIGN", (0,0), (-1,-1), "TOP"),
                  ("LEFTPADDING", (0,0), (-1,-1), 4*mm), ("RIGHTPADDING", (0,0), (-1,-1), 4*mm),
                  ("TOPPADDING", (0,0), (-1,-1), 3*mm), ("BOTTOMPADDING", (0,0), (-1,-1), 2*mm),
              ])),
              Spacer(1, 5 * mm), Paragraph("关键外部事实", s["h2"]),
              Paragraph("CUTLASS Task Scheduling 已能显式表达 tasks、resources、dependencies 与 warp assignments；但其官方 validation 文档明确指出，@cute.jit work body 内部同步仍然不透明。我们的 source importer 正好覆盖这一验证盲区。", s["body"]),
              PageBreak()]

    story += [Paragraph("03 · AI-native kernel control plane", s["h1"]),
              ArchitectureDiagram(doc.width), Spacer(1, 3 * mm),
              Paragraph("设计原则", s["h2"]),
              Paragraph("<b>Semantic locality.</b> Agent 修改的是 warp role、fanout、completion、stage、resource，而不是一段不可解释 PTX。<br/><b>Proof before spend.</b> 静态 verifier 在编译与 GPU 测量前淘汰错误候选。<br/><b>Evidence is typed.</b> correctness、置信区间、硬件、GPU-seconds 与 activation state 属于候选本身。<br/><b>Learning changes the language.</b> 重复错误升级为 rule/primitive；形状相关收益升级为 calibration；接线与回退条件升级为 tactic。", s["body"]),
              Paragraph("这让 Agent 的能力边界从“写代码”移动到“维护一个能够持续学习的内核设计系统”。", s["quote"]),
              PageBreak()]

    story += [Paragraph("04 · 真实 kernel 的协议被恢复，而不是猜出来", s["h1"]),
              ProtocolDiagram(doc.width),
              Paragraph("从冻结版 BF16 fused M128 源码恢复出 6 类 warp role、23 个 barrier group 与 97 个 barrier/elect 事件。Q/K 第一条边的真实协议推翻了早期简化假设：ready 是 block elect-one(count=1)，而 free 是 4 个 compute warp 各 leader-arrive(count=4)；reuse waiter 是 prep，并不要求它同时是数据消费者。", s["body"]),
              Table([
                  ["Rule / Primitive", "拒绝的错误", "知识类型"],
                  [Paragraph("KIR107", s["cell"]), Paragraph("物理 warp role 重叠", s["cell"]), Paragraph("verifier rule", s["cell"])],
                  [Paragraph("KIR713", s["cell"]), Paragraph("arrival count 无法由物理参与者解释", s["cell"]), Paragraph("verifier rule", s["cell"])],
                  [Paragraph("KIR714", s["cell"]), Paragraph("consumer 到 reuse 不存在 completion path", s["cell"]), Paragraph("proof obligation", s["cell"])],
                  [Paragraph("ArrivalMode:<br/>ELECT_ONE_PER_WARP", s["cell"]), Paragraph("错误地把每-warp到达简化为 block leader", s["cell"]), Paragraph("IR primitive", s["cell"])],
              ], colWidths=[45*mm, 77*mm, 45*mm], style=TableStyle([
                  ("FONT", (0,0), (-1,-1), "CJK", 8), ("BACKGROUND", (0,0), (-1,0), NAVY),
                  ("TEXTCOLOR", (0,0), (-1,0), WHITE), ("GRID", (0,0), (-1,-1), 0.5, LINE),
                  ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 2.5*mm),
                  ("BOTTOMPADDING", (0,0), (-1,-1), 2.5*mm),
              ])),
              PageBreak()]

    story += [Paragraph("05 · Typed delta + B300 calibration", s["h1"]),
              Paragraph("29 个 frozen candidate 不再作为字符串文件名进入搜索，而被解析为 29 个唯一物理语义指纹：7 个 scalar-tile、21 个 value-tile、1 个 value64-split。每个候选显式携带 value rows、full-chunk、token extent、persistent tasks、grid stride 与 tile-schedule requirement。", s["body"]),
              SpeedupChart(doc.width, cases),
              Paragraph("结果解读", s["h2"]),
              Paragraph("固定长度与 uniform batch 对 value partition / persistent scheduling 最敏感，收益 1.199×–1.290×；mixed workload 中 H96 几乎持平，bootstrap 95% 下界低于 1，因此 tactic 明确选择 retain-baseline。这里最有价值的不是平均数，而是系统学会了<font color='#2563EB'>在什么形状上不激活优化</font>。", s["body"]),
              Paragraph("测量口径：B300 SM103a；CUPTI；cold L2；20 dry-run + 100 measured iterations。candidate 为 prepared evolution scope，baseline 为 public recurrent_kda scope，因此当前结论用于选择下一轮 activation，而非发布 production claim。", s["small"]),
              PageBreak()]

    story += [Paragraph("06 · B300 算力不是背景信息，而是实验数据", s["h1"]),
              metric_table([("55.32", "保留成功实验 GPU-seconds"), ("0.922", "B300 GPU-minutes"), ("1,980", "CUPTI timing samples"), ("12.73 kJ", "估算分配能耗")], s),
              Spacer(1, 7 * mm),
              Table([
                  ["实验", "墙钟 / GPU-s", "能耗估算", "峰值显存"],
                  *[[item["experiment"], f'{item["wall_seconds"]:.2f}s', f'{item["estimated_allocation_energy_joules"]/1000:.2f} kJ', f'{item["peak_memory_used_mib"]/1024:.2f} GiB'] for item in compute["accounts"] if item["exit_code"] == 0],
              ], colWidths=[72*mm, 34*mm, 31*mm, 30*mm], style=TableStyle([
                  ("FONT", (0,0), (-1,-1), "CJK", 7.7), ("BACKGROUND", (0,0), (-1,0), NAVY),
                  ("TEXTCOLOR", (0,0), (-1,0), WHITE), ("GRID", (0,0), (-1,-1), 0.5, LINE),
                  ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 2.5*mm),
                  ("BOTTOMPADDING", (0,0), (-1,-1), 2.5*mm),
              ])),
              Spacer(1, 6 * mm), Paragraph("如何理解这些数字", s["h2"]),
              Paragraph("55.32 GPU-s 是保留下来的四次成功单卡 allocation 下界，包含首次加载/编译、输入准备、同步与空隙；其中 CUPTI 真正累计计时的 kernel duration 为 0.640s。两者的差值不是浪费，而是说明 kernel search 的主要成本经常落在 compilation/setup/control plane，而非 kernel 本身。", s["body"]),
              Paragraph("早期三个 pre-kernel 环境失败曾复用相同输出路径，未能保留完整账本，因此没有计入 55.32 GPU-s。这个缺陷已转化为 append-only attempt ledger：未来失败也会保留 Slurm job id、节点、功率、显存与 GPU-seconds。报告选择给出可核验下界，而不是用 SSH 墙钟反推 GPU 占用。", s["body"]),
              PageBreak()]

    story += [Paragraph("07 · 真正的新 insight：Deployment tactic 是缺失的中间层", s["h1"]),
              Paragraph("传统 autotuner 的终点是 winner；真正的 AI infra 终点应是 safe activation。一个 winner 至少要经历四种状态：", s["body"]),
              Table([
                  ["状态", "必要证据", "本阶段"],
                  ["Expressible", "typed delta 能表达物理 schedule", "29/29"],
                  ["Verified", "规则、资源与 completion proof 通过", "29/29 corpus；Q/K edge source-proof"],
                  ["Calibrated", "正确性 + 硬件测量 + 置信", "6/6；5 个建议激活"],
                  ["Dispatchable", "public route、回退与同口径复测", "未完成；auto 仍为 cake"],
              ], colWidths=[30*mm, 93*mm, 44*mm], style=TableStyle([
                  ("FONT", (0,0), (-1,-1), "CJK", 8), ("BACKGROUND", (0,0), (-1,0), NAVY),
                  ("TEXTCOLOR", (0,0), (-1,0), WHITE), ("GRID", (0,0), (-1,-1), 0.5, LINE),
                  ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 3*mm),
                  ("BOTTOMPADDING", (0,0), (-1,-1), 3*mm),
              ])),
              Spacer(1, 7 * mm), Paragraph("这会改变 Agent 的搜索目标", s["h2"]),
              Paragraph("Agent 不再最大化离线 benchmark，而是最大化 expected deployed value：收益必须乘以 correctness confidence、shape coverage 与 activation probability，同时扣除 compile/setup/GPU budget。于是“保留基线”成为一种合法且可学习的动作；public adapter、fallback 和 routing predicate 与 warp schedule 一样，都是搜索产物。", s["quote"]),
              Paragraph("这也是比“让 LLM 写 CUDA”更深的护城河：长期资产不是某个 kernel，而是不断增长的语义、证明、硬件校准与部署决策数据库。", s["body"]),
              PageBreak()]

    story += [Paragraph("08 · 下一阶段：从方向性 winner 到生产闭环", s["h1"]),
              Paragraph("下一个阶段不扩展到更多算子，集中完成一条最短生产路径：", s["body"]),
              Table([
                  ["1", "Public adapter", "将 verifier-clean evolution route 作为显式 backend 接入；不静默覆盖 auto。"],
                  ["2", "Scope parity", "candidate 与 cake 使用完全相同 output/final-state、rotation、CUPTI scope。"],
                  ["3", "Tactic activation", "仅激活置信下界 > 1 且 correctness 通过的 5 个形状；H96 mixed 保留 baseline。"],
                  ["4", "Resource-aware objective", "把 compile/setup GPU-s 与 kernel speedup 同时纳入搜索目标。"],
                  ["5", "SM100→SM103 delta", "在保持数学与 protocol proof 的前提下搜索 warp-role/completion topology，而非机械替换 MMA。"],
              ], colWidths=[10*mm, 38*mm, 119*mm], style=TableStyle([
                  ("FONT", (0,0), (-1,-1), "CJK", 8.3), ("BACKGROUND", (0,0), (0,-1), CYAN),
                  ("TEXTCOLOR", (0,0), (0,-1), WHITE), ("GRID", (0,0), (-1,-1), 0.5, LINE),
                  ("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING", (0,0), (-1,-1), 3.5*mm),
                  ("BOTTOMPADDING", (0,0), (-1,-1), 3.5*mm),
              ])),
              Spacer(1, 8 * mm), Paragraph("阶段验收线", s["h2"]),
              Paragraph("只有 public route 同口径复测仍获得稳定收益，才把候选从 Calibrated 升级为 Dispatchable。目标不是追求所有形状都快，而是建立第一个由 typed IR + verifier + calibration + tactic 共同批准的生产优化。", s["quote"]),
              Paragraph("届时我们才能严谨回答最初问题：SM80 MMA 迁移到 SM100/SM103 是否值得。答案不会是架构口号，而是一组带 shape domain、证明边界与算力成本的部署决策。", s["body"]),
              PageBreak()]

    story += [Paragraph("附录 · 证据、边界与参考", s["h1"]),
              Paragraph("可复现实验资产", s["h2"]),
              Paragraph("• source_to_ir_a1418fd1ae.json：真实源码、warp ranges、97 个事件与 completion path。<br/>• evolution_typed_corpus.json：29 个 typed semantic fingerprints。<br/>• b300_evolution6_st_stable.json：6 个候选 correctness 与 20×100 CUPTI samples。<br/>• b300_legacy_cake.json / b300_legacy_auto.json：public baseline 与 routing control。<br/>• b300_stage2_calibration.json：bootstrap confidence、activation tactic 与算力汇总。", s["body"]),
              Paragraph("当前证明边界", s["h2"]),
              Paragraph("源码图证明的是结构化 happens-before 路径，尚未形式化证明每个循环 stage/phase alias 的全程序等价；register usage 仍未由 importer 精确恢复；性能比较的 candidate/public scope 尚未完全同口径。以上边界均被显式保留，没有用“测试通过”替代形式化结论。", s["body"]),
              Paragraph("参考", s["h2"]),
              Paragraph("[1] CAKE: Coding Agents for Kernel Exploration. https://arxiv.org/html/2608.12629<br/>[2] NVIDIA CUTLASS DSL - Task Scheduling Introduction. https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_introduction.html<br/>[3] NVIDIA CUTLASS DSL - Task Scheduling Validation and Verification Gaps. https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/ts_general/ts_validation.html", s["small"]),
              Spacer(1, 18 * mm), Paragraph("Internal position", s["h2"]), Spacer(1, 2 * mm),
              Paragraph("把 kernel optimization 做成会积累知识、会拒绝错误、会核算算力、会控制部署的系统，而不是一次性的代码生成。", s["quote"])]

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
