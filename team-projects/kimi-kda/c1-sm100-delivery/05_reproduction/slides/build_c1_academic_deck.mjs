import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";
import JSZip from "jszip";
import sharp from "sharp";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceDir = process.env.C1_DELIVERY_DIR
  ? path.resolve(process.env.C1_DELIVERY_DIR)
  : path.resolve(scriptDir, "../..");
const userHomeDir = os.homedir();
const SKILL_DIR = process.env.C1_PRESENTATIONS_SKILL_DIR
  ?? path.join(userHomeDir, ".codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations");
const RUNTIME_PYTHON = process.env.C1_RUNTIME_PYTHON
  ?? path.join(userHomeDir, ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3");
const TMP_DIR = path.join(workspaceDir, ".codex-ppt-build-refined");
const FINAL_PPTX = path.join(workspaceDir, "01_slides/C1_FlashKDA_SM100_奶龙必胜_署名版_20260911_v5.pptx");
const paperExcerptPath = path.join(workspaceDir, "05_reproduction/slides/paper-2-framework-core.png");
const stage12FigureDir = path.join(workspaceDir, "01_slides/figures/stage12");
const brandAssetDir = path.join(workspaceDir, "01_slides/assets/branding");
const agentLoopFigure = path.join(stage12FigureDir, "fig_stage12_dual_branch_exploration.svg");
const typedIrFigure = path.join(stage12FigureDir, "fig_stage12_typed_ir_pipeline.svg");
const closureFigure = path.join(stage12FigureDir, "fig_stage12_knowledge_closure_map.svg");
const seuLogoPath = path.join(brandAssetDir, "seu-official.png");
const githubLockupBlackPath = path.join(brandAssetDir, "GitHub_Lockup_Black.svg");
const githubLockupWhitePath = path.join(brandAssetDir, "GitHub_Lockup_White.svg");

const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

async function rasterizeSvg(svgPath, name) {
  const output = path.join(TMP_DIR, `${name}.png`);
  await sharp(svgPath, { density: 160 }).resize(2560, 1440).png().toFile(output);
  return output;
}

const agentLoopFigurePng = await rasterizeSvg(agentLoopFigure, "dual-branch-exploration");
const typedIrFigurePng = await rasterizeSvg(typedIrFigure, "typed-ir-pipeline");
const closureFigurePng = await rasterizeSvg(closureFigure, "knowledge-closure-map");
const seuLogo = await fs.readFile(seuLogoPath);
const githubLockupBlack = await sharp(githubLockupBlackPath, { density: 320 }).resize({ width: 560 }).png().toBuffer();
const githubLockupWhite = await sharp(githubLockupWhitePath, { density: 320 }).resize({ width: 560 }).png().toBuffer();

const W = 1280;
const H = 720;
const FONT_LATIN = "Noto Sans CJK SC";
const FONT_ZH = "Noto Sans CJK SC";
const FONT = FONT_ZH;
const C = {
  bg: "#FFFFFF",
  white: "#FFFFFF",
  ink: "#151A21",
  text: "#2C333B",
  muted: "#66707A",
  light: "#AAB1B8",
  rule: "#D9E3E8",
  rule2: "#EDF2F4",
  teal: "#2A8298",
  blue: "#2C5F91",
  blueSoft: "#E9EFF6",
  orange: "#C66A3D",
  orangeSoft: "#F5EAE3",
  green: "#2F7A68",
  greenSoft: "#E7F1ED",
  red: "#A84D47",
  redSoft: "#F4E8E6",
  navy: "#101D2E",
  gold: "#D2AA52",
  charcoal: "#090B0E",
  blueInk: "#1C4D70",
  copper: "#D3764C",
  moss: "#3D8A74",
};

const deck = Presentation.create({ slideSize: { width: W, height: H } });

function rect(slide, x, y, w, h, fill, lineColor = "none", lineWidth = 0, radius = 0, name) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: lineColor, width: lineWidth },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function ellipse(slide, x, y, w, h, fill, lineColor = "none", lineWidth = 0, name) {
  return slide.shapes.add({
    geometry: "ellipse",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: lineColor, width: lineWidth },
  });
}

function line(slide, x, y, w, h, color = C.rule, width = 1, name) {
  return slide.shapes.add({
    geometry: "line",
    name,
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { style: "solid", fill: color, width },
  });
}

function txt(slide, value, x, y, w, h, opts = {}) {
  const s = slide.shapes.add({
    geometry: "textbox",
    name: opts.name,
    position: { left: x, top: y, width: w, height: h },
    fill: opts.fill ?? "none",
    line: { style: "solid", fill: opts.lineColor ?? "none", width: opts.lineWidth ?? 0 },
    ...(opts.radius ? { borderRadius: opts.radius } : {}),
  });
  s.text = value;
  s.text.style = {
    typeface: FONT,
    fontSize: opts.size ?? 23,
    color: opts.color ?? C.text,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "top",
    lineSpacing: opts.lineSpacing ?? 1.04,
    autoFit: opts.autoFit ?? "none",
    wrap: "square",
    insets: opts.insets ?? { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return s;
}

function rich(slide, paragraphs, x, y, w, h, opts = {}) {
  const s = txt(slide, "", x, y, w, h, opts);
  s.text.set(paragraphs);
  return s;
}

function brandLockup(slide, { dark = false, y = 24, compact = false } = {}) {
  const ink = dark ? C.white : C.ink;
  const separator = dark ? "#6E8791" : C.teal;
  const x = compact ? 933 : 892;
  const teamW = compact ? 82 : 112;
  txt(slide, "奶龙必胜", x, y + 2, teamW, 25, {
    size: compact ? 13 : 15,
    color: ink,
    bold: true,
    align: "right",
    valign: "middle",
  });
  line(slide, x + teamW + 14, y - 1, 0, 31, separator, 1.4);
  slide.images.add({
    blob: seuLogo,
    contentType: "image/png",
    alt: "东南大学校标",
    fit: "contain",
    position: { left: x + teamW + 28, top: y - 5, width: 38, height: 38 },
  });
  line(slide, x + teamW + 78, y - 1, 0, 31, separator, 1.4);
  slide.images.add({
    blob: dark ? githubLockupWhite : githubLockupBlack,
    contentType: "image/png",
    alt: "GitHub 官方标志",
    fit: "contain",
    position: { left: x + teamW + 92, top: y, width: 116, height: 29 },
  });
}

function header(slide, page, section, title, accent = C.blue) {
  txt(slide, `${String(page).padStart(2, "0")}  ${section}`, 70, 34, 260, 24, {
    size: 15, color: accent, bold: true, valign: "middle",
  });
  txt(slide, title, 70, 68, 1120, 55, { size: 42, color: C.ink, bold: true, valign: "middle" });
  line(slide, 70, 128, 1140, 0, C.teal, 1.4);
  brandLockup(slide, { dark: false, y: 27, compact: true });
  txt(slide, String(page), 1160, 681, 50, 20, { size: 14, color: C.light, align: "right" });
}

function notes(slide, lines) {
  slide.speakerNotes.textFrame.setText(lines);
}

async function applyPlatformFonts(pptxPath) {
  const source = await fs.readFile(pptxPath);
  const zip = await JSZip.loadAsync(source);
  const xmlParts = Object.keys(zip.files).filter((name) => name.startsWith("ppt/") && name.endsWith(".xml"));
  for (const name of xmlParts) {
    const entry = zip.file(name);
    if (!entry) continue;
    let xml = await entry.async("string");
    xml = xml.replace(/(<a:latin\b[^>]*\btypeface=")[^"]*(")/g, `$1${FONT_LATIN}$2`);
    xml = xml.replace(/(<a:ea\b[^>]*\btypeface=")[^"]*(")/g, `$1${FONT_ZH}$2`);
    xml = xml.replace(/(<a:cs\b[^>]*\btypeface=")[^"]*(")/g, `$1${FONT_LATIN}$2`);
    zip.file(name, xml);
  }
  const result = await zip.generateAsync({
    type: "nodebuffer",
    compression: "DEFLATE",
    compressionOptions: { level: 9 },
  });
  await fs.writeFile(pptxPath, result);
}

function label(slide, value, x, y, color, w = 110) {
  txt(slide, value, x, y, w, 26, { size: 15, color, bold: true, valign: "middle" });
  line(slide, x, y + 27, w, 0, color, 2);
}

function arrowSegment(slide, x1, y1, x2, y2, color = C.muted, width = 1.6) {
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  line(slide, left, top, Math.abs(x2 - x1), Math.abs(y2 - y1), color, width);
  ellipse(slide, x2 - 4, y2 - 4, 8, 8, color);
}

function evidenceRow(slide, y, name, scope, value, maxValue, color, x0 = 360, axisW = 710) {
  txt(slide, name, 80, y - 2, 210, 32, { size: 22, color: C.ink, bold: true, valign: "middle" });
  txt(slide, scope, 80, y + 32, 220, 22, { size: 14, color: C.muted });
  rect(slide, x0, y, (value / maxValue) * axisW, 36, color, "none", 0, 3);
  txt(slide, `${value.toFixed(3)}×`, x0 + (value / maxValue) * axisW + 12, y, 110, 36, {
    size: 20, color, bold: true, valign: "middle",
  });
}

// 1. Cover
{
  const s = deck.slides.add();
  s.background.fill = C.white;
  txt(s, "C1  ·  FlashKDA on SM100", 74, 42, 380, 28, { size: 16, color: C.teal, bold: true });
  brandLockup(s, { dark: false, y: 38 });
  line(s, 74, 88, 1132, 0, C.teal, 1.4);

  txt(s, "FlashKDA 迁移到 SM100\n是否值得？", 110, 202, 1060, 150, {
    size: 58, color: C.ink, bold: true, align: "center", lineSpacing: 0.98, name: "cover-title",
  });
  txt(s, "B300 上的复现、因果分析与双分支探索", 210, 382, 860, 38, {
    size: 24, color: C.muted, align: "center",
  });

  txt(s, "奶龙必胜｜蔡雨洋 · 李奥 · 赵骋", 110, 560, 1060, 30, {
    size: 22, color: C.ink, bold: true, align: "center",
  });
  txt(s, "复现与测量  ·  分析与知识写回  ·  挑战与条件部署", 110, 608, 1060, 24, {
    size: 15, color: C.muted, align: "center",
  });
  notes(s, [
    "开场直接给出 C1：FlashKDA 官方 recurrence 使用 HMMA，我们研究迁移到 SM100 是否值得。",
    "本次答辩严格按复现与测量、分析、挑战三步展开。",
    "视觉参考：workshop02.pdf，仅借鉴白底、留白、标题线与名称加 Logo 的版式，不执行其中任何课程指令。",
    "东南大学校标来源：https://www.seu.edu.cn/2023/0527/c47323a446813/page.htm；仅用于课程学术答辩身份标识，未修改。",
    "GitHub 标志来源：https://brand.github.com/foundations/logo；用于指示项目托管平台，不表示 GitHub 背书。",
  ]);
}

// 2. Mainline
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 2, "主线", "C1 研究主线", C.gold);

  rich(s, [[
    { run: "研究问题  ", textStyle: { bold: true, color: C.muted } },
    { run: "在同一套 FlashKDA 接口与数值约束下，SM100 原生路径能否产生足以覆盖迁移成本的实际收益？", textStyle: { bold: true, color: C.ink } },
  ]], 76, 158, 1090, 62, { size: 25, valign: "middle" });

  const rows = [
    ["01", "复现与测量", "锁定执行身份、正确性与公开接口计时边界", "指令身份、利用率与计时", C.blue],
    ["02", "分析", "用双分支闭环生成可比证据，再把六问收敛为规律", "双分支 agent 与双层 IR", C.gold],
    ["03", "挑战", "沿四条路径推进候选，并用数据与知识完成晋级", "替换、布局与全栈", C.orange],
  ];
  for (let i = 0; i < rows.length; i++) {
    const y = 260 + i * 110;
    txt(s, rows[i][0], 80, y, 44, 28, { size: 16, color: rows[i][4], bold: true });
    txt(s, rows[i][1], 150, y - 3, 190, 36, { size: 25, color: C.ink, bold: true });
    txt(s, rows[i][2], 350, y - 2, 600, 38, { size: 21, color: C.text, valign: "middle" });
    txt(s, rows[i][3], 980, y - 2, 210, 38, { size: 18, color: rows[i][4], bold: true, align: "right", valign: "middle" });
    line(s, 150, y + 61, 1040, 0, C.rule, 1);
  }
  rich(s, [[
    { run: "回答  ", textStyle: { bold: true, color: C.green } },
    { run: "在已测 B300 工作负载上，SM100 值得以受条件保护的后端方式加入；有效迁移单位是全栈协同，而不是单独替换矩阵乘加指令。", textStyle: { color: C.ink } },
  ]], 150, 594, 1000, 62, { size: 22, valign: "middle", lineSpacing: 1.08 });
  notes(s, [
    "这一页定义整场答辩的主线。后续方法、结果与知识都回到这三步。",
    "最终回答强调 guarded backend 和全栈协同。",
    "来源：C1_FINAL_ASSIGNMENT_OUTLINE_20260910.md。",
  ]);
}

// 3. Operator structure
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 3, "背景", "KDA 递推结构", C.blue);

  label(s, "一个 chunk", 76, 160, C.muted, 120);
  rect(s, 80, 218, 150, 74, C.blueSoft, C.blue, 1, 4);
  txt(s, "state t−1", 92, 218, 126, 74, { size: 23, color: C.blue, bold: true, align: "center", valign: "middle" });
  rect(s, 310, 218, 170, 74, C.white, C.rule, 1, 4);
  txt(s, "K1  prepare", 325, 218, 140, 74, { size: 23, color: C.ink, bold: true, align: "center", valign: "middle" });
  rect(s, 560, 204, 250, 102, C.orangeSoft, C.orange, 1, 4);
  txt(s, "K2\n16-step recurrence", 580, 210, 210, 90, { size: 24, color: C.orange, bold: true, align: "center", valign: "middle", lineSpacing: 0.96 });
  txt(s, "state t", 930, 218, 140, 74, { size: 23, color: C.blue, bold: true, align: "center", valign: "middle" });
  arrowSegment(s, 230, 255, 310, 255, C.muted, 1.8);
  arrowSegment(s, 480, 255, 560, 255, C.muted, 1.8);
  arrowSegment(s, 810, 255, 930, 255, C.muted, 1.8);
  line(s, 1070, 255, 92, 0, C.muted, 1.8);
  txt(s, "next chunk", 1080, 274, 100, 22, { size: 14, color: C.muted, align: "center" });

  line(s, 76, 360, 1125, 0, C.rule, 1);
  const facts = [
    ["CHUNK = 16", "数值范围、16×16 求解与 SM80 MMA 形状共同形成的设计点。", C.orange],
    ["递推正确性", "一次调用要同时验证 output 与 final state；连续调用还要检查 state handoff。", C.blue],
    ["可利用并行", "chunk 内 K2 串行，但 head、value slice、sequence 与角色仍可并行。", C.green],
  ];
  for (let i = 0; i < facts.length; i++) {
    const y = 406 + i * 78;
    txt(s, facts[i][0], 80, y, 190, 30, { size: 20, color: facts[i][2], bold: true });
    txt(s, facts[i][1], 286, y - 2, 870, 38, { size: 21, color: C.text, valign: "middle" });
    if (i < facts.length - 1) line(s, 286, y + 52, 870, 0, C.rule2, 1);
  }
  notes(s, [
    "先解释为什么 KDA 不是普通 GEMM。K2 在 chunk 内递推，state 又把 chunk 串联起来。",
    "因此迁移要同时处理并行结构、数值状态和跨调用 correctness。",
    "来源：https://github.com/MoonshotAI/FlashKDA/blob/master/docs/20260420-flashkda-v1-deep-dive.md",
  ]);
}

// 4. Reproduction identity
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 4, "复现与测量", "官方路径仍使用 HMMA", C.blue);

  label(s, "执行身份", 76, 160, C.blue, 110);
  const chain = [
    ["FlashKDA", "1ce47ea"],
    ["CUTLASS", "5c149f5"],
    ["目标架构", "sm_103a"],
    ["反汇编", "实际加载 .so"],
  ];
  for (let i = 0; i < chain.length; i++) {
    const x = 80 + i * 190;
    txt(s, chain[i][0], x, 225, 150, 30, { size: 20, color: C.ink, bold: true, align: "center" });
    txt(s, chain[i][1], x, 262, 150, 24, { size: 15, color: C.muted, align: "center" });
    if (i < chain.length - 1) arrowSegment(s, x + 152, 246, x + 186, 246, C.rule, 1.5);
  }

  line(s, 850, 170, 0, 415, C.rule, 1);
  txt(s, "3,640", 900, 185, 270, 78, { size: 58, color: C.blue, bold: true, align: "center", valign: "middle" });
  txt(s, "条 HMMA 矩阵乘加指令", 870, 268, 330, 34, { size: 21, color: C.text, align: "center" });
  txt(s, "0", 900, 350, 270, 70, { size: 56, color: C.orange, bold: true, align: "center", valign: "middle" });
  txt(s, "SM100 原生 tcgen05、UTCMMA", 850, 424, 370, 34, { size: 20, color: C.text, align: "center" });
  txt(s, "源码命名不是证据；实际加载二进制的指令身份才是。", 880, 520, 310, 60, {
    size: 18, color: C.muted, align: "center", valign: "middle", lineSpacing: 1.08,
  });

  line(s, 80, 346, 700, 0, C.rule, 1);
  txt(s, "同一复现同时记录", 80, 382, 270, 30, { size: 21, color: C.ink, bold: true });
  const ids = ["实际加载模块路径", "源码与二进制 SHA-256", "Python 导入来源", "CUDA 头文件版本", "实际执行路径"];
  for (let i = 0; i < ids.length; i++) {
    txt(s, String(i + 1).padStart(2, "0"), 80, 434 + i * 38, 30, 24, { size: 14, color: C.blue, bold: true });
    txt(s, ids[i], 125, 432 + i * 38, 320, 28, { size: 18, color: C.text });
  }
  notes(s, [
    "我们不是根据源码名称判断指令，而是冻结 commit、实际加载模块与 hash，再对 SASS 计数。",
    "结果是 3,640 条静态 HMMA，TCGEN/UTCMMA 为 0。",
    "来源：SM100_MAINLINE_DELIVERY.md；C1_REPRODUCTION_ANALYSIS_CHALLENGE_CONTENT_20260910.md。",
  ]);
}

// 5. Measurement contract
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 5, "复现与测量", "计时与正确性边界", C.blue);

  txt(s, "计时边界", 76, 151, 150, 38, { size: 24, color: C.blue, bold: true, valign: "middle" });
  line(s, 76, 190, 150, 0, C.blue, 2.4);
  txt(s, "首次编译与一次性分配", 76, 218, 230, 34, { size: 18, color: C.muted, align: "center" });
  rect(s, 320, 205, 640, 64, C.blueSoft, C.blue, 1, 3);
  const timedStages = ["布局处理", "工作区", "内核执行", "状态回写"];
  for (let i = 0; i < timedStages.length; i++) {
    txt(s, timedStages[i], 320 + i * 160, 205, 160, 64, {
      size: 20, color: C.blue, bold: true, align: "center", valign: "middle",
    });
    if (i < timedStages.length - 1) line(s, 480 + i * 160, 221, 0, 32, "#B7C8DA", 1);
  }
  txt(s, "后续调用", 990, 218, 160, 34, { size: 18, color: C.muted, align: "center" });
  arrowSegment(s, 80, 292, 1160, 292, C.rule, 1.5);
  line(s, 320, 292, 640, 0, C.blue, 3);
  line(s, 320, 286, 0, 12, C.blue, 1.5);
  line(s, 960, 286, 0, 12, C.blue, 1.5);
  txt(s, "计时外", 112, 307, 90, 22, { size: 15, color: C.muted, bold: true, align: "center" });
  txt(s, "公开接口计时范围", 538, 307, 204, 24, { size: 16, color: C.blue, bold: true, align: "center" });

  line(s, 76, 372, 1120, 0, C.rule, 1);
  const rules = [
    ["01", "状态一致", "每次调用使用相同非零初态，重复测量时轮转状态槽位。"],
    ["02", "进程隔离", "每个实现使用独立进程，避免 CUDA 符号覆盖改变实际执行路径。"],
    ["03", "双重正确性", "同时检查输出与最终状态，并覆盖打包、尾块和连续两次调用。"],
    ["04", "正式性能", "冷 L2、CUDA 性能接口（CUPTI）、平衡顺序与 ABBA 交替复测。"],
  ];
  for (let i = 0; i < rules.length; i++) {
    const y = 410 + i * 54;
    txt(s, rules[i][0], 80, y, 36, 24, { size: 14, color: i < 2 ? C.blue : C.orange, bold: true });
    txt(s, rules[i][1], 130, y - 2, 150, 30, { size: 19, color: C.ink, bold: true });
    txt(s, rules[i][2], 285, y - 2, 870, 30, { size: 18, color: C.text });
  }
  notes(s, [
    "正式比较以 public call 为单位。布局、workspace 和 state copy-back 只要每次调用发生，就必须计时。",
    "隔离进程修正了同进程 symbol interposition 风险。",
    "来源：C1_REPRODUCTION_ANALYSIS_CHALLENGE_CONTENT_20260910.md §2。",
  ]);
}

// 6. Why dual-agent
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 6, "分析方法", "双分支的公平性", C.gold);

  txt(s, "架构收益只能从“同预算的最优边界”中被分离出来。", 76, 154, 900, 44, {
    size: 25, color: C.ink, bold: true,
  });
  txt(s, "两条支路的搜索能量可以不同，但机会预算与证据门槛必须一致。", 76, 198, 900, 32, { size: 18, color: C.muted });

  rect(s, 76, 286, 116, 276, C.navy, C.gold, 1.5, 12);
  txt(s, "共同\n问题", 92, 322, 84, 78, { size: 21, color: C.white, bold: true, align: "center", valign: "middle", lineSpacing: 0.95 });
  txt(s, "同一\nworkload", 92, 438, 84, 60, { size: 15, color: "#C9D3DC", align: "center", valign: "middle", lineSpacing: 0.95 });

  // 两条连续水平支路使用不同线宽表达搜索投入，避免视觉上出现断裂。
  line(s, 192, 338, 530, 0, C.blue, 8);
  ellipse(s, 330, 326, 24, 24, "#7DB9DF");
  ellipse(s, 548, 323, 30, 30, "#7DB9DF");
  ellipse(s, 704, 326, 24, 24, C.green);
  txt(s, "HMMA", 222, 276, 150, 32, { size: 25, color: C.blue, bold: true });
  txt(s, "H0  官方路径", 294, 365, 160, 26, { size: 17, color: C.muted });
  txt(s, "H1  同预算最优", 500, 365, 190, 26, { size: 18, color: C.blue, bold: true });
  txt(s, "局部闭包", 650, 298, 100, 24, { size: 16, color: C.green, bold: true, align: "right" });

  line(s, 192, 482, 530, 0, C.orange, 12);
  ellipse(s, 330, 468, 28, 28, "#F0A078");
  ellipse(s, 548, 464, 36, 36, "#F0A078");
  ellipse(s, 704, 470, 24, 24, C.green);
  txt(s, "tcgen05", 222, 526, 180, 32, { size: 25, color: C.orange, bold: true });
  txt(s, "T0  同预算接入", 286, 430, 190, 26, { size: 17, color: C.muted });
  txt(s, "T1 / T2  布局与全栈", 492, 520, 230, 26, { size: 18, color: C.orange, bold: true });

  line(s, 816, 250, 0, 372, C.rule, 1);
  txt(s, "共享证据合同", 866, 262, 350, 40, { size: 27, color: C.ink, bold: true });
  const shared = [
    ["工作负载", "q / k / v、状态与打包边界一致", C.blue],
    ["机会预算", "候选数与搜索机会按轮匹配", C.gold],
    ["证据门槛", "身份、双重正确性与 public-full", C.orange],
  ];
  for (let i = 0; i < shared.length; i++) {
    const y = 326 + i * 78;
    txt(s, String(i + 1).padStart(2, "0"), 866, y, 34, 24, { size: 14, color: shared[i][2], bold: true });
    txt(s, shared[i][0], 914, y - 4, 210, 30, { size: 20, color: shared[i][2], bold: true });
    txt(s, shared[i][1], 914, y + 27, 286, 30, { size: 15, color: C.text });
  }
  line(s, 866, 568, 354, 0, C.rule, 1);
  rich(s, [[
    { run: "比较对象  ", textStyle: { bold: true, color: C.green } },
    { run: "两条支路在同一合同下得到的最优边界。", textStyle: { color: C.ink } },
  ]], 866, 590, 354, 50, { size: 18, lineSpacing: 1.08 });
  notes(s, [
    "双分支 Agent 的动机是公平性。HMMA 和 tcgen05 各自独立优化，再用同一 workload 和同一 gate 比较。",
    "H0/H1 分离旧路径的优化空间，T0/T1/T2 分离 tcgen05 接入与全栈路径。",
    "来源：FRAMEWORK_SCOPE_CORRECTION_20260910.md。",
  ]);
}

// 7. Agent loop
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  const svg = await fs.readFile(agentLoopFigurePng);
  s.images.add({
    blob: svg,
    contentType: "image/png",
    alt: "双分支 Agent 的知识驱动探索闭环。互补角色生成 typed patch，HMMA 与 TCGEN05 分支在共同证据合同下实验，判断写回 scoped Experience IR，并改变下一轮搜索空间。",
    fit: "contain",
    position: { left: 0, top: 0, width: W, height: H },
  });
  brandLockup(s, { dark: false, y: 25, compact: true });
  notes(s, [
    "这一页强调 Agent 的核心产物是判断和可复用知识，不是一串 benchmark 数字。",
    "五类角色只提交 typed patch；协调器先做冲突、物理协议和语义去重检查，再把合法候选分配到 HMMA 与 TCGEN05 两条支路。",
    "证据合同检查实际路径、output 与 final state，以及同 workload 和同计时范围。判断写回 scoped Experience IR，已关闭旋钮不再重复采样。",
    "来源：STAGE12_TYPED_IR_SEARCH_CLOSURE_20260911.md §2.3、§3.3。",
  ]);
}

// 8. Dual IR
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  const svg = await fs.readFile(typedIrFigurePng);
  s.images.add({
    blob: svg,
    contentType: "image/png",
    alt: "P3、BF16 ROUND 与 P4 的 Typed IR 流水。P3 和 P4 共用一次 TMEM 分配，在 ROUND 语义边界通过 32 字节 swizzled shared carrier 传递 BF16 值。",
    fit: "contain",
    position: { left: 0, top: 0, width: W, height: H },
  });
  brandLockup(s, { dark: false, y: 25, compact: true });
  notes(s, [
    "Typed IR 把 shape、dtype、layout、carrier、pipeline 和 lifetime 变成可检查字段。",
    "P3 与 P4 只分配一次 TMEM，并复用同一累加列。P3 的 FP32 D fragment 在明确 BF16 ROUND 后，通过 32B swizzled shared carrier 形成 P4 的输入。",
    "ROUND 是语义边界，规范化器不得跨过；D fragment 到 A TMEM 也必须有显式 layout proof。",
    "来源：STAGE12_TYPED_IR_SEARCH_CLOSURE_20260911.md §3.1–3.2、§4.3。",
  ]);
}

// 9. Six questions become three laws
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 9, "分析结论", "三个关键结论", C.gold);

  const rows = [
    {
      y: 164, q: "Q1 · Q5", title: "数值约束决定分块设计", color: C.blue,
      body: "CHUNK16 同时约束指数恢复、16×16 求解和 BF16 状态舍入；扩大到 C32/C64 需要重新缩放或分块求解。",
      ev: "输出、最终状态、连续两次调用",
    },
    {
      y: 326, q: "Q2 · Q4", title: "分块合法不等于整体高效", color: C.orange,
      body: "tcgen05 还要承担张量内存（TMEM）、地址描述符、同步和读回；H12 受低并行度与关键路径限制。",
      ev: "V128 0.920×   计算/显存 2.64%/1.24%",
    },
    {
      y: 488, q: "Q3 · Q6", title: "并行机会决定迁移和部署方式", color: C.green,
      body: "注意力头、数值维、序列和生产/消费角色仍可并行；收益经工作负载认证后进入条件后端。",
      ev: "H12 value 切片时延 −27.0%",
    },
  ];
  for (const r of rows) {
    txt(s, r.q, 80, r.y, 110, 26, { size: 15, color: r.color, bold: true });
    txt(s, r.title, 210, r.y - 4, 390, 38, { size: 24, color: C.ink, bold: true });
    txt(s, r.body, 210, r.y + 45, 720, 64, { size: 19, color: C.text, lineSpacing: 1.1 });
    txt(s, r.ev, 950, r.y + 22, 245, 54, { size: 17, color: r.color, bold: true, align: "right", valign: "middle" });
    line(s, 210, r.y + 132, 985, 0, C.rule, 1);
  }
  notes(s, [
    "课程六问不需要在台上逐条念。把它们归纳成数值合同、成本结构、并行与部署三条规律，右侧给关键证据。",
    "Q1/Q5 对应 CHUNK 与 BF16；Q2/Q4 对应 tile 与瓶颈；Q3/Q6 对应并行度与 v2 决策。",
    "来源：team/TASK.md；C1_REPRODUCTION_ANALYSIS_CHALLENGE_CONTENT_20260910.md §3。",
  ]);
}

// 10. Challenge map
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 10, "挑战", "四条挑战路径的投入与回报", C.orange);

  txt(s, "核心探针与 public-full 分开标注，1.0× 表示各自基线。", 76, 154, 780, 32, { size: 18, color: C.muted });

  const plotLeft = 240;
  const plotRight = 1138;
  const baselineY = 474;
  line(s, plotLeft, 224, 0, 344, C.rule, 1.2);
  line(s, plotLeft, 568, plotRight - plotLeft, 0, C.rule, 1.2);
  line(s, plotLeft, baselineY, plotRight - plotLeft, 0, C.ink, 1.2);
  txt(s, "1.0×", 176, baselineY - 12, 52, 24, { size: 14, color: C.ink, bold: true, align: "right" });
  txt(s, "实测加速比", 78, 224, 130, 28, { size: 17, color: C.muted });

  const xDirect = 330;
  const xV16 = 555;
  const xH3 = 790;
  const xCake = 1040;
  const yDirect = 490;
  const yV16Core = 330;
  const yV16Full = 520;
  const yH3 = 496;
  const yCake = 250;

  ellipse(s, xDirect - 12, yDirect - 12, 24, 24, C.red, C.white, 2);
  txt(s, "0.920×", xDirect - 54, yDirect - 48, 108, 28, { size: 18, color: C.red, bold: true, align: "center" });

  line(s, xV16, yV16Core, 0, yV16Full - yV16Core, C.orange, 3);
  ellipse(s, xV16 - 14, yV16Core - 14, 28, 28, C.green, C.white, 2);
  ellipse(s, xV16 - 12, yV16Full - 12, 24, 24, C.red, C.white, 2);
  txt(s, "core  1.615×", xV16 - 124, yV16Core - 42, 118, 28, { size: 17, color: C.green, bold: true, align: "right" });
  txt(s, "public  0.779×", xV16 + 18, yV16Full - 12, 150, 28, { size: 17, color: C.red, bold: true });
  txt(s, "承载成本", xV16 + 20, 402, 104, 26, { size: 15, color: C.orange, bold: true });

  ellipse(s, xH3 - 12, yH3 - 12, 24, 24, C.red, C.white, 2);
  txt(s, "0.906×", xH3 - 54, yH3 - 48, 108, 28, { size: 18, color: C.red, bold: true, align: "center" });

  ellipse(s, xCake - 38, yCake - 38, 76, 76, C.greenSoft, C.green, 1.2);
  ellipse(s, xCake - 18, yCake - 18, 36, 36, C.green, C.white, 3);
  txt(s, "2.482×", xCake - 62, yCake - 72, 124, 30, { size: 23, color: C.green, bold: true, align: "center" });
  txt(s, "9 / 9 获益", xCake - 64, yCake + 46, 128, 26, { size: 17, color: C.green, bold: true, align: "center" });

  const paths = [
    [xDirect, "01  Direct", "矩阵乘加替换"],
    [xV16, "02  V16", "布局与 carrier"],
    [xH3, "03  H3", "数据流重排"],
    [xCake, "04  CAKE", "全栈协同"],
  ];
  for (const [x, title, subtitle] of paths) {
    line(s, x, 560, 0, 16, C.rule, 1.2);
    txt(s, title, x - 92, 586, 184, 28, { size: 18, color: C.ink, bold: true, align: "center" });
    txt(s, subtitle, x - 100, 616, 200, 24, { size: 14, color: C.muted, align: "center" });
  }

  rich(s, [[
    { run: "轨迹展示的不是线性进步。 ", textStyle: { bold: true, color: C.orange } },
    { run: "Direct 和 H3 留下负边界，V16 揭示 carrier 问题，CAKE 才将这些知识组成完整收益。", textStyle: { color: C.ink } },
  ]], 80, 662, 1090, 32, { size: 17, valign: "middle" });
  notes(s, [
    "挑战不是只写一个 tcgen kernel，而是用四条互相证伪的路径逐步扩大改造范围。",
    "Direct 和 H3 失败，V16 揭示 core/carrier 差异，CAKE 证明全栈协同存在显著工程收益。",
    "来源：SM100_MAINLINE_DELIVERY.md；ROUND1_REVIEW.md。",
  ]);
}

// 11. tcgen mechanism result
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 11, "挑战结果", "tcgen05 的承载成本", C.orange);

  txt(s, "每一行使用自己的基线和计时范围", 76, 154, 600, 28, { size: 17, color: C.muted });
  const x0 = 340;
  const max = 1.8;
  const axisW = 690;
  for (const tick of [0, 0.5, 1.0, 1.5]) {
    const x = x0 + (tick / max) * axisW;
    line(s, x, 196, 0, 10, C.rule, 1);
    txt(s, `${tick.toFixed(1)}×`, x - 25, 171, 50, 20, { size: 13, color: C.muted, align: "center" });
  }
  const bx = x0 + (1 / max) * axisW;
  line(s, bx, 207, 0, 360, C.ink, 1.3);

  const rows = [
    [220, "Direct V128 直接替换", "完整承载层", 0.919742, C.red],
    [340, "V16 首选布局", "核心探针 L0", 1.615021, C.green],
    [460, "V16 标量重构", "承载层 L1", 0.778523, C.red],
  ];
  for (const r of rows) {
    txt(s, r[1], 78, r[0], 220, 32, { size: 20, color: C.ink, bold: true });
    txt(s, r[2], 78, r[0] + 36, 160, 22, { size: 14, color: C.muted });
    const bw = (r[3] / max) * axisW;
    rect(s, x0, r[0], bw, 38, r[4], "none", 0, 3);
    txt(s, `${r[3].toFixed(3)}×`, x0 + bw + 12, r[0], 110, 38, { size: 21, color: r[4], bold: true, valign: "middle" });
  }

  line(s, 1080, 186, 0, 388, C.rule, 1);
  txt(s, "承载层成本", 1098, 208, 124, 30, { size: 19, color: C.ink, bold: true, align: "center" });
  for (const [i, item] of ["地址描述符", "张量内存", "提交 / 等待", "结果读回", "数据布局"].entries()) {
    txt(s, item, 1098, 270 + i * 52, 120, 28, { size: 16, color: i === 4 ? C.orange : C.muted, align: "center" });
    if (i < 4) line(s, 1110, 308 + i * 52, 96, 0, C.rule2, 1);
  }

  rich(s, [[
    { run: "知识  ", textStyle: { bold: true, color: C.orange } },
    { run: "tcgen05 核心计算有潜力，但布局、张量内存与同步协议决定公开路径是否胜出。", textStyle: { color: C.ink } },
  ]], 78, 610, 1030, 42, { size: 21, valign: "middle" });
  notes(s, [
    "V16 preferred-layout core 达到 1.615×，说明 native compute 有潜力；加入 scalar rematerialization 的 carrier 后只有 0.779×。",
    "因此迁移优化的重点不是 opcode，而是 carrier 与布局。",
    "来源：experiments/sm100_open_round1/TCGEN_OPTION_VALUE.md。",
  ]);
}

// 12. Full-stack result
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 12, "挑战结果", "CAKE 全栈路径：9/9 获益", C.green);

  txt(s, "公开接口完整加速比 = 官方时延 / 候选时延", 78, 154, 620, 28, { size: 17, color: C.muted });
  txt(s, "图示代表工作负载；H3 为局部重排，CAKE 为全栈协同", 730, 154, 450, 28, { size: 16, color: C.muted, align: "right" });
  const x0 = 360;
  const axisW = 720;
  const max = 2.6;
  const baseline = x0 + (1 / max) * axisW;
  line(s, baseline, 198, 0, 356, C.ink, 1.3);
  txt(s, "1.0", baseline - 20, 560, 40, 22, { size: 13, color: C.ink, bold: true, align: "center" });
  evidenceRow(s, 214, "H3 · H12", "公开接口完整计时", 0.9055, max, C.red, x0, axisW);
  evidenceRow(s, 306, "H3 · H96", "公开接口完整计时", 0.7581, max, C.red, x0, axisW);
  evidenceRow(s, 414, "CAKE · H12", "完整计时 · T2", 2.4823, max, C.green, x0, axisW);
  evidenceRow(s, 506, "CAKE · H96", "完整计时 · T2", 2.2532, max, C.green, x0, axisW);

  line(s, 80, 598, 1110, 0, C.rule, 1);
  rich(s, [[
    { run: "知识  ", textStyle: { bold: true, color: C.green } },
    { run: "代数重排不足以缩短完整路径；驻留、角色、同步屏障、流水、布局与分发的协同才形成稳定收益。", textStyle: { color: C.ink } },
  ]], 80, 618, 1090, 42, { size: 20, valign: "middle" });
  notes(s, [
    "H3 correctness 通过，但 public-full H12/H96 几何平均分别为 0.9055× 和 0.7581×。",
    "CAKE/official 在九个 profile 全部正收益，H12/H96 几何平均分别为 2.4823× 和 2.2532×。",
    "这使全栈协同成为本项目最关键的迁移知识。",
    "来源：https://arxiv.org/abs/2608.12629；SM100_MAINLINE_DELIVERY.md。",
  ]);
}

// 13. Dual-agent discoveries
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  const svg = await fs.readFile(closureFigurePng);
  s.images.add({
    blob: svg,
    contentType: "image/png",
    alt: "有限 Typed Grammar 的局部知识闭包。声明域内的候选均有终态证据，missing_design_ids 为空；完整 P1 到 P6 public-call、可验证 D 到 A 映射和新算法族仍在开放边界外。",
    fit: "contain",
    position: { left: 0, top: 0, width: W, height: H },
  });
  brandLockup(s, { dark: false, y: 25, compact: true });
  notes(s, [
    "闭包只覆盖图中的有限 typed grammar 与已测 profile。每个候选都以 qualified、below gate 或 correctness rejected 结束，证书中的 missing_design_ids 为空。",
    "这不是全局 GPU 最优性证明。完整 P1 到 P6 public-call、可验证 D 到 A TMEM 映射，以及新的 M64、M128、split、slab 算法族仍在开放边界外。",
    "长驻留 P3/P4 探针给出 1.43 到 1.45 倍的机制上界；是否成为完整 SM100 后端仍由 public-full 正确性和同口径计时决定。",
    "来源：STAGE12_TYPED_IR_SEARCH_CLOSURE_20260911.md §1、§4–6；04_evidence/agent_rounds/round9_closure_certificate.json。",
  ]);
}

// 14. Deployment
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  header(s, 14, "结论", "证据驱动的条件部署", C.green);

  txt(s, "每个请求依次通过执行身份、递推正确性和 public-full 证据，再决定后端。", 76, 154, 1080, 40, { size: 22, color: C.ink, bold: true });

  ellipse(s, 82, 388, 84, 84, C.navy, C.gold, 1.8);
  txt(s, "运行\n请求", 96, 398, 56, 64, { size: 20, color: C.white, bold: true, align: "center", valign: "middle", lineSpacing: 0.95 });
  txt(s, "device · profile", 62, 484, 126, 24, { size: 14, color: C.muted, align: "center" });

  // 学术化的路由图：一条连续证据主线，末端分成认证路径和兼容回退。
  line(s, 166, 430, 574, 0, C.muted, 4);
  const checks = [
    [320, "执行身份", "实际 module", C.blue, 330],
    [510, "递推正确性", "output + final state", C.orange, 468],
    [700, "公开完整计时", "public-full", C.green, 330],
  ];
  for (const [x, title, subtitle, color, labelY] of checks) {
    ellipse(s, x - 13, 417, 26, 26, color, C.white, 2);
    txt(s, title, x - 90, labelY, 180, 28, { size: 19, color, bold: true, align: "center" });
    txt(s, subtitle, x - 110, labelY + 31, 220, 24, { size: 14, color: C.muted, align: "center" });
  }

  line(s, 740, 350, 0, 170, C.muted, 2.5);
  line(s, 740, 350, 410, 0, C.green, 10);
  line(s, 740, 520, 410, 0, C.blue, 7);
  ellipse(s, 1129, 338, 24, 24, C.green);
  ellipse(s, 1132, 510, 20, 20, C.blue);
  txt(s, "全部通过", 760, 310, 110, 24, { size: 15, color: C.green, bold: true });
  txt(s, "否则", 760, 534, 70, 24, { size: 15, color: C.blue, bold: true });

  txt(s, "SM100 专用后端", 872, 254, 278, 38, { size: 27, color: C.green, bold: true, align: "right" });
  txt(s, "T2 · 仅认证 profile", 872, 298, 278, 28, { size: 18, color: C.green, align: "right" });
  txt(s, "HMMA 兼容后端", 872, 552, 278, 38, { size: 27, color: C.blue, bold: true, align: "right" });
  txt(s, "默认回退 · 保持跨代可用", 842, 594, 308, 28, { size: 17, color: C.muted, align: "right" });

  line(s, 76, 646, 1110, 0, C.rule, 1);
  rich(s, [[
    { run: "路由原则  ", textStyle: { bold: true, color: C.orange } },
    { run: "已认证 profile 才进入 SM100，未覆盖或证据不足时保留 HMMA。", textStyle: { color: C.ink } },
  ]], 80, 666, 1080, 30, { size: 18, valign: "middle" });
  notes(s, [
    "最终交付不是一个孤立 benchmark，而是一个可以实现的 dispatcher。",
    "资格 profile 进入 SM100 T2，其余条件走 HMMA 兼容路径。每个 route 都由执行身份、递推 correctness 和 public-full 证据晋级。",
    "来源：C1_FINAL_ASSIGNMENT_OUTLINE_20260910.md §7–9。",
  ]);
}

// 15. Paper excerpt and final answer
{
  const s = deck.slides.add();
  s.background.fill = C.white;
  txt(s, "论文方法图", 72, 48, 650, 50, { size: 40, color: C.ink, bold: true });
  txt(s, "论文图 1  方法框架摘录", 900, 60, 306, 24, { size: 15, color: C.muted, align: "right" });
  brandLockup(s, { dark: false, y: 24, compact: true });

  const bytes = await fs.readFile(paperExcerptPath);
  rect(s, 72, 138, 720, 400, C.white, C.rule, 1, 3).shadow = "shadow-lg";
  s.images.add({
    blob: bytes,
    contentType: "image/png",
    alt: "论文 Figure 1 核心片段：HMMA 与 tcgen05 双分支、可比较证据和双层记忆",
    fit: "contain",
    position: { left: 88, top: 152, width: 690, height: 370 },
  });

  label(s, "图中方法的中文释义", 842, 148, C.orange, 190);
  const points = [
    ["双分支搜索", "HMMA 与 tcgen05 独立优化"],
    ["证据式晋级", "实际路径、正确性、成本同时携带"],
    ["双层中间表示", "把运行数据转成带适用范围的经验"],
    ["部署策略", "已认证 SM100 路径，未覆盖 profile 回退 HMMA"],
  ];
  for (let i = 0; i < points.length; i++) {
    const y = 208 + i * 82;
    txt(s, points[i][0], 842, y, 190, 28, { size: 20, color: i % 2 === 0 ? C.blue : C.orange, bold: true });
    txt(s, points[i][1], 842, y + 34, 340, 36, { size: 17, color: C.text });
  }

  line(s, 72, 590, 1135, 0, C.rule, 1);
  rich(s, [[
    { run: "最终回答  ", textStyle: { bold: true, color: C.orange } },
    { run: "SM100 值得以受条件保护的后端方式加入 FlashKDA。", textStyle: { bold: true, color: C.ink } },
  ]], 74, 614, 930, 44, { size: 24, valign: "middle" });
  txt(s, "提问　奶龙必胜", 1010, 668, 195, 22, { size: 16, color: C.muted, align: "right" });
  notes(s, [
    "收尾回到论文方法图。双分支搜索、证据晋级和双层 IR 把迁移问题从一次性 benchmark 变成可积累的探索闭环。",
    "最终回答：SM100 值得以 guarded backend 的方式加入 FlashKDA。",
    "图片来源：output/pdf/runtime-profile-evolution-mainline-20260910.pdf，第 2 页 Figure 1 核心片段。",
  ]);
}

const previewDir = path.join(TMP_DIR, "academic-preview-refined-v5");
await fs.mkdir(previewDir, { recursive: true });
for (let i = 0; i < deck.slides.length; i++) {
  const slide = deck.slides.getItemAt(i);
  const png = await deck.export({ slide, format: "png", scale: 1.5 });
  await fs.writeFile(path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.layout.json`), await layout.text());
}
const montage = await deck.export({ format: "png", montage: true, scale: 0.72 });
await fs.writeFile(path.join(previewDir, "montage.png"), new Uint8Array(await montage.arrayBuffer()));

const requirements = {
  explicitTotalSlideCount: 15,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  requiredEmbeddedWorkbookChartOwnerSlides: [],
  materializeLiteralChartWorkbooks: false,
};
const fontPolicy = { basis: "design", families: [FONT_ZH], scriptFonts: { ea: FONT_ZH } };
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "C1_FlashKDA_SM100_奶龙必胜_署名版_candidate_20260911_v5.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await applyPlatformFonts(candidatePath);

const result = await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, `${path.basename(FINAL_PPTX)}.validation.json`),
});

console.log(JSON.stringify({ final: FINAL_PPTX, previewDir, result }, null, 2));
