# MARPE paper

本目录包含 C1 的 8 页论文固定版 [`main.pdf`](main.pdf)、LaTeX 源文件
[`main.tex`](main.tex)、参考文献 [`references.bib`](references.bib) 和模板来源说明
[`TEMPLATE_PROVENANCE.md`](TEMPLATE_PROVENANCE.md)。

署名：Yuyang Cai（蔡雨洋）、Ao Li（李奥）、Cheng Zhao（赵骋）；团队
“奶龙必胜”。

论文采用单栏 Letter 版式，叙述 FlashKDA SM80-HMMA→SM100 主线、
H0/H1/T0/T1/T2 挑战、开放 proposal、Program IR、带作用域的 Experience IR、
执行身份，以及 B300 上八轮双分支闭环。结果明确区分 qualified result、scoped
screen 与 mechanism probe；没有在缺少单 Agent 配对消融时声称 multi-agent
优于单 Agent。

## 构建

安装 Tectonic 后，在本目录运行：

```bash
make
```

需要强制重建时运行：

```bash
make rebuild
```

也可以直接运行 `tectonic main.tex`。产物写入本目录的 `main.pdf`。

主线范围修正见
[`../03_reports/FRAMEWORK_SCOPE_CORRECTION_20260910.md`](../03_reports/FRAMEWORK_SCOPE_CORRECTION_20260910.md)，
最终知识饱和与停止边界见
[`../03_reports/DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md`](../03_reports/DEFAULT_KNOWLEDGE_SATURATION_REVIEW_20260910.md)。
