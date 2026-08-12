# 科学写作 LLM Judge 论文：从当前结果到投稿的执行指南

> 版本：2026-08-04
>
> 适用项目：本仓库中的科学写作评价、CoT 监督、Paper Align 与 RAFT 实验
>
> 目标：不等待“拼凑新方法”，先把已有稳定发现写成一篇论点完整、证据边界清楚、可以投稿的实证研究论文。

---

## 使用本指南前：先把论文工程搭起来

这一部分回答最具体的问题：**用什么软件、在哪个目录写、每个文件放什么、怎样把结果变成表图、怎样编译 PDF，以及如何让 AI 参与但不制造错误。**

### A. 本项目唯一推荐的写作方式

采用下面这套约定：

> **所有论文工作都从 `paper_workspace/` 执行。论文源文件只放在 `manuscript/` 中，以 LaTeX 为正式稿；中文思考放 `manuscript/notes/`，英文投稿正文放 `manuscript/sections/`。实验结果仍以上一级的 `../eval_output/` 为原始来源，任何表图必须由 `analysis/` 中的脚本生成。**

不要把正式正文写在以下位置：

- 项目根目录 `../README.md`：这是研究日志，不是论文；
- `literature/`：这里保存相关论文和阅读笔记，不是自己的稿件；
- 聊天窗口：AI 输出只能作为候选草稿，确认后才进入 `.tex`；
- Word、Overleaf、本地 LaTeX 三处同时修改：会立即产生版本分叉；
- `../eval_output/results/evaluation_analysis.md`：这是自动分析产物，不是论文 Results。

### B. 当前机器有什么工具

截至 2026-08-04，下面的写作工具已经实际安装并通过编译验证，不需要
`sudo`，也不需要再创建环境：

| 工具 | 当前状态 | 用途 |
|---|---|---|
| Cursor/VS Code 命令 | 已安装 | 编辑 `.tex`、`.md`、`.py`，查看 Git diff |
| LaTeX Workshop 10.16.1 | 已安装并配置 | 保存、编译和预览论文 |
| Git | 已安装 | 版本管理、回退、标记投稿版本 |
| Python 3.13 + `../.venv-paper` | 已安装并验证 | 表格、统计检验、作图、PDF 检查 |
| `uv` | 已安装 | 为论文分析创建独立 Python 环境 |
| TeX Live 2026 + `latexmk` 4.88 | 已安装到 `../.tools/texlive/2026/` | 用官方 ACL 模板编译 PDF |
| Tectonic 0.17.0 | 已安装到 `../.tools/tectonic/` | 备用 LaTeX 引擎，当前不作为默认编译器 |
| Poppler 24.02.0 | 已安装到 `../.tools/poppler/` | 检查 PDF 元数据、纸张和字体嵌入 |
| Pandoc | 未安装且当前不需要 | 仅在期刊明确要求 DOCX 时再安装 |
| Zotero | 可选桌面工具 | 文献较多时管理元数据；不影响现在开始写作 |

本项目推荐工具组合：

| 工作 | 推荐工具 | 最终产物 |
|---|---|---|
| 写正文 | Cursor 或 VS Code + LaTeX Workshop | `manuscript/sections/*.tex` |
| 管文献 | Zotero + Better BibTeX | `manuscript/references.bib` |
| 记录论证 | Markdown + CSV | `manuscript/notes/*.md/csv` |
| 数据分析 | Python、pandas、SciPy、statsmodels | `analysis/derived/*.csv` |
| 作图 | matplotlib + seaborn | `manuscript/figures/*.pdf` |
| 表格 | Python 生成 LaTeX | `manuscript/tables/*.tex` |
| 编译 | 官方 ACL template + `latexmk` | `manuscript/build/main.pdf` |
| 版本管理 | Git | commit、tag、submission snapshot |
| 投稿 | OpenReview（ARR）或期刊系统 | 匿名 PDF、supplementary、metadata |

### C. 已创建的目录

论文工程已经创建完成。不要再运行初始化脚本，直接按照下面的职责使用：

```text
my-small-paper/
├── paper_workspace/                    # 所有论文写作资产
│   ├── README.md                       # 统一入口
│   ├── Makefile                        # 编译和检查命令
│   ├── docs/                           # 指南、研究分析与综述
│   ├── literature/                     # 参考论文和阅读笔记
│   ├── manuscript/                     # 论文唯一正式源文件
│   │   ├── main.tex                    # 标题、摘要和章节入口
│   │   ├── references.bib              # 唯一参考文献库
│   │   ├── sections/                   # 英文正文
│   │   ├── notes/                      # 中文论证与证据表
│   │   ├── figures/                    # 投稿图片
│   │   ├── tables/                     # 自动生成表格
│   │   ├── appendix/                   # 附录
│   │   └── build/                      # 可再生编译产物
│   ├── analysis/                       # 统计、图表脚本与派生数据
│   ├── reproducibility/                # 环境、数据和图表溯源
│   └── submission/                     # 投稿快照
├── training/                           # 原项目训练代码，不移动
├── data/                               # 原项目数据，不移动
├── eval_output/                        # 原始评测输出，不移动
└── MoDE/                               # 原项目模块，不移动
```

进入 `paper_workspace/` 后，内部目录直接使用 `manuscript/`、`analysis/`、
`literature/`、`reproducibility/` 和 `submission/`；原项目资源使用
`../training/`、`../data/` 和 `../eval_output/`。不要移动原项目目录。

根目录 `.gitignore` 已经忽略项目本地环境和构建产物：

```gitignore
# Paper-local environments and build artifacts
.venv-paper/
paper_workspace/manuscript/build/
```

不要全局忽略 `*.pdf`，因为最终论文图片本来就建议保存为 PDF。

### D. 已安装的 LaTeX 与编辑器支持

本项目使用用户目录内的 TeX Live，不依赖系统包管理器。Cursor 的 LaTeX
Workshop 已配置为调用这套 `latexmk + pdfLaTeX` 工具链。首次开始只需：

```bash
cd /home/messi/pyprojects/paper/my-small-paper/paper_workspace
source analysis/activate.sh
make paper-env-check
make paper-check
```

成功后会看到 `PASS`，PDF 位于：

```text
manuscript/build/main.pdf
```

`analysis/activate.sh` 会同时激活上一级的 `../.venv-paper`、项目缓存目录和
TeX Live 路径，因此不要手工修改全局 `PATH`。工作区 `Makefile` 也使用绝对路径，
即使忘记激活环境，`make paper` 仍能编译。

当前配置不需要 Overleaf。若以后多人协作改用 Overleaf，必须明确唯一真源：
要么所有正文都以本仓库为准，要么所有正文都以 Overleaf 为准，不能两边同时
修改。

### E. 已取得官方 ACL 模板

官方 `acl.sty` 与 `acl_natbib.bst` 已原样放在 `manuscript/`，`main.tex`
已经按匿名 review 模式组装好。模板来源、版本 commit 和文件校验信息记录在
`manuscript/TEMPLATE_PROVENANCE.md`。不要修改这两个官方样式文件；若投稿轮次
发布了更新模板，先对比官方版本，再整体替换。

`main.tex` 只保留入口内容。章节通过 `\input{}` 引入，例如：

```latex
\documentclass[11pt]{article}
\usepackage[review]{acl}

% 保留官方模板要求的 package，不修改 acl.sty。
\title{When Chain-of-Thought Supervision Hurts Scientific-Writing Judges}
\author{Anonymous ACL submission}

\begin{document}
\maketitle

\begin{abstract}
% 摘要最后写。投稿前这里不能有占位文本。
\end{abstract}

\input{sections/01_introduction}
\input{sections/02_related_work}
\input{sections/03_task_and_setup}
\input{sections/04_main_results}
\input{sections/05_mechanism_analysis}
\input{sections/06_discussion}
\input{sections/09_conclusion}
\input{sections/07_limitations}
\input{sections/08_ethics}

\bibliography{references}
\appendix
\input{appendix/full_results}

\end{document}
```

这里的 preamble 只是结构示意。真正创建 `main.tex` 时，以下载的官方 `acl_latex.tex` 为基础，不能用这段示意替换官方必需设置。

### F. 已创建论文专用 Python 环境

`../.venv-paper` 已安装 NumPy、pandas、SciPy、statsmodels、matplotlib、seaborn、
Pillow 和 pypdf，版本锁定在 `analysis/requirements-lock.txt`。使用时执行：

```bash
source analysis/activate.sh
python -c "import numpy, pandas, scipy, statsmodels, matplotlib, seaborn, pypdf"
```

分析依赖清单维护在 `analysis/requirements.in`；只有脚本确实需要新库时才添加，
不要把作图依赖装进模型训练环境。若环境损坏，可按 `analysis/README.md` 中的
重建步骤恢复。

正式图表生成流程固定为：

```text
../eval_output/results/**/predictions.jsonl + metrics.json
                         |
                         v
analysis/scripts/*.py
                         |
                         +--> analysis/derived/*.csv
                         +--> manuscript/tables/*.tex
                         +--> manuscript/figures/*.pdf
```

不要从 `comparison_table.md` 手工复制数字进论文。Markdown 表适合阅读，不适合作为最终数字源。

### G. 安装和配置 Zotero

桌面端使用 Zotero 管理文献，建议安装 Better BibTeX 插件。建立一个单独 collection，例如 `Scientific-Writing-Judge-Paper`，并自动导出到：

```text
manuscript/references.bib
```

文献进入正文前必须完成四步：

1. 将 PDF 和正确元数据收入 Zotero；
2. 至少阅读与本文主张相关的原文段落；
3. 在 `manuscript/notes/02_literature_matrix.csv` 记录它支持什么、原文位置和不能支持什么；
4. 通过 citation key 在正文引用，例如 `\citep{wei2022chain}`。

建议的文献矩阵字段：

```csv
citation_key,title,topic,claim_supported,evidence_location,comparison_to_ours,limitation,human_verified
```

不要把 Google Scholar、AI 或论文摘要自动生成的 BibTeX 直接视为正确。作者、题名、年份、venue 和 DOI 都要回到 DOI 页面、ACL Anthology、出版社或论文原页核对。

### H. 写作语言和落笔位置

采用“两层写作”，但避免最后整篇机器翻译：

1. **中文思考层：** 在 `manuscript/notes/` 写事实、论证和段落目标；
2. **英文投稿层：** 确认逻辑后，直接在 `manuscript/sections/*.tex` 写英文。

例如，先在 `00_paper_contract.md` 写：

```markdown
## RQ2

主张：CoT 退化至少包含输出接口失败和有效标签边界漂移。

证据：
- Positioning Check：overall 与 valid-only 大幅分离；新增受损主要为 null/format。
- Grounding/Verifiability：无无效输出，但存在方向一致的标签迁移。

不能写：已经证明模型内部存在梯度冲突。
```

然后在 `05_mechanism_analysis.tex` 写英文段落。每个段落严格使用：

```text
主题句（回答哪个 RQ）
→ 定量证据（表/图、差值、CI、两个 seed）
→ 解释（最小充分解释）
→ 边界（还有什么不能确定）
```

一个结果段落的 LaTeX 骨架：

```latex
\paragraph{Output failures dominate positioning-check degradation.}
Under direct-label inference, the CoT-supervised model shows a large gap
between overall and valid-only performance (Table~\ref{tab:main-results}).
Paired prediction analysis attributes most newly introduced errors to null or
malformed outputs rather than incorrect valid labels
(Figure~\ref{fig:error-decomposition}). This pattern supports an interface
mismatch explanation for this task, but it does not explain the valid-label
shifts observed on Grounding and Verifiability.
```

这段只能在表图数字最终冻结后补入精确数值，不能凭记忆填写。

### I. 每个章节具体怎么开始

第一轮写作不追求语言漂亮，只追求“事实齐、论证闭合”。按下面顺序操作。

#### I.1 `03_task_and_setup.tex`：第一个写

打开并同时参考：

- `../training/configs/<task>/*.yaml`：训练条件；
- `../eval_output/results/**/resolved_config.json`：实际执行配置；
- `../data/<task>/`：数据字段和样本；
- `../training/supervision/*.py`：损失实现；
- `../training/trainers/*.py`：trainer 行为。

先写表格，不先写散文：

| Task | Scientific-writing construct | Input | Label | Train/Test | Metric |
|---|---|---|---|---|---|
| `rw_gen_coherence` | Coherence | 待从数据样本确认 | 待确认 | 待统计 | 待确认 |
| `rw_gen_positioning_type` | Positioning type | 待确认 | 待确认 | 待统计 | 待确认 |
| `rw_gen_positioning_check` | Positioning validity | 待确认 | 待确认 | 待统计 | 待确认 |
| `rev_util_actionability` | Review actionability | 待确认 | 待确认 | 待统计 | 待确认 |
| `rev_util_helpfulness` | Review helpfulness | 待确认 | 待确认 | 待统计 | 待确认 |
| `rev_util_grounding_specificity` | Grounding/specificity | 待确认 | 待确认 | 待统计 | 待确认 |
| `rev_util_verifiability` | Review verifiability | 待确认 | 待确认 | 待统计 | 待确认 |

这里的“待确认”是写作任务，不是允许进入投稿 PDF 的占位符。每项都必须从数据和配置中读取后替换。

随后写五个小节：Tasks and Data、Training Views、Learning Objectives、Inference Protocols、Metrics and Statistical Tests。

#### I.2 `04_main_results.tex`：第二个写

唯一数字入口应是：

```text
../eval_output/results/evaluation_analysis_records.json
```

但写入正文前，必须回查相应任务目录中的 `metrics.json` 与 `predictions.jsonl`。创建 `analysis/scripts/build_main_results.py`，输出：

```text
analysis/derived/main_results.csv
manuscript/tables/main_results.tex
```

在 `.tex` 中只写：

```latex
\input{tables/main_results}
```

不要复制粘贴 105 条记录，也不要手工维护两份表。

#### I.3 `05_mechanism_analysis.tex`：第三个写

创建三个脚本，每个脚本只负责一种分析：

```text
analysis/scripts/build_paired_transitions.py
analysis/scripts/build_error_decomposition.py
analysis/scripts/build_raft_diagnostics.py
```

预期输出：

```text
analysis/derived/paired_transitions.csv
analysis/derived/error_decomposition.csv
analysis/derived/raft_diagnostics.csv
manuscript/figures/paired_transitions.pdf
manuscript/figures/error_decomposition.pdf
manuscript/figures/raft_diagnostics.pdf
```

先确保同一 `sample_id` 可以跨条件对齐，再画图。对不上的记录必须显式报告，不能按行号强行匹配。

#### I.4 `02_related_work.tex`：结果稳定后写

先在文献矩阵中建立四组：LLM-as-a-Judge、CoT/rationale supervision、rationale harm/task dependence、ordinal/regression-aware training。每组回答：

1. 已有工作解决了什么；
2. 使用了什么数据和对照；
3. 与本文最接近的结论是什么；
4. 本文仍然填补哪个缺口。

Related Work 不是论文摘要拼接。正文每段只围绕一个比较轴组织。

#### I.5 `01_introduction.tex`：主体完成后写

先只写六个 bullet，再展开成六段：背景、已有做法、缺口、问题本质、本文设计、贡献。Introduction 中的每条数字必须已经存在于 Results，贡献必须与 `00_paper_contract.md` 一致。

#### I.6 Abstract 和标题：最后一天写

Abstract 不单独建章节文件，放在 `main.tex`。从 Results 选 2–3 个最有解释力的数字，不列出所有任务。标题、摘要、Introduction 贡献和 Conclusion 做一次四向对照，确保说的是同一篇论文。

### J. 每天的操作流程

每次开始写作：

```bash
cd /home/messi/pyprojects/paper/my-small-paper/paper_workspace
git status --short
source analysis/activate.sh
cursor .
```

写一个章节时只打开四类材料：该章节 `.tex`、`00_paper_contract.md`、对应的证据表、需要引用的原文。不要同时打开整个仓库后凭印象写。

每完成一个可读单元：

```bash
make paper-check
```

编译输出固定在 `manuscript/build/`；不要绕过工作区 `Makefile` 另写一套命令。

每天结束前：

1. 打开 `manuscript/build/main.pdf`，不要只看编译成功；
2. 检查当日新增数字是否都在 evidence map 中；
3. 检查新 citation key 是否在 `references.bib`；
4. 运行 `git diff --check`；
5. 只提交一个语义完整的变更，例如 `paper: draft task and setup section`；
6. 在 `04_open_issues.md` 更新未解决问题。

建议用小 commit：Setup、主结果表、错误分解图分别提交。不要等整篇写完才做第一次 commit。

### K. 如何使用 Codex 辅助写作

不要只说“帮我写 Introduction”。每次给 Codex 一个章节、一个主张和一组授权读取的证据。

方法部分提示词示例：

```text
读取 ../training/configs、../training/supervision、../training/trainers 和三个代表任务的
resolved_config.json。只修改 manuscript/sections/03_task_and_setup.tex。
写清 Label-only、CoT、Paper Align、RAFT 的数据视图、损失和推理协议。
所有超参数必须来自文件；找不到就记录到 manuscript/notes/04_open_issues.md，
不要猜。先给证据表，再写英文正文。
```

结果部分提示词示例：

```text
读取 ../eval_output/results/evaluation_analysis_records.json 和对应 metrics.json。
按 RQ1/RQ2/RQ3 起草 manuscript/sections/04_main_results.tex。
每个段落必须是 claim-evidence-interpretation-boundary，保留两个 seed，
不把数值变化写成显著性，不提出新机制。只引用自动生成的 LaTeX 表格。
```

机制部分提示词示例：

```text
读取原始 predictions.jsonl 和 literature/mechanism-analysis-literature-map.md。
先审计 sample_id 是否能跨条件配对，再生成 paired transition 和 error
decomposition 脚本。只把行为证据称为 behavioral mechanism；没有梯度证据时
禁止写 gradient conflict。输出脚本、派生 CSV、PDF 图和对应英文分析段落。
```

语言润色提示词示例：

```text
只润色 manuscript/sections/05_mechanism_analysis.tex，不改变数字、引用、
因果强度和术语。列出所有可能改变科学含义的句子，等待我确认；
其余仅修复语法、衔接和重复表达。
```

每次 AI 修改后必须查看 Git diff。AI 可以起草、重组和检查，但不能代替作者确认数据、引用、领域解释、作者贡献和披露。

### L. 编译、清理和版本标记

常用命令：

```bash
# 激活论文分析与 PDF 检查工具
source analysis/activate.sh

# 编译
make paper

# 编译并检查 A4、匿名元数据和字体嵌入
make paper-check

# 清理全部可再生构建产物，包括 PDF；不删除源文件
make paper-clean

# 查看本次修改
git diff -- manuscript analysis reproducibility

# 检查空白和冲突标记
git diff --check
```

不要使用 `rm -rf` 清理编译目录；使用 `make paper-clean`，之后运行
`make paper-check` 即可重新生成并检查 PDF。

内部版本建议：

```text
v0.1：Setup + 主结果初稿
v0.2：机制分析 + Mix
v0.3：完整八页初稿
v0.4：内部审稿后
v0.5：匿名投稿候选版
```

真正提交后再打 Git tag，例如：

```bash
git tag -a arr-2026-10-12-submitted -m "ARR submission snapshot"
```

只有确认系统上传版本与仓库 PDF 完全一致后才打 tag。

### M. 最终提交文件写在哪里

正式稿仍然在 `manuscript/` 修改。`submission/` 只存某一次提交的不可变快照：

```text
submission/arr-2026-10-12/
├── anonymous-paper.pdf
├── supplementary.zip
├── submission-metadata.md
├── compliance-checklist.md
└── sha256sums.txt
```

`submission-metadata.md` 记录：标题、摘要、作者顺序（本地私有版本）、领域、关键词、贡献类型、OpenReview submission ID、提交时间和目标 venue。公开仓库中不得提交含作者隐私或未公开 submission 信息的文件。

生成快照前：

1. 从 `manuscript/build/main.pdf` 复制为 `anonymous-paper.pdf`；
2. 用系统 PDF 阅读器逐页检查；
3. 确认匿名、页数、字体嵌入、引用、Limitations 和补充材料；
4. 计算 SHA-256，确认上传后下载的 PDF 与本地相同；
5. 提交后不覆盖该目录，后续 revision 建新目录。

---

## 0. 先给结论

现在应该开始写论文，不应该先停下来发明一个新方法。

当前工作已经有两个稳定 seed、七类科学写作评价任务以及两项足以支撑论文主线的发现：

1. CoT 监督对科学写作 Judge 不是普遍有益，而是产生明显的任务依赖性；
2. 这种退化同时包含“输出接口/格式失配”和“有效输出中的评分边界漂移”，Paper Align 能恢复相当一部分直接评分能力，而 RAFT 的序数回归目标并不能自动解决接口失配。

因此，论文最合适的定位不是“提出一个全面超过现有方法的新算法”，而是：

> **一篇以新问题与系统实证发现为核心、以轻量缓解方案为辅的分析型论文。**

推荐的中心论断是：

> **CoT supervision is not uniformly beneficial for scientific-writing judges. It induces task-dependent specialization to the reasoning interface, manifested as both output-schema failures and valid-score boundary drift. Balanced direct/reason supervision restores much of the direct-scoring ability, whereas regression-aware scoring alone does not resolve the interface mismatch.**

中文工作版本：

> CoT 监督并不会一致提升科学写作评价器，而会使模型对推理输出接口产生任务依赖的专门化。这种专门化既表现为输出格式失败，也表现为有效输出内部的评分边界漂移。平衡直接评分与推理监督能够恢复大部分直接评分能力，但单独采用回归感知的评分目标并不能解决接口失配。

这句话是全文的主轴。所有实验、图表和机制分析都应服务于它，而不是把每个结果写成彼此独立的观察。

---

## 1. 论文类型与投稿形态

### 1.1 论文类型

按技术论文的逻辑分类，本工作更接近 **New Problem/Setting + Empirical Analysis**，而不是纯 Technique paper：

| 项目 | 本论文的内容 |
|---|---|
| 新问题/场景 | CoT 监督在科学写作 Judge 中是否稳定有效 |
| 关键经验发现 | CoT 的效果取决于评价任务与推理接口；总体分数会掩盖不同失效机制 |
| 分析方法 | 训练/推理模式对照、成对样本迁移、格式失败与有效评分漂移分解 |
| 配套缓解 | Paper Align 的 Direct/Reason 平衡监督 |
| 对照目标 | RAFT/序数回归是否能解决该问题 |
| 不应强行声称 | 一个全新的通用训练算法、已经证明的梯度冲突、所有 Judge 场景均成立 |

### 1.2 推荐投稿形态

首稿按照 ACL 系长文组织，目标正文 8 页。这样即使最终转向信息科学期刊，核心论证、表格和分析也已经成型，只需扩展领域背景、数据说明和讨论。

候选路线：

| 路线 | 适合条件 | 当前建议 |
|---|---|---|
| 下一轮 ARR，面向 NAACL/COLING 等 | 能补齐 Mix 对照，最好再有一个模型规模或家族 | 学术匹配度最高，优先准备 |
| IPM/JASIST 等信息科学期刊 | 更重视科学写作评价、信息质量与应用价值；需要更完整的数据和讨论 | 毕业时间更看重正式 online 时可考虑 |
| 中文高质量期刊 | 学校认定确定、时间压力较大 | 作为独立中文稿路线，不与英文稿同时投稿 |

截至 2026-08-04，官方 ARR 日期页显示：8 月轮提交日为 2026-08-03，已经过去；下一轮提交日为 **2026-10-12**，该日期也是页面列出的 NAACL 2027 / COLING 2027 最终 ARR 提交日。所有日期均应在真正投稿前再次核对。

---

## 2. 写作前先冻结的“论文合同”

写作最常见的问题不是句子写不好，而是写到一半改变研究问题、表格口径或贡献定义。正式写正文前，先冻结下面四项。

### 2.1 一句话研究目标

> 本文系统研究 CoT 监督如何影响科学写作 Judge 的直接评分能力，区分输出接口失配与有效评分边界漂移，并检验平衡 Direct/Reason 监督和回归感知目标能否缓解这些问题。

### 2.2 三个研究问题

**RQ1：CoT 监督是否稳定提升科学写作评价？**

比较不同任务、seed、训练条件与推理方式，回答收益是否普遍，以及平均指标掩盖了什么。

**RQ2：CoT 监督造成的退化具体发生在哪里？**

把退化分成：无法解析/格式错误、有效输出中的标签迁移、不同任务的方向性偏差，并用成对样本追踪 LL→CL→Paper Align 的变化。

**RQ3：Paper Align 与 RAFT 分别解决了什么，没有解决什么？**

检验 Paper Align 的恢复来自加入 Direct 样本还是视图平衡损失；检验 RAFT 的连续预测、离散预测与概率饱和现象，明确“序数建模”和“输出接口对齐”并非同一问题。

如果 Mix 对照暂时没有完成，RQ3 的表述必须保守：只能说 Paper Align 与恢复相关，不能断言恢复一定来自其损失设计。

### 2.3 三条贡献

最终 Introduction 中只保留三条贡献，避免把实验步骤写成贡献：

1. **Problem and evidence.** 首次或系统地检验 CoT 监督在多维科学写作评价中的任务依赖效应，并显示总体平均性能不足以描述其风险。是否使用“首次”必须经过完整文献核查。
2. **Failure decomposition.** 提出并执行成对失效分解，将退化定位为输出接口失败与有效评分边界漂移，说明两类机制在不同科学写作维度上占比不同。
3. **Mitigation diagnosis.** 比较 Paper Align 与 RAFT，表明保留 Direct/Reason 两种监督接口比单纯改变标签损失更直接；Mix 对照完成后，再判断能否把效果归因于视图平衡目标。

### 2.4 证据与主张边界

| 主张 | 当前证据状态 | 正文允许怎样写 |
|---|---|---|
| CoT 效果具有任务依赖性 | 两个 seed、多任务稳定 | 可以作为主要发现 |
| Positioning Check 的退化主要来自格式/空输出 | seed 43 中新受损样本 141/146 属于 null/format；另一个 seed 有一致趋势 | 可以定量陈述，同时报告两个 seed |
| Grounding/Verifiability 的退化不是格式问题 | 无无效输出但存在系统标签迁移 | 可以作为“有效评分边界漂移”的证据 |
| CoT 推理路径能恢复部分 CoT 模型性能 | 多任务有恢复，Actionability 不是全部恢复 | 可以说明接口专门化，但不能说解释了全部退化 |
| Paper Align 恢复 LL→CL 翻转 | 两 seed 约恢复 74.5%/76.0% | 可以作为主要结果 |
| Paper Align 的损失设计导致恢复 | 还缺同数据普通 SFT 的 Mix 对照 | 暂时不能作因果归因 |
| RAFT 不能解决接口失配 | 离散/连续结果与概率饱和支持 | 可以在当前模型和设置范围内陈述 |
| CoT 导致梯度冲突 | 尚无梯度证据 | 不得作为已证实机制 |
| 结论适用于所有模型与所有 Judge 数据 | 当前主要是 Qwen3-4B | 不得泛化；需要跨模型实验后再扩大范围 |

---

## 3. 整篇论文的逻辑骨架

### 3.1 从问题到贡献的映射

| 研究障碍 | 论文采用的分析模块 | 产出 |
|---|---|---|
| 训练监督与推理提示容易混在一起 | 训练条件 × 推理路径的对照 | 判断是能力损失还是接口专门化 |
| 总体分数看不出错误发生在哪里 | 成对样本迁移 + 格式/语义分解 | 定位不同任务的失效类型 |
| Paper Align 的恢复原因不清楚 | Mix 对照：同 Direct/Reason 数据、普通 token-averaged SFT | 区分数据混合效应与损失效应 |
| RAFT 的离散结果可能掩盖序数信息 | 连续期望分、离散分、概率分布和校准共同报告 | 区分序数建模能力与接口对齐能力 |

### 3.2 四个一致性检查

写完大纲和每轮修改后都检查：

- **Challenge–Method：** 每个研究障碍都有对应实验或分析，不能只在 Introduction 提出。
- **Method–Result：** 每个分析模块都有表格、图或统计结果，不能只有方法描述。
- **Result–Contribution：** 每条贡献都有结果支撑，不能把未来工作写成已完成贡献。
- **Scope–Claim：** 结论限定于当前模型、数据、评分标签和推理协议；跨模型证据不足时主动收窄措辞。

---

## 4. ACL 风格长文的详细章节设计

下面按照 8 页正文设计。页数只是写作预算，不是最终模板规则；投稿前必须以目标 venue 最新模板为准。

### 4.1 标题

标题要同时出现现象、对象和核心解释，不要把 RAFT 或 Paper Align 强行放进标题。

候选标题：

1. **When Chain-of-Thought Supervision Hurts Scientific-Writing Judges**
2. **Reasoning Is Not Always Judging: Diagnosing CoT-Supervised Evaluators for Scientific Writing**
3. **Does Chain-of-Thought Supervision Improve Scientific-Writing Evaluation? A Task-Level Failure Analysis**

在跨模型实验完成前，标题不要使用 *general*, *universal* 或 *fundamental*。

### 4.2 Abstract（约 180–220 词，最后写）

使用五句结构：

1. 背景：LLM-as-a-Judge 越来越多地用于科学写作评价，而 CoT 监督通常被认为能提升可解释性或判断质量。
2. 缺口：现有结果没有清楚区分 CoT 的训练效果、推理接口效果以及不同写作维度上的失败类型。
3. 方法：说明七类任务、两 seed、训练/推理对照、成对失效分解以及 Paper Align/RAFT 比较。
4. 结果：给出 2–3 个最关键的定量结果，不罗列所有指标。
5. 结论：说明 CoT 监督应被视为一种任务和接口相关的设计选择，而非默认改进。

摘要完成标准：每一个数字都能在正文表格中找到；不出现正文没有验证的因果词。

### 4.3 Introduction（约 1.0 页）

按六段写，不要先写 Related Work：

1. **背景与实例。** 科学写作评价涉及 coherence、grounding、verifiability、positioning、actionability 等不同判断过程；一个 Judge 往往既要给分，也可能被要求解释。
2. **现有做法的限制。** CoT 监督常被当作统一改进，但“会解释”不等于“在直接评分接口下稳定给分”，不同任务也可能受影响不同。
3. **问题本质。** 真正的问题不是 CoT 总体上加分还是减分，而是它改变了什么能力、在哪些任务上改变、变化来自格式还是标签边界。
4. **三个挑战。** 训练/推理混杂、平均指标掩盖失效、缓解方法归因不清。
5. **本文方案。** 简述对照矩阵、成对迁移分析、Paper Align/RAFT/Mix。
6. **贡献。** 使用第 2.3 节的三条贡献，避免重复实验设置。

Introduction 中至少放一个具体例子：同一科学写作输入在 Label-only、CoT-supervised direct path、CoT path 下分别输出什么。例子必须来自真实样本，不能为了叙事人工编造。

### 4.4 Related Work（约 0.8–0.9 页）

建议组织为三个问题链，而不是论文清单：

1. **LLM-as-a-Judge 与细粒度写作评价：** 现有 Judge 研究如何处理评分可靠性、偏差、标尺和解释；科学写作维度为何比单一偏好判断更异质。
2. **Reasoning/CoT supervision：** 区分训练时 rationale supervision 与推理时 rationale elicitation；比较已有工作报告的提升、忠实性问题和任务依赖性。
3. **Ordinal/regression-aware learning：** 说明科学写作标签具有顺序结构，RAFT 类目标解决的是标签几何；随后指出标签几何不等于输出协议对齐。

写法要求：每段最后明确“这些工作没有回答本文哪个 RQ”。参考论文用于确定概念、对照和证据标准，不是照搬它们的章节名称或叙事。

已有文献线索见 [literature/mechanism-analysis-literature-map.md](../literature/mechanism-analysis-literature-map.md)。正式引用前逐一核对题名、作者、年份、venue、DOI/ACL Anthology 链接与原文主张。

### 4.5 Task and Experimental Design（约 1.2 页）

本节必须使读者在不看代码的情况下理解比较是否公平。

#### 4.5.1 Scientific-Writing Evaluation Tasks

逐项交代：

- 七项任务的名称、输入对象、输出标签和标签语义；
- 每项任务衡量科学写作的哪个属性；
- 数据来源、构造过程、训练/验证/测试规模；
- 是否存在作者、文档、论文或来源层面的数据泄漏；
- 标签由谁产生、是否人工复核、类分布如何；
- 为什么这些任务需要分别报告，不能只取宏平均。

不要只写数据集名称。至少给每项任务一个压缩后的真实输入/标签示例，完整例子放附录。

#### 4.5.2 Training Conditions

用统一符号定义所有条件，并在全文保持一致：

- Label-only / baseline；
- CoT-supervised；
- Paper Align；
- RAFT；
- Mix baseline（补充后）；
- 其他只用于消融的条件。

必须说明每种条件看到的训练样本数量、Direct/Reason 比例、目标 token、损失归一化方式以及推理输出格式。否则 Paper Align 与 CoT 的差异可能被审稿人质疑为数据量差异。

#### 4.5.3 Inference Protocols

明确区分：

- Direct scoring path；
- CoT/reasoning path；
- 解析规则与容错；
- temperature、max tokens、解码方式；
- null、格式错误、越界标签如何记分。

#### 4.5.4 Metrics and Statistics

至少同时报告：

- 任务主指标，例如 QWK、Macro-F1 或 Accuracy；
- 有效输出率；
- valid-only 指标；
- 标签迁移矩阵；
- RAFT 连续期望分与离散预测的对应指标；
- 两个 seed 的逐 seed 结果，不要只给均值。

### 4.6 Main Results（约 1.4–1.5 页）

按 RQ 写，不按模型名写。

#### 4.6.1 RQ1：CoT 是否普遍有益

先给主结果表，再写三个层次：

1. 总体趋势；
2. 不同任务的异质性；
3. 两个 seed 的稳定性。

不要把“有显著差异”和“数值更高”混用。统计检验未通过时写“numerically higher/lower”。

#### 4.6.2 RQ2：损失来自格式还是有效标签漂移

必须同时展示 overall 与 valid-only。例如 Positioning Check 中 CoT 条件 overall 明显下降，但 valid-only 接近完整正确率，这支持格式失配；Grounding/Verifiability 没有无效输出却仍有标签迁移，则支持评分边界改变。

#### 4.6.3 RQ3：Paper Align 与 RAFT

先陈述观察：Paper Align 与 Label-only 预测高度一致，并恢复大量 LL→CL 翻转；RAFT 连续值可能优于取整结果，但总体未超过最佳分类/Paper Align 条件，且标签分布高度饱和。

如果没有 Mix，只能在段尾写：

> This recovery may arise from exposure to the direct-scoring view, the view-balanced objective, or both; we isolate these factors in the following analysis / leave this attribution as a limitation.

### 4.7 Mechanism Analysis（约 1.8–2.0 页）

机制分析不是为了让论文“看起来复杂”，而是为了把“指标下降”升级成可检验、可复用的认识。它回答审稿人最自然的三个问题：为什么下降、为什么不同任务不一样、为什么某种缓解有效。

本论文不需要声称神经网络内部的完整因果机制。更准确的名字是 **behavioral mechanism analysis** 或 **failure mechanism analysis**。

#### Analysis A：训练路径 × 推理路径

比较同一 checkpoint 在 Direct 与 CoT 推理路径下的结果。当前结果显示多个任务在 CoT 路径恢复，但 Actionability 并未完全恢复。

可支持的结论：模型部分专门化到训练时的输出接口。

不可支持的结论：所有性能下降都只是格式问题。

#### Analysis B：成对样本迁移

对同一测试样本记录：

```text
gold | Label-only pred | CoT-direct pred | CoT-path pred | Paper Align pred | validity
```

至少计算：

- LL 正确、CL 错误的样本数；
- 其中 Paper Align 恢复的比例；
- 新增错误中 null/format、相邻标签漂移、跨级漂移的比例；
- 不同任务的迁移方向是否偏向更高或更低标签。

#### Analysis C：输出失败与语义失败分解

使用统一的互斥类别：

1. null/empty；
2. malformed/unparseable；
3. out-of-range；
4. valid but adjacent-label error；
5. valid but non-adjacent error；
6. correct。

所有条件使用同一个解析器重新跑统计，避免解析规则改变造成假差异。

#### Analysis D：RAFT 概率与连续分

当前 CoT-RAFT 的最大标签概率接近 1，期望分与离散标签差异极小。这意味着模型没有真正利用连续标签空间，至少在当前温度、目标和数据下如此。

报告：

- 最大概率分布；
- 预测熵；
- `|expected score - argmax label|`；
- 连续分和离散分各自的 Pearson/Spearman/QWK；
- 按 gold label 分组的校准曲线。

不要仅凭“概率饱和”声称优化失败；更稳妥的表述是“当前训练设置下，回归感知目标没有产生可观的软序数预测”。

#### Analysis E：Mix 归因实验（最高优先级）

补一个与 Paper Align 使用完全相同 Direct/Reason 样本的普通 token-averaged SFT：

| 对照结果 | 解释 |
|---|---|
| Mix ≈ Paper Align，且二者都优于 CoT | 恢复主要来自加入 Direct 训练视图 |
| Mix ≈ CoT，Paper Align 更好 | 视图平衡/损失设计是关键 |
| Mix 介于两者之间 | 数据视图与损失设计共同作用 |

这是当前最值得补的一个实验，因为它决定论文能否解释 Paper Align 的有效成分。

### 4.8 Discussion（约 0.5 页）

围绕三个设计含义写：

1. 科学写作 Judge 的评价维度不应被视为同质任务；
2. 训练时要求解释可能改变模型对输出协议的依赖，Direct 与 Reason 两条路径都应验证；
3. 有序标签损失解决标签几何，不能替代输出接口对齐。

然后说明实际建议：部署科学写作 Judge 时，同时报告有效输出率、valid-only 质量和路径敏感性，而不是只看一个总体分数。

### 4.9 Conclusion、Limitations 与 Ethics

Limitations 至少写：

- 当前主要模型规模/家族有限；
- 两个 seed 足以说明当前趋势稳定，但不能替代跨模型泛化；
- 数据标签与任务定义可能带有领域和标注偏差；
- 行为分析不能证明内部梯度或表征因果机制；
- Mix 未完成时，Paper Align 的有效成分不能完全归因；
- 自动科学写作评价不应直接替代专家审稿或高风险教育决策。

Ethics 讨论数据许可、隐私、自动评价误用、偏差和人类监督。Conclusion 只重述已经得到证据支持的答案，不引入新实验。

当前 ARR 规则要求单独的 Limitations section，缺失可能直接被拒；投稿时应再次核对最新模板和 Responsible NLP checklist。

---

## 5. 应该准备的图表

### 5.1 正文最小图表集

| 编号 | 内容 | 回答的问题 |
|---|---|---|
| Figure 1 | 训练条件 × 推理路径的实验设计和主要发现概览 | 本文到底区分了哪些因素 |
| Table 1 | 七项任务定义、规模、标签、类分布 | 数据是什么、为什么与科学写作有关 |
| Table 2 | 两 seed 的全任务主结果，包含 overall 与 valid rate | CoT 是否普遍有效 |
| Figure 2 | LL→CL→Paper Align 的样本迁移流或成对翻转图 | Paper Align 恢复了哪些错误 |
| Figure 3 | 任务级错误分解：格式失败、相邻漂移、跨级漂移 | 不同任务为什么下降 |
| Figure 4 | RAFT 连续/离散结果、预测熵或校准 | RAFT 是否使用了序数信息 |
| Table 3 | Mix/Paper Align/CoT 的归因消融 | 恢复来自数据视图还是损失 |

页面放不下时，优先保留 Table 2、Figure 2、Figure 3 和 Mix 消融。完整逐任务数字移到附录。

### 5.2 每张图的完成标准

- 图注单独阅读即可理解样本、指标、方向和误差线；
- 颜色之外还使用形状、线型或纹理，保证灰度打印可读；
- 不用截屏表格；
- 坐标轴不截断或明确标注截断；
- 所有数字由脚本从同一结果表生成；
- 正文提到每张图最重要的一个结论，不逐点复述。

### 5.3 附录内容

- 每任务、每 seed、每训练/推理条件的完整指标；
- 训练提示、推理提示、输出 schema；
- 超参数、训练步数、最佳 checkpoint 选择规则；
- 各类真实失败案例；
- 数据构造与泄漏检查；
- RAFT 标签损失公式和实现伪代码；
- 解析器规则和无效输出示例；
- Mix 及其他消融；
- 计算资源与运行时间。

---

## 6. 统计分析方案

两个 seed 已经能够支持“结果在当前设置下可复现”，但统计推断不要把两个 seed 当作足够大的独立样本做普通 t 检验。

建议：

1. 对同一测试集上的模型差值做 **paired bootstrap**，每个 seed 单独重采样样本 10,000 次，报告 95% CI；
2. 对正确/错误状态的成对翻转使用 **McNemar test**；
3. 多任务同时检验时使用 Holm 校正，并同时报告未校正 p 值和效应量；
4. 序数任务报告 QWK，并补充 Macro-F1/Accuracy 与标签偏移方向；
5. RAFT 连续输出报告 Pearson 和 Spearman，但不要用相关性代替分类/序数一致性；
6. 两个 seed 逐项展示，再给均值；不要只给“mean ± std”隐藏方向；
7. 失败案例抽样要预先规定规则，例如每个代表任务、每类错误固定抽取 30 条，并保存样本 ID，避免挑例子。

统计显著不等于实际重要。主文同时报告：差值、置信区间、受影响样本数和错误类型占比。

---

## 7. 还需要做什么实验

### 7.1 投稿前强烈建议完成

**第一优先：Mix baseline。** 这是因果归因最关键的缺口，成本通常低于发明新方法。

**第二优先：一个额外模型规模或家族。** 选 Grounding、Verifiability、Coherence 等代表任务，不必重复全部七项任务。目的不是刷更高性能，而是验证“接口失败 + 标签漂移”的模式是否跨模型存在。

**第三优先：统一重新生成机制统计。** 用最终解析器从原始预测生成迁移矩阵、错误分解、有效率和置信区间，确保正文数字有单一来源。

### 7.2 有时间再做

- rationale 删除、截断或替换干预；
- 标签损失系数/温度小规模 sweep；
- 置信度与校准分析；
- 少量表征或梯度相似度分析。

这些可以增强论文，但不是开始写作的前置条件。没有内部证据时，将机制称为“behavioral failure mechanism”，已经足够严谨。

### 7.3 不建议现在做

- 为了多一个方法名改造复杂网络；
- 第三个 seed 替代跨模型验证；
- 没有明确假设的大规模超参数搜索；
- 只挑总体分数最高的设置，忽略无效输出；
- 同时维护内容差异很大的会议稿、期刊稿和中文稿。

---

## 8. 实际写作顺序

不要按论文阅读顺序写。推荐顺序：

1. **冻结结果表和图。** 建立唯一的结果清单，给每个数字记录脚本、输入文件和 commit。
2. **写 Task and Experimental Design。** 这是最客观、最少依赖修辞的部分。
3. **写 Main Results。** 每个 RQ 使用“结论→证据→限定”的段落结构。
4. **写 Mechanism Analysis。** 先写已完成分析；Mix 未完成处明确标为内部 TODO，不进入投稿 PDF。
5. **写 Discussion 和 Limitations。** 主动限定外推范围。
6. **整理 Related Work。** 用文献回答概念和证据缺口，不堆引用。
7. **写 Introduction。** 这时挑战、方法和贡献已经稳定。
8. **最后写标题、摘要和结论。** 三者必须使用同一组主张。

每一结果段可套用下面的内部模板，投稿时删除标签：

```text
Claim: 本段回答 RQ 的哪一部分。
Evidence: 哪张表/图、差值、置信区间、两个 seed 是否一致。
Interpretation: 结果支持什么解释。
Boundary: 结果不能支持什么更强结论。
```

---

## 9. 论文文件结构的执行规则

实际目录以本指南开头“使用本指南前：先把论文工程搭起来”中的完整目录树为准，不再维护第二套结构。

其中三条规则最重要：

1. `manuscript/` 是论文正文的唯一真源；
2. `analysis/` 负责把 `../eval_output/` 转成表图，禁止手工抄数字；
3. `reproducibility/experiment_manifest.md` 必须记录代码 commit、模型 checkpoint、数据版本、seed、配置文件、训练命令、推理命令、原始预测路径和汇总脚本。

审稿回复放在对应的提交轮次下，例如 `submission/arr-2026-10-12/response/rebuttal_matrix.md`，不要混进正文目录。

---

## 10. 从现在到可投稿版本的时间表

### 阶段 A：2026-08-04 至 2026-08-10，冻结论文设计

- 确定英文主标题工作版、三条 RQ、三条贡献；
- 决定主投会议路线还是期刊路线；
- 整理七项任务的数据说明表；
- 建立结果到脚本/文件的映射；
- 固定 Mix 与跨模型实验配置；
- 向学院书面确认目标 venue、Main/Findings/索引和毕业认定。

### 阶段 B：2026-08-11 至 2026-08-24，完成初稿主体

- 先写 Setup、Results、Mechanism Analysis；
- 完成 Table 1、Table 2、Figure 2、Figure 3；
- 跑 Mix；
- 所有数字至少由另一条汇总检查或人工抽查一次；
- 形成一份能从头读到尾、允许语言粗糙但论证完整的 v0.1。

### 阶段 C：2026-08-25 至 2026-09-07，补关键证据

- 完成一个额外模型规模/家族的代表任务；
- 跑 paired bootstrap、McNemar 和多任务校正；
- 写 Related Work、Discussion、Limitations 和 Ethics；
- 把所有失败案例替换成真实、可追踪的匿名样本。

### 阶段 D：2026-09-08 至 2026-09-21，形成投稿稿

- 写 Introduction、Abstract、Conclusion；
- 做第一次完整内部审稿；
- 修复主张过强、表图不可读、数据定义不清等问题；
- 准备匿名代码与补充材料；
- 检查所有引用能否直接支持相邻句子。

### 阶段 E：2026-09-22 至 2026-10-05，模拟审稿与终审

- 找至少一位不了解项目的人复述论文主张；若复述不出来，重写摘要和 Introduction；
- 用审稿人问题清单做红队审查；
- 完成可复现性、伦理、数据许可、AI 使用披露和匿名化检查；
- 使用官方模板编译最终 PDF，并逐页检查。

### 阶段 F：2026-10-06 至 2026-10-12，提交窗口

- 不再增加新主实验，只修正确性和表达问题；
- 锁定作者顺序和所有作者 OpenReview 信息；
- 至少提前 48 小时上传草稿，检查 PDF 与补充材料；
- 在官方系统确认提交成功、作者注册和 reviewer duty 要求。

如果选择滚动期刊，不必等待 10 月，但也不要提交一个尚未完成 Mix 归因和数据说明的版本。

---

## 11. 投稿前完整流程：ACL Rolling Review 路线

以下依据 2026-08-04 可见的官方 ARR/EMNLP 页面整理，投稿当天重新核对。

### 11.1 投稿前 4–8 周

1. 确认目标 venue 接受哪一轮 ARR review；
2. 下载当轮官方模板，不沿用旧模板；
3. 所有作者完善 OpenReview profile、机构邮箱、冲突域和 ORCID（如要求）；
4. 提前锁定作者名单。ARR 当前规则不允许提交后更改作者列表；
5. 明确代码、数据、模型权重是否能匿名开放；
6. 核查数据许可、隐私和第三方模型许可；
7. 确认没有同时投往其他 archival venue。

### 11.2 PDF 与匿名化

- 长文当前内容页上限为 8 页，短文为 4 页；参考文献和规定位置的 Limitations 不计入内容页，但以当轮规则为准；
- 删除作者、单位、致谢、项目号和可识别自引措辞；
- 匿名代码/补充材料链接，不使用带访问追踪的链接；
- 自引使用第三人称，不用“our previous work”暴露身份；
- 检查 PDF metadata、文件名、图中路径、日志和仓库 commit author；
- 必须包含独立 Limitations section；
- 按要求填写 Responsible NLP checklist、ethics 与预印本状态。

### 11.3 OpenReview 提交

1. 填写标题、摘要、领域、贡献类型和关键词；
2. 上传主 PDF 与允许的补充材料；
3. 核对作者顺序、冲突关系和匿名状态；
4. 所有作者按期完成 reviewer registration。当前 ARR 页面明确提示全部作者都要注册，违规论文可能 desk reject；
5. 下载或截图保存 submission ID 和提交确认；
6. 截止前重新下载系统中的 PDF，确认不是旧版本或损坏版本。

### 11.4 审稿与作者回复

收到评论后建立矩阵：

| Reviewer | Comment | Severity | Planned response | New evidence | Manuscript change | Status |
|---|---|---|---|---|---|---|

回复顺序：

1. 先回答事实错误和核心有效性问题；
2. 对每个问题直接给结论，再给证据；
3. 承认可证实的限制，不用情绪化语言；
4. 新实验必须真正回答评论，不能用更多无关数字淹没问题；
5. 明确说明改了论文哪一节、哪张表；
6. 不承诺在截止前无法可靠完成的工作。

### 11.5 Meta-review 与 venue commitment

ARR review 完成后并不等于会议录用。作者需要在兼容会议的 commitment 窗口把 review package 提交给目标 venue，随后才会得到 Main/Findings/拒稿决定。

在 commitment 前检查：

- 目标会议主题是否匹配；
- 学院是否认可 Main、Findings 或对应索引；
- 是否需要提交 revision notes；
- 作者、标题和摘要是否满足 commitment 规则；
- 是否存在新的 dual-submission 限制。

### 11.6 Camera-ready

- 根据 meta-review 和决定信完成承诺修改；
- 恢复作者、单位、致谢和资助；
- accepted paper 如允许增加一页，只用于必要修改，不塞入未经审查的新主张；
- 再做引用、表图、伦理、Limitations、版权和 AI disclosure 检查；
- 所有作者完成版权协议、注册和展示义务；
- 保存正式 proceedings 页面、DOI/Anthology 页面和索引证明。

---

## 12. 投稿前完整流程：期刊路线

### 12.1 选刊，不要只看影响因子

对每个候选期刊核对：

- scope 是否接收 LLM evaluation、scientific writing、information quality 或 scholarly communication；
- article type 和篇幅；
- 是否双盲；
- 数据/代码开放政策；
- AI 写作披露政策；
- 版面费、开放获取费用；
- 学院认定、分区、正式 online 与检索时间要求。

期刊规则变化频繁，所有结论必须以投稿当天官网 author guidelines 为准。

### 12.2 期刊稿应比会议稿增加什么

- 更完整的科学写作与信息科学理论背景；
- 数据构造、标签定义和应用情境；
- 全七任务和全消融结果；
- 更充分的效度讨论；
- 对自动评价在科研、教育和编辑流程中的影响；
- 完整数据、代码和复现说明。

不能仅把会议式 8 页稿“拉长”。期刊稿需要更强的问题背景和解释深度。

### 12.3 提交材料

通常包括：

- blinded manuscript；
- title page；
- cover letter；
- highlights / graphical abstract（若要求）；
- CRediT author contributions；
- conflict of interest；
- funding statement；
- data/code availability；
- ethics statement；
- AI-assisted writing disclosure；
- 推荐/回避审稿人及理由。

Cover letter 只写四件事：研究问题、最重要的 2–3 个发现、为什么适合该刊、无一稿多投及合规声明。不要把摘要复制一遍。

### 12.4 返修

逐条回复所有评论，即使不采纳也要说明证据和理由。回复信使用：

```text
Reviewer comment:
Response:
Change made:
Location in revised manuscript:
```

每轮返修保留：提交版本、带修订版本、干净版本、回复信和编辑决定。论文 online 后保存 DOI、online publication date、卷期页码（如有）和检索证明。

---

## 13. 最容易导致拒稿的问题

### 13.1 研究层面

- 只有“某方法高/低几个点”，没有回答为什么；
- 把格式错误和有效评分错误混为一谈；
- Paper Align 与 baseline 的数据量、训练 token 或采样比例不公平；
- 用两个 seed 声称跨模型普适；
- 没有 Mix 却把恢复归因于新损失；
- 把相关性或行为证据写成梯度/表征因果机制；
- 科学写作任务只列名称，没有定义其真实应用意义。

### 13.2 写作层面

- Abstract、Introduction 和 Conclusion 的主张不一致；
- 贡献写成“我们做了大量实验”；
- 一段中同时讨论多个不相关结论；
- 只报最好 seed 或只报平均数；
- 结论使用 *prove*, *always*, *universal* 等证据不支持的词；
- 引用只在段尾堆积，无法判断支持哪句话；
- 把未核查的 AI 生成引用写入 `.bib`。

### 13.3 Desk reject 层面

- 超页或模板错误；
- 缺少 Limitations；
- 匿名化失败；
- 作者名单/账户/注册不合规；
- 一稿多投；
- 预印本声明或匿名政策违规；
- 数据、伦理或 AI 使用披露缺失；
- 补充材料含作者身份或跟踪链接。

---

## 14. 完稿后的内部审稿清单

### 14.1 内容正确性

- [ ] 每个 RQ 在 Results/Analysis 中有明确答案；
- [ ] 每条贡献至少对应一张表或图；
- [ ] 所有正文数字能从最终结果表复现；
- [ ] seed、样本数、标签范围和解析规则全篇一致；
- [ ] overall、valid rate、valid-only 没有混用；
- [ ] RAFT 连续分和离散分没有混用；
- [ ] 观察、解释和因果结论被明确区分；
- [ ] Limitations 覆盖模型、数据、统计和机制边界。

### 14.2 引用完整性

- [ ] 每个非公知事实有来源；
- [ ] 每篇引用都已打开原文核对；
- [ ] 作者、题名、年份、venue、页码/DOI 正确；
- [ ] 没有把摘要中的弱结论改写成正文强结论；
- [ ] 没有虚构或无法访问的引用；
- [ ] “首次”“很少研究”等 novelty 声明有系统检索支持。

### 14.3 可复现性

- [ ] 代码 commit 和环境文件已固定；
- [ ] 数据版本、许可和划分规则已记录；
- [ ] 所有 checkpoint 有唯一标识；
- [ ] 训练与推理命令可运行；
- [ ] 原始输出未被汇总脚本覆盖；
- [ ] 表格和图片可由脚本重建；
- [ ] 计算资源、时长和 seed 已报告。

### 14.4 投稿合规

- [ ] 所有作者同意论文内容、作者顺序和目标 venue；
- [ ] 作者单位与学校毕业要求一致；
- [ ] 无一稿多投；
- [ ] 匿名 PDF、补充材料和仓库均通过身份检查；
- [ ] Limitations、Ethics、Data Availability、COI、Funding、CRediT 和 AI disclosure 按 venue 要求提供；
- [ ] 页数、字体、参考文献和文件大小符合最新规则；
- [ ] 学院已经书面确认会议层级/期刊、Main/Findings 与索引认定。

---

## 15. 审稿人最可能问的十个问题

投稿前必须能用一段话和一张表/图回答每个问题：

1. 这不是普通的格式遵循问题吗？
2. 为什么它是科学写作 Judge 特有或尤其重要的问题？
3. CoT 训练和 CoT 推理的影响是否被混淆？
4. Paper Align 的收益是否只是因为见过更多 Direct 样本？
5. 不同条件使用的训练 token、数据量和计算量是否公平？
6. 两个 seed 能说明什么，不能说明什么？
7. 结果能否迁移到另一个模型或模型规模？
8. RAFT 的实现是否忠实，连续预测为什么没有带来收益？
9. 数据标签是否可靠，是否存在文档/作者泄漏？
10. 自动评价科学写作会产生哪些误用和公平性风险？

如果第 4 问没有 Mix，第 7 问没有额外模型，论文仍可写，但投稿层级和措辞必须相应保守。

---

## 16. 你本人必须做的事情

AI 可以辅助整理表格、画图、检查一致性和起草文字，但以下判断必须由作者完成：

1. 确认每项科学写作任务的真实定义、数据来源和标签有效性；
2. 确认所有实验数字与原始输出一致；
3. 决定作者顺序、单位、目标 venue 和毕业路线；
4. 逐篇阅读并批准论文中的关键引用；
5. 判断案例分析是否符合领域语义，而不仅是标签匹配；
6. 批准所有因果与泛化措辞；
7. 向学院取得书面认定，尤其是 Main/Findings、检索状态和正式 online 的要求；
8. 所有作者阅读并批准最终稿、披露和投稿表单。

毕业相关的 venue 风险和认定建议见 [GRADUATION_VENUE_GUIDE.md](GRADUATION_VENUE_GUIDE.md)。以学院最新书面答复为准，不以网页排名、口头承诺或录用通知替代。

---

## 17. 接下来 72 小时的具体动作

### 第 1 天：锁定论文，不写漂亮句子

- 把本指南第 2 节复制为内部 claim sheet；
- 给七项任务补全“输入、标签、规模、科学写作意义、数据来源”；
- 建立所有实验条件的统一缩写表；
- 确定主投 ARR 还是期刊；
- 冻结 Mix 配置。

### 第 2 天：先做表和图的空壳

- 创建 Table 1 任务表；
- 创建 Table 2 主结果表；
- 生成 Figure 2 成对迁移图的数据表；
- 生成 Figure 3 错误分解的数据表；
- 每张图先写一句结论式 caption。

### 第 3 天：开始正文

- 写完 Task and Experimental Design 初稿；
- 按 RQ1/RQ2 写 Main Results，不写 Introduction；
- 把仍缺数字的位置放进单独 TODO 清单，不让 TODO 混入投稿正文；
- 启动 Mix 训练并保存配置、日志与 commit。

一周后的验收标准不是“写了多少页”，而是：一个不知情读者能从 Table 1、Table 2、Figure 2 和 Figure 3 看懂数据、现象和机制分解。

---

## 18. 当前官方规则与项目资料

投稿规则会变化，以下链接仅代表 2026-08-04 核对的版本：

- [ACL Rolling Review Call for Papers](https://aclrollingreview.org/cfp)
- [ACL Rolling Review Dates and Venues](https://aclrollingreview.org/dates)
- [ACL Rolling Review Author Guidelines](https://aclrollingreview.org/authors)
- [EMNLP 2026 Main Conference Call for Papers](https://2026.emnlp.org/calls/main_conference_papers/)

项目内资料：

- [机制分析与相关工作地图](../literature/mechanism-analysis-literature-map.md)
- [毕业与投稿 venue 指南](GRADUATION_VENUE_GUIDE.md)
- [已有研究分析](RESEARCH_ANALYSIS.md)

`RESEARCH_ANALYSIS.md` 中偏“先提出新方法”的内容可以作为早期备选思路，但不应覆盖本指南的当前策略：先完成实证论文主线，通过 Mix 和跨模型证据提高归因与外部有效性，而不是为了方法名增加不必要复杂度。

---

## 19. 最终完成定义

只有同时满足以下条件，才算“写完并可以提交”：

- 论文用一句话能说清问题、发现和意义；
- 三个 RQ 全部有证据回答；
- 所有核心结果两个 seed 一致展示；
- Mix 对照完成，或正文明确降低 Paper Align 的归因强度；
- 至少完成一次跨模型代表任务验证，或将外部有效性写为主要限制；
- 所有表图来自冻结脚本和原始输出；
- 所有引用逐篇核查，无虚构、错引和过度转述；
- 全文没有内部 TODO、占位数字或未定义缩写；
- 匿名、页数、Limitations、伦理和作者注册符合目标 venue 当轮规则；
- 所有作者批准最终 PDF；
- 学院书面确认该投稿路线符合毕业成果认定要求。

到这一步，下一动作是提交，不是继续无限追加实验。
