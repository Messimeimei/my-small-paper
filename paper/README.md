# 论文工作区

与论文写作直接相关的内容统一放在本目录。上一级项目目录只保留训练代码、
数据、评测输出和项目入口。

## 立即开始

```bash
cd /home/messi/pyprojects/paper/my-small-paper/paper_workspace
source analysis/activate.sh
make paper-check
cursor .
```

编译后的匿名草稿位于 `manuscript/build/main.pdf`。

## 目录说明

| 目录 | 内容 |
|---|---|
| `docs/` | 写作指南、研究分析、文献综述和投稿规划 |
| `literature/` | 参考论文 PDF 与阅读笔记 |
| `manuscript/` | ACL LaTeX 正文、证据表、图表和编译产物 |
| `analysis/` | 统计、作图、表格生成脚本与派生数据 |
| `reproducibility/` | 环境、数据来源和结果到图表的溯源记录 |
| `submission/` | 各投稿轮次的冻结快照与检查清单 |

## 路径约定

所有写作命令都从本目录运行：

- 工作区内部使用 `manuscript/`、`analysis/`、`literature/` 等路径；
- 原项目训练代码使用 `../training/`；
- 原项目数据使用 `../data/`；
- 原始评测结果使用 `../eval_output/`；
- 本地 TeX、Python 和 PDF 工具由 `analysis/activate.sh` 自动连接。

正文写作从 [manuscript/START_HERE.md](manuscript/START_HERE.md) 开始；完整流程见
[docs/PAPER_WRITING_AND_SUBMISSION_GUIDE.md](docs/PAPER_WRITING_AND_SUBMISSION_GUIDE.md)。

## Related Work 资料：评价科学写作与同行评审

更新时间：2026-08-11。以下资料服务于
`Evaluating Scientific Writing and Peer Review` 小节。筛选时优先保留正式发表且能
定义评价对象、评价构念或评价有效性的工作；预印本单独列出，不能与正式论文混写。
“摘要介绍”均为对论文摘要或正文的中文转述，不是原文翻译或直接引语。

### 建议的讲故事逻辑

这一小节不宜按论文年份逐篇罗列，建议写成三段：

1. **从单一语言质量扩展到多对象、多维评价。** 早期工作主要判断科学文本是否
   需要语言编辑，后续资源把对象扩展到 revision、citation/related work、完整论文和
   peer-review comment。由此定义本节的基本观点：科学写作质量不是一个不加说明的
   总分，而是依赖对象和任务的多个构念。
2. **从“生成得像”转向“评价得对”。** LLM 反馈可能与人类意见重合并被作者认为
   有用，但 overlap、流畅度和主观 helpfulness 都不能替代科学正确性。细粒度研究
   因而分别检查 limitation identification、claim grounding、actionability、
   verifiability、focus coverage 和 robustness；这些结果共同说明 aggregate score 会
   掩盖不同失败模式。
3. **由专用评价器收束到本文缺口。** RevUtil 和 related-work evaluation 将具体
   artifact 拆成可操作 rubric，SciRM 再把多类科学写作任务统一为跨任务 reward
   modeling。现有工作主要构建 benchmark、生成器或更强 evaluator，尚未系统隔离
   `rationale supervision` 与推理时 `rationale elicitation/interface` 的作用；本文将
   这些既有构念作为受控判断任务，而不是再提出一个审稿生成器或总质量基准。

推荐的正文引文密度是每段 3--5 组代表工作；其余文献用于定义术语、回应审稿人或
补充讨论。必须保留以下证据边界：人类评论并非无噪声金标准；观点重合不等于事实
正确；评论有用不等于离散评分可靠；评论质量和分数质量也不能由 rationale 的流畅度
替代。

### A. 测量学基础：同行评审质量如何被定义

#### 1. Development of the Review Quality Instrument (RQI) for Assessing Peer Reviews of Manuscripts

- **地址/发表：** [Journal of Clinical Epidemiology, 1999](https://doi.org/10.1016/S0895-4356(99)00047-5)，正式期刊论文。
- **摘要介绍：** 论文开发多维 Review Quality Instrument，用于考察审稿意见是否充分
  处理研究重要性、方法、结果呈现和解释，以及评论是否建设性、是否给出充分论证。
- **与本小节的关系：** 它是“审稿质量需要先操作化再测量”的经典起点，可防止把
  review quality 简化成长度、语气或单一总体印象。

#### 2. Tools Used to Assess the Quality of Peer Review Reports: A Methodological Systematic Review

- **地址/发表：** [BMC Medical Research Methodology, 2019](https://doi.org/10.1186/s12874-019-0688-x)，正式期刊系统综述。
- **摘要介绍：** 该综述发现既有工具覆盖多个不同领域，但多数工作没有先清楚定义
  “质量”，开发过程、效度与信度证据也普遍不足。
- **与本小节的关系：** 它提供直接的方法学依据：不同评价维度不能在缺少构念说明时
  合并成总分，自动 evaluator 也需要报告其标签和量表从何而来。

#### 3. Development of ARCADIA: A Tool for Assessing the Quality of Peer-Review Reports in Biomedical Research

- **地址/发表：** [BMJ Open, 2020](https://doi.org/10.1136/bmjopen-2019-035604)，正式期刊论文。
- **摘要介绍：** ARCADIA 通过编辑和作者调查形成 14 项清单，覆盖研究重要性、方法
  稳健性、解释与讨论、报告透明度以及评论本身的特征。
- **与本小节的关系：** 它清楚区分“审稿人是否正确评价论文内容”和“评论文本是否
  具有良好属性”，对应本文需要分开的 score quality 与 explanation quality。

#### 4. Is the Quality of Reviews Reflected in Editors' and Authors' Satisfaction with Peer Review?

- **地址/发表：** [Learned Publishing, 2021](https://doi.org/10.1002/leap.1344)，正式期刊论文。
- **摘要介绍：** 论文分析多领域期刊的稿件和评论，发现作者满意度会受到接收建议
  影响，而编辑判断、作者感受和 RQI 测得的评论质量并不等价。
- **与本小节的关系：** 它说明 preference/helpfulness 标签可能受推荐极性干扰，不能
  被无条件当作科学评价的金标准。

#### 5. Defining Quality in Peer Review Reports: A Scoping Review

- **地址/发表：** [Knowledge and Information Systems, 2025](https://doi.org/10.1007/s10115-025-02435-0)，正式期刊综述。
- **摘要介绍：** 该综述汇总大量原始研究，将稿件评价归纳为结构、方法、文体、伦理、
  科学价值和总体适切性，并总结建设性、具体性、公平性、充分性、一致性、客观性和
  可读性等评论属性。
- **与本小节的关系：** 这是当前较完整的 review-quality taxonomy，可为“多维构念”
  提供跨学科依据，但不应被误写成已经统一公认的单一量表。

### B. 数据与评价对象：从句子编辑到论文和审稿意见

#### 6. A Report on the Automatic Evaluation of Scientific Writing Shared Task

- **地址/发表：** [BEA 2016 / ACL Anthology](https://aclanthology.org/W16-0506/)，DOI
  [`10.18653/v1/W16-0506`](https://doi.org/10.18653/v1/W16-0506)，正式会议论文。
- **摘要介绍：** AESW shared task 使用专业编辑前后的平行科学文本，要求模型判断
  句子是否需要修改，所涉及问题不只包括基本语法，也包括科学文体的适切性。
- **与本小节的关系：** 适合作为历史起点，说明早期 automated scientific-writing
  evaluation 更接近语言编辑需求检测，而非证据、定位或反馈效用判断。

#### 7. A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications

- **地址/发表：** [NAACL 2018 / ACL Anthology](https://aclanthology.org/N18-1149/)，DOI
  [`10.18653/v1/N18-1149`](https://doi.org/10.18653/v1/N18-1149)，正式会议论文。
- **摘要介绍：** PeerRead 汇集论文、专家评论、方面评分和接收决定，并研究接收结果与
  aspect score 的预测，是计算同行评审研究的基础数据集之一。
- **与本小节的关系：** 它建立了 outcome/aspect prediction 范式，但这些任务仍不能
  直接回答反馈是否具体、可验证、可执行或有科学根据。

#### 8. DISAPERE: A Dataset for Discourse Structure in Peer Review Discussions

- **地址/发表：** [NAACL 2022 / ACL Anthology](https://aclanthology.org/2022.naacl-main.89/)，DOI
  [`10.18653/v1/2022.naacl-main.89`](https://doi.org/10.18653/v1/2022.naacl-main.89)，正式会议论文。
- **摘要介绍：** DISAPERE 对 review--rebuttal 讨论中的句子进行话语功能、立场和论证
  关系标注，使审稿意见与作者回应之间的结构可以被计算建模。
- **与本小节的关系：** 它把“意见能否被定位和回应”转化为细粒度对象，为
  actionability 与 argument grounding 提供数据基础。

#### 9. arXivEdits: Understanding the Human Revision Process in Scientific Writing

- **地址/发表：** [EMNLP 2022 / ACL Anthology](https://aclanthology.org/2022.emnlp-main.641/)，DOI
  [`10.18653/v1/2022.emnlp-main.641`](https://doi.org/10.18653/v1/2022.emnlp-main.641)，正式会议论文。
- **摘要介绍：** 论文构建科学文章多版本语料和细粒度修订标注，研究句子对齐、编辑
  片段识别及 edit intent 分类。
- **与本小节的关系：** 它把评价对象从“是否要改”推进到“为何修改以及修改承担什么
  写作功能”，但仍不等同于最终 revision quality 判断。

#### 10. CORWA: A Citation-Oriented Related Work Annotation Dataset

- **地址/发表：** [NAACL 2022 / ACL Anthology](https://aclanthology.org/2022.naacl-main.397/)，DOI
  [`10.18653/v1/2022.naacl-main.397`](https://doi.org/10.18653/v1/2022.naacl-main.397)，正式会议论文。
- **摘要介绍：** CORWA 标注 related work 中不同来源与不同类型的 citation text
  fragments，揭示 related work 是变长、多来源的篇章结构，而非彼此独立的摘要句。
- **与本小节的关系：** 它为 coherence 和 positioning 提供语言学基础，但本身是
  annotation/generation resource，不是质量评价器。

#### 11. NLPeer: A Unified Resource for the Computational Study of Peer Review

- **地址/发表：** [ACL 2023 / ACL Anthology](https://aclanthology.org/2023.acl-long.277/)，DOI
  [`10.18653/v1/2023.acl-long.277`](https://doi.org/10.18653/v1/2023.acl-long.277)，正式会议论文。
- **摘要介绍：** NLPeer 跨多个来源收集论文与评论，并保留稿件版本、结构化表示和
  元数据，同时提供多类审稿辅助任务。
- **与本小节的关系：** 它支持跨领域、版本感知和 paper-grounded 的评价研究，弥补
  单领域、单版本 peer-review 数据的限制。

#### 12. CiteBench: A Benchmark for Scientific Citation Text Generation

- **地址/发表：** [EMNLP 2023 / ACL Anthology](https://aclanthology.org/2023.emnlp-main.455/)，DOI
  [`10.18653/v1/2023.emnlp-main.455`](https://doi.org/10.18653/v1/2023.emnlp-main.455)，正式会议论文。
- **摘要介绍：** CiteBench 统一多个 citation-text 数据集和任务定义，比较跨数据集
  迁移，并指出既有评价设置与输入条件缺少一致性。
- **与本小节的关系：** 它说明 citation text 的质量同时依赖被引来源和 citing context，
  通用 NLG 相似度不足以覆盖 coherence 与 positioning。

#### 13. Systematic Task Exploration with LLMs: A Study in Citation Text Generation

- **地址/发表：** [ACL 2024 / ACL Anthology](https://aclanthology.org/2024.acl-long.265/)，DOI
  [`10.18653/v1/2024.acl-long.265`](https://doi.org/10.18653/v1/2024.acl-long.265)，正式会议论文。
- **摘要介绍：** 论文系统改变任务说明和输入配置，并结合参考文本、自动指标和人工
  评价分析 citation generation，显示任务定义与指标之间不存在简单的一一对应关系。
- **与本小节的关系：** 它直接支持科学写作评价必须显式给定 input、rubric 和 construct，
  不能只报告一个 aggregate generation score。

#### 14. Related Work and Citation Text Generation: A Survey

- **地址/发表：** [EMNLP 2024 / ACL Anthology](https://aclanthology.org/2024.emnlp-main.767/)，DOI
  [`10.18653/v1/2024.emnlp-main.767`](https://doi.org/10.18653/v1/2024.emnlp-main.767)，正式会议论文。
- **摘要介绍：** 该综述梳理 related-work/citation generation 的任务定义、方法与长期
  挑战，强调需要形成连贯的文献叙事并说明当前工作的相对位置。
- **与本小节的关系：** 可用于定义 related work 的两个职责：综合既有研究，以及定位
  本文工作；它不是本文三个 related-work 判断任务的直接数据来源。

### C. LLM 科学反馈：从现实效用到细粒度可靠性

#### 15. Can Large Language Models Provide Useful Feedback on Research Papers? A Large-Scale Empirical Analysis

- **地址/发表：** [NEJM AI, 2024](https://ai.nejm.org/doi/10.1056/AIoa2400196)，DOI
  [`10.1056/AIoa2400196`](https://doi.org/10.1056/AIoa2400196)，正式期刊论文。
- **摘要介绍：** 研究在数千篇 Nature-family 和 ICLR 论文上比较 GPT-4 与人类审稿
  观点的重合，并通过研究者调查考察反馈的主观帮助程度；同时发现模型倾向模板化建议，
  对方法设计的深入批评较弱。
- **与本小节的关系：** 它是自动科研反馈具有现实价值的强证据，但 overlap 和作者认为
  helpful 都不能直接证明离散评分或科学判断可靠。

#### 16. Automated Focused Feedback Generation for Scientific Writing Assistance

- **地址/发表：** [Findings of ACL 2024](https://aclanthology.org/2024.findings-acl.580/)，DOI
  [`10.18653/v1/2024.findings-acl.580`](https://doi.org/10.18653/v1/2024.findings-acl.580)，正式会议论文。
- **摘要介绍：** SWIF2T 使用 planner、investigator、reviewer 和 controller 生成具体、
  可执行且连贯的科学写作反馈，并在真实弱点评论上进行人工评价。
- **与本小节的关系：** 它支持 specificity、actionability、reading comprehension 和
  helpfulness 是不同维度；但研究目标是反馈生成，不是评分模型。

#### 17. Is LLM a Reliable Reviewer? A Comprehensive Evaluation of LLM on Automatic Paper Reviewing Tasks

- **地址/发表：** [LREC-COLING 2024 / ACL Anthology](https://aclanthology.org/2024.lrec-main.816/)，DOI
  [`10.63317/48d359hjdvog`](https://doi.org/10.63317/48d359hjdvog)，正式会议论文。
- **摘要介绍：** 论文系统评测 score prediction、review generation 和基于
  review--revision 的问题回答，发现模型在长文理解、zero-shot scoring 和关键反馈方面
  仍有明显不足。
- **与本小节的关系：** 它完成从“LLM 能生成评论”到“LLM 是否是可靠 reviewer”的
  问题转向，可用于反驳仅凭生成流畅度判断 evaluator 能力。

#### 18. LLMs Assist NLP Researchers: Critique Paper (Meta-)Reviewing

- **地址/发表：** [EMNLP 2024 / ACL Anthology](https://aclanthology.org/2024.emnlp-main.292/)，DOI
  [`10.18653/v1/2024.emnlp-main.292`](https://doi.org/10.18653/v1/2024.emnlp-main.292)，正式会议论文。
- **摘要介绍：** 论文构建 ReviewCritique，把人类与 LLM 评论切分并标注局部缺陷及其
  解释，同时评估模型作为 reviewer 和 meta-reviewer 的能力。
- **与本小节的关系：** 它将整体相似度推进为 localized deficiency evaluation，与本文
  区分标签判断和理由质量最为接近。

#### 19. Automated Peer Reviewing in Paper SEA: Standardization, Evaluation, and Analysis

- **地址/发表：** [Findings of EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.595/)，DOI
  [`10.18653/v1/2024.findings-emnlp.595`](https://doi.org/10.18653/v1/2024.findings-emnlp.595)，正式会议论文。
- **摘要介绍：** 论文将多来源评论标准化，生成建设性评论，并以 mismatch/consistency
  信号检查 paper--review 是否一致，再利用该信号进行自校正。
- **与本小节的关系：** 它把“评论是否由论文支持”明确为独立评价目标，说明
  paper-grounding 不能被评论的语言质量替代。

#### 20. Identifying Reliable Evaluation Metrics for Scientific Text Revision

- **地址/发表：** [ACL 2025 / ACL Anthology](https://aclanthology.org/2025.acl-long.335/)，DOI
  [`10.18653/v1/2025.acl-long.335`](https://doi.org/10.18653/v1/2025.acl-long.335)，正式会议论文。
- **摘要介绍：** 论文以人工标注检验 scientific text revision 指标，显示 ROUGE、
  BERTScore 等偏向文本相似，LLM judge 对 instruction following 较敏感但在 correctness
  上仍不稳定，并建议结合任务专用指标。
- **与本小节的关系：** 它把评价对象扩展到修订文本，并为“rubric/task-specific metric
  与通用 judge 互补”提供直接证据。

#### 21. LazyReview: A Dataset for Uncovering Lazy Thinking in NLP Peer Reviews

- **地址/发表：** [ACL 2025 / ACL Anthology](https://aclanthology.org/2025.acl-long.165/)，DOI
  [`10.18653/v1/2025.acl-long.165`](https://doi.org/10.18653/v1/2025.acl-long.165)，正式会议论文。
- **摘要介绍：** 论文构建细粒度 lazy-thinking 评论语料，发现 zero-shot LLM 难以稳定
  检测这类模式，而专门训练可改善识别，并能支持更完整、可执行的评论修订。
- **与本小节的关系：** 它说明 review quality 还涉及认知捷径与推理充分性，而非只有
  文体和礼貌；SciRM 也将其视为相邻的科学评审评价工作。

#### 22. Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers

- **地址/发表：** [ACL 2025 / ACL Anthology](https://aclanthology.org/2025.acl-long.1009/)，DOI
  [`10.18653/v1/2025.acl-long.1009`](https://doi.org/10.18653/v1/2025.acl-long.1009)，正式会议论文。
- **摘要介绍：** 论文提出 limitation taxonomy 和 LIMITGEN，其中同时包含受控扰动的
  合成数据与真实人工局限，并研究文献检索对识别关键研究缺陷的帮助。
- **与本小节的关系：** 它把总体 review quality 下钻到“能否识别实质性 limitation”，
  说明高质量科学评价不能停留在语言流畅度或总体建议。

#### 23. CLAIMCHECK: How Grounded Are LLM Critiques of Scientific Papers?

- **地址/发表：** [Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1185/)，DOI
  [`10.18653/v1/2025.findings-emnlp.1185`](https://doi.org/10.18653/v1/2025.findings-emnlp.1185)，正式会议论文。
- **摘要介绍：** CLAIMCHECK 将论文 claim、审稿 weakness 与二者的链接进行专家标注，
  并设置 claim-centric 的定位、对齐和验证任务；当前模型在需要专家判断的部分仍明显
  受限。
- **与本小节的关系：** 它把 grounding 操作化为 critique--claim 对齐和 claim
  verification，为“评价理由需要证据链”提供最直接的基准之一。

#### 24. Mind the Blind Spots: A Focus-Level Evaluation Framework for LLM Reviews

- **地址/发表：** [EMNLP 2025 / ACL Anthology](https://aclanthology.org/2025.emnlp-main.1805/)，DOI
  [`10.18653/v1/2025.emnlp-main.1805`](https://doi.org/10.18653/v1/2025.emnlp-main.1805)，正式会议论文。
- **摘要介绍：** 论文把评论关注点表示为 target（如 problem、method、experiment）和
  aspect（如 validity、clarity、novelty）的分布；实验发现通用 LLM 倾向技术有效性，
  却显著遗漏 novelty。
- **与本小节的关系：** 它说明即使局部评论看似正确，整体 focus coverage 仍可能偏移，
  aggregate score 会掩盖系统性盲区。

#### 25. ReviewEval: An Evaluation Framework for AI-Generated Reviews

- **地址/发表：** [Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1120/)，DOI
  [`10.18653/v1/2025.findings-emnlp.1120`](https://doi.org/10.18653/v1/2025.findings-emnlp.1120)，正式会议论文。
- **摘要介绍：** ReviewEval 从 human alignment、factual accuracy、analytical depth、
  constructiveness 和 guideline adherence 等维度评价 AI 生成评论，并将评价框架用于
  改进生成系统。
- **与本小节的关系：** 它代表面向完整 AI review 的综合框架，与 RevUtil 的作者效用
  维度互补，但目标仍主要是评价生成出的 review。

#### 26. Where Do LLMs Go Wrong? Diagnosing Automated Peer Review via Aspect-Guided Multi-Level Perturbation

- **地址/发表：** [CIKM 2025](https://doi.org/10.1145/3746252.3761274)，正式会议论文。
- **摘要介绍：** 论文对 paper、review 和 rebuttal 的贡献、可靠性、呈现、语气及完整性
  进行定向扰动，发现错误评论、强拒绝结论和 rebuttal 表达可系统性影响 reviewer 与
  meta-reviewer，且部分偏差在 CoT prompting 下仍存在。
- **与本小节的关系：** 它提供反事实 robustness 视角，说明可靠评价既要对实质变化
  敏感，也应对不相关表面变化保持不变。

#### 27. The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors

- **地址/发表：** [EMNLP 2025 / ACL Anthology](https://aclanthology.org/2025.emnlp-main.1476/)，DOI
  [`10.18653/v1/2025.emnlp-main.1476`](https://doi.org/10.18653/v1/2025.emnlp-main.1476)，正式会议论文。
- **摘要介绍：** RevUtil 使用人工与合成评论，将作者效用拆分为 actionability、
  grounding and specificity、helpfulness 与 verifiability，并训练自动评价器。
- **与本小节的关系：** 它是本文四个 ordinal peer-review 任务的直接 construct 和数据
  来源，因此应是正文核心引用，而不只是宽泛背景。

#### 28. OpenReviewer: A Specialized Large Language Model for Generating Critical Scientific Paper Reviews

- **地址/发表：** [NAACL 2025 System Demonstrations](https://aclanthology.org/2025.naacl-demo.44/)，DOI
  [`10.18653/v1/2025.naacl-demo.44`](https://doi.org/10.18653/v1/2025.naacl-demo.44)，正式会议论文。
- **摘要介绍：** OpenReviewer 使用大规模专家评论微调开源模型，并在真实论文上考察
  评论批判性与推荐分布，将系统定位为投稿前辅助工具。
- **与本小节的关系：** 它说明领域训练能够改变 review 风格和推荐分布，但“更像人类”
  仍不自动等于证据层面更正确。

#### 29. Peer Review in the Age of Artificial Intelligence: A Comparative Study of Human and AI-Generated Review Reports

- **地址/发表：** [Postgraduate Medical Journal, 2026](https://doi.org/10.1093/postmj/qgag005)，正式期刊论文。
- **摘要介绍：** 研究对真实论文的人类和 AI 评论进行盲评比较，发现 AI 往往覆盖章节
  更全面、格式更规整，而人类在解释深度、原创性和情境适用性上更强。
- **与本小节的关系：** 它将当前差异概括为 surface coverage 与 epistemic depth 的分离，
  支持采用多维评价而不是笼统宣称 AI 已经超过或不如人类。

### D. 最接近本文的专用评价器

#### 30. Reward Modeling for Scientific Writing Evaluation (SciRM)

- **地址/发表：** [ACL 2026 / ACL Anthology](https://aclanthology.org/2026.acl-long.567/)，DOI
  [`10.18653/v1/2026.acl-long.567`](https://doi.org/10.18653/v1/2026.acl-long.567)，正式会议论文。
- **摘要介绍：** SciRM 面向动态 criteria/rubric 的科学写作评价，先优化科学评价偏好，
  再强化 reasoning，并通过多 aspect、多任务联合训练实现跨任务及未见设置泛化。
- **与本小节的关系：** 这是本文最接近的领域基线。SciRM 关注如何构建更强、可复用的
  multi-task reward model；本文应突出 matched training/inference interface 下 rationale
  supervision 的作用与负迁移诊断，而不能再声称首次提出 scientific-writing evaluator。

### E. 值得保留但尚未正式发表的工作

以下条目只在其直接定义本文任务或提供目前没有正式论文替代的新评价设计时保留。
投稿前必须再次检查版本和发表状态。

#### 31. Expert Preference-Based Evaluation of Automated Related Work Generation

- **地址/状态：** [arXiv:2508.07955](https://arxiv.org/abs/2508.07955)，预印本。
- **摘要介绍：** 论文面向自动 related-work generation 建立专家偏好评价，将质量拆为
  连贯性、研究定位等细粒度维度，并用对比样例帮助评价与专家标准对齐。
- **与本小节的关系：** 它直接定义本文的 coherence、positioning type 和 positioning
  check 三个任务，也是 SciRM 复用的直接来源，因此即使未正式发表也必须保留。

#### 32. PRISM: A Multi-Dimensional Benchmark for Evaluating LLM Peer Reviewers

- **地址/状态：** [arXiv:2605.26730](https://arxiv.org/abs/2605.26730)，预印本。
- **摘要介绍：** PRISM 从分析深度、新颖性评价、缺陷识别与优先级、建设性四个维度
  统一比较多种自动 reviewer 和人类评论，并用 argument mining、retrieval verification
  与 consensus scoring 减少对表面指标的依赖。
- **与本小节的关系：** 它是较新的综合 benchmark，可强化“不同系统有不同盲区”的
  论点，但它评估完整 review system，不是本文七项判断任务的训练来源。

#### 33. Automatic Reviewers Fail to Detect Faulty Reasoning in Research Papers: A New Counterfactual Evaluation Framework

- **地址/状态：** [arXiv:2508.21422](https://arxiv.org/abs/2508.21422)，预印本。
- **摘要介绍：** 论文通过向研究文本注入可控 faulty reasoning，检查自动 reviewer 能否
  对实质性错误作出方向正确的反应，从而把评测从自然数据相关性推进到反事实能力测试。
- **与本小节的关系：** 它为本文未来的 counterfactual diagnostics 提供设计依据，但
  目前只宜作为 robustness 补充，不能承担正式发表证据的核心位置。

#### 34. Do Methods Support the Claims? Intra-Paper Verification for Peer Review

- **地址/状态：** [arXiv:2607.26066](https://arxiv.org/abs/2607.26066)，预印本。
- **摘要介绍：** 论文把论文内部验证拆为 claim 抽取、方法证据定位和 claim--method
  支持关系判断，使科学评价的中间证据可以被显式检查。
- **与本小节的关系：** 它为结构化、可验证的 rationale 提供比自由文本“三步推理”
  更有科学含义的模板，适合放在讨论或未来工作，而不是当前核心相关工作。

### 正文选文建议

若小节最终只有三段，优先引用下面的最小集合：

- **第一段（对象与多维构念）：** Liang et al. (2024)、SWIF2T、RevUtil、
  scientific text revision metrics；
- **第二段（总分不足与可靠性）：** LIMITGEN、CLAIMCHECK、Mind the Blind Spots、
  aspect-guided perturbation；
- **第三段（专用 evaluator 与本文差异）：** related-work expert-preference evaluation、
  RevUtil、SciRM。

PeerRead、NLPeer、RQI/ARCADIA、AESW/arXivEdits/CORWA/CiteBench 等保留在文献库中，
只有在需要交代数据谱系、测量依据或回应审稿意见时再进入正文。SWIF2T、OpenReviewer、
ReviewEval 等生成系统相关论文只能证明现实需求或评价维度，不能用来证明 rationale
supervision 或 reward-model scoring 有效。

## Related Work 资料：训练 LLM-as-a-Judge

更新时间：2026-08-11。本组资料服务于 `Training LLM-as-a-Judge` 小节，重点区分
评价协议、训练监督和输出接口。除 Prometheus 使用 OpenReview 的正式会议信息外，
其余条目均由 ACL Anthology、NeurIPS Proceedings 或 DOI 元数据核验。

### 建议的讲故事逻辑

1. **先定义协议，而不是先列模型。** Pointwise/direct assessment 对单个候选输出
   绝对分数，pairwise assessment 在两个候选之间做相对选择。两者的标签空间、提示
   输入和可比较结果不同，不能视为同一种任务的两种无关紧要的输出格式。
2. **再说明 prompted judge 的接口会影响判断。** G-Eval、MT-Bench 等工作表明 LLM
   可以执行 rubric-conditioned scoring 或 response comparison；但解释请求、候选顺序
   和评测模板本身都可能改变结果，因此 interface 也是 evaluator 定义的一部分。
3. **最后转向 fine-tuned open judge。** Prometheus、Prometheus 2 和 Foundational
   Autoraters 通过评价数据训练开放模型，但它们同时改变任务混合、rubric、反馈文本和
   分数格式。TRACT 在联合序列 CE 上额外加入数值回归损失。由此可提出更精确的问题：
   在 rubric-conditioned binary/ordinal scoring 中，rationale supervision、推理时
   rationale elicitation 和 score objective/readout 的作用需要分别考察。

### A. 评价协议与 prompted judges

#### 1. Can Large Language Models Be an Alternative to Human Evaluations?

- **地址/发表：** [ACL 2023](https://aclanthology.org/2023.acl-long.870/)，DOI
  [`10.18653/v1/2023.acl-long.870`](https://doi.org/10.18653/v1/2023.acl-long.870)，正式会议论文。
- **摘要介绍：** 论文将人类评价使用的相同任务说明、样本和问题交给 LLM，让模型对
  故事生成和对抗攻击文本作答和评分，并比较 LLM evaluation 与专家评价的一致性。
- **与本小节的关系：** 这是 TRACT 用来定义 LLM-as-a-Judge/LLM evaluation 的早期
  direct-assessment 来源，适合支撑本小节的第一句，而不是支撑微调或 CoT 监督。

#### 2. A Closer Look into Using Large Language Models for Automatic Evaluation

- **地址/发表：** [Findings of EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.599/)，DOI
  [`10.18653/v1/2023.findings-emnlp.599`](https://doi.org/10.18653/v1/2023.findings-emnlp.599)，正式会议论文。
- **摘要介绍：** 论文比较不同 LLM evaluation 流程，发现 G-Eval 的自动 CoT 并不总能
  提高与人类评分的相关性，而要求模型解释自身评分在其测试设置中更稳定地改善相关性。
- **与本小节的关系：** 它直接证明“是否输出解释”不是中性的排版选择，也提醒不能从
  rationale-first 结果直接推断训练阶段的 rationale supervision 有效。

#### 3. G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment

- **地址/发表：** [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.153/)，DOI
  [`10.18653/v1/2023.emnlp-main.153`](https://doi.org/10.18653/v1/2023.emnlp-main.153)，正式会议论文。
- **摘要介绍：** G-Eval 使用自然语言评价标准和自动生成的 evaluation steps，引导
  GPT-4 对 NLG 输出评分，并利用输出分数的 token probabilities 聚合评价结果。
- **与本小节的关系：** 它是 rubric/step-conditioned pointwise judge 的代表，适合说明
  推理时先产生评价步骤与最终如何读出分数是两个不同设计环节。

#### 4. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

- **地址/发表：** [NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)，DOI
  [`10.52202/075280-2020`](https://doi.org/10.52202/075280-2020)，正式会议论文。
- **摘要介绍：** 论文研究强 LLM 作为开放式对话 judge 的可行性，构建 MT-Bench 与
  Chatbot Arena，并分析 position、verbosity、self-enhancement 和推理能力等局限。
- **与本小节的关系：** 它是 single-answer grading 与 pairwise comparison 的核心
  范式来源，也说明 pairwise 结果不能脱离候选顺序和偏差控制来解释。

#### 5. Large Language Models Are Not Fair Evaluators

- **地址/发表：** [ACL 2024](https://aclanthology.org/2024.acl-long.511/)，DOI
  [`10.18653/v1/2024.acl-long.511`](https://doi.org/10.18653/v1/2024.acl-long.511)，正式会议论文。
- **摘要介绍：** 论文系统研究 pairwise LLM evaluators 的顺序偏差，并提出在交换候选
  顺序时综合两次结果的校准方法，以降低评价次序对最终选择的影响。
- **与本小节的关系：** 它为“评测协议与输出接口会改变 judge 可靠性”提供直接证据；
  这类 pairwise position bias 与本文 direct scoring 的 readout 问题相关但并不相同。

### B. Fine-tuned open judges 与 rationale/score 监督

#### 6. Prometheus: Inducing Fine-Grained Evaluation Capability in Language Models

- **地址/发表：** [ICLR 2024 / OpenReview](https://openreview.net/forum?id=8euJaTveKw)，正式会议论文。
- **摘要介绍：** Prometheus 使用用户给定 rubric、reference answer、评价反馈和分数
  数据训练开放模型，使其能够针对细粒度标准产生 feedback 并给出绝对评分。
- **与本小节的关系：** 它与本文最接近的通用监督模板之一，因为 rationale-like
  feedback 与 score 被联合输出；但其设计不能单独识别哪部分监督带来评分收益。

#### 7. Prometheus 2: An Open Source Language Model Specialized in Evaluating Other Language Models

- **地址/发表：** [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.248/)，DOI
  [`10.18653/v1/2024.emnlp-main.248`](https://doi.org/10.18653/v1/2024.emnlp-main.248)，正式会议论文。
- **摘要介绍：** Prometheus 2 分别针对 direct assessment 和 pairwise ranking 进行
  微调，再通过模型权重合并，使同一开放 judge 支持绝对评分和相对比较。
- **与本小节的关系：** 它说明两类协议可以由同一 judge 支持，但并未因此消除两种
  标签空间、提示接口和评价目标的差异。

#### 8. Foundational Autoraters: Taming Large Language Models for Better Automatic Evaluation

- **地址/发表：** [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.949/)，DOI
  [`10.18653/v1/2024.emnlp-main.949`](https://doi.org/10.18653/v1/2024.emnlp-main.949)，正式会议论文。
- **摘要介绍：** 论文提出 FLAMe 模型家族，以一百多种质量评价任务和大规模公开人类
  判断训练通用 autorater，考察其在留出任务上的迁移和进一步微调能力。
- **与本小节的关系：** 它代表“扩大任务混合以训练可复用 judge”的路线；与本文逐任务
  控制训练目标的路线不同，任务混合收益不能回答 rationale supervision 的独立作用。

#### 9. TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge

- **地址/发表：** [ACL 2025](https://aclanthology.org/2025.acl-long.147/)，DOI
  [`10.18653/v1/2025.acl-long.147`](https://doi.org/10.18653/v1/2025.acl-long.147)，正式会议论文。
- **摘要介绍：** TRACT 面向 direct ordinal assessment，先生成可复用的 self-CoT，
  再对 `[CoT, gold score]` 完整序列计算 cross-entropy，并根据 score token 概率计算
  期望分数和额外的 regression-aware loss，从而显式考虑分数之间的数值距离。
- **与本小节的关系：** 它是当前最接近“rationale supervision 如何与 score learning
  交互”的方法工作；但其完整方法同时改变 CoT 来源、训练阶段和数值损失，不能单凭
  最终性能确定 rationale tokens 本身的因果贡献。

### C. 可扩展的生成式 open judges

#### 10. CritiqueLLM: Towards an Informative Critique Generation Model for Evaluation of Large Language Model Generation

- **地址/发表：** [ACL 2024](https://aclanthology.org/2024.acl-long.704/)，DOI
  [`10.18653/v1/2024.acl-long.704`](https://doi.org/10.18653/v1/2024.acl-long.704)，正式会议论文。
- **摘要介绍：** CritiqueLLM 构造多类 critique 数据，以生成式多任务训练支持有参考和
  无参考的 direct/pairwise 评价，并输出 critique 及数值分数或偏好标签。
- **与本小节的关系：** 它进一步说明 explanation-like output 与判断标签经常被联合
  监督；但 critique 的信息量不能直接证明其改善了最终标签的学习。

#### 11. PandaLM: An Automatic Evaluation Benchmark for LLM Instruction Tuning Optimization

- **地址/发表：** [ICLR 2024 / OpenReview](https://openreview.net/forum?id=5Nn2BLV7SB)，正式会议论文。
- **摘要介绍：** PandaLM 构建 instruction-following response 的成对评价数据，训练
  开放 evaluator 输出比较理由、winner/tie 判断及参考回答，并用于模型选择。
- **与本小节的关系：** 它对应 TRACT 所述“以开放数据微调 open-source judge”的
  pairwise 路线，但其 preference label 与本文的绝对 binary/ordinal score 不同。

#### 12. Generative Judge for Evaluating Alignment

- **地址/发表：** [ICLR 2024 / OpenReview](https://openreview.net/forum?id=gtkFw6sZGS)，正式会议论文。
- **摘要介绍：** Auto-J 将 direct scoring 和 pairwise comparison 统一为结构化的
  critique-and-judgment 生成任务，并以合成评价数据微调开放模型。
- **与本小节的关系：** 它代表以统一生成接口覆盖多种判断协议的路线，也进一步说明
  比较不同 judge 时不能把 critique、标签格式和任务混合的作用混为一谈。

#### 13. JudgeLM: Fine-Tuned Large Language Models Are Scalable Judges

- **地址/发表：** [ICLR 2025 / OpenReview](https://openreview.net/forum?id=xsELpEPn4A)，正式会议论文。
- **摘要介绍：** JudgeLM 主要面向 pairwise judge，使用合成判断微调开放模型，并通过
  candidate swap、reference manipulation 等设计分析或缓解位置、知识和格式偏差。
- **与本小节的关系：** 它说明 open-judge training 需要显式控制数据与接口偏差；但其
  主要目标仍是成对选择，不直接回答 rationale supervision 对绝对分数学习的影响。

### 正文选文与证据边界

正文第一段使用 G-Eval 和 MT-Bench 定义范式，再用 Chiang and Lee 与 order-bias
研究说明 interface 会改变判断。第二段使用 Prometheus、Prometheus 2、Foundational
Autoraters 和 TRACT 比较训练监督。PandaLM、JudgeLM、CritiqueLLM 与 Auto-J 作为
扩展背景保留在文献库中，但在当前两段篇幅内不会增加新的论证职责，因此不进入正文。

需要保持四个边界：direct 与 pairwise 不是同一标签任务；推理时要求 explanation 不等于
训练时优化 rationale tokens；feedback/rationale 的可读性不等于 score correctness；
TRACT 的改进来自组合方法，不能归因于其中单个组件，除非有匹配的消融证据。

## Related Work 资料：CoT Elicitation 与 Rationale Supervision

### 本小节的讲述逻辑

这一小节不再重复介绍 open judge 的模型谱系，而是解释一个控制问题。首先区分
推理时 `rationale elicitation` 与训练时 `rationale supervision`；随后并列呈现 CoT
有效、无效和有害的条件性证据；最后指出已有方法通常同时改变 rationale target、
score objective 和 inference interface，因此无法把评分变化归因于某一个因素。

正文只在定位层面引出本文的设计：需要交叉比较 score-only/rationale-supervised
training 与 direct/rationale-first inference。具体损失、数据配对和实验结果应留在
Method 与 Experiments，而不应提前写入 Related Work。

### A. 推理时 CoT 的任务依赖性

#### 1. To CoT or Not to CoT? Chain-of-Thought Helps Mainly on Math and Symbolic Reasoning

- **地址/发表：** [ICLR 2025](https://openreview.net/forum?id=w6nlcS8Kkn)，
  正式会议论文；[arXiv](https://arxiv.org/abs/2409.12183)。
- **摘要介绍：** 论文结合百余篇论文的定量分析与跨模型、跨任务实验，发现 CoT 的
  明显收益主要集中在数学、逻辑和符号执行任务，在其他任务上的平均收益较小。
- **与本小节的关系：** 它支持“CoT 收益依赖任务结构”，但研究对象主要是推理时
  prompting，不能用于证明 rationale SFT 会提高或损害科学写作评分。

#### 2. Improving LLM-as-a-Judge Inference with the Judgment Distribution

- **地址/发表：** [Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1259/)，
  DOI [`10.18653/v1/2025.findings-emnlp.1259`](https://doi.org/10.18653/v1/2025.findings-emnlp.1259)，
  正式会议论文。
- **摘要介绍：** 论文比较从 judgment-token distribution 读出 pointwise、pairwise
  和 listwise 判断的方法，发现均值读出优于仅取众数；其 CoT 实验还显示，CoT 可能
  压缩判断分布的离散程度并降低评价性能。
- **与本小节的关系：** 这是“CoT 可能伤害评分”的 judge-specific 证据，但它讨论的
  是推理和读出，不是训练时的 rationale supervision。

### B. 训练时 rationale supervision 的条件性证据

#### 3. Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes

- **地址/发表：** [Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.507/)，
  DOI [`10.18653/v1/2023.findings-acl.507`](https://doi.org/10.18653/v1/2023.findings-acl.507)，
  正式会议论文。
- **摘要介绍：** 论文把标签预测和 LLM 生成 rationale 组织为带不同任务前缀的
  multi-task supervision，使较小模型在四个 NLP benchmark 上以更少训练数据取得
  优于标准微调或蒸馏的表现。
- **与本小节的关系：** 它是重要的正面证据，也说明“使用 rationale”不必等价于把
  长 rationale 与短标签串成一个未分解的生成目标。

#### 4. Investigating the Impact of Rationales for LLMs on Natural Language Understanding

- **地址/发表：** [arXiv:2510.16686](https://arxiv.org/abs/2510.16686)，
  截至 2026-08-11 为预印本，未核验到正式会议或期刊版本。
- **摘要介绍：** 论文在多类中文 NLU 任务上比较 Label-Only、Reason、Explain、Mix
  和 Align。多数直接加入 rationale 的训练方法弱于 label-only；分别归一化并加权
  标签与 rationale 目标的 Align 则取得较稳定的改善。
- **与本小节的关系：** 它最接近“训练目标如何改变判别能力”的受控研究，提示问题
  可能来自目标结构和损失分配，而不能概括为 rationale 本身必然有害。

#### 5. Supervised Fine-tuning with Synthetic Rationale Data Hurts Real-World Disease Prediction

- **地址/发表：** [arXiv:2606.10279](https://arxiv.org/abs/2606.10279)，
  截至 2026-08-11 为预印本，未核验到正式会议或期刊版本。
- **摘要介绍：** 论文在真实纵向健康记录的疾病预测中比较 label-only 与两类合成
  rationale SFT，报告 rationale 条件在多种配置下明显落后；相似解释作为 few-shot
  demonstration 却可能改善推理时表现。
- **与本小节的关系：** 它直接说明“推理时解释有用”与“解释适合作为 SFT target”
  不是同一命题。其 token dilution 与目标冲突仍属于待进一步验证的机制解释。

#### 6. Rethinking Supervised Fine-Tuning: Emphasizing Key Answer Tokens for Improved LLM Accuracy

- **地址/发表：** [arXiv:2512.21017](https://arxiv.org/abs/2512.21017)，
  截至 2026-08-11 为预印本，未核验到正式会议或期刊版本。
- **摘要介绍：** SFTKey 先在完整 reasoning--answer 序列上训练，再增加仅优化答案
  token 的阶段，以缓解长 CoT 中最终答案监督相对不足的问题。
- **与本小节的关系：** 它把“答案 token 是否得到足够优化”变成直接基线，但两阶段
  训练同时改变训练预算，因此仍不能单独确定 rationale token 的因果作用。

### 正文证据边界

本小节可以写“CoT 的评分收益具有任务和接口依赖性”以及“naive rationale SFT 在
部分判别任务中弱于 label-only”。不能据此写“CoT 普遍伤害评分”，也不能把
`rationale elicitation`、`rationale supervision`、解释忠实性和 score correctness
视为同一个变量。最接近的 TRACT 已在上一小节介绍；这里引用它的职责只是说明，
即便显式增加 score-aware loss，CoT 来源、训练阶段和推理读出仍可能同时变化。
