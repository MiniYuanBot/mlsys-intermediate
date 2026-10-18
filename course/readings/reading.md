---
title: 如何把 200 篇变成可执行的研究地图
tags: [reading, research]
updated: 2026-10-18
---

# 如何把 200 篇变成可执行的研究地图

[路线](../guides/roadmap.md) · [方向选择](../guides/directions.md) · [资料总表](resources.md) · [论文卡](paper-card.md)

“先读 200 篇”可能是希望你拓宽视野，也可能包含导师对阅读深度的其他预期，不能自行理解为每篇都要完整复现。可以把下面的分层记录带给导师，确认需要达到什么深度；本课程不替代他的要求。

## 三层阅读

| 层级        | 时间        | 写出什么                                                         | 什么情况下升级               |
| ----------- | ----------- | ---------------------------------------------------------------- | ---------------------------- |
| S：筛选     | 15–25min/篇 | 问题、关键约束、机制一句话、最关键的图、保留或排除的理由         | 与当前实验变量直接相关       |
| D：精读     | 2–4h/篇     | 目标函数、正确性条件、baseline、硬件和数据、主要消融、缺失的证据 | 要作为最近邻或复现对象       |
| R：复现机制 | 8–20h 起    | 源码入口、固定 commit、一个机制的复现、与原文的差异说明          | 主方向值得继续，资源也够得着 |

第一轮建议 24–32 篇 S、6–8 篇 D、1–2 个 R；之后每周 6–10 篇 S、2 篇 D，逐步扩成 200 篇的主题地图。三层是包含关系，同一篇论文不能算三次。如果导师明确要求 200 篇都精读，需要重新讨论工期和实验产出。

## 第一轮必读：每个方向 3 篇

下表从 [资料总表](resources.md) 的种子论文里每个方向挑 3 篇，九个方向共 27 篇，正好是第一轮 S 的量。读完 S 后，在自己的主选方向（见[方向选择](../guides/directions.md)）里挑 2–3 篇升到 D。标「苗」的是苗旭鹏老师参与的论文，合作关系见资料总表。

| 方向                                                        | 第 1 篇（先读）                                        | 第 2 篇                                                                                                                        | 第 3 篇                                                                       | 为什么是这三篇                                                      |
| ----------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [D1 KV 缓存](../modules/01-kv-cache.md)                     | [KIVI](https://arxiv.org/abs/2402.02750)               | [StreamingLLM](https://arxiv.org/abs/2309.17453)                                                                               | [PQCache](https://arxiv.org/abs/2407.12820)（苗）                             | 量化、淘汰、检索三条路线各一篇，正好对应 lab1 的 A/B、B1、C         |
| [D2 低比特 kernel](../modules/02-low-bit-kernels.md)        | [AWQ](https://arxiv.org/abs/2306.00978)                | [Marlin](https://arxiv.org/abs/2408.11743)                                                                                     | [QServe](https://arxiv.org/abs/2405.04532)                                    | 数值方法、单个 kernel、完整系统三个层次                             |
| [D3 动态编译](../modules/03-dynamic-compilation.md)         | [PyTorch 2](https://doi.org/10.1145/3620665.3640366)   | [Relax](https://arxiv.org/abs/2311.02103)                                                                                      | [Mirage](https://arxiv.org/abs/2405.05751)（苗）                              | guard 与重编译、符号形状、超优化，覆盖讲义的主线和组内方向          |
| [D4 推测解码](../modules/04-speculative-slo.md)             | [Leviathan 等](https://arxiv.org/abs/2211.17192)       | [SpecInfer](https://arxiv.org/abs/2305.09781)（苗）                                                                            | [AdaServe](https://arxiv.org/abs/2501.12162)（苗）                            | 算法基础、树形验证、多 SLO 服务；后两篇是组内工作                   |
| [D5 前缀缓存](../modules/05-prefix-cache.md)                | [PagedAttention](https://arxiv.org/abs/2309.06180)     | [SGLang](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | [Preble](https://openreview.net/forum?id=meKEKDhdnx)                          | 块表、radix tree、多副本路由，对应 lab5 的 A、B、C                  |
| [D6 低显存训练](../modules/06-memory-efficient-training.md) | [ZeRO](https://arxiv.org/abs/1910.02054)               | [Checkmate](https://arxiv.org/abs/1910.02653)                                                                                  | [FlexLLM](https://www.usenix.org/conference/nsdi26/presentation/oliaro)（苗） | 字节账、重计算规划、推理与微调共服务                                |
| [D7 MoE 卸载](../modules/07-moe-offload.md)                 | [Mixtral-offloading](https://arxiv.org/abs/2312.17238) | [Pre-gated MoE](https://arxiv.org/abs/2308.12066)                                                                              | [Fiddler](https://openreview.net/forum?id=N5fVv6PZGz)                         | 缓存、预取、CPU 计算三种应对 miss 的办法                            |
| [D8 rollout 调度](../modules/08-rollout-scheduling.md)      | [HybridFlow](https://arxiv.org/abs/2409.19256)         | [AReaL](https://arxiv.org/abs/2505.24298)                                                                                      | [DARTS](https://arxiv.org/abs/2605.30859)（苗）                               | 同步编排、全异步、冗余采样，对应 lab8 的 C7、C5/C6、B4              |
| [D9 异构 serving](../modules/09-hetero-serving.md)          | [Helix](https://arxiv.org/abs/2406.01566)（苗）        | [Mélange](https://arxiv.org/abs/2404.14527)                                                                                    | [Coral](https://arxiv.org/abs/2605.04357)（苗）                               | 单模型放置、按价格选卡、多模型规划；Helix 到 Coral 是组内的一条主线 |

只看一个方向时，先读该行的 3 篇，再按讲义「前沿与 baseline 对照」表里标“实现”或“简化”的条目往下读。

## 读一篇系统论文的顺序

1. 摘要与问题图：优化的究竟是吞吐、尾延迟、成本还是质量？约束是什么？
2. 评测章节：硬件、模型、输入/输出长度、到达过程、质量指标、baseline 版本。先判断自己能复现到哪一层。
3. 核心机制：把状态、事件和不变量画出来，试着写 20 行伪代码。
4. 消融和局限：去掉某个组件，理论上哪项指标会变？哪张图真正检验了它？
5. 查源码：沿 benchmark 的调用栈找关键实现，固定 commit 并记下 symbol（入口见[代码地图](code-map.md)），不需要从仓库第一行读起。
6. 写一条反证：在什么负载或设备上，作者的机制可能没有收益？据此设计实验，不要把没验证过的猜测写成已发现的缺陷。

## 查新

每一讲的「前沿与 baseline 对照」节末尾都有一段**查新结论**，检索日期是 2026-10-18。它点名了每个研究问题最近的 3 篇以上工作，并写明还剩什么空白。正式选题时，从这段结论出发往后补，不要从零开始搜。

| 方向                                             | 查新结论的要点（详见讲义）                                                                                            |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| [D1](../modules/01-kv-cache.md)                  | GQA 与 QK-Norm 下 K 的量化轴是最便宜的空白；淘汰加量化的混合已经拥挤；融合 kernel 里 per-channel K 的代价缺少公开测量 |
| [D2](../modules/02-low-bit-kernels.md)           | 四个候选题排序为 (b) 形状相关分发 > (a) Ada 上 FP8 对 INT4 > (d) 组大小取舍 > (c) 权重与 KV 融合的持久 kernel         |
| [D3](../modules/03-dynamic-compilation.md)       | 三项：trace 驱动的捕获尺寸选择、漂移下的在线重编译准入、跨进程共享编译产物                                            |
| [D4](../modules/04-speculative-slo.md)           | 按 SLO 自适应选推测长度已经很拥挤；尚未被用上的是接受长度的删失结构                                                   |
| [D5](../modules/05-prefix-cache.md)              | 预填充代价非线性、副本异构时，复制热前缀还是排队；现有路由不计复制本身的代价                                          |
| [D6](../modules/06-memory-efficient-training.md) | LoRA 少存激活与优化器压缩都已饱和；单卡按长度选 checkpoint/offload 计划、8GB 卡上推理与微调共服务仍有空白             |
| [D7](../modules/07-moe-offload.md)               | (a) 跨层预取、(b) CPU/GPU 混合执行、(c) 混合精度专家缓存、(d) 统一 harness 下的替换策略，各有一段                     |
| [D8](../modules/08-rollout-scheduling.md)        | 四个候选按可行性排序：partial rollout 的重要性修正、长度预测调度、有界陈旧度流水线、去中心化工作窃取                  |
| [D9](../modules/09-hetero-serving.md)            | 六篇最近邻；空白在于代价模型误差 $\eta$ 对放置决策的影响，以及从笔记本迁移到数据中心 GPU                              |

补查的步骤：

1. **读结论，列出最近邻。** 把查新结论里点名的工作抄进对照表：最接近的工作、相同假设、不同假设、相同机制、你能拿出的新证据。
2. **向后查引用。** 在 Semantic Scholar 或 Google Scholar 里查这几篇最近邻的被引列表，只看 2026-10-18 之后出现的论文。
3. **查作者和会议。** 看最近邻作者的主页与 DBLP；翻目标会议（OSDI、SOSP、ASPLOS、MLSys、EuroSys、NSDI、ATC、SIGMOD）最新一期的录用列表；看苗老师的 [publications 页](https://hsword.github.io/publications/)。
4. **按“约束 + 目标 + 方法”检索 arXiv。** 例如 `KV mixed precision budget per head latency`，不要只搜自己想好的方法名。
5. **更新对照表并做决定。** 有新工作覆盖了你的空白，就按讲义「研究问题」里写的“继续或停止的证据”判断是换题、缩小范围，还是写成负结果。

**“把两个现有模块拼起来”只是实验的起点，本身不构成贡献。** 有价值的贡献可以是：可靠地刻画一个新瓶颈；给出既有假设的反例；提出带正确性与成本分析的新策略；或者在真实约束下系统地超过强基线。

## 资料汇总

[论文卡模板](paper-card.md) · [外部课程](courses-and-labs.md) · [源码定位](code-map.md) · [可视化](visualizations.md)
