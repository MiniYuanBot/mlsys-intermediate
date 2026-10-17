---
title: 论文、课程、代码与工具总表
tags: [references, moc]
checked: 2026-10-17
---

# 资料汇总

[课程索引](../index.md) · [课程与开放 lab](courses-and-labs.md) · [可视化](visualizations.md) · [固定源码入口](code-map.md)

核验日期为 **2026-10-17**。资料优先取自作者、大学、会议和官方文档，年份区分首次预印本与会议发表时间。这里的前沿资料包括仍可作为强基线的既有方法，以及 2026 年公开的工作，不保证覆盖每个方向的全部最新论文。正式选题前，需按 [查新流程](../guides/reading.md) 更新最接近的相关工作。

## 论文与项目

| 模块  | 论文/来源                                                                                                                                      | 代码                                                                 | 在本课程中的用途与边界                                             |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------ |
| D1    | [KIVI，ICML 2024](https://arxiv.org/abs/2402.02750)                                                                                            | [作者实现](https://github.com/jy-yuan/KIVI)                          | K/V 分组轴与残余窗口；本 lab 用 4bit 缩小机制                      |
| D1    | [PQCache，SIGMOD 2025](https://arxiv.org/abs/2407.12820)                                                                                       | [作者实现](https://github.com/HugoZHL/PQCache)                       | PQ 检索与标量量化的差异；读取完整数据路径                          |
| D1/D2 | [FlashInfer，MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)  | [作者实现](https://github.com/flashinfer-ai/flashinfer)              | 定制 attention/kernel 后端与真实布局                               |
| D2    | [AWQ，MLSys 2024](https://arxiv.org/abs/2306.00978)                                                                                            | [作者实现](https://github.com/mit-han-lab/llm-awq)                   | 数值缩放、packed layout 与硬件实现配合                             |
| D2    | [QServe，MLSys 2025](https://arxiv.org/abs/2405.04532)                                                                                         | [OmniServe](https://github.com/mit-han-lab/omniserve)                | 完整量化系统；旧 qserve URL 已重定向                               |
| D3    | [Mirage，OSDI 2025](https://arxiv.org/abs/2405.05751)                                                                                          | [Mirage](https://github.com/mirage-project/mirage)                   | 多层优化与合法变换；不以两算子融合冒充复现                         |
| D3    | [MPK，OSDI 2026](https://arxiv.org/abs/2512.22219)                                                                                             | [Mirage/MPK](https://github.com/mirage-project/mirage)               | 编译与运行时协作；源码已固定快照                                   |
| D3    | [Event Tensor，MLSys 2026](https://arxiv.org/abs/2604.13327)                                                                                   | 先从论文项目入口核对 artifact                                        | 动态依赖表达；本课程不假定其原实验能在 4060 跑通                   |
| D4    | [SpecInfer，ASPLOS 2024](https://arxiv.org/abs/2305.09781)                                                                                     | [FlexFlow Serve](https://github.com/flexflow/flexflow-serve)         | tree 验证的基础；本讲专注控制层                                    |
| D4    | [AdaServe，EuroSys 2026](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/AdaServe_EuroSys_26_abs.shtml) / [论文](https://arxiv.org/abs/2501.12162) | [artifact](https://github.com/zikun-li/AdaServe-Artifact-Evaluation) | 多 SLO 推测；公开 artifact 的资源超过本机                          |
| D5    | [SGLang，NeurIPS 2024](https://arxiv.org/abs/2312.07104)                                                                                       | [作者实现](https://github.com/sgl-project/sglang)                    | exact prefix sharing 与生命周期                                    |
| D5    | [CacheBlend，EuroSys 2025](https://www.microsoft.com/en-us/research/uploads/prod/2024/09/eurosys25-final999.pdf)                               | [LMCache](https://github.com/LMCache/LMCache)                        | 非前缀复用需要额外机制；工程版与论文版分清                         |
| D6    | [QLoRA，NeurIPS 2023](https://arxiv.org/abs/2305.14314)                                                                                        | [QLoRA](https://github.com/artidoro/qlora)                           | 参数高效微调与低比特基座；lab 未实现 NF4                           |
| D6    | [FlexLLM，NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/oliaro)                                                             | [artifact](https://github.com/FlexLLM/artifact)                      | 训练/推理共服务，不能只优化一端                                    |
| D7    | [MoE-Infinity（2024 初稿，2025 修订）](https://arxiv.org/abs/2401.14361)                                                                       | [作者实现](https://github.com/EfficientMoE/MoE-Infinity)             | 专家缓存、offload 与预取；此项不强填未由本次一手页面确认的会议标签 |
| D7    | [MegaBlocks，MLSys 2023](https://arxiv.org/abs/2211.15841)                                                                                     | [作者实现](https://github.com/stanford-futuredata/megablocks)        | 区分稀疏计算优化与数据移动优化                                     |
| D8    | [HybridFlow，EuroSys 2025](https://i.cs.hku.hk/~cwu/papers/gmsheng-eurosys25.pdf)                                                              | [verl](https://github.com/verl-project/verl)                         | 训练与 rollout 的系统分工；旧地址可重定向                          |
| D8    | [DARTS，ICML 2026](https://arxiv.org/abs/2605.30859)                                                                                           | [作者实现](https://github.com/PKU-DAIR/DARTS)                        | 主动改变采样分布的路线；需要质量验证                               |
| 总览  | [高效生成式 LLM 服务综述](https://arxiv.org/abs/2312.15234)                                                                                    | —                                                                    | 建立主题地图；不是将综述逐页当成论文清单                           |

会议归属优先依据官方 proceedings/作者 artifact，其余与作者 [publication list](https://hsword.github.io/publications/) 交叉核对。文献题目可能在预印本与最终版间变化，阅读时记录实际版本。

## 课程与作业入口

- [Stanford CS336](https://stanford-cs336.github.io/)；[Systems 作业](https://github.com/stanford-cs336/assignment2-systems)；[Alignment 作业](https://github.com/stanford-cs336/assignment5-alignment)。
- [CMU Deep Learning Systems](https://dlsyscourse.org/)；[CMU Machine Learning Systems](https://mlsyscourse.org/)。
- [MIT 6.5940 2024 官方课程](https://hanlab.mit.edu/courses/2024-fall-65940)；不误标为 2025 开课。
- [LCPU Infra Seminars](https://infra.seminars.lcpu.dev/)；[活动仓库](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars)。
- [原 foundations](https://github.com/MiniYuanBot/mlsys-foundations)：本课程的能力前置和样式参照，不要求重做。

## 官方工程文档与作者博客

| 资源                                                                                           | 用法                                      |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------- |
| [PyTorch compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | D3 graph break、dynamic shape、首次编译   |
| [PyTorch checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)                     | D6 重入、RNG 与梯度语义                   |
| [Checkpoint 技术博客](https://pytorch.org/blog/activation-checkpointing-techniques/)           | 对比不同重计算粒度与内存权衡              |
| [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)    | 实际 trace，而非只看 wall clock           |
| [CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html)                             | stream、同步、显存口径                    |
| [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)            | D2 bfe/shuffle 等语义，以编译目标支持为准 |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)                  | 时间线与 overlap；可用工具但不是开源项目  |
| [HF KV cache](https://huggingface.co/docs/transformers/en/kv_cache)                            | cache API、策略、支持条件                 |
| [SGLang 作者博客](https://lmsys.org/blog/2024-01-17-sglang/)                                   | prefix reuse 与 RadixAttention 图解       |
| [FlashInfer 作者博客](https://flashinfer.ai/)                                                  | kernel/layout 的工程解读                  |

## 核验与更新

[upstream-snapshots.json](upstream-snapshots.json) 记录 13 个仓库的 commit 与相关文件路径。需要检查外部链接时，在 `labs/` 下运行 `python tools/audit_external_links.py`，结果写入本地 `results/external-links.json`。HTTP 200 仅表示页面可访问，不能证明方法已复现、许可已完整审查或硬件兼容。遇到 403/429 或网络失败时，优先尝试作者 GitHub 或论文入口；限流不代表论文不存在。

重跑某篇论文之前固定模型、数据、代码 commit、依赖和硬件。对外引用使用实际论文版本，实验报告中应说明本课程简化实现与原论文的差异。新论文如果只有标题或新闻，尚未公开全文或代码，就不列为必做实验。
