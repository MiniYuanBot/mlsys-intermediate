---
title: 论文、课程、代码与工具总表
tags: [references, moc]
checked: 2026-10-18
---

# 资料汇总

[课程索引](../index.md) · [课程与开放 lab](courses-and-labs.md) · [可视化](visualizations.md) · [固定源码入口](code-map.md) · [阅读方法](reading.md)

核验日期 **2026-10-18**。下表是九个方向的种子论文，每个方向 12–15 篇，均取自各讲「前沿与 baseline 对照」节，题目、会议和链接在 Phase 2 由作者页、会议 proceedings 或 arXiv 页面核对过。会议栏写正式录用的会议；只有预印本的写 arXiv 和年份。讲义里的对照表更长，含更多 2026 年预印本，选题时以讲义为准，本页只放第一轮必须知道的工作。正式选题前按 [查新流程](reading.md#查新) 再查一遍。

**苗组标记。** 「苗」表示作者列表里有苗旭鹏老师，依据是他的 [publications 页](https://hsword.github.io/publications/) 或 arXiv 作者列表。括号里写清合作关系：Mirage、MPK、Event Tensor、FlexLLM 是他在 Purdue 期间与 CMU Zhihao Jia 组的合作；NetMoE、FlexMoE 是训练侧工作；他没有量化 kernel 或前缀缓存方面的论文。没有标记的论文不代表与他无关，只代表作者列表里没有他。

## 论文与项目

### D1 KV 缓存（[讲义](../modules/01-kv-cache.md) · [lab1](../../labs/lab1/README.md)）

| 模块  | 论文/来源                                                                                                                                     | 代码                                                     | 在本课程中的用途与边界                                                       |
| ----- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| D1    | [KIVI，ICML 2024](https://arxiv.org/abs/2402.02750)                                                                                           | [作者实现](https://github.com/jy-yuan/KIVI)              | K 按通道、V 按 token 分组，外加残余窗口；lab1 A1/A2、B2–B5 实现其 4 bit 版本 |
| D1    | [KVQuant，NeurIPS 2024](https://arxiv.org/abs/2401.18079)                                                                                     | [作者实现](https://github.com/SqueezeAILab/KVQuant)      | pre-RoPE K 量化与非均匀码本；Qwen3 的 cache 已在 RoPE 之后，lab 未覆盖       |
| D1    | [PQCache，SIGMOD 2025](https://arxiv.org/abs/2407.12820)                                                                                      | [作者实现](https://github.com/HugoZHL/PQCache)           | **苗**（第 5 作者，Bin Cui 组）。PQ 检索取回原始 KV；lab1 C1–C4 实现数值部分 |
| D1    | [StreamingLLM，ICLR 2024](https://arxiv.org/abs/2309.17453)                                                                                   | [作者实现](https://github.com/mit-han-lab/streaming-llm) | attention sink 加最近窗口；lab1 B1 的 sink/recent 规则                       |
| D1    | [H2O，NeurIPS 2023](https://arxiv.org/abs/2306.14048)                                                                                         | [作者实现](https://github.com/FMInference/H2O)           | 按累计注意力分数淘汰；B1 只接收外部分数，是简化版                            |
| D1    | [SnapKV，NeurIPS 2024](https://arxiv.org/abs/2404.14469)                                                                                      | [作者实现](https://github.com/FasterDecoding/SnapKV)     | prefill 末尾窗口投票选 KV；未覆盖，作淘汰类强基线                            |
| D1    | [Ada-KV，NeurIPS 2025](https://arxiv.org/abs/2407.11550)                                                                                      | [作者实现](https://github.com/FFY0/AdaKV)                | 按 head 分配淘汰预算；未覆盖                                                 |
| D1    | [Quest，ICML 2024](https://arxiv.org/abs/2406.10774)                                                                                          | [作者实现](https://github.com/mit-han-lab/Quest)         | 按 page 的查询感知稀疏读取；与 C 阶段的 PQ 检索是两条路线                    |
| D1    | [ShadowKV，ICML 2025](https://arxiv.org/abs/2410.21465)                                                                                       | [作者实现](https://github.com/ByteDance-Seed/ShadowKV)   | 低秩 K 加 CPU 上的 V；未覆盖，研究题 (c) 的近邻                              |
| D1    | [TurboQuant，ICLR 2026](https://arxiv.org/abs/2504.19874)                                                                                     | 无官方代码（LMDeploy `quant_policy=42`）                 | 随机旋转后用固定码本，与数据无关，绕开了分组轴问题；未覆盖                   |
| D1    | [BitDecoding，arXiv 2025](https://arxiv.org/abs/2503.18773)                                                                                   | [作者实现](https://github.com/OpenBitSys/BitDecoding)    | 用 Tensor Core 做低比特 KV decode；R 阶段只用 CUDA core，是它的简化          |
| D1    | [RDKV，arXiv 2026](https://arxiv.org/abs/2605.08317)                                                                                          | 无公开代码                                               | 率失真反向注水，在淘汰与全精度之间分配比特；研究题 (b) 的拥挤程度参照        |
| D1    | [Multi-Segment Attention（AsymCache），arXiv 2026](https://arxiv.org/abs/2606.02964)                                                          | 代码待核实                                               | **苗**（合作者）。KV 分段管理；组内与 D1 最接近的近期工作                    |
| D1/D3 | [FlashInfer，MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html) | [作者实现](https://github.com/flashinfer-ai/flashinfer)  | 可定制 attention kernel 与 JIT；R 阶段用 SDPA 或自写 kernel，不依赖它        |

### D2 低比特权重 kernel（[讲义](../modules/02-low-bit-kernels.md) · [lab2](../../labs/lab2/README.md)）

| 模块 | 论文/来源                                                                        | 代码                                                      | 在本课程中的用途与边界                                               |
| ---- | -------------------------------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------- |
| D2   | [GPTQ，ICLR 2023](https://arxiv.org/abs/2210.17323)                              | [作者实现](https://github.com/IST-DASLab/gptq)            | 二阶信息的逐列量化；未覆盖，读它的误差补偿思路                       |
| D2   | [AWQ，MLSys 2024](https://arxiv.org/abs/2306.00978)                              | [作者实现](https://github.com/mit-han-lab/llm-awq)        | 激活感知的通道缩放；lab2 C1–C5 是单层缩小版                          |
| D2   | [SmoothQuant，ICML 2023](https://arxiv.org/abs/2211.10438)                       | [作者实现](https://github.com/mit-han-lab/smoothquant)    | 把激活的离群值迁移到权重；未覆盖，W8A8 背景                          |
| D2   | [QServe，MLSys 2025](https://arxiv.org/abs/2405.04532)                           | [OmniServe](https://github.com/mit-han-lab/omniserve)     | W4A8KV4 全系统；lab2 B 只做 W4A32，旧 qserve 地址已重定向            |
| D2   | [Marlin，PPoPP 2025](https://arxiv.org/abs/2408.11743)                           | [作者实现](https://github.com/IST-DASLab/marlin)          | INT4 Tensor Core GEMM 与交错布局；B 是 SIMT 简化版，挑战 2/3 对照    |
| D2   | [QuaRot，NeurIPS 2024](https://arxiv.org/abs/2404.00456)                         | [作者实现](https://github.com/spcl/QuaRot)                | Hadamard 旋转消除离群值；未覆盖                                      |
| D2   | [QuIP#，ICML 2024](https://arxiv.org/abs/2402.04396)                             | [作者实现](https://github.com/Cornell-RelaxML/quip-sharp) | 格码本向量量化；未覆盖，代表 2 bit 以下的路线                        |
| D2   | [FLUTE，EMNLP 2024 Findings](https://arxiv.org/abs/2407.10960)                   | [作者实现](https://github.com/HanGuo97/flute)             | 查表式 LUT GEMM；未覆盖                                              |
| D2   | [T-MAC，EuroSys 2025](https://arxiv.org/abs/2407.00088)                          | [作者实现](https://github.com/microsoft/T-MAC)            | CPU 上的查表低比特 GEMM；未覆盖                                      |
| D2   | [Ladder/BitBLAS，OSDI 2024](https://github.com/microsoft/BitBLAS)                | [BitBLAS](https://github.com/microsoft/BitBLAS)           | 混合精度 GEMM 的类型变换与调度；未覆盖，论文链接待核实               |
| D2   | [FP8 Formats for Deep Learning，arXiv 2022](https://arxiv.org/abs/2209.05433)    | —                                                         | E4M3/E5M2 的定义；R 阶段 FP8 编码器的规格来源                        |
| D2   | ["Give Me BF16 or Give Me Death"?，arXiv 2024](https://arxiv.org/abs/2411.02355) | 无公开代码                                                | 不同格式在精度与速度上的大规模对比；R 基准是它在 kernel 级的缩影     |
| D2   | [MicroMix，ICLR 2026](https://arxiv.org/abs/2508.02343)                          | [作者实现](https://github.com/lwy2020/MicroMix)           | MX 格式混合精度，面向 Blackwell；4060 无 FP4，未覆盖                 |
| D2   | [Quantized Side Tuning，ACL 2024](https://arxiv.org/abs/2401.07159)              | [作者实现](https://github.com/YouAreSpecialToMe/QST)      | **苗**（合作者）。组内唯一的量化方法论文，属于微调侧，与 kernel 无关 |

### D3 动态形状与编译（[讲义](../modules/03-dynamic-compilation.md) · [lab3](../../labs/lab3/README.md)）

| 模块 | 论文/来源                                                                                                                        | 代码                                                                                       | 在本课程中的用途与边界                                                                                  |
| ---- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| D3   | [PyTorch 2，ASPLOS 2024](https://doi.org/10.1145/3620665.3640366)                                                                | [PyTorch](https://github.com/pytorch/pytorch)                                              | Dynamo guard 与 Inductor；lab3 C1–C4 直接调用                                                           |
| D3   | [Triton，MAPL 2019](https://doi.org/10.1145/3315508.3329973)                                                                     | [Triton](https://github.com/triton-lang/triton)                                            | 块级编程模型；lab 只读 Inductor 生成的 Triton，不手写                                                   |
| D3   | [Ansor，OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/zheng)                                                  | [TVM auto_scheduler](https://github.com/apache/tvm/tree/v0.14.0/python/tvm/auto_scheduler) | 每个形状单独搜索的代价参照；未覆盖                                                                      |
| D3   | [Nimble，MLSys 2021](https://arxiv.org/abs/2006.03031)                                                                           | 代码待核实                                                                                 | 动态模型的 VM 与形状函数；未覆盖                                                                        |
| D3   | [DietCode，MLSys 2022](https://proceedings.mlsys.org/paper_files/paper/2022/hash/f89b79c9a28d4cae22ef9e557d9fa191-Abstract.html) | 无公开代码                                                                                 | 动态形状的 micro-kernel 共享；B1 只选上界，不生成 kernel                                                |
| D3   | [Hidet，ASPLOS 2023](https://arxiv.org/abs/2210.09603)                                                                           | [作者实现](https://github.com/hidet-org/hidet)                                             | 任务映射式调度；未覆盖                                                                                  |
| D3   | [BladeDISC，PACMMOD 2023](https://doi.org/10.1145/3617327)                                                                       | [作者实现](https://github.com/alibaba/BladeDISC)                                           | 动态形状编译器；未覆盖                                                                                  |
| D3   | [Relax，ASPLOS 2025](https://arxiv.org/abs/2311.02103)                                                                           | [TVM](https://github.com/apache/tvm)                                                       | 符号形状贯穿整个 IR；lab 只在 batch 维分桶                                                              |
| D3   | [Mirage，OSDI 2025](https://arxiv.org/abs/2405.05751)                                                                            | [作者实现](https://github.com/mirage-project/mirage)                                       | **苗**（时在 Purdue，与 CMU 合作）。多层超优化与等价验证；lab 只对照“搜索加验证”的流程                  |
| D3   | [MPK，OSDI 2026（据苗老师主页）](https://arxiv.org/abs/2512.22219)                                                               | [Mirage/MPK](https://github.com/mirage-project/mirage)                                     | **苗**（时在 Purdue，与 CMU 合作）。整段 decode 编成一个 persistent megakernel；正式 proceedings 待核实 |
| D3   | [Event Tensor，MLSys 2026](https://arxiv.org/abs/2604.13327)                                                                     | 无公开代码                                                                                 | **苗**（时在 Purdue，与 CMU 合作）。动态形状 megakernel；不假定能在 4060 上复现                         |
| D3   | [Medusa（编译期），ASPLOS 2025](https://doi.org/10.1145/3669940.3707285)                                                         | [代码镜像](https://github.com/MachineLearningSystem/25ASPLOS-Medusa)                       | 缩短 LLM serving 冷启动；对应查新 (c)，注意与 D4 的 Medusa 不是同一篇                                   |
| D3   | [Foundry，arXiv 2026](https://arxiv.org/abs/2604.06664)                                                                          | 无公开代码                                                                                 | 编译产物的复用与共享；对应查新 (c)                                                                      |
| D3   | [vLLM CUDA graph 设计文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/)                                                   | [vLLM](https://github.com/vllm-project/vllm)                                               | piecewise capture 与捕获尺寸；R 阶段的 `vllm_default` 基线（commit `371745eb`）                         |

### D4 推测解码与 SLO（[讲义](../modules/04-speculative-slo.md) · [lab4](../../labs/lab4/README.md)）

| 模块 | 论文/来源                                                                                                                                                                         | 代码                                                                 | 在本课程中的用途与边界                                                                             |
| ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| D4   | [Speculative Decoding（Leviathan 等），ICML 2023](https://arxiv.org/abs/2211.17192)                                                                                               | 无公开代码                                                           | 拒绝采样保持目标分布；R1 `coupled_step`，A 阶段把 i.i.d. 推广到逐位置 $a_j$                        |
| D4   | [SpecInfer，ASPLOS 2024](https://arxiv.org/abs/2305.09781)                                                                                                                        | [FlexFlow Serve](https://github.com/flexflow/flexflow-serve)         | **苗**（第一作者）。token tree 验证；lab 只做链                                                    |
| D4   | [Medusa，ICML 2024](https://proceedings.mlr.press/v235/cai24b.html)                                                                                                               | [作者实现](https://github.com/FasterDecoding/Medusa)                 | 多头草稿，草稿成本接近 0；作 $c \to 0$ 的对照                                                      |
| D4   | [EAGLE-2，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.422/)                                                                                                              | [作者实现](https://github.com/SafeAILab/EAGLE)                       | 用草稿置信度动态建树；R 阶段记录 top-1 概率，控制器未使用                                          |
| D4   | [Sequoia，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea1f5f0878d43ff4fb8bf64ef4a2326c-Abstract-Conference.html)                                     | [作者实现](https://github.com/Infini-AI-Lab/Sequoia)                 | 硬件感知的树形 DP；B6 的 DP 在请求之间分预算，不在树上                                             |
| D4   | [SpecDec++，COLM 2025](https://arxiv.org/abs/2405.19715)                                                                                                                          | [作者实现](https://github.com/Kaffaljidhmah2/SpecDec_pp)             | 每个位置决定是否停止草稿；lab 每轮开始时一次决定 $k$                                               |
| D4   | [MagicDec，ICLR 2025](https://arxiv.org/abs/2408.11049)                                                                                                                           | [作者实现](https://github.com/Infini-AI-Lab/MagicDec)                | 大 batch 长上下文下推测仍有收益；lab 只有 batch 1                                                  |
| D4   | [DistServe，OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)                                                                                        | [作者实现](https://github.com/LLMServe/DistServe)                    | goodput 定义与 P/D 分离；C4 用截止时间达标率和 p95 TPOT                                            |
| D4   | [BanditSpec，ICML 2025](https://arxiv.org/abs/2505.15141)                                                                                                                         | [作者实现](https://github.com/sail-sg/BanditSpec)                    | 用 bandit 选推测配置；R 阶段的 `ArmUcb` 是简化版                                                   |
| D4   | [AdaSpec，SoCC 2025](https://arxiv.org/abs/2503.05096)                                                                                                                            | [作者实现](https://github.com/cerebellumking/AdaSpec)                | batch 级自适应 $k$；未覆盖                                                                         |
| D4   | [TETRIS，ACL 2025](https://aclanthology.org/2025.acl-long.1598/)                                                                                                                  | 无公开代码                                                           | batch 内按边际收益分配草稿 token；B5 贪心、B6 精确 DP                                              |
| D4   | [AdaServe，EuroSys 2026](https://arxiv.org/abs/2501.12162)                                                                                                                        | [artifact](https://github.com/zikun-li/AdaServe-Artifact-Evaluation) | **苗**（合作者）。多 SLO 的推测服务；B5/B6 的共享预算与 C 的 SLO 指标，无树；artifact 资源超过本机 |
| D4   | [HedgeSpec / Not-a-Bandit，ICLR 2026](https://arxiv.org/abs/2510.20064)                                                                                                           | 无公开代码                                                           | 利用接受长度的删失结构做全信息评估；R4 是它在长度维的删失版本                                      |
| D4   | [Speculative Decoding: Performance or Illusion?，MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/554e056fe2b6d9fd27ffcd3367ae1267-Abstract-Conference.html) | 无公开代码                                                           | 真实引擎上的加速比往往低于论文；对应 R 阶段“实测与回放”的对比                                      |

### D5 前缀缓存（[讲义](../modules/05-prefix-cache.md) · [lab5](../../labs/lab5/README.md)）

| 模块 | 论文/来源                                                                                                                                                     | 代码                                                 | 在本课程中的用途与边界                                                                           |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| D5   | [PagedAttention / vLLM，SOSP 2023](https://arxiv.org/abs/2309.06180)                                                                                          | [vLLM](https://github.com/vllm-project/vllm)         | 分页 KV 与引用计数；B 阶段的引用计数与租约                                                       |
| D5   | [SGLang / RadixAttention，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | [作者实现](https://github.com/sgl-project/sglang)    | radix tree 上的精确前缀共享；A4/A5 的叶子语义、C2 的 SGLang 规则                                 |
| D5   | [Prompt Cache，MLSys 2024](https://arxiv.org/abs/2311.04934)                                                                                                  | [作者实现](https://github.com/yale-sys/prompt-cache) | 模块化的位置无关复用；未覆盖，`position_offset` 只是替身                                         |
| D5   | [CachedAttention，ATC 2024](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost)                                                                | 无公开代码                                           | 多轮对话 KV 存到分层存储；未覆盖（讲义 §8）                                                      |
| D5   | [CacheGen，SIGCOMM 2024](https://arxiv.org/abs/2310.07240)                                                                                                    | [作者实现](https://github.com/UChi-JCL/CacheGen)     | KV 编码后经网络传输；未覆盖                                                                      |
| D5   | [Mooncake，FAST 2025](https://www.usenix.org/conference/fast25/presentation/qin)                                                                              | [作者实现](https://github.com/kvcache-ai/Mooncake)   | 以 KV 为中心的 P/D 分离；R2 读取其 trace 格式（未下载原始 trace，许可证待核实）                  |
| D5   | [Preble，ICLR 2025](https://openreview.net/forum?id=meKEKDhdnx)                                                                                               | 代码仓库待核实                                       | 前缀感知的多副本调度；C3 的 Hybrid 只有阈值，没有淘汰代价 $M$                                    |
| D5   | [CacheBlend，EuroSys 2025](https://arxiv.org/abs/2405.16444)                                                                                                  | 在 [LMCache](https://github.com/LMCache/LMCache) 中  | 非前缀复用需要选择性重算；未覆盖，工程版与论文版要分开看                                         |
| D5   | [Marconi，MLSys 2025](https://arxiv.org/abs/2411.19379)                                                                                                       | [作者实现](https://github.com/ruipeterpan/marconi)   | 混合架构下的准入与淘汰；研究题 (b) 的近邻                                                        |
| D5   | [EPIC，ICML 2025](https://proceedings.mlr.press/v267/hu25j.html)                                                                                              | 代码待核实                                           | 位置无关缓存，重算每块开头几个 token；未覆盖                                                     |
| D5   | [T-LRU，NeurIPS 2025](https://arxiv.org/abs/2510.15152)                                                                                                       | 代码待核实                                           | 考虑代价的 LRU 变体；讲义 §5 说明其线性代价假设在长上下文不成立                                  |
| D5   | [DualMap，arXiv 2026](https://arxiv.org/abs/2602.06502)                                                                                                       | [作者实现](https://github.com/ASISys/DualMap)        | 双候选哈希做缓存亲和路由；C3 Hybrid 只有一个亲和候选                                             |
| D5   | [MapFlow（Attention Map Reuse），SIGMOD 2027](https://arxiv.org/abs/2610.14688)                                                                               | 代码待核实                                           | **苗**（末位作者）。跨模型复用 attention map 而非 KV；组内没有直接的前缀缓存论文，这是最近的一篇 |

### D6 低显存训练（[讲义](../modules/06-memory-efficient-training.md) · [lab6](../../labs/lab6/README.md)）

| 模块 | 论文/来源                                                                                                | 代码                                                  | 在本课程中的用途与边界                                                                                           |
| ---- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| D6   | [ZeRO，SC 2020](https://arxiv.org/abs/1910.02054)                                                        | [DeepSpeed](https://github.com/microsoft/DeepSpeed)   | 按参数、梯度、优化器状态分片；单卡未覆盖，讲义 §2 用它推导字节账                                                 |
| D6   | [ZeRO-Offload，ATC 2021](https://arxiv.org/abs/2101.06840)                                               | [DeepSpeed](https://github.com/microsoft/DeepSpeed)   | 优化器状态与计算卸到 CPU；C 阶段只卸载激活                                                                       |
| D6   | [Checkmate，MLSys 2020](https://arxiv.org/abs/1910.02653)                                                | [作者实现](https://github.com/parasj/checkmate)       | 重计算的 ILP；B2/B4 是链上的背包                                                                                 |
| D6   | [Reducing Activation Recomputation（selective recompute），MLSys 2023](https://arxiv.org/abs/2205.05198) | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)  | 只重算 attention 的大张量；未覆盖，讲义 §3.3 的公式来源                                                          |
| D6   | [LoRA，ICLR 2022](https://arxiv.org/abs/2106.09685)                                                      | [作者实现](https://github.com/microsoft/LoRA)         | 低秩适配；A 阶段实现                                                                                             |
| D6   | [QLoRA，NeurIPS 2023](https://arxiv.org/abs/2305.14314)                                                  | [作者实现](https://github.com/artidoro/qlora)         | NF4 基座加 LoRA；lab 基座为 FP32，未实现 NF4                                                                     |
| D6   | [LoRA-FA，arXiv 2023](https://arxiv.org/abs/2308.03303)                                                  | 无公开代码                                            | 冻结 $A$ 以少存激活；R1 实现                                                                                     |
| D6   | [LoRAFusion，EuroSys 2026](https://arxiv.org/abs/2510.00206)                                             | [作者实现](https://github.com/CentML/lorafusion)      | 融合 LoRA kernel 与多任务调度；可选挑战的模板                                                                    |
| D6   | [Cut Cross-Entropy，ICLR 2025](https://arxiv.org/abs/2411.09009)                                         | [作者实现](https://github.com/apple/ml-cross-entropy) | 不物化 logits 的交叉熵；未覆盖，说明大词表的激活峰值                                                             |
| D6   | [MEMO，SIGMOD 2025](https://arxiv.org/abs/2407.12117)                                                    | 无公开代码                                            | 长序列训练的细粒度激活管理；未覆盖                                                                               |
| D6   | [FlexLLM，NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/oliaro)                       | [artifact](https://github.com/FlexLLM/artifact)       | **苗**（时在 Purdue，与 CMU 合作）。推理与 PEFT 微调共服务；R 阶段是其“少存激活”一半的单卡练习，图剪枝细节待核实 |
| D6   | [Hetu v2，OSDI 2026](https://arxiv.org/abs/2504.20490)                                                   | [Hetu](https://github.com/PKU-DAIR/Hetu)              | **苗**（合作者）。层次化、异构的 SPMD 标注；多卡，未覆盖                                                         |

### D7 MoE 卸载（[讲义](../modules/07-moe-offload.md) · [lab7](../../labs/lab7/README.md)）

| 模块 | 论文/来源                                                                                                                                    | 代码                                                            | 在本课程中的用途与边界                                                                                  |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| D7   | [Switch Transformers，JMLR 2022](https://jmlr.org/papers/v23/21-0998.html)                                                                   | [Mesh TensorFlow](https://github.com/tensorflow/mesh)           | top-1 路由；A1 的 SoftmaxNoRenorm                                                                       |
| D7   | [GShard，ICLR 2021](https://openreview.net/forum?id=qrwe7XHTmYb)                                                                             | 无官方代码                                                      | top-2 门控与容量；A1 的 TopkSoftmax                                                                     |
| D7   | [DeepSeek-V3 技术报告，2024](https://arxiv.org/abs/2412.19437)                                                                               | [EPLB](https://github.com/deepseek-ai/EPLB)                     | sigmoid 打分、偏置只参与选择；A1 的 SigmoidRenorm                                                       |
| D7   | [MegaBlocks，MLSys 2023](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html) | [作者实现](https://github.com/databricks/megablocks)            | 块稀疏的专家计算布局，不涉及卸载；A2/A5 只做布局                                                        |
| D7   | [Mixtral-offloading，arXiv 2023](https://arxiv.org/abs/2312.17238)                                                                           | [作者实现](https://github.com/dvmazur/mixtral-offloading)       | LRU 专家缓存加量化；B1 的 LRU                                                                           |
| D7   | [Pre-gated MoE，ISCA 2024](https://arxiv.org/abs/2308.12066)                                                                                 | [作者实现](https://github.com/ranggihwang/Pregated_MoE)         | 提前一层预测专家以便预取；需改模型，未覆盖                                                              |
| D7   | [Fiddler，ICLR 2025](https://openreview.net/forum?id=N5fVv6PZGz)                                                                             | [作者实现](https://github.com/efeslab/fiddler)                  | miss 时在 CPU 上算专家而不搬运权重；R5 的判定式                                                         |
| D7   | [MoE-Lightning，ASPLOS 2025](https://dl.acm.org/doi/10.1145/3669940.3707267)                                                                 | [artifact](https://github.com/caoshiyi/artifacts/tree/asplos25) | 批处理吞吐导向的 CPU/GPU 流水线；lab 是 batch 1 延迟，未覆盖                                            |
| D7   | [KTransformers，SOSP 2025](https://dl.acm.org/doi/10.1145/3731569.3764843)                                                                   | [作者实现](https://github.com/kvcache-ai/ktransformers)         | CPU/GPU 混合推理；对照 R5 的 `cpu_speedup` 扫描                                                         |
| D7   | [Cache Management for MoE LLMs，Euro-Par 2025](https://arxiv.org/abs/2509.02408)                                                             | 无公开代码                                                      | 专家缓存替换策略的系统比较；对照 B 阶段                                                                 |
| D7   | [MoE-Infinity，arXiv 2024（v3 2025）](https://arxiv.org/abs/2401.14361)                                                                      | [作者实现](https://github.com/EfficientMoE/MoE-Infinity)        | 基于激活追踪的预取与缓存；工程版与论文版有差异                                                          |
| D7   | [NetMoE，ICLR 2025](https://openreview.net/forum?id=1qP3lsatCR)                                                                              | 代码待核实                                                      | **苗**（训练侧）。按路由局部性移动样本以减少 All-to-All；与 D7 共享“局部性决定放置”的思路，不是推理卸载 |
| D7   | [FlexMoE，SIGMOD 2023](https://dl.acm.org/doi/10.1145/3588964)                                                                               | —                                                               | **苗**（训练侧）。运行时按负载重排专家到设备；同上，不是推理卸载                                        |

### D8 RL rollout 调度（[讲义](../modules/08-rollout-scheduling.md) · [lab8](../../labs/lab8/README.md)）

| 模块 | 论文/来源                                                           | 代码                                                      | 在本课程中的用途与边界                                                                |
| ---- | ------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| D8   | [HybridFlow / verl，EuroSys 2025](https://arxiv.org/abs/2409.19256) | [verl](https://github.com/verl-project/verl)              | 训练与 rollout 的编排；C7 的单进程闭环                                                |
| D8   | [AReaL，NeurIPS 2025](https://arxiv.org/abs/2505.24298)             | [作者实现](https://github.com/inclusionAI/AReaL)          | 全异步 RL 与解耦 PPO 目标；C5、C6 实现                                                |
| D8   | [DAPO，NeurIPS 2025](https://arxiv.org/abs/2503.14476)              | [作者实现](https://github.com/BytedTsinghua-SIA/DAPO)     | 动态采样与 token 级损失；讲义 §7 的选样例子                                           |
| D8   | [Dr. GRPO，COLM 2025](https://arxiv.org/abs/2503.20783)             | [作者实现](https://github.com/sail-sg/understand-r1-zero) | 去掉长度与 std 归一化的偏差；A4 提供两种 std                                          |
| D8   | [DARTS，ICML 2026](https://arxiv.org/abs/2605.30859)                | [作者实现](https://github.com/PKU-DAIR/DARTS)             | **苗**（合作者，Bin Cui 组）。冗余采样、先完成者入选；B4 的选择规则，R 量化其长度偏差 |
| D8   | [QLPO，COLM 2026](https://arxiv.org/abs/2607.21793)                 | 无公开代码                                                | **苗**（合作者）。按长度与奖励象限加权采样；讲义 §7 的“有意偏差”例子                  |
| D8   | [RollPacker，NSDI 2026](https://arxiv.org/abs/2509.21009)           | [作者实现](https://github.com/Farrrrland/RollPacker)      | 长尾 rollout 打包；R 的 `keep_first_n` 与讲义讨论                                     |
| D8   | [Seer，OSDI 2026](https://arxiv.org/abs/2511.14617)                 | 无公开代码                                                | 按 prompt 预测长度做调度；R2/R3 的简化对象                                            |
| D8   | [Laminar，EuroSys 2026](https://arxiv.org/abs/2510.12633)           | 无公开代码                                                | 轨迹级异步与权重同步；C7 的简化对象                                                   |
| D8   | [StaleFlow，SIGMOD 2027](https://arxiv.org/abs/2601.12784)          | [作者实现](https://github.com/psrl-project/psrl)          | Bin Cui / Fangcheng Fu 组，作者中**没有**苗老师。陈旧度准入；C1/C2/C5                 |
| D8   | [APRIL，arXiv 2025](https://arxiv.org/abs/2509.18521)               | [作者实现](https://github.com/RLsys-Foundation/APRIL)     | 主动 partial rollout；B1、C6                                                          |
| D8   | [PipelineRL，TMLR 2026](https://arxiv.org/abs/2509.19128)           | [作者实现](https://github.com/ServiceNow/PipelineRL)      | 生成中途更新权重；讲义 §6.3 的对照，未覆盖                                            |

### D9 异构与多模型 serving（[讲义](../modules/09-hetero-serving.md) · [lab9](../../labs/lab9/README.md)）

| 模块 | 论文/来源                                                                                   | 代码                                                        | 在本课程中的用途与边界                                                                    |
| ---- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| D9   | [Helix，ASPLOS 2025](https://arxiv.org/abs/2406.01566)                                      | [作者实现](https://github.com/Thesys-lab/Helix-ASPLOS25)    | **苗**（与 CMU Vinayak/Jia 组合作）。max-flow 层放置；B1–B7 用 DP 加局部搜索代替 MILP     |
| D9   | [HexGen，ICML 2024](https://proceedings.mlr.press/v235/jiang24f.html)                       | [作者实现](https://github.com/Relaxed-System-Lab/HexGen)    | 异构 GPU 上的非对称并行；lab 只做 PP                                                      |
| D9   | [Mélange，arXiv 2024](https://arxiv.org/abs/2404.14527)                                     | [作者实现](https://github.com/tyler-griggs/melange-release) | 按价格选 GPU 组合的 ILP；C2 的覆盖问题，正式会议待核实                                    |
| D9   | [MuxServe，ICML 2024](https://proceedings.mlr.press/v235/duan24a.html)                      | [作者实现](https://github.com/hao-ai-lab/MuxServe)          | 多 LLM 空间与时间复用；C1 只做时间份额                                                    |
| D9   | [AlpaServe，OSDI 2023](https://arxiv.org/abs/2302.11665)                                    | [作者实现](https://github.com/alpa-projects/mms)            | 突发流量下的统计复用；C3 的 bursty trace                                                  |
| D9   | [SpotServe，ASPLOS 2024](https://arxiv.org/abs/2311.15566)                                  | [作者实现](https://github.com/Hsword/SpotServe)             | **苗**（第一作者）。抢占式实例上的动态重配置；lab 是静态规划，未覆盖                      |
| D9   | [DistServe，OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)         | [作者实现](https://github.com/LLMServe/DistServe)           | P/D 分离与 goodput；讲义 §10 的建模讨论                                                   |
| D9   | [Splitwise，ISCA 2024](https://arxiv.org/abs/2311.18677)                                    | [模拟器](https://github.com/Mutinifni/splitwise-sim)        | 不同阶段用不同卡型；其 Azure 2023 trace 可用于 C3                                         |
| D9   | [Demystifying Cost-Efficiency，ICML 2025](https://proceedings.mlr.press/v267/jiang25c.html) | 无公开代码                                                  | **苗**（合作者）。异构 GPU 的成本效率规划；C2/C4 的预算与基线结构                         |
| D9   | [Coral，arXiv 2026](https://arxiv.org/abs/2605.04357)                                       | 无公开代码                                                  | **苗**（合作者，Helix 团队）。多模型两阶段 ILP；C2 的模板加 B&B 是其缩小版，会议待核实    |
| D9   | [OpWeave，arXiv 2026](https://arxiv.org/abs/2609.14237)                                     | 无公开代码                                                  | **苗**（合作者，Helix 团队）。算子级拆分加解析代价模型；lab 停在层粒度，会议待核实        |
| D9   | [Janus，SOSP 2026](https://yangtonghome.github.io/uploads/Janus_SOSP2026.pdf)               | [作者实现](https://github.com/Janus2026/Janus)              | **苗**（合作者，PKU 与京东）。多 LLM serving；其预测误差量级（中位约 7.8%）是研究题的参照 |
| D9   | [KernelSight-LM，arXiv 2026](https://arxiv.org/abs/2606.28565)                              | 无公开代码                                                  | roofline 加学习效率项的跨 GPU 预测；A3/R5 是其极简版                                      |
| D9   | [ShuntServe，arXiv 2026](https://arxiv.org/abs/2606.18600)                                  | 无公开代码                                                  | 方法与 lab A2 加 B4/B5 几乎相同；选题时必须对比                                           |

### 总览

| 模块 | 论文/来源                                                   | 代码 | 在本课程中的用途与边界                   |
| ---- | ----------------------------------------------------------- | ---- | ---------------------------------------- |
| 总览 | [高效生成式 LLM 服务综述](https://arxiv.org/abs/2312.15234) | —    | 建立主题地图，不要把综述当成逐篇必读清单 |

会议归属优先依据官方 proceedings 和作者 artifact，其余与苗老师的 [publication list](https://hsword.github.io/publications/) 交叉核对。预印本与最终版的题目可能不同，阅读时记下实际读的版本。

## 课程与作业入口

外部课程、开放 lab 与本课程模块的对应关系见 [courses-and-labs.md](courses-and-labs.md)。常用入口：

- [MIT 6.5940 Fall 2026](https://hanlab.mit.edu/courses/2026-fall-65940)：2025 年秋季没有开课（`2025-fall-65940` 页面返回 404），旧的公开 Colab 在 [2024 课程页](https://hanlab.mit.edu/courses/2024-fall-65940)。
- [CMU 15-779 LLM Edition](https://www.cs.cmu.edu/~zhihaoj2/15-779/schedule.html)、[CMU 15-442/642](https://mlsyscourse.org/)。
- [Stanford CS336](https://cs336.stanford.edu/)；[Systems 作业](https://github.com/stanford-cs336/assignment2-systems)；[Alignment 作业](https://github.com/stanford-cs336/assignment5-alignment)。
- [LCPU Infra Seminars](https://infra.seminars.lcpu.dev/)；[活动仓库](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars)。
- [原 foundations](https://github.com/MiniYuanBot/mlsys-foundations)：本课程的能力前置和样式参照，不要求重做。

## 官方工程文档与作者博客

| 资源                                                                                           | 用法                                                      |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [PyTorch compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | D3 graph break、dynamic shape、首次编译                   |
| [PyTorch checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)                     | D6 重入、RNG 与梯度语义                                   |
| [Checkpoint 技术博客](https://pytorch.org/blog/activation-checkpointing-techniques/)           | 对比不同重计算粒度与内存权衡                              |
| [PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)    | 看实际 trace，而不只看 wall clock                         |
| [CUDA notes](https://docs.pytorch.org/docs/stable/notes/cuda.html)                             | stream、同步、显存口径                                    |
| [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)            | D2 `bfe`、shuffle、FP8 转换指令的语义，以编译目标支持为准 |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)                  | 时间线与 overlap；WSL2 下只有 CLI 预览，本课程未用它出图  |
| [HF KV cache](https://huggingface.co/docs/transformers/en/kv_cache)                            | cache API、策略、支持条件                                 |
| [vLLM APC 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)                     | D5 块哈希与 free queue                                    |
| [SGLang 作者博客](https://lmsys.org/blog/2024-01-17-sglang/)                                   | prefix reuse 与 RadixAttention 图解                       |
| [FlashInfer 作者博客](https://flashinfer.ai/)                                                  | kernel 与 layout 的工程解读                               |

## 核验与更新

上游仓库的固定 commit 与源码链接见[代码地图](code-map.md)。需要检查外部链接时，从仓库根目录运行 `python course/readings/audit_external_links.py`，结果写入本地 `course/readings/results/external-links.json`，完整用法见[阅读资料说明](README.md)。HTTP 200 只说明页面能打开，不能证明方法已复现、许可已审查或硬件兼容。遇到 403/429 或网络失败，先换作者 GitHub 或论文入口；被限流不等于论文不存在。

重跑某篇论文之前，固定模型、数据、代码 commit、依赖和硬件。对外引用写实际读的论文版本，实验报告里说明本课程的简化实现与原论文的差异。只有标题或新闻、还没有全文或代码的新工作，不列为必做实验。

**截至 2026-10-18 仍待核实的条目**（来自各方向的核验记录；引用前先补齐）：

- D1：AsymCache 的代码仓库；BitDecoding 仓库内的具体文件路径。
- D2：AWQ 在固定 commit 下的上游搜索网格；Tilus、LiquidGEMM 的代码仓库；RTX 4060 的 Tensor Core TFLOPS（目前是由规格推算的）；lab2 中 INT4 慢于 FP8 的解释尚未用 SASS 证实。
- D3：vLLM/SGLang 的默认捕获尺寸与已安装版本是否一致（本机未安装二者）；vLLM `fused_add_rms_norm` 的源码位置；PyTorch 2 论文 §3.3/§3.4 以外的节号；Nimble 的代码链接；MPK 的正式会议（苗老师主页写 OSDI 2026）；CS336 2026 A2 中 Triton 部分的细节。
- D4：SLOs-Serve、Nightjar、FASER 未核实，暂不收录。
- D5：Preble、EPIC、Cake、T-LRU 的代码仓库；Hydragen 的会议（目前按 arXiv 2024）；4060 的实际 PCIe 有效带宽；Mooncake trace 的许可证（仓库为 Apache-2.0，trace 未声明，lab5 只附同 schema 的生成器）。
- D6：FlexLLM 图剪枝的实现细节。
- D7：NetMoE 代码（`PKU-DAIR/NetMoE` 返回 404）。
- D8：TRL GRPO 指南中的版本、4–7 GB 峰值与单步时间；vLLM 在驱动 566.24 上能否运行；OSDI 2026 的 Weave、RobustRL 只有程序页标题，论文链接未公开。
- D9：HexGen-2 的代码；Mélange 的正式会议；Coral、OpWeave 的会议与代码；A10G 规格（用 A10 代替）；GCP 价格；Helix 的 commit 固定。
- 苗组：MapFlow 的代码仓库（D5 笔记记为 `PKU-DAIR/MapFlow`，D9 笔记记为暂无链接，两者冲突）。
