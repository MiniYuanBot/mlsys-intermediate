---
title: 九个方向的选题与硬件地图
tags: [research, decision, hardware]
---

# 九个方向的选题与硬件地图

[课程索引](../index.md) · [36h/周路线](roadmap.md) · [从 lab 到真实系统](real-systems.md)

你在数学、系统基础和执行力方面有优势，也能稳定投入约 36h/周。接下来要建立研究中的反馈：把“读懂一个领域”拆成“每周测出一个能解释的现象”。现阶段优先选**单机就能验证核心机制、成功失败都能提供信息、与目标导师组的在研方向相连**的题目。

以下是课程设计建议，不承诺新颖性或发表概率。苗旭鹏老师现为北京大学计算机学院助理教授，公开工作覆盖推理、编译、训练和数据管理（[主页](https://hsword.github.io/)、[论文目录](https://hsword.github.io/publications/)，2026-10-18 核验）。具体题目是否与组内在做的课题重复，要通过组内讨论确认；下表“组内对应工作”只列作者名单里有苗旭鹏、并经论文页或主页核对过的工作。

## 优先级总表

| 优先级         | 方向与切入点                                                                            | 本机最小实验                                                                                      | 值得研究的条件变化                                                               | 最近邻/竞争工作                                                                    | 组内对应工作                                                                                                                                              | 升级条件                                                          |
| -------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| A              | **D4**：利用 $A(k)=\min(L,k)$ 删失结构的探索；在线接受模型的 SLO 预算                   | lab4 A/B/C 在 CPU；R 在 4060 采 Qwen3-0.6B 到 1.7B 的接受 trace（GPU 约 5 分钟）                  | 任务切换导致接受率漂移；验证耗时非线性；队列压力；eager 与 CUDA graph 两种成本比 | BanditSpec、HedgeSpec、TurboSpec/vLLM DSD、AdaSpec                                 | SpecInfer（ASPLOS'24，苗旭鹏一作）、AdaServe（EuroSys'26，作者之一）                                                                                      | 树结构、多请求共享预算和高并发，需要组内 GPU 与 AdaServe artifact |
| A              | **D8**：partial rollout 的 KV 语义与逐 token 重要性修正；长度预测误差下的调度界         | lab8 全部用 CPU（Rust）；R 回放 4060 采的 512 条 Qwen3-0.6B 样本；TRL GRPO 指南可选（8GB 未验证） | 长度预测误差、策略版本滞后、“先完成者入选”造成的长度偏移                         | AReaL、APRIL、PipelineRL、Seer、RollPacker、StaleFlow                              | DARTS（ICML'26）、QLPO（COLM'26），作者均含苗旭鹏                                                                                                         | RL 学习曲线的多 seed 验证和跨机迁移需要多卡                       |
| A              | **D9**：规划器对代价模型 $\eta$ 误差的鲁棒性；$\eta$ 从笔记本迁移到数据中心 GPU         | lab9 A/B/C 在 CPU 上仿真异构集群；R 在 4060 标定 copy、GEMV、GEMM                                 | $\eta$ 的系统性偏差、价格表切换、到达突发度（CV）、多模型共置                    | KernelSight-LM、ShuntServe、Hetis、Mélange、HexGen                                 | Helix（ASPLOS'25）、SpotServe（ASPLOS'24，苗旭鹏一作）、Demystifying（ICML'25）、Janus（SOSP'26）；Coral、OpWeave（arXiv'26，作者含苗旭鹏，尚未列入主页） | 迁移验证至少要在一种数据中心 GPU 上实测 $\eta$                    |
| A-（底层备选） | **D3**：显存预算下按 trace 选 CUDA graph 捕获尺寸；漂移下的在线重编译准入               | lab3 A 到 C4 在 CPU；C5–C7 和 R 在 4060（WSL）                                                    | 形状分布漂移、捕获显存池、冷启动与稳态分开、跨进程编译缓存                       | vLLM/SGLang 固定网格、TensorRT-LLM 调参博客、DietCode、PyTorch mega-cache、Foundry | Mirage（OSDI'25）、MPK（OSDI'26）、Event Tensor（MLSys'26），苗旭鹏在 Purdue 期间合作；关系是“编译期搜索与运行时调度”，不是捕获尺寸选择本身               | 完整 28 层模型加公开 trace；不必先上多卡                          |
| B              | **D1**：GQA/QK-Norm 下 K 的量化轴（Qwen3）；融合 kernel 内 per-channel K 的代价         | Qwen3-0.6B/1.7B 的 K/V 轴探针；lab1 R 的 int4 融合 decode kernel                                  | 上下文 4K 到 32K、split 数、量化轴与组大小                                       | KIVI、KVQuant、QServe、TurboQuant、Kitty、BitDecoding                              | PQCache（SIGMOD'25）、Multi-Segment Attention / AsymCache（arXiv 2606.02964），作者均含苗旭鹏                                                             | RULER/LongBench 级的长上下文质量评测需要更大显存                  |
| B              | **D6**：按序列长度分桶的 checkpoint/offload 计划；8GB 卡上以显存为中心的推理–微调共服务 | lab6 正确性在 CPU；4060 上测峰值与 saved bytes                                                    | 序列长度波动；KV 页面与微调激活争同一块显存                                      | MEMO、PyTorch `activation_memory_budget`、LoRA-FA、Unsloth、LLMStation             | FlexLLM（NSDI'26）、Hetu v2（OSDI'26）；Quantized Side Tuning（ACL'24）为近邻；作者均含苗旭鹏                                                             | 真实微调质量和共服务实验要更大显存                                |
| B              | **D7**：统一 harness 下专家缓存替换与预取的净收益                                       | 合成加真实路由 trace；4–32 MiB 专家的真实 H2D 与 overlap                                          | 相位突变、无局部性 trace、单 copy engine 争用                                    | mixtral-offloading（LRU）、Euro-Par'25 LLRU、SpecMD、FlashMoE                      | 没有 MoE 推理或卸载论文；NetMoE（ICLR'25）、FlexMoE（SIGMOD'23）在**训练侧**利用路由局部性做放置                                                          | 真 MoE 权重受主机内存和磁盘约束（OLMoE trace 约需 13.8 GB）       |
| C              | **D2**：形状/格式相关的低比特 kernel 分发；Ada 上 FP8 对 INT4 的 kernel 级交叉点        | 4060（sm_89，有 FP8，无 FP4）上的 GEMV 形状与 batch 扫描                                          | batch、组大小、层形状、同一权重存两种布局的显存                                  | GemLite、vLLM `choose_mp_linear_kernel`、BitBLAS/Ladder、TileLang、Marlin、QServe  | 没有量化 kernel 论文；最近的是 Quantized Side Tuning（ACL'24）；做 (b) 时可以表述成 Mirage/MPK/Event Tensor 的编译对象（推断，非组内公开方向）            | 工程竞争强，需要超过 Marlin/cuBLAS 级 baseline                    |
| C              | **D5**：异构副本上按实测非线性 prefill 代价决定复制前缀还是排队；深度感知淘汰           | lab5 A/B/C 在 CPU；R 在 4060 拟合 Qwen3-0.6B 的 $T(p,n)$                                          | 热点迁移、长前缀占满缓存、副本异构                                               | Preble、SGLang 网关、Dynamo、llm-d、DualMap、CacheRoute、T-LRU                     | 没有前缀缓存论文；最近的是 MapFlow（SIGMOD'27，跨模型 attention map 复用）、PQCache（SIGMOD'25）                                                          | 先解决真实 trace（Mooncake trace 的许可证待核实），再谈多卡       |

A/B/C 是**起步推荐等级**，不是对领域学术价值的排序。如果特别喜欢底层代码，D3 是首选的底层方向，D2 可以作为它的备选；如果对 RL 更有动力，就从 D8 开始，但纯调度模拟不能当作 RL 收敛结果。

## 为什么这样排

排序综合了三类证据：各讲“前沿与 baseline 对照”里的**查新结论**、lab R 阶段在 4060 上的**实测结果**，以及与组内在研工作的距离。实测数字都来自 2026-10-18 的参考实现运行，只是单机、单次 trace 上的结果。

- **D4（A）**：方向本身拥挤，BanditSpec、HedgeSpec、TurboSpec、AdaServe、AdaSpec 各占一块，但它和 SpecInfer、AdaServe 一脉相承，讨论时有共同语言。实测给出一个有信息量的负结果：CUDA graph 下 $c\approx0.50$ 时，真实 trace 上没有哪个自适应策略赢过事后最好的固定 $k=2$（乐观控制器慢约 2.3%，独立臂 UCB 慢 7.1%，均值控制器卡在 $k=0$ 慢 19%）；生存概率 $S_j$ 从 0.81 降到 0.40，条件接受率 $a_j$ 却从 0.81 升到 0.90。研究题因此很具体：先找到或构造确实存在漂移的负载，再证明删失结构的价值。
- **D8（A）**：DARTS、QLPO 是组内正在推进的 RL 线。查新结论里 (c)“重算 KV、保留旧 KV、丢弃重来三种语义下的 IS 比值分布”没有找到专门比较的工作（负结果，不保证不存在），0.5B 模型在 8GB 卡上可做。实测回放里，冗余 2 倍、先完成的 4 条入选比从 8 条中无偏取 4 条快 18.5%（99.4 s 对 122.0 s），代价是回答长度短 12.9%（190.3 对 218.5 token）。速度和样本语义的取舍在这个数里看得很清楚。
- **D9（A）**：Helix、Coral、OpWeave 是同一团队从层级放置、多模型模板到算子级拆分的一条线，lab9 就是沿这条线设计的。查新结论：KernelSight-LM 已做跨 GPU 预测，ShuntServe 已做 roofline 驱动的 DP 放置，但都没有分析 $\eta$ 误差传到 max-flow 或 ILP 规划器之后会不会翻转方案排名。实测：$\eta_c=0.67$、$\eta_m=0.91$，GEMV held-out 误差 9.5%；把全部 GPU 的 $\eta$ 缩到 0.75 倍时，排名第一的方案已经翻转（Kendall 距离 3），最便宜的共置方案在 Poisson 到达下 SLO 达成率只有 73%。这只是一个场景的一次运行，可以作为研究的起点。
- **D3（A-）**：Mirage、MPK、Event Tensor 是苗旭鹏在 Purdue 期间合作的编译线，适合想做底层的同学。实测提醒了代价模型的差距：模型推算里 DP 的 regret 为 1–3%，固定网格为 4–10%；真实回放（每步同步）里 stable trace 上 DP 1.409 ms/step，vLLM 默认网格 1.411，shift trace 上 DP 1.887 对 1.864，默认网格反而更快。要做出贡献，必须用完整模型和公开 trace，并写出 piecewise graph 下的显存模型。
- **D1（B）**：(d)“GQA 和 QK-Norm 下 K 的量化轴”是查新里最便宜的空白，本机探针也说明它有实际意义（per-token int4 K 让 Qwen3-0.6B 的 NLL 从 5.27 升到 8.36，per-channel 只到 5.29）。它排在 B 是因为：(b) 淘汰加量化已经很拥挤；质量结论需要 RULER 级评测，8GB 卡做不完整。组内有 PQCache 和 AsymCache 可以对接。
- **D6（B）**：(b) LoRA 少存反传、(d) 优化器状态压缩都已饱和；(a) 单卡按长度分桶的计划和 (c) 8GB 卡上的推理–微调共服务还有空白，并与 FlexLLM 相连。实测 4 层、8192 token 时，checkpoint 把峰值增量从 eager 的 1188 MiB 降到 577 MiB。
- **D7（B）**：查新空白清楚，统一 harness 下的替换策略排序、命中率能否转成端到端时间，都还没人回答。实测：单 copy engine 下单槽 overlap 恰为 0，两槽及以上 0.75–0.91；标量 CPU 算专家 7.74 ms，远慢于拷贝加 GPU 的 1.51 ms。但组里没有 MoE 推理工作，真实 MoE 权重也受主机内存约束。
- **D2（C）**：(b) 形状相关分发要同时对上 GemLite、vLLM、BitBLAS/Ladder、TileLang，只做“按 batch 选 kernel”会和 GemLite 撞车；主页上没有量化 kernel 工作。实测数据有用（FP8 SIMT 对 FP16 cuBLAS 的交叉 batch 在 $8192^2$ 上为 2–4、up_proj 上为 4–8，远低于模型推算的 23），但更适合当 D3 的测量工具，不适合单独成题。
- **D5（C）**：路由一侧有 Preble、SGLang 网关、Dynamo、llm-d、DualMap、CacheRoute，淘汰一侧有 Marconi、T-LRU、KVFlow；讲义自己也写明空白是“根据笔记推断，提给导师之前必须重新检索”。组里没有前缀缓存论文。若仍想做，推荐把它并到 D9（异构副本上的前缀亲和路由）。实测 prefill 代价 $T(p,n)$ 的二次项交叉点约 4.3k token，可以直接给 D9 用。

## 硬件边界与起步配置

| 设备                                             | 现在适合做什么                                                                    | 不能据此推断什么                                                                                               |
| ------------------------------------------------ | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 16GB 统一内存 Mac mini                           | 阅读；C++/Rust CPU 测试；trace 回放与数据分析；Apple Silicon 上可另测 MPS         | 统一内存不等于 16GB CUDA 显存；MPS 性能不能替代 NVIDIA kernel 结果                                             |
| RTX 4060 Laptop 8GB，Windows + WSL2              | 全部 lab 的全部阶段；CUDA/PTX、torch.compile、Qwen3-0.6B/1.7B 短上下文、微型 LoRA | 不能把 H100 的 FP8/TMA/多卡通信结果按比例搬来；当前驱动跑不了新版 vLLM，见[从 lab 到真实系统](real-systems.md) |
| Intel UHD                                        | 日常显示；CPU 程序与文档                                                          | “128MB”不能与 NVIDIA 显存合并用于 CUDA                                                                         |
| [北大 HPC](https://hpc.pku.edu.cn/) / 组内服务器 | 研究问题确认后，做规模与可迁移性验证                                              | 尚未确认账户资格、可用 GPU/分区/价格/配额，不预设已经可以申请                                                  |

起步预算：CUDA microbench 一般低于 512MiB；lab6 微型网络以实测峰值为准。Qwen3-0.6B bf16 权重约 1.5GB，Qwen3-1.7B 约 4.1GB，另需 KV、临时工作区、框架和 CUDA context。先把 `batch=1,input<=512,output<=64` 跑通，再逐级扩大；实际可用显存低于 8GB。

KV 估算用模型 config 的 `num_key_value_heads`，不要把 query heads 当成 KV heads。训练预算还要算梯度、优化器状态和激活；全参数 Adam 与冻结基座的 LoRA 差别很大。各平台实际验证到哪一步，见[验证边界](../../labs/docs/validation.md)。

## 选题打分：用证据替代“看起来热门”

每项 0–2 分：兴趣、两周内 baseline 可运行、有可用 workload、能写出反证条件、与导师在研工作契合。**新颖性要对照已有工作判断**：另列最近三篇论文，逐项比较场景、约束、机制和评价。说不清差异时，先做复现与测量，不要称为新算法。

## 带去讨论的候选假设

下面五条按优先级排列。每条都给出本机已有的实测起点和停止条件；起点数字是参考实现的单次结果，正式实验要按[实验规范](experiments.md)重测。

1. **H-Spec（D4）**：在分段平稳、按任务切换的真实 Qwen3-0.6B 到 1.7B 贪心 trace 上，利用 $A(k)=\min(L,k)$ 的乐观探索控制器，在每次切换后 200 轮内的累计 regret 比独立臂 UCB 低至少 30%，平稳段 regret 不高于 5%。
   - 实测起点：平稳 trace 上没有自适应策略赢过最好的固定 $k=2$（乐观控制器 $-2.3\%$）；回放与真实运行的轮次一致率 98.4%。
   - 先与 AdaServe、BanditSpec 的问题定义逐项比对。任务切换次数加倍后仍不能比最好的固定 $k$ 快 3% 以上，就停下，改做“用草稿 top-1 概率等上下文信号预测下一轮难度”。
2. **H-Rollout（D8）**：在 partial rollout 下，“重算 KV、保留旧 KV、丢弃重来”三种语义产生的 IS 比值分布和训推概率失配可以区分开，并且会改变 GRPO 的学习曲线。
   - 实测起点：先完成者入选比无偏子集快 18.5%，回答长度短 12.9%，奖励偏移很小（0.813 对 0.816），这份 trace 的权重从未更新（`policy_version=0`）。
   - 需要真实的多版本训练 trace。若三种语义的 IS 比值差异落在 seed 间噪声内，报告负结果，转向长度预测误差下的 makespan 上界。
3. **H-Hetero（D9）**：(a) 把非实测 GPU 的 $\eta$ 同比例缩到 0.75 到 1.0 之间（KernelSight-LM 报告的约 12% 误差量级），就足以让 B&B 最优方案的排名第一翻转，并让按名义 $\eta$ 选出的方案在“真实”系数下的美元后悔值超过 10%；(b) 从功耗受限的笔记本卡标定的 $\eta$，迁移到数据中心卡后误差可控。
   - 实测起点：全部 GPU 缩到 0.75 倍时排名第一已翻转（Kendall 距离 3），只是一个场景一次运行。
   - 翻转只在 0.5 附近出现，或价格表切换造成的变化远大于 $\eta$，就停掉 $\eta$ 方向，改做价格和可用量的不确定性。4060 自身的 held-out 误差超过 25% 时，先回去改代价模型。(b) 必须在至少一种数据中心 GPU 上实测，本机只能给出假设。
4. **H-Compile（D3）**：在 Qwen3-1.7B 完整 28 层 decode step 上，用历史 trace 加 DP 选出的 8 个捕获尺寸，在 shift trace 上的每步延迟比 vLLM 默认网格的前 8 个尺寸低至少 3%，共享池显存不超过默认网格。
   - 实测起点：小模型真实回放里 DP 与默认网格几乎相同（1.409 对 1.411 ms/step），shift trace 上默认网格反而更快（1.887 对 1.864），说明代价模型漏掉了某些项。
   - 先解释这个差距。如果收益只在不合理地重置编译缓存，或者只在模型推算里出现，就放弃这一假设。动态分桶已有工作，差异要来自新的目标、约束或能解释的反例。
5. **H-KV（D1）**：在 GQA 加 QK-Norm 的 Qwen3 上，K 的量化轴（per-channel 对 per-token）比统一位宽的选择对质量影响更大，而 per-channel K 在融合 kernel 里的额外代价可以控制在 per-token 的 10% 以内。
   - 实测起点：31 token 探针上 per-token int4 K 让 NLL 从 5.27 升到 8.36，per-channel 只到 5.29；lab1 R 的融合 kernel 在 4K 上下文为 70 us（FP16 106 us，先反量化再算 131 us）。
   - 先排除 KIVI、Kitty 和 mixed-precision KV 工作。若 RULER 级评测里两种轴差别落在噪声内，停止增加策略复杂度。

## 申请服务器前的一页说明

- 本机复现了什么，原始数据（`labs/results/` 里的记录）在哪里；哪个结论尚不成立？
- 需要验证的唯一规模变量是什么（模型规模、上下文长度、GPU 型号、卡数）？为什么本机不够？
- 请求的 GPU 型号与显存/数量、主机内存、数据集与磁盘、预估 GPU-hours、停止条件。
- 需要的软件栈：驱动和 CUDA 版本、能否装 vLLM/SGLang、是否允许容器、能否访问 Hugging Face 或镜像。
- 队列系统、存储路径和使用政策以实际平台说明为准。使用 Slurm 前核对分区名，不要照抄不存在的名称。确认事项清单见[从 lab 到真实系统](real-systems.md)的最后一节。

## 资料汇总

[讲义九模块](../index.md) · [论文与代码](../readings/resources.md) · [前沿对照与提案模板](../templates/proposal.md) · [实验设计](experiments.md) · [从 lab 到真实系统](real-systems.md)
