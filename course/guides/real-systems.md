---
title: 从 lab 到真实系统
tags: [real-systems, serving, measurement, traces]
---

# 从 lab 到真实系统

[课程索引](../index.md) · [选题地图](directions.md) · [36h/周路线](roadmap.md) · [实验规范](experiments.md)

lab 的 R 阶段已经在 RTX 4060 Laptop 上碰到了真实模型、真实 kernel 或真实 trace。本页回答下一步的问题：在这台笔记本上还能跑哪些真实系统，怎样测得可信，用哪些公开 trace，以及每个方向走出课程的第一个实验是什么。版本与驱动信息于 2026-10-18 核验；没有在本机试过的项目都标了“待核实”。

## 1. 这台机器能跑什么

| 组件                                               | 本机状态                                                                                                                                                | 依据                                                                                                                                         |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Windows 驱动 566.24（R565 分支）                   | `nvidia-smi` 显示 `CUDA Version: 12.7`（2026-10-18 本机）                                                                                               | CUDA Toolkit 发布说明的驱动下限表：12.6 GA 要求 Windows 驱动不低于 560.76，12.8 GA 不低于 570.65，12.9 GA 不低于 576.02，13.0 属于 R580 分支 |
| CUDA Toolkit 12.6（WSL 内 `/usr/local/cuda-12.6`） | 可用；全部 lab 的 CUDA 阶段都在用                                                                                                                       | [环境配置](../../labs/setup.md)                                                                                                              |
| torch 2.14.1+cu126、transformers 5.18.0            | 可用；用到它们的 lab3/4/5/6/8 R 阶段已在 WSL 跑通                                                                                                       | [验证边界](../../labs/docs/validation.md)                                                                                                    |
| vLLM                                               | **未安装**。当前 0.32.0 钉死 `torch==2.13.0`，与课程环境的 torch 2.14.1 冲突；稳定版文档写默认 wheel 按 CUDA 12.9 编译（另有 12.8、13.0），不再提 cu126 | D4、D8 调研笔记（vLLM 安装文档与 PyPI 元数据）。cu129 wheel 能否靠 minor version compatibility 在 566.24 上运行：**未测试，待核实**          |
| SGLang                                             | **未安装、未核验**；对 torch 和驱动的要求待核实，预计与 vLLM 类似                                                                                       | —                                                                                                                                            |
| TRL GRPO                                           | lab8 只给指南，`use_vllm=False`，用单独的 `~/venvs/mlsys-rl`                                                                                            | [lab8 GUIDE](../../labs/lab8/real/GUIDE.md)（8GB 上的峰值与步时未验证）                                                                      |

因此有三条路，建议按顺序尝试：

1. **不动驱动，用最小 harness（先做这条）**。lab4 的 [`real/engine.py`](../../labs/lab4/real/engine.py) 已经是一个能用的单请求引擎：Qwen3 草稿与目标模型都用 `StaticCache`，每个输入宽度捕获一张 CUDA graph，推测循环每轮只同步一次 host。在它外面加一个请求队列和逐 token 时间戳，就能测 TTFT、TPOT 和排队。lab5 的 `real/measure.py` 测 prefill 代价 $T(p,n)$，可以直接复用。它没有 continuous batching 和 PagedAttention，绝对吞吐不能和 vLLM 比，但机制实验的结论可以成立。
2. **升级 Windows 驱动，另建 venv 装 vLLM**。CUDA 12.9 的 wheel 需要驱动不低于 576.02，CUDA 13.0 需要 R580 分支。在单独的 venv（例如 `~/venvs/mlsys-serve`）里装，不要改 `~/venvs/mlsys`。升级后重跑 `bash tools/setup_env.sh --lab all` 之后的 `check_env`，再跑一个 lab 的 R 阶段 reference 测试；驱动变化前后的计时不要放在同一张表里比较。
3. **用组内服务器或北大 HPC**。适合需要多卡、大模型或数据中心 GPU 的实验，申请前的确认事项见第 5 节。

## 2. 怎样给小模型做 benchmark

**模型**：WSL 的 `~/models` 下已有 Qwen3-0.6B（共 751.6M 参数，bf16 权重约 1.5GB）和 Qwen3-1.7B（2.03B，约 4.1GB），两者都是 Apache-2.0，tokenizer 完全相同，可以作为同 tokenizer 的草稿–目标对。chat template 默认开启思考模式，lab 用 `enable_thinking=False`，因为思考 trace 会改变接受率和长度分布；贪心实验显式传 `do_sample=False`，采样实验记录温度、top-k、top-p。

**指标**（与 [D4 §5](../modules/04-speculative-slo.md) 和[实验规范](experiments.md)一致）：

- TTFT：$t_{\mathrm{first}}-t_{\mathrm{arrive}}$，包含排队和 prefill。只测 prefill kernel 的时间不能叫 TTFT。
- TPOT：$(t_{\mathrm{finish}}-t_{\mathrm{first}})/(N_{\mathrm{out}}-1)$，$N_{\mathrm{out}}<2$ 的请求单列。
- ITL：相邻可见 token 的间隔分布。推测一轮提交多个 token 时，同轮 token 的时间戳相同，ITL 会变得突发，不能用整轮均摊代替。
- goodput：至少有四种定义（TurboSpec、DistServe、AdaServe、lab4 C 阶段），写报告时说明用的是哪一种。默认用请求级定义

$$
G=\frac{\left\lvert\{\, r \mid \mathrm{TTFT}_r\le S_{\mathrm{TTFT}},\ \mathrm{TPOT}_r\le S_{\mathrm{TPOT}} \,\}\right\rvert}{T_{\mathrm{window}}},
$$

其中 $S_{\mathrm{TTFT}}$、$S_{\mathrm{TPOT}}$ 是事先定好的 SLO，$T_{\mathrm{window}}$ 是观测窗口；窗口结束时还没完成的请求是删失样本，要说明怎么处理。

**三步测量**：

1. 单请求延迟：输入 128、512、2048 token，输出 64、128 token，各测 prefill 时间与每个 decode 步的时间。模型加载、CUDA graph 捕获、预热分别记录。
2. 闭环：一个请求结束再发下一个，得到无排队时的 TPOT 分布。
3. 开环：按 Poisson 到达或第 3 节的公开 trace 发请求，测排队后的 TTFT 和 goodput。单请求引擎相当于 FCFS 单服务台，结论要限定在这个前提下。

**测量规则**（详见 [authoring §5](../../labs/docs/authoring.md#5-测量规则)）：

- 每组计时前让 GPU 连续忙至少 300 ms。笔记本 GPU 一空闲就降到低 P-state，RTX 4060 Laptop 上实测显存频率会从 8000 MHz 掉到 810 MHz，几次普通 warmup 拉不回来。
- 每行输出 `stable` 列，判据是 $p_{75}\le1.5\,p_{25}$；不稳定的行重跑后再用。排查时用 `nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv -lms 50` 记录时钟。
- 一张卡同时只跑一个计时实验。`run.py --device cuda` 会自动拿 `/tmp/mlsys-gpu.lock`；手动运行时用 `flock /tmp/mlsys-gpu.lock <命令>`。后台有负载时测到的数即使很稳定也不能用。
- 由公式、规格表或仿真得到的数字标 `modeled, not measured`，只有本机实测的标 `measured`，两者不放在同一列比较。

## 3. 公开 trace 与许可证

| Trace                                                                                                               | 内容                                                                                                                           | 许可证与引用要求                                       | 课程里谁用                                                            |
| ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ | --------------------------------------------------------------------- |
| [Azure LLM Inference 2023](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md) | code、conv 两份 CSV：`TIMESTAMP, ContextTokens, GeneratedTokens`，不含文本                                                     | CC-BY；要求引用 Splitwise（ISCA'24）                   | lab9 可选挑战（替换 `make_trace`）；D3 研究题 (a)                     |
| [Azure LLM Inference 2024](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md) | 一周的 code、conv 两份，字段同上                                                                                               | CC-BY；要求引用 DynamoLLM（HPCA 2025）                 | 同上                                                                  |
| [BurstGPT](https://github.com/HPMLL/BurstGPT)                                                                       | Azure OpenAI GPT 服务 213 天、约 1031 万条；v2 加了 Session ID；含失败请求（Response tokens 为 0）                             | CC-BY-4.0；引用 BurstGPT 论文                          | lab9 可选挑战；D3 研究题 (a)                                          |
| [Mooncake traces (FAST'25)](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)              | conversation 12,031 条、toolagent 23,608 条、synthetic；到达时间、输入输出长度、512-token 块的哈希 ID；平均输入约 8K–15K token | 仓库为 Apache-2.0；trace 是否另有许可证**待核实**      | lab5 R2/R3 的格式（课程只提交同格式的合成 trace，没有下载真实 trace） |
| [ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)             | 约 53K 条对话，有文本，没有到达时间                                                                                            | HF 页面标 Apache-2.0，但上传者匿名，原始来源和授权不清 | 课程不使用；若用，只取长度分布，并在报告里写明这一风险                |
| 课程自带                                                                                                            | lab4 `real/prompts.jsonl` 的 36 个 prompt；lab8 用 Qwen3-0.6B 采的 512 条样本                                                  | 课程自写，随仓库的 MIT 许可证                          | lab4 R、lab8 R                                                        |

使用注意：

- Mooncake trace 同一请求里的块 ID 可能重复（例如 `[7,7]`），`input_length` 也不总等于块数乘 512。它的平均输入超过 4060 能实测的范围（约 8k token），外推的 $T(p,n)$ 不可信。DualMap 只用了前 4k/8k 个请求，以保证有限容量下缓存行为有意义。
- Azure 和 BurstGPT 只有长度和时间，没有文本，做接受率或质量实验时要另配 prompt。D4 可用的 prompt 集里，`lmsys/mt_bench_human_judgments` 标 CC-BY-4.0，`tatsu-lab/alpaca` 是 CC-BY-NC-4.0（仅限非商业）。
- 不要把下载的 trace 提交进仓库；在报告里写清版本、下载日期和截取范围。

## 4. 每个方向的第一次真实系统实验

“本机实测”是参考实现在 2026-10-18 的单次结果，供你校对自己的实现；“下一步”是课程之外的第一个实验，对应路线中的 W8。

| 方向 | lab R 阶段测什么                                                             | 本机实测（参考实现）                                                                                                                                           | 课程之外的下一步                                                                                                                              | 环境                             |
| ---- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| D1   | [lab1](../../labs/lab1/README.md) R：int4 KV 融合 decode-attention kernel    | 4K 上下文：融合 70 us，FP16 106 us，先反量化再算 131 us；32K：382/737/1549 us                                                                                  | 在 Qwen3-0.6B/1.7B 上用 RULER 或 LongBench 的子集比较 per-token 与 per-channel K 的质量；对照 llama.cpp、vLLM Triton decode kernel 的分组方式 | 本机；完整长上下文评测需更大显存 |
| D2   | [lab2](../../labs/lab2/README.md) R：sm_89 上的 FP8 权重 GEMV                | up_proj batch 1：INT4 B3 83 us，FP8 R3 87 us，FP16 cuBLAS 139 us；FP8 SIMT 与 FP16 cuBLAS 的交叉 batch 在 $8192^2$ 上为 2–4、up_proj 上为 4–8，远低于推算的 23 | 在 Qwen3-1.7B 的完整 decode step 里替换线性层，与 GemLite 或 vLLM 的 FP8 Marlin 同口径比较                                                    | vLLM 需升级驱动（第 1 节）       |
| D3   | [lab3](../../labs/lab3/README.md) R：CUDA graph 分桶                         | 真实回放 DP 1.409 ms/step，vLLM 默认网格 1.411；shift trace 1.887 对 1.864；每条捕获 stream 多占约 8 MiB cuBLAS 工作区                                         | 把 R 的规划接到 Qwen3-1.7B 完整 28 层 decode，用 BurstGPT 或 Azure trace 的 batch 分布选捕获尺寸                                              | 本机                             |
| D4   | [lab4](../../labs/lab4/README.md) R：Qwen3-0.6B 到 1.7B 的接受 trace 与成本  | eager $c\approx0.96$，graph $c\approx0.50$；平稳 trace 上没有自适应策略赢过最好的固定 $k=2$                                                                    | 采集按任务切换的多组 trace；之后在 vLLM/SGLang 的草稿模型推测模式下复测，与 AdaServe artifact 对照                                            | 本机；框架部分需升级驱动         |
| D5   | [lab5](../../labs/lab5/README.md) R：拟合 $T(p,n)$，回放 Mooncake 格式 trace | 底噪 22.1 ms，线性项 66.3 us/token，二次项 15.3 ns/token$^2$，交叉点约 4.3k token                                                                              | 下载 Mooncake toolagent trace 的前 4k–8k 个请求（先核实许可证），比较 SGLang 网关式阈值路由与代价感知路由                                     | 本机 CPU 回放                    |
| D6   | [lab6](../../labs/lab6/README.md) R：比 eager autograd 存得更少              | 8192 token、4 层：峰值增量 eager 1188 MiB，LoRA-FA 804，fused 1125，checkpoint 577                                                                             | 用 PEFT 在 Qwen3-0.6B 上做真实 LoRA 微调，按序列长度分桶，与按桶扫 `activation_memory_budget` 的基线比较                                      | 本机                             |
| D7   | [lab7](../../labs/lab7/README.md) R：MB 级专家的 pinned 流水线               | pinned H2D 12.6–12.8 GB/s；单槽 overlap 为 0，两槽及以上 0.75–0.91；标量 CPU 专家 7.74 ms 对拷贝加 GPU 1.51 ms                                                 | 采一份真实 OLMoE 路由 trace（约 13.8 GB 磁盘与内存预算），与 llama.cpp 的 `--n-cpu-moe` 静态切分同口径比较                                    | 受主机内存约束                   |
| D8   | [lab8](../../labs/lab8/README.md) R：回放实测 trace、长度预测与分布偏移      | 4 个 worker：先完成的 4 条 99.4 s，无偏 4 条 122.0 s，长度短 12.9%                                                                                             | 按 [GUIDE](../../labs/lab8/real/GUIDE.md) 在单独 venv 里跑 TRL GRPO 几步，采一份带策略版本的真实 trace                                        | 本机（8GB 可行性未验证）         |
| D9   | [lab9](../../labs/lab9/README.md) R：在 RTX 4060 上标定 $\eta$               | copy 227–231 GB/s；WMMA GEMM 22.1 TFLOPS（cuBLAS 25.4）；$\eta_c=0.67$，$\eta_m=0.91$；$\eta$ 缩到 0.75 倍时最优方案翻转                                       | 在一种数据中心 GPU 上重测 copy/GEMV/GEMM，检验 $\eta$ 能否从笔记本迁移；用 Azure 或 BurstGPT trace 替换 C3 的到达过程                         | 需要组内服务器或 HPC             |

## 5. 申请北大 HPC 或组内服务器之前

本机能回答机制问题，规模和可迁移性要到别的机器上验证。课程没有确认过账户资格、可用 GPU、分区、价格或配额，申请前先逐项确认：

- 账户：本科生能否申请，需要谁担保，是否计费。
- 硬件：GPU 型号与显存、单节点卡数、主机内存、本地盘与共享存储的容量。
- 软件：驱动与 CUDA 版本（决定能装哪一版 vLLM/SGLang）、是否允许容器、能否自建 Python 环境。
- 作业：队列系统、分区名、单作业时长上限、交互式调试是否可行。
- 网络：能否访问 Hugging Face 或镜像，模型与数据要不要先传上去。

各平台的基本配置见[环境配置](../../labs/setup.md)。写申请时用[选题地图的一页说明](directions.md#申请服务器前的一页说明)，写清唯一要验证的规模变量和停止条件。迁移后先重测 baseline，不同卡上的绝对时间不能直接相比。

## 资料汇总

[实验规范](experiments.md) · [测量规则（authoring §5）](../../labs/docs/authoring.md#5-测量规则) · [验证边界](../../labs/docs/validation.md) · [论文与代码](../readings/resources.md) · [固定源码阅读入口](../readings/code-map.md)
