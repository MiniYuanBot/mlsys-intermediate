# 上游代码阅读地图

[资料总表](resources.md) · [阅读资料说明](README.md)

本页按上游仓库列出各讲和各 lab 指定读的源码入口，截至 2026-10-18。每节第一行写版本：

- **固定 commit** 或 **固定 tag**：链接指向那个版本，内容不会随主分支变化。
- **未固定 commit（待固定）**：链接指向 `main`/`master`，路径在 2026-10-18 核对过，但以后可能移动。阅读时先记下当天的 commit，再写进自己的笔记。

这些链接是当日的工程快照，不一定对应论文 artifact 的 tag，不能用 main 分支的现状推断论文的原始实验。可选的本地快照放在 `results/upstream-snapshots.json`，由 Git 忽略。

读完一个入口，记下五件事：输入和输出、调用者、状态不变量、对应的论文机制、在 8GB 卡上缩小了什么。本项目只引用这些公开代码，没有整段复制其实现。

## vllm-project/vllm

版本因讲而异：D3 固定 commit `371745ebd6c12f314402c6c319de2e7ef82c2981`；D5 固定 tag `v0.32.0`；D1、D2 的入口为未固定 commit（待固定）。本机没有安装 vLLM（当前版本要求比 566.x 更新的驱动），默认值以源码为准，没有实跑核对。

- D1 R2/R3：[vllm/v1/attention/ops/triton_decode_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py) 的 `_fwd_grouped_kernel_stage1` 与 `_fwd_kernel_stage2`，即 split-KV 加 LSE 合并。
- D2 R3：[vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py)，FP8 权重走 Marlin 路径时的打包与转换。
- D3 R1：[vllm/v1/cudagraph_dispatcher.py](https://github.com/vllm-project/vllm/blob/371745ebd6c12f314402c6c319de2e7ef82c2981/vllm/v1/cudagraph_dispatcher.py) 的 `_compute_bs_to_padded_graph_size`，batch 补到哪个捕获尺寸。
- D3 R 基线：[vllm/config/compilation.py](https://github.com/vllm-project/vllm/blob/371745ebd6c12f314402c6c319de2e7ef82c2981/vllm/config/compilation.py) 中 `max_cudagraph_capture_size` 的 docstring，`vllm_default` 基线由此而来。
- D4 R1：[spec decode metrics 文档](https://docs.vllm.ai/en/stable/api/vllm/v1/spec_decode/metrics/) 中逐位置接受率的定义，对应 `hazard_from_survival`。
- D5 A1：[vllm/v1/core/kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_utils.py) 的 `hash_block_tokens`，链式块哈希。
- D5 C2：[vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/kv_cache_manager.py) 中命中长度最多取 $n-1$ 的上限。
- D5 A4 对照：[vllm/v1/core/block_pool.py](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/block_pool.py) 的 free queue。

## sgl-project/sglang

版本因讲而异：D3 固定 commit `57db5089c207ae4ce189cc092a89a2027c8aea29`；D5 固定 tag `v0.5.21`。

- D3 R 基线：[python/sglang/srt/arg_groups/cuda_graph_hook.py](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/arg_groups/cuda_graph_hook.py) 的 `generate_decode_cuda_graph_batch_sizes`，`sglang_default` 基线由此而来。
- D3 R1/R2：[python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py) 的 `_pad_to_bucket`。
- D5 A4、B2/B3：[python/sglang/srt/mem_cache/radix_cache.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/radix_cache.py) 的 `match_prefix`、`evict`、`inc_lock_ref`。
- D5 A4：[python/sglang/srt/mem_cache/evict_policy.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/mem_cache/evict_policy.py) 的 `LFUStrategy`。
- D5 C2：[python/sglang/srt/managers/schedule_batch.py](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/managers/schedule_batch.py) 的 `_compute_max_prefix_len`。
- D5 C3：仓库内 `sgl-model-gateway/src/policies/cache_aware.rs`，失衡判定与 `cache_threshold`；路径按 lab5 资料记录，深链接待固定。

## ggml-org/llama.cpp

未固定 commit（待固定）。

- D1 R1：[ggml/src/ggml-cuda/fattn-common.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh) 的 `vec_dot_fattn_vec_KQ_q4_0`，最常用的 int4 KV 点积。
- D7 R5：[tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) 的 `-ot` 与 `--n-cpu-moe`，静态切分基线。注意正则 `exps` 匹配不到共享专家的 `_shexp`。

## InternLM/lmdeploy

未固定 commit（待固定）。D1 R1–R3：TurboMind 的在线 int4/int8 KV，V 布局与 lab1 R 相同，K 布局不同。

- [src/turbomind/kernels/attention/decoding.cu](https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/attention/decoding.cu)
- [src/turbomind/kernels/attention/quantization.h](https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/attention/quantization.h)
- [src/turbomind/kernels/attention/reduce.cu](https://github.com/InternLM/lmdeploy/blob/main/src/turbomind/kernels/attention/reduce.cu)

## jy-yuan/KIVI

固定 commit：`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`（前两项）。D1：找到 group/scale 的布局与解码入口，解释追加时残余窗口的行为，不要只看 API。

- [models/utils_quant.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/utils_quant.py)：`AsymGroupedQuantizer` 与 `AsymGroupedQuantizerByChannel`。
- [quant/csrc/gemv_cuda.cu](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/csrc/gemv_cuda.cu)
- B2：[models/llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/main/models/llama_kivi.py)（main 分支链接，未固定 commit（待固定）），残余窗口满后何时 flush。

## HugoZHL/PQCache

固定 commit：`0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828`。D1 C4：画出 cache manager 的构建、候选检索和原始 KV 取回路径。

- [vq_method/retrieval_based/cache_manager.py](https://github.com/HugoZHL/PQCache/blob/0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828/vq_method/retrieval_based/cache_manager.py)

## huggingface/transformers

D4、D5 固定 tag `v5.18.0`（与本机安装版本一致）；模型文件为未固定 commit（待固定）。

- D1 R 与 probe：[models/qwen3/modeling_qwen3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) 的 `Qwen3Attention.forward`，QK-Norm 之后才写 cache。
- D4 B3：[generation/candidate_generator.py](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/generation/candidate_generator.py) 的 `AssistedCandidateGenerator.update_candidate_strategy`，即 `HfHeuristic` 的来源。
- D4 R1：[generation/utils.py](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/generation/utils.py) 的 `_speculative_sampling`。
- D5 R1：[cache_utils.py](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/cache_utils.py) 的 `crop`。
- D7 A1：[models/mixtral/modeling_mixtral.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py)（`MixtralTopKRouter`）、[models/olmoe/modeling_olmoe.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/olmoe/modeling_olmoe.py)、[models/deepseek_v3/modeling_deepseek_v3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py)：三种归一化约定；`e_score_correction_bias` 只参与选择，不参与加权。

## IST-DASLab/marlin

未固定 commit（待固定）。D2 B1/B3 与挑战 2。

- [marlin/marlin_cuda_kernel.cu](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu) 的 `dequant`：用 `lop3` 和魔数把 int4 转成 FP16。
- [marlin/**init**.py](https://github.com/IST-DASLab/marlin/blob/master/marlin/__init__.py) 的 `_get_perms`：离线交错布局。

## mit-han-lab/llm-awq

固定 commit：`d6e797a42b9ef7778de8ee2352116e0f48a78d61`。

- D2 C1–C3：[awq/quantize/auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py) 的 `_search_module_scale`，通道缩放的网格搜索（该 commit 下的网格细节待核实）。
- D2 B：[awq/kernels/csrc/quantization/gemv_cuda.cu](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/kernels/csrc/quantization/gemv_cuda.cu)：nibble 解码、scale 读取、归约；精度与本项目的 W4A32 不同。

## NVIDIA/FasterTransformer

未固定 commit（待固定；仓库已不再更新）。D2 讲义 §3.3：

- [cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h](https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h)：交错布局下 int4 到 FP16 的快速转换。

## dropbox/gemlite

未固定 commit（待固定）。D2 B5：[仓库](https://github.com/dropbox/gemlite)中的 `gemlite/core.py` 按 batch 在 GEMV、split-K GEMV 和 GEMM 之间分发；lab2 只写四条路径，不做自动选择。

## mit-han-lab/omniserve

固定 commit：`02b2925aa6fa3b92b06316a1524b7f38922cd9c8`。D2：读量化后端与 scheduler 的接口；旧 qserve 地址已重定向。

- [omniserve/core/scheduler.py](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/omniserve/core/scheduler.py)

## mirage-project/mirage

固定 commit：`f9eb70c254acefc9f3667b2a973d0dcf25471fce`。D3：追踪图、编译、launch 和 profiling，逐步缩小实验规模，不要直接照搬大模型 demo。

- [python/mirage/mpk/persistent_kernel.py](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/python/mirage/mpk/persistent_kernel.py)：`PersistentKernel`、`get_compile_command`。

## pytorch/pytorch

未固定 commit（待固定）。下列行为已在本机 torch 2.14.1+cu126 上实测（`CheckpointPolicy` 含 `MUST_CPU_OFFLOAD`/`PREFER_CPU_OFFLOAD`；`activation_memory_budget` 默认 1.0）。

- D6 B1/C3：[torch/utils/checkpoint.py](https://github.com/pytorch/pytorch/blob/main/torch/utils/checkpoint.py)：非重入 checkpoint 与 `CheckpointPolicy`。
- D6 B4 基线：[torch/_functorch/config.py](https://github.com/pytorch/pytorch/blob/main/torch/_functorch/config.py) 的 `activation_memory_budget`。PyTorch 博客写的 `torch._dynamo.config` 拼法在 2.14.1 中不存在。
- D6 R 的出发点：[tools/autograd/gen_variable_type.py](https://github.com/pytorch/pytorch/blob/main/tools/autograd/gen_variable_type.py) 的 `guard_for`，eager 已经只保存反传需要的操作数。

## huggingface/peft

未固定 commit（待固定）。D6 A1–A3：

- [src/peft/tuners/lora/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) 的 `Linear.forward` 与 `merge`。

## unslothai/unsloth

未固定 commit（待固定）。D6 R2：

- [unsloth/kernels/fast_lora.py](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) 的 `LoRA_MLP`，手写反传的融合 LoRA MLP；lab6 R2 写的是 GELU 版本。

## FlexLLM/artifact

固定 commit：`9c01a2ae5a2c004292add9fe38050776178d3045`。D6：先读 README 的测量目标、子模块与 figure 脚本；不要执行 teardown 删除命令。

- [README.md](https://github.com/FlexLLM/artifact/blob/9c01a2ae5a2c004292add9fe38050776178d3045/README.md)

## flexflow/flexflow-serve

固定 commit：`7f07ba2bf98c505c86dd4c1bacb6384454fa37a5`。D4 B5/B6：

- [src/runtime/request_manager.cc](https://github.com/flexflow/flexflow-serve/blob/7f07ba2bf98c505c86dd4c1bacb6384454fa37a5/src/runtime/request_manager.cc) 的 `add_tokens_toward_slo`：按 SLO 分配草稿 token。

## zikun-li/AdaServe-Artifact-Evaluation

固定 commit：`dbae1c30319f70c73b0f512ba2e5710003382786`。D4：先读各图的实验入口与机器配置，再进递归子模块。artifact wrapper 的 commit 不等于子模块内部的源码版本。

- [README.md](https://github.com/zikun-li/AdaServe-Artifact-Evaluation/blob/dbae1c30319f70c73b0f512ba2e5710003382786/README.md)

## databricks/megablocks

固定 commit：`952db33d6eac334d22c61e47a0d5d41446298784`。

- [megablocks/layers/moe.py](https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/layers/moe.py)：`indices_and_bins` 对应 A2 的计数排序；`permute_and_compute` 对应 A4/C2 的 gather、MLP、scatter。

## EfficientMoE/MoE-Infinity

固定 commit：`96a95527028f434f94a616c487094714ef10436b`。D7：读预取控制与真实 overlap microbench；当前工程版与论文版有差异。

- [core/prefetch/archer_prefetch_handle.cpp](https://github.com/EfficientMoE/MoE-Infinity/blob/96a95527028f434f94a616c487094714ef10436b/core/prefetch/archer_prefetch_handle.cpp)
- [benchmarks/expert_io_microbench/bench_overlap_prefetch.py](https://github.com/EfficientMoE/MoE-Infinity/blob/96a95527028f434f94a616c487094714ef10436b/benchmarks/expert_io_microbench/bench_overlap_prefetch.py)
- [docs/configuration.md](https://github.com/EfficientMoE/MoE-Infinity/blob/96a95527028f434f94a616c487094714ef10436b/docs/configuration.md)

## PKU-DAIR/DARTS

固定 commit：`e07825aee39612e0659ce7e5a38f8ae02a2250b2`（[仓库快照](https://github.com/PKU-DAIR/DARTS/tree/e07825aee39612e0659ce7e5a38f8ae02a2250b2)）。D8。**该 commit 不能按原样运行**：缺少 `ray_trainer_stream*.py`。只读代码，不要试图复现训练。

- B4/R1：[darts/recipe/darts/ray_trainer_darts.py](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/darts/recipe/darts/ray_trainer_darts.py)：`RayTrainerDarts`、冗余分配、`ScoreFilter`（docstring 写 EMA，实际是累计均值，适合当读码练习）。`dynamic_batch_reorder`（LPT）在 `fit` 中的调用被注释掉，对应 A3。
- B4：`verl/workers/rollout/vllm_rollout/vllm_async_server.py` 的 `RedundancyCounter`：前 $n$ 条完成即保留、其余 abort。路径相对于仓库内的 verl 副本，深链接待固定。
- 继承来的 one-step-off 代码：[darts/recipe/darts/ray_trainer.py](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/darts/recipe/darts/ray_trainer.py) 的 `OneStepOffRayTrainer`、`sync_rollout_weights`、`_async_gen_next_batch`。这些不是论文新增逻辑。
- 入口：[darts/recipe/darts/main_ppo.py](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/darts/recipe/darts/main_ppo.py)，`algorithm.trainer_name` 决定选哪个 trainer。

## verl-project/verl

固定 commit：`18cfb74852a053a27dcf301d3318a093be81b628`。D8。

- A4/C4：[verl/trainer/ppo/core_algos.py](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/verl/trainer/ppo/core_algos.py) 的 `compute_grpo_outcome_advantage` 与 `compute_policy_loss`。
- C6：[docs/algo/rollout_corr.md](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/docs/algo/rollout_corr.md)，rollout 与训练策略不一致时的重要性校正。
- C1/C2/C7：[verl/experimental/fully_async_policy/fully_async_rollouter.py](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/verl/experimental/fully_async_policy/fully_async_rollouter.py)。
- 总入口：[verl/trainer/main_ppo.py](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/verl/trainer/main_ppo.py)、[docs/algo/grpo.md](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/docs/algo/grpo.md)。

## Thesys-lab/Helix-ASPLOS25

未固定 commit（待固定）。D9 B1–B7：[仓库](https://github.com/Thesys-lab/Helix-ASPLOS25)的 README simulator tutorial；`simulator/model_manager`（每种 GPU 的“层数到吞吐”profiling 表）；`config/machine_profile.ini`（NIC 速度、显存）；`FakeClusterGenerator` 的集群生成方式。

## hao-ai-lab/MuxServe

未固定 commit（待固定）。D9 C1：[仓库](https://github.com/hao-ai-lab/MuxServe)中 LLM unit 的放置与自适应 batch 调度；lab9 只做时间份额，没有 MPS 空间复用。

## tyler-griggs/melange-release

未固定 commit（待固定）。D9 C2：[仓库](https://github.com/tyler-griggs/melange-release)中成本感知装箱的 ILP 建模，以及每种 GPU 在给定请求长度与 SLO 下的最大请求率表。

## Mutinifni/splitwise-sim

未固定 commit（待固定）。D9 C3：[仓库](https://github.com/Mutinifni/splitwise-sim)中迭代级集群模拟的事件循环；lab9 没有 P/D 拆分和抢占。

## microsoft/vidur

未固定 commit（待固定）。D9 A2/A3：[仓库](https://github.com/microsoft/vidur)中算子级 profiling 加预测模型；README 写明每个 replica 只有一种设备，不支持异构。

## stanford-cs336/assignment2-systems

固定 commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`。D2/D3/D6：先读 benchmark/profiling 任务与硬件假设，挑相关子任务，不需要移植整套作业。2026 版 §3 “Single-GPU Memory”与 lab6 B 高度重合，可当并行阅读。

- [cs336_assignment2_systems.pdf](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)

## stanford-cs336/assignment5-alignment

固定 commit：`c2734a26308710949fe13226960a1e8cece94b7e`。D8：用 GRPO 单元测试理解数学语义；原作业按 B200 小时计预算，4060 上不能复现训练。

- [tests/test_grpo.py](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/tests/test_grpo.py)
- [cs336_spring2026_assignment5_alignment.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_alignment.pdf)

## 从上游函数回到本项目实验

- KIVI 的 `AsymGroupedQuantizer.forward` 与 `AsymGroupedQuantizerByChannel.forward`：看 reshape 和归约轴。lab1 额外规定了常量组与不整除尾组的行为，不能假设上游所有 helper 都支持任意尺寸。
- llama.cpp 的 `vec_dot_fattn_vec_KQ_q4_0` 与 vLLM 的 `_fwd_grouped_kernel_stage1`：对照 lab1 R 的融合 kernel，看 int4 解包放在点积内部还是外部，split 之间怎么合并。
- Marlin 的 `dequant` 与 FasterTransformer 的交错转换：对照 lab2 B 的 `bfe` 解包，数清每个 int4 权重花几条指令。
- vLLM 的 `_compute_bs_to_padded_graph_size` 与 SGLang 的 `_pad_to_bucket`：对照 lab3 R1，两者都是“向上补到最近的捕获尺寸”，区别在尺寸列表怎么生成。
- flexflow-serve 的 `add_tokens_toward_slo`：对照 lab4 B5/B6 的共享预算，看它按什么顺序给请求加 token。
- SGLang 的 `match_prefix`、`evict`、`inc_lock_ref`，vLLM 的 `hash_block_tokens`：对照 lab5 的 match/insert/acquire/release。比较时，不变量比容器类型更重要。
- PEFT 的 `Linear.forward`/`merge` 与 unsloth 的 `LoRA_MLP`：对照 lab6 A 和 R2，数清每种写法在反传时保存了哪些张量。
- MegaBlocks 的 `indices_and_bins` 与 MoE-Infinity 的 `ArcherPrefetchHandle`：前者是计算布局，后者是数据移动；lab7 的历史预测器只是一种可替换的教学策略。
- DARTS 的 `RedundancyCounter` 与 `RayTrainerDarts`：先确认冗余采样在哪里截断、被 abort 的样本是否还计入统计，再对照 lab8 B4 与 R1 的 `KeepFirstN`。
- Helix 的 `model_manager` profiling 表：对照 lab9 A 的代价模型和 R 的 $\eta$ 标定，注意它是查表，lab9 是拟合。
