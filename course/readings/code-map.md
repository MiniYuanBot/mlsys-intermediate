# 上游代码阅读地图

[资料总表](resources.md) · [阅读资料说明](README.md)

截至 2026-10-17，已通过公开 GitHub API 固定下列 HEAD commit，并核验文件路径。这些链接指向当日工程快照，不一定对应论文的 artifact tag。阅读前需核对论文版本，不能根据 main 分支的变化推断论文原始实验。

可选的本地快照保存在 `results/upstream-snapshots.json`，由 Git 忽略。下文的固定 commit 和源码链接可直接使用。

每项源码阅读完成后，都要记录输入/输出、调用者、状态不变量、对应的论文机制，以及在 8GB 卡上缩小的部分。本项目只引用这些公开代码，没有整段复制其实现。

## jy-yuan/KIVI

固定 commit：`876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`。D1：找到 group/scale 的布局与解码入口，解释追加时的残余窗口行为，不能只看 API。

- [models/utils_quant.py](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/utils_quant.py)
- [quant/csrc/gemv_cuda.cu](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/csrc/gemv_cuda.cu)

## HugoZHL/PQCache

固定 commit：`0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828`。D1：画 cache manager 的构建、候选检索、原始 KV 访问路径。

- [vq_method/retrieval_based/cache_manager.py](https://github.com/HugoZHL/PQCache/blob/0b74e125207dc3f24da3bbaaf84e8a5f1d3b1828/vq_method/retrieval_based/cache_manager.py)

## mit-han-lab/llm-awq

固定 commit：`d6e797a42b9ef7778de8ee2352116e0f48a78d61`。D2：标出 nibble 解码、scale 读取、归约；与本项目 W4A32 的精度不同。

- [awq/kernels/csrc/quantization/gemv_cuda.cu](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/kernels/csrc/quantization/gemv_cuda.cu)

## mit-han-lab/omniserve

固定 commit：`02b2925aa6fa3b92b06316a1524b7f38922cd9c8`。D2/D4：读量化后端与 scheduler 的接口；qserve 旧地址已重定向。

- [omniserve/core/scheduler.py](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/omniserve/core/scheduler.py)

## mirage-project/mirage

固定 commit：`f9eb70c254acefc9f3667b2a973d0dcf25471fce`。D3：追踪图、编译、launch 和 profiling，逐步缩小实验规模，不直接照搬大模型 demo。

- [python/mirage/mpk/persistent_kernel.py](https://github.com/mirage-project/mirage/blob/f9eb70c254acefc9f3667b2a973d0dcf25471fce/python/mirage/mpk/persistent_kernel.py)

## zikun-li/AdaServe-Artifact-Evaluation

固定 commit：`dbae1c30319f70c73b0f512ba2e5710003382786`。D4：先读各图的实验入口与机器配置，再进入其递归子模块。artifact wrapper 的 commit 不等于子模块内部源码版本。

- [README.md](https://github.com/zikun-li/AdaServe-Artifact-Evaluation/blob/dbae1c30319f70c73b0f512ba2e5710003382786/README.md)

## sgl-project/sglang

固定 commit：`456231e2517dd29e78c77d20126c893b3f1a6675`。D5：搜索 match_prefix、evict、inc_lock_ref/dec_lock_ref 或该版本等价函数，追请求生命周期。

- [python/sglang/srt/mem_cache/radix_cache.py](https://github.com/sgl-project/sglang/blob/456231e2517dd29e78c77d20126c893b3f1a6675/python/sglang/srt/mem_cache/radix_cache.py)
- [python/sglang/srt/managers/scheduler.py](https://github.com/sgl-project/sglang/blob/456231e2517dd29e78c77d20126c893b3f1a6675/python/sglang/srt/managers/scheduler.py)

## FlexLLM/artifact

固定 commit：`9c01a2ae5a2c004292add9fe38050776178d3045`。D6：先查 README 的测量目标、子模块与 figure 脚本；不要执行 teardown 删除命令。

- [README.md](https://github.com/FlexLLM/artifact/blob/9c01a2ae5a2c004292add9fe38050776178d3045/README.md)

## EfficientMoE/MoE-Infinity

固定 commit：`96a95527028f434f94a616c487094714ef10436b`。D7：读预取控制与真实 overlap microbench；当前工程版和论文版有差异。

- [core/prefetch/archer_prefetch_handle.cpp](https://github.com/EfficientMoE/MoE-Infinity/blob/96a95527028f434f94a616c487094714ef10436b/core/prefetch/archer_prefetch_handle.cpp)
- [benchmarks/expert_io_microbench/bench_overlap_prefetch.py](https://github.com/EfficientMoE/MoE-Infinity/blob/96a95527028f434f94a616c487094714ef10436b/benchmarks/expert_io_microbench/bench_overlap_prefetch.py)

## PKU-DAIR/DARTS

固定 commit：`e07825aee39612e0659ce7e5a38f8ae02a2250b2`。D8：从 recipe/darts 进入，区分继承的 trainer 与论文新增采样/预算逻辑。

- [darts/recipe/darts/ray_trainer.py](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/darts/recipe/darts/ray_trainer.py)
- [darts/recipe/darts/main_ppo.py](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/darts/recipe/darts/main_ppo.py)

## verl-project/verl

固定 commit：`18cfb74852a053a27dcf301d3318a093be81b628`。D8：跟踪主训练入口的 rollout/训练调用边界，检查版本与 group。

- [verl/trainer/main_ppo.py](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/verl/trainer/main_ppo.py)
- [docs/algo/grpo.md](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/docs/algo/grpo.md)

## stanford-cs336/assignment2-systems

固定 commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`。D2/D3/D6：先读 benchmark/profiling 任务与硬件假设，选取相关子任务，无需移植整套作业。

- [cs336_assignment2_systems.pdf](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)

## stanford-cs336/assignment5-alignment

固定 commit：`c2734a26308710949fe13226960a1e8cece94b7e`。D8：用 GRPO 独立测试理解数学语义，保留原作业对硬件的说明。

- [tests/test_grpo.py](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/tests/test_grpo.py)
- [cs336_spring2026_assignment5_alignment.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_alignment.pdf)

## 从实际函数回到本项目实验

- KIVI 的 `AsymGroupedQuantizer.forward` 与 `AsymGroupedQuantizerByChannel.forward`：观察 reshape/reduction 轴；本项目补充常量组与不整除尾组的显式行为，不能假设上游所有 helper 都支持任意尺寸。
- AWQ 的 `gemv_kernel_g64` / `gemv_kernel_g128`：观察向量加载、zero/scale 与 nibble 循环；本实验采用更易读的 byte/warp 布局，没有照搬其参数与精度路径。
- SGLang 的 `match_prefix`、`evict`、`inc_lock_ref`、`dec_lock_ref`：对照 lab5 的 match/insert/acquire/release，比较时，不变量比容器类型更重要。
- Mirage 的 `PersistentKernel`、`get_compile_command`：对照 lab3 中“成本/编译/launch”的边界；局部 RMSNorm 融合不足以检验整个运行时。
- MoE-Infinity 的 `ArcherPrefetchHandle`：从 policy 配置与 acquire/prefetch 生命周期理解真实异步实现；lab7 的历史预测器只是一种可替换的教学策略。
- DARTS 的 `OneStepOffRayTrainer`、`_async_gen_next_batch`、`sync_rollout_weights`：先确认数据与策略版本如何跨阶段流动，再读采样预算；lab8 的账本与归一化只是正确性起点。
