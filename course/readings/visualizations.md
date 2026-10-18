# 可视化与作者技术资料

[课程索引](../index.md) · [总资源表](resources.md)

按当前正在读的小节挑一份材料即可。先写下自己预计会看到什么关系，再看图或调参数，最后用一两句话解释看到的结果。各讲已把读图任务和随堂自测放在相关小节旁边；本页只是入口索引，不要求另花时间通读所有外链。

每条材料标了类型：

- **教学图解**：解释机制的示意图或动画，不带测量数据。
- **实测图**：作者在自己的硬件、输入和实现上测出的性能或质量曲线，不能当作本项目的测量结果。
- **工具**：能打开自己数据的交互工具或 profiler。
- **公式演示**：给定假设、调参数看后果的计算器或交互页。

以下入口于 2026-10-18 核对；具体观察任务以各讲正文为准。

## D1 KV 缓存（[讲义](../modules/01-kv-cache.md)）

| 资源                                                                                           | 类型     | 指定观察                                                                                                                      |
| ---------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| [HF：KV Caching Explained](https://huggingface.co/blog/not-lain/kv-caching)                    | 教学图解 | 逐步跟踪 cache 怎样随 decode 增长，对比有无 cache 时每步的计算量                                                              |
| [Flash-Decoding 博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)                 | 教学图解 | `parallelization_kv` 动画：KV 先按 split 并行，再用 log-sum-exp 合并；对应 lab1 R2/R3                                         |
| [HAN Lab：StreamingLLM 博客](https://hanlab.mit.edu/blog/streamingllm)                         | 实测图   | 逐出最前面几个 token 后 PPL 跳升；注意力集中在 token 0；对应 B1 为什么要保留 `sink`                                           |
| [Lilian Weng：推理优化](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) | 教学图解 | “Quantization granularity”图（per-tensor / per-token / per-channel）。它讲的是权重和激活，但这是能找到的最接近 K/V 分组轴的图 |
| [kvpress 排行榜](https://huggingface.co/spaces/nvidia/kvpress-leaderboard)                     | 实测图   | RULER 4K 上各种淘汰方法的质量对比；挑战中用它确定基线                                                                         |

## D2 低比特权重 kernel（[讲义](../modules/02-low-bit-kernels.md)）

| 资源                                                                                                                                                             | 类型     | 指定观察                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Marlin README](https://github.com/IST-DASLab/marlin) 的 `assets/peak.png`                                                                                       | 实测图   | 不同 batch 下与其他 4-bit kernel 的比较（A10，$g=128$）；读出加速比接近 4 倍的 batch 范围，与讲义 §1.2 的 $B^\ast$ 对照。硬件不同，不能当作 4060 的结果 |
| [Transformer Engine FP8 入门（release 2.13）](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.13/user-guide/examples/fp8_primer.html) | 教学图解 | Fig.1 的 E4M3/E5M2 位布局；Fig.6/8 的 scaling 方案                                                                                                      |
| [NVIDIA NVFP4 博客](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)                                     | 教学图解 | Fig.2 的两级缩放；Fig.5 的格式对比。4060 没有 FP4 硬件，只读格式                                                                                        |
| [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)                                                    | 教学图解 | 对称与非对称量化、clipping 如何改变误差；对应 lab2 A                                                                                                    |
| [HF：4-bit 与 QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)                                                                                 | 教学图解 | FP8 一节的指数/尾数取舍                                                                                                                                 |
| [Modal GPU Glossary：roofline](https://modal.com/gpu-glossary/perf/roofline-model)                                                                               | 公式演示 | 小 batch GEMV 为什么落在带宽斜线上；用 4060 的带宽和算力自己画一遍                                                                                      |

## D3 动态形状与编译（[讲义](../modules/03-dynamic-compilation.md)）

| 资源                                                                                                                                                                                                     | 类型     | 指定观察                                                                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------ |
| [tlparse](https://github.com/pytorch/tlparse)                                                                                                                                                            | 工具     | 把 `TORCH_TRACE` 日志转成重编译报告，找出哪个 guard 失败触发了重编译                             |
| [vLLM CUDA graph 设计文档](https://docs.vllm.ai/en/latest/design/cuda_graphs/)                                                                                                                           | 教学图解 | piecewise 切分图：attention 留在 graph 外，其余部分按捕获尺寸录制                                |
| [Triton fused softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) / [layer norm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | 教学图解 | 与 lab3 生成的 `build/inductor_*.py` 对照，看 Inductor 选的块大小和归约方式                      |
| [TensorRT-LLM blog20：调 CUDA graph batch size](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog20_Tuning_CUDA_Graph_Batch_Sizes_for_Higher_Output_Throughput.md)       | 实测图   | 捕获尺寸网格疏密与吞吐的权衡表；对应 lab3 R 的规划问题                                           |
| [PyTorch 编译 profiling](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)                                                                       | 工具     | 在 trace 里找首次编译、graph break 与 CPU/GPU 事件的对应；链接为 main 文档，执行前按安装版本核对 |

## D4 推测解码与 SLO（[讲义](../modules/04-speculative-slo.md)）

| 资源                                                                                                                        | 类型     | 指定观察                                                                                                                                                           |
| --------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Speculative Decoding, Visualized（Adaptive ML）](https://www.adaptive-ml.com/post/speculative-decoding-visualized)         | 教学图解 | $\min(p, q)$ 接受规则的动画；讲义 §1                                                                                                                               |
| [Leviathan 等原论文](https://arxiv.org/pdf/2211.17192)                                                                      | 教学图解 | Figure 1 的接受、拒绝与纠正；Figure 3 是不同 $c$ 下最优 $\gamma$ 随 $\alpha$ 的曲线，在图上标出讲义 §2 给的两个 $(\alpha, c)$ 点。图假设 i.i.d. 接受和常数验证成本 |
| [EAGLE 项目页](https://sites.google.com/view/eagle-llm)                                                                     | 教学图解 | Fig.5：链式草稿与树形草稿的区别                                                                                                                                    |
| [Medusa 博客（Together）](https://www.together.ai/blog/medusa)                                                              | 教学图解 | tree attention mask 怎样让一次前向验证多条候选                                                                                                                     |
| [Google Research：Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/) | 教学图解 | 方法演进的时间线，适合读完讲义后回顾                                                                                                                               |

## D5 前缀缓存（[讲义](../modules/05-prefix-cache.md)）

| 资源                                                                                                        | 类型     | 指定观察                                                       |
| ----------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------- |
| [vLLM APC 设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)                                  | 教学图解 | 按 Time 1–6 推演 block 状态，再看 free queue 图；对应 A1 与 A4 |
| [SGLang 博客（2024-01-17）](https://www.lmsys.org/blog/2024-01-17-sglang/)                                  | 教学图解 | Fig.3/4：radix tree 上的共享前缀、引用与淘汰                   |
| [FlashInfer cascade inference 博客](https://flashinfer.ai/2024/02/02/cascade-inference.html)                | 教学图解 | 共享前缀只读一次的两级 attention                               |
| [LMCache 架构文档](https://docs.lmcache.ai/developer_guide/architecture.html)                               | 教学图解 | KV 在 GPU、CPU、磁盘之间的分层与传输                           |
| [Mooncake FAST25-release README](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md) | 实测图   | trace 字段与分布；lab5 R2 读取同一 schema                      |

## D6 低显存训练（[讲义](../modules/06-memory-efficient-training.md)）

| 资源                                                                                           | 类型     | 指定观察                                                          |
| ---------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------- |
| [HF：Visualize and understand GPU memory in PyTorch](https://huggingface.co/blog/train_memory) | 实测图   | 峰值从前向移到 optimizer step 的过程                              |
| [PyTorch：激活 checkpoint 技术](https://pytorch.org/blog/activation-checkpointing-techniques/) | 教学图解 | SAC 保留算子图与 memory budget 扫描图；对应 B4 的预算基线         |
| [PyTorch：Understanding GPU Memory 1](https://pytorch.org/blog/understanding-gpu-memory-1/)    | 教学图解 | snapshot 时间线上每个分配的存活区间                               |
| [memory_viz](https://docs.pytorch.org/memory_viz)                                              | 工具     | 打开自己的 snapshot，在 Active Memory Timeline 找跨阶段存活的分配 |
| [HF Ultra-Scale Playbook](https://nanotron-ultrascale-playbook.static.hf.space/)               | 公式演示 | 显存计算器：调序列长度、重计算和 ZeRO 阶段，看各项字节怎样变化    |

## D7 MoE 卸载（[讲义](../modules/07-moe-offload.md)）

| 资源                                                                                                                           | 类型     | 指定观察                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------- |
| [NVIDIA：How to Overlap Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)   | 实测图   | C1060 那一栏只有一个 copy engine，与 4060 在 WDDM 下 `asyncEngineCount=1` 的情况相同；讲义 §5                 |
| [NVIDIA：How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/) | 实测图   | pageable 内存要先经 pinned 缓冲中转；对照 lab7 R4 实测的 12.6 GB/s（pinned）与 11.6 GB/s（pageable）；讲义 §2 |
| [HF：Mixture of Experts Explained](https://huggingface.co/blog/moe)                                                            | 教学图解 | `01_moe_layer.png` 的层结构、`11_expert_matmuls.png` 的专家矩阵乘；讲义 §1/§3                                 |
| [A Visual Guide to Mixture of Experts](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)      | 教学图解 | router、top-k、加权求和三步；负载均衡与容量；讲义 §1                                                          |

## D8 RL rollout 调度（[讲义](../modules/08-rollout-scheduling.md)）

| 资源                                                                                        | 类型     | 指定观察                                                          |
| ------------------------------------------------------------------------------------------- | -------- | ----------------------------------------------------------------- |
| [PPO 论文 Figure 1](https://arxiv.org/pdf/1707.06347#page=3)                                | 教学图解 | 优势为正和为负时，裁剪目标在哪一段变平                            |
| [HF Deep RL Course Unit 8](https://huggingface.co/learn/deep-rl-course/unit8/visualize)     | 教学图解 | 裁剪的六种情况表；与 lab8 C3/C4 的梯度分支逐行对应                |
| [AReaL 论文](https://arxiv.org/abs/2505.24298)                                              | 教学图解 | 图 3 的生成管理时间线：竖线是训练就绪时刻，叉是被中断的请求       |
| [verl fully_async 文档](https://verl.readthedocs.io/en/latest/advance/fully_async.html)     | 教学图解 | rollouter 与 trainer 解耦后的数据流图；结构图不证明真的发生了重叠 |
| [HF 博客：Keep the Tokens Flowing](https://huggingface.co/blog/async-rl-training-landscape) | 教学图解 | 各框架的陈旧度与异步方式对比表                                    |
| [RLHF Book：Policy Gradients](https://rlhfbook.com/c/11-policy-gradients)                   | 教学图解 | 图 5 PPO 目标随 $\rho$ 的变化；图 7 GRPO                          |

## D9 异构与多模型 serving（[讲义](../modules/09-hetero-serving.md)）

| 资源                                                     | 类型     | 指定观察                                                           |
| -------------------------------------------------------- | -------- | ------------------------------------------------------------------ |
| [VisuAlgo Network Flow](https://visualgo.net/en/maxflow) | 公式演示 | 单步运行 Dinic，看分层图怎样建立、反向边怎样让流退回；对应 lab9 B2 |
| [MuxServe 博客](https://haoailab.com/blogs/muxserve/)    | 教学图解 | 空间复用与时间复用示意图；lab9 C1 只做时间份额                     |

## 通用交互与分析工具

| 资源                                                                                                                                            | 类型                      | 推荐操作                                                                        | 对应模块              |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------- | --------------------- |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) / [源码](https://github.com/poloclub/transformer-explainer)          | 教学图解（开源交互）      | 观察 token 与 attention 数据流，再标出 KV 的位置                                | D1/D5；只用于建立直觉 |
| [LLM Visualization](https://bbycroft.net/llm) / [源码](https://github.com/bbycroft/llm-viz)                                                     | 教学图解（开源 3D 交互）  | 沿矩阵运算和维度看一次前向                                                      | D2/D3                 |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles)                                                                                             | 工具（开源 notebook）     | 画 lane 到地址的映射，检查 reduction 与 mask                                    | D2/D3                 |
| [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles)                                                                                    | 工具（开源练习）          | 对比 block/mask 与 CUDA 编程模型                                                | D1/D2/D3              |
| [Perfetto UI](https://ui.perfetto.dev/) / [操作文档](https://perfetto.dev/docs/visualization/perfetto-ui)                                       | 工具（开源 trace viewer） | 打开兼容 trace，W/S 缩放、A/D 平移，点击事件看持续时间                          | D3/D4/D6/D7/D8        |
| [PyTorch Memory Visualizer](https://docs.pytorch.org/memory_viz) / [snapshot 文档](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) | 工具（官方浏览器工具）    | Active Memory Timeline 找跨阶段存活的分配，再到 Allocator State History 查来源  | D6                    |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)                                                                   | 工具（厂商，**非开源**）  | 检查 H2D 与计算是否真的重叠；WSL2 下只有 CLI 预览，本课程的验证没有用它出时间线 | D2/D3/D7              |

Perfetto 需要兼容的事件 trace，Memory Visualizer 需要内存 snapshot；`tools/run.py` 保存的 JSON 是测试与 benchmark 的执行记录，不能直接喂给这两个工具。没有自己的 trace 时，可以先用工具自带的示例练习读图；要报告本机行为，仍要采集自己的数据。

## 让可视化服务于实验

每张图回答一个明确的问题：trace 图定位 launch gap，内存快照分析生命周期，Pareto 图比较质量与成本。附 profiler 截图时，同时给出时间区间、计时事件、输入大小和原始数值。

实验图用 [plot.py](../../labs/tools/plot.py) 导出 PNG；误差条是样本标准差，不是置信区间。写论文时的统计分析，要按独立请求、trace seed 或训练 seed 设计样本，不能把一次循环里的每个 token 都当作独立样本。
