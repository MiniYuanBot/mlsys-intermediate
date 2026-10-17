# 可视化与作者技术资料

[课程索引](../index.md) · [总资源表](resources.md)

按当前正在读的小节选择一份材料即可。先写出自己预计会看到的关系，再看图或调整参数，最后用一两句话解释观察结果。各讲义已把读图任务和随堂自测放在相关小节旁；本页用于查找入口，不要求另花时间通读所有外链。

下面区分三类材料：教学图解释机制，公式演示展示给定假设的后果，profiler 则分析实际采集的运行记录。作者文章中的性能图有自己的硬件、输入和实现条件，不能当作本项目的测量结果。

## 本地交互演示

直接用浏览器打开 [visuals/index.html](../visuals/index.html)，不需要服务端、外网或安装包。可从下面的链接定位到对应图：

| 面板                                               | 一次具体操作                                                                | 留下的记录                                             |
| -------------------------------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------ |
| [KV 字节账](../visuals/index.html#kvPlot)，D1      | 固定 context=4096、4bit，将 group 从 64 改为 32；分别看 payload 与 metadata | 写出两种 group 大小的总字节，并指出面板没有测量化误差  |
| [推测成本](../visuals/index.html#specPlot)，D4     | 固定接受率与成本，把每轮上界降到 0.5，再恢复为 3.0                          | 记录何时没有可行 k，解释为何不能忽略约束只看最高柱     |
| [编译摊销](../visuals/index.html#compilePlot)，D3  | 固定 C=800ms，对比调用 20000 与 20100 次                                    | 区分打平与严格回本，并列出多 shape 场景还要计入的成本  |
| [保存与重计算](../visuals/index.html#keepPlot)，D6 | 对比预算 0、10、20 时的保留集合                                             | 记下剩余重计算成本，解释为什么这个图不是显存峰值预测器 |

所有面板都基于已标明假设的公式模型，不展示本机实测结果。先用面板形成预测，再到 lab 中验证。

## 八讲的读图入口

以下入口于 2026-10-17 核对。指定图和观察任务已写入各讲；这里保留主题、来源和需要带回笔记的结论。

| 讲义                                                        | 图解或可视化入口                                                                                                                                                                                                   | 观察任务与适用范围                                                                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| [D1 KV 缓存](../modules/01-kv-cache.md)                     | [vLLM PagedAttention 动画](https://vllm.ai/blog/2023-06-20-vllm)、[HF KV 量化图解](https://huggingface.co/blog/kv-cache-quantization)                                                                              | 分别标出分页空位与量化字节开销；把机制示意、作者质量曲线和本地预算公式分开。                                         |
| [D2 低比特算子](../modules/02-low-bit-kernels.md)           | [NVIDIA warp shuffle 图](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)、[AWQ 作者论文图 2–4](https://arxiv.org/html/2306.00978v5)                                                           | 画归约伙伴与有效 lane；再区分通道缩放、roofline 和打包布局。论文中的 ARM NEON 布局不能直接当作本 lab 的 CUDA 格式。  |
| [D3 动态编译](../modules/03-dynamic-compilation.md)         | [PyTorch 编译 profiling 图](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html)、[Perfetto UI 操作图解](https://perfetto.dev/docs/visualization/perfetto-ui) | 找首次编译、graph break 与 CPU/GPU 事件之间的关联。PyTorch 链接为 main 文档，执行命令前按安装版本核对。              |
| [D4 推测解码](../modules/04-speculative-slo.md)             | [原论文 Figure 1](https://arxiv.org/pdf/2211.17192)、[HF assisted generation 图解](https://huggingface.co/blog/assisted-generation)                                                                                | 标出草稿、验证、拒绝后的处理；对照采样设置与延迟图，区分接受收益和验证成本。                                         |
| [D5 前缀缓存](../modules/05-prefix-cache.md)                | [SGLang RadixAttention 图 3/4](https://www.lmsys.org/blog/2024-01-17-sglang/)、[vLLM APC 状态图](https://docs.vllm.ai/en/latest/design/prefix_caching/)                                                            | 沿共享前缀跟踪引用与释放，再按 Time 1–6 推演 block 状态；树和块表的外形不同，不妨碍比较生命周期约束。                |
| [D6 低显存训练](../modules/06-memory-efficient-training.md) | [PyTorch checkpoint 图解](https://pytorch.org/blog/activation-checkpointing-techniques/)、[内存时间线导读](https://pytorch.org/blog/understanding-gpu-memory-1/)                                                   | 在前向/反向图上标记保存与重算，再在 snapshot 时间线找存活区间；示意图不提供自己的峰值显存。                          |
| [D7 MoE 卸载](../modules/07-moe-offload.md)                 | [Switch Transformers 图 2](https://jmlr.org/papers/volume23/21-0998/21-0998.pdf#page=4)、[NVIDIA 复制与计算时间线](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)                          | 追踪 token→router→专家；画出双槽资源依赖。Switch 的路由归一化与本 lab 不同，历史 GPU 的时间线也不能外推到 RTX 4060。 |
| [D8 rollout 调度](../modules/08-rollout-scheduling.md)      | [verl 执行结构图](https://verl.readthedocs.io/en/latest/hybrid_flow.html#overall-execution-diagram)、[PPO Figure 1](https://arxiv.org/pdf/1707.06347#page=3)                                                       | 沿 rollout/更新路径标出 group 与版本；对正负优势分别找裁剪目标的平坦区。结构图不证明重叠，目标函数图也不是学习曲线。 |

## 通用交互与分析工具

| 资源                                                                                                                                                   | 开放形式                         | 推荐操作                                                                              | 对应模块              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------------------- | --------------------- |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) / [源码](https://github.com/poloclub/transformer-explainer)                 | 开源交互                         | 观察 token 与 attention 数据流，再标出 KV 的位置                                      | D1/D5；只用于机制直觉 |
| [LLM Visualization](https://bbycroft.net/llm) / [源码](https://github.com/bbycroft/llm-viz)                                                            | 开源 3D 交互                     | 沿矩阵运算与维度看一次前向                                                            | D2/D3                 |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles)                                                                                                    | 开源 notebook 与可视化           | 画 lane→地址映射、检查 reduction/mask                                                 | D2/D3                 |
| [Triton-Puzzles](https://github.com/gpu-mode/Triton-Puzzles)                                                                                           | 开源练习                         | 对比 block/mask 与 CUDA 编程模型                                                      | D1/D2/D3              |
| [Perfetto UI](https://ui.perfetto.dev/) / [操作文档](https://perfetto.dev/docs/visualization/perfetto-ui) / [源码](https://github.com/google/perfetto) | 开源 trace viewer                | 打开示例或兼容 trace，用 W/S 缩放、A/D 平移，点击事件查看持续时间                     | D3/D4/D6/D7/D8        |
| [PyTorch Memory Visualizer](https://pytorch.org/memory_viz) / [snapshot 文档](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)             | 官方浏览器工具，PyTorch 开源生态 | 在 Active Memory Timeline 找到跨阶段存活的分配，再到 Allocator State History 查其来源 | D6                    |
| [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)                                                                          | 厂商工具，**非开源**             | 检查 H2D/compute 是否真正重叠                                                         | D2/D3/D7              |
| [SGLang 作者博客](https://lmsys.org/blog/2024-01-17-sglang/)                                                                                           | 图解文章                         | 手动标注 radix prefix 的共享与释放                                                    | D5                    |
| [PyTorch checkpoint 博客](https://pytorch.org/blog/activation-checkpointing-techniques/)                                                               | 官方技术文章与示意图             | 比较不同策略的内存—速度空间                                                           | D6                    |
| [FlashInfer](https://flashinfer.ai/)                                                                                                                   | 作者技术博客与项目入口           | 结合源码阅读 layout 与 attention kernel 设计                                          | D1/D2                 |

Perfetto 需要兼容的事件 trace，Memory Visualizer 需要内存 snapshot；`tools/run.py` 保存的 JSON 是测试与 benchmark 执行记录，不能直接当作这两种输入。没有自己的 trace 时，可以先用工具示例练习读图；要报告本机行为，仍需采集自己的数据。

## 让可视化服务于实验

每张图应回答一个明确的问题：用 trace 图定位 launch gap，用内存快照分析生命周期，用 Pareto 图比较质量与成本。附上 profiler 截图时，还需提供时间区间、计时事件、输入大小和原始数值。

实验图使用 [plot.py](../../labs/tools/plot.py) 导出 PNG；误差条是样本标准差，不是置信区间。用于论文的统计分析，应按独立请求、trace seed 或训练 seed 设计，不能将一次循环中的每个 token 都当作独立样本。
