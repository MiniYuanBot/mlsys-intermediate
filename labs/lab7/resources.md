# Lab 7 资源与设计来源

核查日：2026-10-17。以下已读取主源；本lab实现为独立教学实现，没有复制完整第三方运行时。固定commit便于你复读同一版本。

| 主源                             | 实际阅读入口                                                                                                                                                                                    | 本lab借鉴                                                               | 明确简化                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| MegaBlocks                       | [moe.py @952db33](https://github.com/databricks/megablocks/blob/952db33d6eac334d22c61e47a0d5d41446298784/megablocks/layers/moe.py)，`indices_and_bins`、`permute_and_compute`、`forward_once`   | A2 histogram/prefix-sum布局、A4 gather/scatter逆映射、A5/C2专家分段计算 | CPU counting sort与小CUDA matvec；没有block-sparse GEMM或expert-parallel通信 |
| MoE-Infinity                     | [配置与overlap admission](https://github.com/EfficientMoE/MoE-Infinity/blob/96a95527028f434f94a616c487094714ef10436b/docs/configuration.md)，`overlap_prefetch_policy`、window/inflight bytes   | B1/B6/B7按字节限制预取与显存驻留；C4验证真实载入/消费依赖               | 单串行copy engine事件模型、固定带宽；没有多层、多GPU、磁盘与完整trace预测器  |
| MoE-Infinity industry microbench | [bench_prefetch_issuance.py](https://github.com/EfficientMoE/MoE-Infinity/blob/96a95527028f434f94a616c487094714ef10436b/benchmarks/dflash/bench_prefetch_issuance.py)，`run_issuance_benchmark` | 将enqueue成本与实际传输/执行成本分开；C报告明确端到端计时               | 本lab不比较pybind批量enqueue，也不复现该基准数值                             |
| CUDA官方编程指南                 | [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)                                                                                                                  | C3/C4跨stream事件依赖、异步传输与pinned源buffer生命周期                 | 不使用CUDA Graph、不承诺任何硬件一定重叠                                     |
| 论文                             | [MoE-Infinity](https://arxiv.org/abs/2401.14361)、[MegaBlocks](https://arxiv.org/abs/2211.15841)                                                                                                | 区分参数移动与稀疏计算布局两个研究问题                                  | 这里只重建核心机制，不是全文系统复现                                         |

## 阅读动作（包含于26h）

1. 在MegaBlocks的`indices_and_bins`旁写出本地A2对应的count/prefix/scatter三步；解释其GPU排序与本labCPU排序的差异。
2. 在`permute_and_compute`找到gather→MLP→scatter，对照C2。检查route权重是在哪一步应用的，避免up与down各乘一次权重。
3. 阅读MoE-Infinity配置中window byte budget与inflight byte budget的区别。本labB6限制单次选择字节，B1控制驻留预算；没有实现单独的可取消native队列上限，写出这会遗漏哪种拥塞。
4. 阅读issuance microbench，指出“提交更快”和“传输完成更快”为何不同。C4的wall time比它覆盖更多阶段，不能直接对照其结果。

## 可视化与进一步实验

- [Nsight Systems文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：真实H2D/kernel时间线。它是厂商工具，不称作开源软件。
- [Perfetto](https://ui.perfetto.dev/)：挑战项把B7事件导成trace JSON后查看；不得把模拟trace标签改成CUDA。
- [本课程资源库](../../course/readings/visualizations.md)：已有可视化与工具边界。

资源汇总：以上链接以及[项目源码版本对照](../../course/readings/code-map.md)。如果继续复现论文，先固定模型、全部依赖/submodule、设备、数据与baseline，再沿用这里的小型契约测试作为排错入口。
