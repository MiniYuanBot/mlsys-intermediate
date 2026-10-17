# Lab 2 资料、具体借鉴与边界

检索和阅读日期：2026-10-17。以下是实际检查过的主源；实现为本课自行编写的缩小版本，没有复制整套上游框架。阅读时间已包含在任务预算中。

| 主源                                                                                                | 精读位置与问题                                                                         | 本课整合                                                      | 未移植/为何                                                                                 |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [AWQ auto_scale.py](https://github.com/mit-han-lab/llm-awq/blob/main/awq/quantize/auto_scale.py)    | `_search_module_scale` 的 activation statistics、ratio 网格、每轮恢复原参数、输出 loss | C1–C3 的通道尺度和校准输出误差搜索；C4 冻结结果应用 held-out  | 完整 Transformer block hooks、层间 scale folding、model zoo；在 8GB 上先验证机制            |
| [AWQ 论文](https://arxiv.org/abs/2306.00978)                                                        | 区分 salient channels 与直接保留高精度权重                                             | 校准目标讨论、alpha=0 对照                                    | 不把本课对称 W4A32 称为论文完整 W4A16 复现                                                  |
| [Marlin CUDA kernel](https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda_kernel.cu) | `dequant`、`lop3`、`scale`、token 维度复用、异步搬运                                   | B3 byte 粒度解码、B4 解码后跨 token 复用、B5 并行度与归约折中 | Tensor Core fragment permutation、cp.async 多级流水、跨 block 全局锁；当前布局/dtype 不兼容 |
| [Stanford CS336 systems 作业](https://github.com/stanford-cs336/assignment2-systems)                | 作业 handout 中 benchmark、profiling、正确性与 kernel 优化的递进安排                   | 独立 oracle→不规则 shapes→配对计时→解释负结果                 | 多卡分布式作业与完整 attention 重复 foundations 内容，不作为本 lab 必修                     |
| [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)                 | `bfe` 位域提取、寄存器约束与线程执行语义                                               | B1 的实际 inline PTX                                          | 不使用目标设备不支持的 Hopper/Blackwell 指令                                                |
| [MIT Efficient ML 2024](https://hanlab.mit.edu/courses/2024-fall-65940)                             | 量化误差、数据分布、校准与性能并看                                                     | A 的格式/误差证书与 C 的数据依赖校准区分                      | 完整课程训练流程另行阅读；不要求重复已完成 foundations                                      |

建议顺序：A 前看 MIT 量化材料与 nibble 手算；B 完成后用 45min 追踪 Marlin `dequant→scale→mma`；C 前用 45min 读 AWQ search。阅读笔记须写出一处本课与上游的实质差异和一个可验证实验，不能仅摘录摘要。
