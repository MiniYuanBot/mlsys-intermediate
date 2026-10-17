# Lab 1 资源与改编说明

检索与代码阅读日期：2026-10-17。这里借鉴公开机制，实验代码为小规模独立教学实现；保留上游链接，不复制大段原实现。

| 资源                                                                                                                                                             | 具体阅读位置                                                                                  | 转化为本 lab 的任务                                                | 简化边界                                                                                                 |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| [KIVI 论文](https://arxiv.org/abs/2402.02750) / [llama_kivi.py](https://github.com/jy-yuan/KIVI/blob/main/models/llama_kivi.py)                                  | `key_states_full` 达到 residual_length 的量化分支；`value_full_length > residual_length` 分支 | A1/A2 K-channel、V-token 的不对称量化；B2 已压缩块与高精度尾部并存 | CPU INT4、单头；本实验两者按 block 同时 flush，原 K/V 条件不同；无原 Triton kernel                       |
| [PQCache 论文](https://arxiv.org/abs/2407.12820) / [cache_manager.py](https://github.com/HugoZHL/PQCache/blob/master/vq_method/retrieval_based/cache_manager.py) | `add_new_token`、`gpu_diff`、`fetch_and_concat_kv_w_cache`                                    | C1–C3 显式 PQ 码本/ADC；C4 query→IDs→精确 KV 收集的完整路径        | 原系统含 CPU/GPU 缓存、块资格、异步传输；实验仅实现数值检索与 CPU 物理 gather，索引 uint16，未复现原吞吐 |
| [Stanford CS336 Assignment 2](https://github.com/stanford-cs336/assignment2-systems)                                                                             | attention 基准、数值对照、性能分析作业结构                                                    | B5 流式 softmax 与密集 oracle 分离，分清准备时间和稳态时间         | 不复制 GPU FlashAttention 作业；本任务使用 C++ CPU 分块 attention                                        |
| [FAISS product quantizer 实现](https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ProductQuantizer.cpp)                                              | 子量化器训练、code 生成和 inner-product lookup                                                | C1 Lloyd 子空间训练、C2 冻结编码、C3 lookup 累加                   | 不依赖 FAISS 求解；省去 SIMD、预计算表优化、索引服务与大规模训练                                         |

建议先读论文图和上述函数，再填写[代码地图](../../course/readings/code-map.md)的机制差异表。先完成[讲义 §1](../../course/modules/01-kv-cache.md)的 KV 字节预算手算，再用实验 CSV 绘制曲线；公式估算与实测结果分别标注。相关图解见[可视化阅读索引](../../course/readings/visualizations.md)。

资源入口可能随上游变化；论文版本、上游固定快照记录见[代码阅读地图](../../course/readings/code-map.md)。本仓库运行证据记录的是本地源码 hash。
