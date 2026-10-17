# Lab 8 资源与设计来源

核查日：2026-10-17。资源选择以课程官方仓库与论文作者代码为主；本lab重新实现小型机制，不直接运行大型分布式依赖。

| 主源                           | 已读文件/符号                                                                                                                                                                                                                                                                                                                                                                                        | 本lab保留的设计                                              | 简化与不声称复现的部分                                                                                    |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| Stanford CS336 Alignment 2026  | [tests/adapters.py @c2734a2](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/tests/adapters.py)，`run_compute_group_normalized_rewards`、`run_compute_policy_gradient_loss`；[tests/test_grpo.py](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/tests/test_grpo.py)                    | A4 group优势；A6保存behavior概率；C3显式梯度、独立测试与对照 | Rust两动作policy，无tokenizer/LLM、KL/entropy/value model；人口方差为本lab约定                            |
| verl fully asynchronous policy | [fully_async_rollouter.py @18cfb74](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/verl/experimental/fully_async_policy/fully_async_rollouter.py)，`get_sample_collection_ratio`、`wait_for_enough_samples`、message queue调用；[fully_async.md](https://github.com/verl-project/verl/blob/18cfb74852a053a27dcf301d3318a093be81b628/docs/advance/fully_async.md) | C1/C2有界队列、C4生产/训练反馈与版本准入                     | 单进程确定事件循环；lab的整数max_lag不是verl staleness_threshold配置的等价映射；无Ray actor或参数网络传输 |
| DARTS                          | [README @e07825a](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/README.md)；[ray_trainer_darts.py](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/darts/recipe/darts/ray_trainer_darts.py)，`dynamic_batch_reorder`、`RayTrainerDarts`、streaming生成路径                                                                         | 用历史预测研究排程，C4同时观察分布/学习变化；B1显式分段语义  | 没有实现冗余轨迹选择或主动长度分布塑形；不是DARTS完整复现                                                 |
| 官方课程说明                   | [CS336课程](https://cs336.stanford.edu/)、[Assignment 5仓库](https://github.com/stanford-cs336/assignment5-alignment)                                                                                                                                                                                                                                                                                | adapter契约→实现→测试→学习实验的递进结构                     | 无需先跑原课程的大模型实验即可完成本lab                                                                   |
| 论文                           | [HybridFlow](https://arxiv.org/abs/2409.19256)、[DARTS](https://arxiv.org/abs/2605.30859)                                                                                                                                                                                                                                                                                                            | 区分系统调度、并行执行、数据分布改变                         | token事件效率不是论文加速数值                                                                             |

## 带问题读源码（1h）

1. CS336 adapter把reward、group归一化和policy loss拆开。对照本lab的`sample/advantages/update`，逐项写出shape与归一化约定。
2. 检查CS336的off-policy/clipped测试需要保存什么。解释“将behavior概率重算成当前概率”为什么会破坏ratio。
3. 在verl里区分queued sample数、trainer要求的样本数与版本新鲜度；不要把queue大小阈值直接解释成完整group验证。本lab额外显式维护group。
4. DARTS README说明主动改变rollout分布。找到trainer里的动态重复/选择位置，并回答它与本labB3只改变执行顺序的差别。

## 可视化与拓展

- [Perfetto](https://ui.perfetto.dev/) 可读取扩展后的B3合成事件trace。明确标注token units。
- C benchmark直接输出真实policy参数产生的expected reward；用[绘图工具](../tools/plot.py)或自己的脚本画曲线，label写contextual bandit，不写LLM准确率。
- [CS336官方作业PDF](https://github.com/stanford-cs336/assignment5-alignment/blob/c2734a26308710949fe13226960a1e8cece94b7e/cs336_spring2026_assignment5_alignment.pdf)是后续将小闭环迁移到语言模型的完整课程任务入口。

资料汇总：[实验规范](../../course/guides/experiments.md)、[前沿代码索引](../../course/readings/code-map.md)、上表固定版本与课程入口。
