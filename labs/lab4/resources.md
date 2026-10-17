# Lab 4 公开资料与整合说明

核验日期：2026-10-17。以下为论文、作者代码或官方文档。实验代码独立编写，不复制上游实现。读源码应固定commit；链接可访问不代表已复现其结果。

| 来源                                                                                                                                                    | 阅读位置                                                                 | 本lab如何借鉴                                          | 明确省略                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------- |
| [AdaServe论文](https://arxiv.org/abs/2501.12162)                                                                                                        | SLO约束与推测资源分配                                                    | B3/B5区别局部与共享预算，C区分round与请求指标          | token tree、真实batch、完整论文优化器             |
| [AdaServe artifact README固定版](https://github.com/zikun-li/AdaServe-Artifact-Evaluation/blob/dbae1c30319f70c73b0f512ba2e5710003382786/README.md)      | 运行环境、各图实验                                                       | 把成本/SLO/到达率分开改变                              | 其完整实验是多A100配置，8GB显存不作为完整复现入口 |
| [负载波动实验脚本](https://github.com/zikun-li/AdaServe-Artifact-Evaluation/blob/dbae1c30319f70c73b0f512ba2e5710003382786/exps/fig14/run_qwen_fluc.sh)  | arrival trace、fixed speculative baseline                                | C的三种到达间隔和固定k基线                             | 不采用上游模型权重与设备参数                      |
| [对应子模块request_manager.cc](https://github.com/flexflow/flexflow-serve/blob/7f07ba2bf98c505c86dd4c1bacb6384454fa37a5/src/runtime/request_manager.cc) | `prune_token_tree`、`add_tokens_toward_slo`、`add_tokens_toward_goodput` | 阅读“先满足约束、再使用剩余预算”的边界；对比B5贪心反例 | B5并非上游算法；C单服务台不调用B5                 |
| [SpecInfer作者论文](https://arxiv.org/abs/2305.09781)                                                                                                   | 树验证的执行成本                                                         | 解释verify[k]为何需要独立校准                          | 不重新实现GPU树注意力                             |
| [CacheBench官方实验配置方法](https://cachelib.org/docs/Cache_Library_User_Guides/Configuring_cachebench_parameters/)                                    | workload/replay与参数控制                                                | 借鉴显式工作负载、对照参数、统计口径的组织方法         | 不是CacheBench实现；模拟延迟不能等同硬件计时      |

## 带问题阅读，而非整仓浏览

1. artifact根仓库只包实验脚本，查看[子模块声明](https://github.com/zikun-li/AdaServe-Artifact-Evaluation/blob/dbae1c30319f70c73b0f512ba2e5710003382786/.gitmodules)并固定上表子模块commit。记录两个SHA。
2. 在 `request_manager.cc` 中画request约束→候选预算→token选择→观测统计的流程；指出与B3/B5/C3的三个不同点。
3. 写一张“成本表来自哪里”的卡片：当前A是合成数据，下一步可以用何种真实计时替换，哪些条件必须固定？
4. 预估真实artifact资源，仅将读取/小机制重写计入24h，完整多GPU运行不计入本lab。

公开大学课程补充入口见[统一课程与lab资料](../../course/readings/courses-and-labs.md)。本lab主源选择作者代码，因为前沿调度机制与现有foundations推测正确性实验衔接更直接。
