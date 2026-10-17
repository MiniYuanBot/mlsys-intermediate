# Lab 5 评分与验收

[任务书](README.md) · [独立测试](tests/exercises.rs) · [参考导读](../../solutions/lab5/README.md)

总分100。A/B/C三个阶段都应有实际实现；源码检查与输出检查结合。参考实现已按同一测试运行；测试不是全输入形式证明，报告中的新策略仍需新增反例。

| 部分 | 验收                                                                 | 分值 |
| ---- | -------------------------------------------------------------------- | ---: |
| A    | A1 namespace/完整块6；A2缺父即停4；A3叶节点约束6；A4坏拓扑6；A5统计3 |   25 |
| B    | B1失败全量回滚8；B2/B3双lease7；B4namespace失效5；B5缩容事务5        |   25 |
| C    | C1有界准入6；C2年龄与LPM6；C3并发事件/守恒/隔离12；C4加权指标6       |   30 |
| 测量 | A12/B8/C36行原始CSV、容量与租户分组、三幅图                          |   10 |
| 推理 | in-flight miss反例、插入失败≠准入拒绝、负结果、来源边界/实耗时       |   10 |

## TODO与测试

| TODO  | 对应测试                                                          |
| ----- | ----------------------------------------------------------------- |
| A1    | `a_keys_namespace_and_tail`                                       |
| A2    | `a_match_stops_at_gap`                                            |
| A3    | `a_leaf_victims`                                                  |
| A4    | `a_invariants_detect_corruption`                                  |
| A5    | `a_accounting_multiple_leases`                                    |
| B1    | `b_transaction_rollback_and_leaf_order`                           |
| B2/B3 | `b_shared_leases`、`b_release_preserves_reuse`                    |
| B4    | `b_namespace_invalidation`                                        |
| B5    | `b_resize_transaction_with_pins`                                  |
| C1    | `c_admission_pollution_and_bound`                                 |
| C2    | `c_scheduler_age_and_prefix`                                      |
| C3    | `c_replay_hand_timeline`、`c_conservation_isolation_and_capacity` |
| C4    | `c_metrics_weighting_and_empty`                                   |

A/B/C各5个测试，共15个。关键底线：不能淘汰pin块；失败不能留下半个路径；未完成prefill不能产生可复用KV；所有请求最终完成且所有lease释放。完成正确但TwoHit/LPM更慢仍能得满分，需解释并保留结果。不把模拟ms计为CPU/GPU实测，不设置性能倍数门槛。

验收先运行分阶段测试，再all和bench。看cache.validate、请求ID集合、统计分母；保留namespace隔离规则。开放挑战另行评价，不挤占25h必修预算。
