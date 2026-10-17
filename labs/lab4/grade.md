# Lab 4 评分与验收

[任务书](README.md) · [测试源码](tests/exercises.rs) · [参考导读](../../solutions/lab4/README.md)

总分100，必须分别提交A/B/C代码。下列测试已在参考overlay实际执行；正确性分按独立测试组发放，报告分需人工检查，不能仅以进程返回0判满分。任何伪造GPU耗时或把未实现stub跳过的结果均不能算对应项完成。

| 部分 | TODO与验收                                                               | 分值 |
| ---- | ------------------------------------------------------------------------ | ---: |
| A    | A1分布枚举8；A2/A3矩及分位数5；A4插值6；A5校准缺列6                      |   25 |
| B    | B1删失/衰减6；B2小样本下界5；B3独立穷举6；B4漂移4；B5共享预算及贪心反例4 |   25 |
| C    | C1提交守恒5；C2到达一次5；C3手算时间线/空队列/过载12；C4独立指标8        |   30 |
| 实验 | A20/B120/C60行原始数据、三图、配对seed/单位/真实边界                     |   10 |
| 分析 | round≠请求SLO反例、无探索冷启动负结果、固定源码映射、实耗时              |   10 |

## 测试映射

| TODO  | 必过测试                                               |
| ----- | ------------------------------------------------------ |
| A1    | `a_distribution_oracle`                                |
| A2/A3 | `a_moments_and_quantiles`                              |
| A4    | `a_profile_interpolation`                              |
| A5    | `a_calibration_grouping_and_missing`                   |
| B1    | `b_censored_observation_and_decay`                     |
| B2    | `b_lower_bound_and_ranking`                            |
| B3    | `b_choice_exhaustive_and_infeasible`                   |
| B4    | `b_drift_drop_only`                                    |
| B5    | `b_joint_budget_and_nonoptimal_counterexample`         |
| C1    | `c_commit_cap_and_timestamps`                          |
| C2    | `c_arrivals_once`                                      |
| C3    | `c_hand_timeline`、`c_conservation_empty_and_overload` |
| C4    | `c_metrics_independent`、守恒测试                      |

A/B/C分别4/5/5个测试，共14个；测试源在学生侧独立文件。浮点断言只在解析值使用1e-12量级误差，随机仿真不设置硬件相关速度阈值。过载时goodput可以为0；正确报告负结果不扣分。挑战不计入100分，不作为必修完成条件。

验收时先运行各阶段，再all，检查CSV行数及请求/token总量守恒。额外参数扫描需保留baseline；不得把失败请求删掉、改变deadline定义、偷看真实alpha来提升分数。
