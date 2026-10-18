# Lab 4 评分与验收

[任务书](README.md) · [测试源码](tests/exercises.rs) · [R 阶段检查](real/test_real.py) · [参考导读](../../solutions/lab4/README.md)

满分 100。正确性分按独立测试组发放，报告分人工检查，不能只看进程是否返回 0。伪造 GPU 耗时、把 modeled 结果写成 measured、跳过未实现的 stub，对应项一律不得分。

| 项                           | 分值 | 自动或人工证据                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------- | ---: | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1/A2 提交分布与矩           |    6 | `a_distribution_oracle`（$2^k$ 枚举）、`a_moments_and_quantiles`（手算均值、方差）                                                                                                                                                                                                                                                                                                                                                                          |
| A3–A5 分位数、插值、校准     |    6 | `a_moments_and_quantiles`、`a_profile_interpolation`、`a_calibration_grouping_and_missing`（缺 $k$ 报错）                                                                                                                                                                                                                                                                                                                                                   |
| B1/B2 删失计数与 Wilson 下界 |    8 | `b_censored_observation_and_decay`、`b_wilson_numeric_and_ranking`（二分法数值 oracle）                                                                                                                                                                                                                                                                                                                                                                     |
| B3/B4 单轮选择与漂移检测     |    6 | `b_choice_exhaustive_and_infeasible`（900 例穷举 + 并列）、`b_drift_exact_events`（逐步触发位置）                                                                                                                                                                                                                                                                                                                                                           |
| B5/B6 共享预算               |    8 | `b_greedy_trace_feasible_and_bounded`（契约、可行、不超最优、极大）、`b_exact_matches_exhaustive`（穷举最优与并列规则）                                                                                                                                                                                                                                                                                                                                     |
| C1/C2 提交与到达             |    4 | `c_commit_cap_and_timestamps`、`c_arrivals_once`                                                                                                                                                                                                                                                                                                                                                                                                            |
| C3 事件引擎                  |   12 | `c_hand_timeline`、`c_infeasible_counts_all_policies`、`c_mean_policy_shadow_replay`、`c_drift_reset_replay`、`c_per_position_beats_pooled_iid`、`c_conservation_empty_and_overload`                                                                                                                                                                                                                                                                        |
| C4 指标                      |    4 | `c_metrics_independent`（手算 p95 TTFT/E2E/TPOT、goodput、利用率）                                                                                                                                                                                                                                                                                                                                                                                          |
| R 真实系统                   |   26 | R1 6 分：`real/test_real.py` 的 R1 组，含 `--device cuda` 的真实循环比对；R2 2 分：假时钟检查；R3 10 分：`r_runs_bruteforce`、`r_trace_env_censors`、`r_fixed_schedule_matches_engine`、`r_reference_rounds_match_measured`、`r_reference_trace_replay_conservation`；R4 3 分：`r_ucb_upper_bisection`、`r_ucb_policy_behaviour`；自采 trace 与第 R 实验第 2–5 项 5 分（原始 JSONL、成本表、测量方法：warmup 次数、重复次数、cudaEvent 区间、中位数与 IQR） |
| 实验与报告                   |   20 | A/B/C 的图；贪心反例与 DP 对照；过载反例；R 阶段实测与模型的误差分析；至少一个负结果；与 baseline（最好固定 $k$、`arm_ucb`、`hf_heuristic`）同口径对比                                                                                                                                                                                                                                                                                                      |
| 环境证据                     |    — | 附 `bash tools/check_env.sh --lab 4` 的完整输出（Windows 原生附 `check_env.ps1 -Lab 4`）；缺失则 R 阶段不给分                                                                                                                                                                                                                                                                                                                                               |

## 必交命令

```bash
bash tools/check_env.sh --lab 4
python tools/run.py --lab 4 --impl student --mode test --part A
python tools/run.py --lab 4 --impl student --mode test --part B
python tools/run.py --lab 4 --impl student --mode test --part C
python tools/run.py --lab 4 --impl student --mode test --part R --device cuda   # 没有 GPU 时去掉 --device cuda，R 实验第 1 项记 0 分
python tools/run.py --lab 4 --impl student --mode bench --part all
python tools/run.py --lab 4 --impl student --mode bench --part R --device cuda
```

保留 `results/` 下每条命令的 RECORD JSON 和 `results/lab4-real/` 下的 trace。没有 GPU 的同学可以只用参考 trace 完成 R3/R4 和第 3–5 项，R 实验的“自采 trace”部分不得分，其余照常评分。

## 必查错误

- 把未走到的后缀记成失败，或把 vLLM 的 per-position 接受率 $P(A\ge j)$ 当成条件接受率 $a_j$。
- 校准缺列时填 0；中位数用插值而不是 nearest-rank，导致 Rust 与 Python 两端对不上。
- 只有返回 `None` 的策略计 infeasible，`Fixed(k)` 超预算却不计。
- 回放直接返回 $L(e)$ 而不是 $\min(L(e),k)$，等于偷看删失数据；固定 $k$ 回放的下一轮起点差一位。
- 计时样本包含 warmup；把 HF 默认带置信度阈值的 assisted generation 当成固定 $k$。
- 把模拟或回放的数字标成 measured，或者只报告最快的 seed。
- 为了 goodput 删掉迟到的请求、改截止时间定义、让控制器读 `Request::accept`。

浮点断言只在解析值上用 $10^{-12}$ 量级的容差，Monte Carlo 检查用 4 倍标准差，真实回放用“分叉前至少 95% 的轮次相同”。不设置与硬件相关的加速阈值；R 阶段测出“推测不划算”是合法结果，照实报告不扣分。挑战不计入 100 分。
