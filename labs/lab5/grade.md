# Lab 5 评分与验收

[任务书](README.md) · [独立测试](tests/exercises.rs) · [R1 检查](real/test_real.py) · [参考导读](../../solutions/lab5/README.md)

满分 100。测试通过是必要条件，不是全部：报告要说明每个数字的口径，A/B/C 的时间都是 `modeled`，只有 R 阶段本机测得的系数和延迟是 `measured`。不设性能倍数门槛，实现正确但某个策略更慢照样可以拿满分，只要解释清楚并保留结果。

| 项                                   | 分值 | 自动或人工证据                                                                                                                                                                                           |
| ------------------------------------ | ---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1 链式块哈希                        |    5 | `a_hash_chain_extra_keys`：逐块精确哈希、LoRA (name, path) 每块、salt 只在第 0 块、多模态重叠与偏移、链式性质                                                                                            |
| A2–A6 键、匹配、淘汰候选、审计、统计 |   10 | `a_keys_namespace_and_tail`、`a_match_stops_at_gap`、`a_leaf_victims`（含 300 例随机 oracle）、`a_invariants_detect_corruption`（按 `InvariantError` 变体比较）、`a_accounting_multiple_leases`          |
| B1–B5 事务与所有权                   |   20 | 5 个 `b_*` 测试：失败插入的完整快照、自身路径受保护、LRU 经 acquire 刷新、双 lease、引用下溢必 panic、失效与缩容的回滚                                                                                   |
| C1–C2 准入与全命中规则               |    6 | `c_admission_lru_history`（LRU 与 FIFO 可区分、容量参数生效、租户隔离）、`c_full_hit_rule`（两种规则与暴力枚举一致）                                                                                     |
| C3–C6 路由、回放、指标               |   16 | `c_routing_policies`（2000 例随机 oracle）、`c_replay_hand_timeline`、`c_replay_routing_oracle`、`c_replay_conservation_and_isolation`、`c_metrics_and_duplication`                                      |
| R 真实系统                           |   15 | R1：`real/test_real.py` 全部 PASS，提交本机 `prefill-*.json` 与 `cost-*.json`（有 GPU 时）并说明测量方法；R2/R3：`r_mooncake_adapter`、`r_compute_saved`、`r_reference_trace_replay`；R 的 16 行回放 CSV |
| 实验与报告                           |   28 | 见下方“必交证据”；四幅图、负结果、与 baseline 同口径的比较、modeled/measured 分列、实际用时                                                                                                              |
| 环境证据                             |    — | 附 `bash tools/check_env.sh --lab 5` 的完整输出；缺失则 R 阶段不给分                                                                                                                                     |

## 必交命令

在 `labs/` 下运行，提交完整输出和 `results/` 里对应的 RECORD JSON：

```bash
bash tools/check_env.sh --lab 5
python tools/run.py --lab 5 --impl student --mode test --part all
python tools/run.py --lab 5 --impl student --mode bench --part all
python tools/run.py --lab 5 --impl student --mode test --part R --device cuda    # WSL，有 GPU 时
python tools/run.py --lab 5 --impl student --mode bench --part R --device cuda   # WSL，有 GPU 时
```

没有 GPU 的学习者提交前两条与 CPU 版的 `--part R`，在报告里注明 R 使用的是仓库提交的 RTX 4060 参考拟合（设备、驱动、日期写在 `real/data/cost-qwen3-0.6b-rtx4060.json` 的 `meta` 里）。

## 必交证据

1. A benchmark 12 行：Vec 键载荷与链式哈希键大小随 prompt 长度的变化，推导 $O(T^2/B)$。
2. 手算三张表：in-flight miss 时间线（SGLang 完成于 6、vLLM 完成于 7）、32 token / $B=16$ 的全命中重算量、一个 Hybrid 分流的路由序列。
3. 跨租户隔离：举一个违反 `c_replay_conservation_and_isolation` 中命中上界的错误实现，说明它在真实系统里意味着什么（讲义 §9 的时序侧信道）。
4. C 的 16 行与 R 的 16 行 CSV，四幅图：A 键载荷、C 的命中率与 p95 TTFT、R1 的拟合残差、R 的路由对比（token 命中率与省下的 prefill 时间并列）。
5. 至少一个负结果，例如 TwoHit 在无扫描负载上全面变差，或纯亲和性命中最高但 p95 TTFT 变差。
6. R1：本机拟合与参考拟合的逐点差异，$p^*=b/c$ 与 FLOPs 估计的对比，以及 launch 下限 $T_0$ 对小请求收益的影响。

## 必查错误

- 用描述字符串而不是 `InvariantError` 变体判断不变量。
- 只淘汰最冷的节点而不检查它是不是叶子、有没有被 pin、是否在当前插入路径上。
- 失败的插入、失效或缩容改动了 clock 或容量。
- 完整命中按零成本计费，或者 vLLM 规则没有按块取整。
- 准入历史容量写死在实现里，或者历史按首次插入而不是最近观察淘汰。
- 路由器忽略负载阈值，或者把排队请求排除在负载之外。
- 请求在完成前就发布 KV，两个并发未命中的请求互相“命中”。
- 命中率按每请求平均而不是按 token 加权；失衡度只在忙碌副本上求平均。
- R1 把下限附近的点放进回归，或者丢掉二次 attention 项；把 modeled 与 measured 的数字放在同一列比较。
