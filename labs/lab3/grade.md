# Lab 3 评分与验收（100 分）

[任务](README.md) · [资料](resources.md) · [参考实现](../../solutions/lab3/README.md)

| 项           | 分值 | 自动或人工证据                                                                                                   |
| ------------ | ---: | ---------------------------------------------------------------------------------------------------------------- |
| A1/A2        |    6 | `pipeline_test.py` A1/A2：按时间切分不泄漏；中位数、MAD、premium 截断、非法 profile                              |
| A3/A4/A5     |    8 | A3 逐区间手算；A4 分数用例对 `Fraction` oracle；A5 按频次加权的 77/88/2                                          |
| B1           |   10 | `dp_exhaustive`：全部划分穷举、随机非线性 profile、两个并列规则                                                  |
| B2–B5        |   10 | B2 比值恰好为 2 的边界和 1–34 全部长度；B3/B4 pin 事务与 LRU 顺序；B5 恰好回本时拒绝                             |
| C1–C4        |   14 | `execution_test.py`：真实 Inductor artifact、两条执行路径、异常时 release、每个 held-out 输出对 FP64 oracle      |
| C5–C7        |   12 | `test.cu`：6 种 blockDim 的精确归约、40 组 FP32 与 32 组 FP16 形状、residual 写回逐位相等                        |
| R1–R4        |   15 | `test_r.py`：选择规则、真实 CUDA graph 回放与 eager 一致、不别名、规划结果对全部子集暴力枚举                     |
| 实验与报告   |   20 | C 三组 trace 五次独立运行；C6/C7 对两 kernel 和 Inductor 的同口径表；R 的 modeled/measured 两列和 regret；负结果 |
| 可复现与工时 |    5 | 环境、缓存状态、seed、命令、RECORD 文件、实际工时                                                                |
| 环境证据     |    — | 附 `bash tools/check_env.sh --lab 3` 的完整输出；缺失则 R 阶段不给分                                             |

## 必交命令

```bash
bash tools/check_env.sh --lab 3
python tools/run.py --lab 3 --impl student --mode test --device cpu --part all
python tools/run.py --lab 3 --impl student --mode test --device cuda --part all
python tools/run.py --lab 3 --impl student --mode bench --device cuda --part all
```

每条命令在 `results/` 下留一份 RECORD JSON，报告里写出路径。只有 CPU 时，C5–C7 和 R 的真实 graph 部分没有验收，要在报告里写明。

## 必查错误

- A4 只在 0 和 1 上测过；TV 漏除以 2 时，分数用例会失败。
- B1 回溯差一位，最后一个上界不再是最大长度；或者容量约束被忽略。
- B2 把比值恰好为 2 的请求送去 fallback，或者补到更小的 bucket。
- B3 淘汰了被 pin 的项，或者插入失败后改变了状态。
- B5 把不等号写反，或者恰好回本时也准入。
- C1 计时不同步，只量到 enqueue 时间。
- C4 只检查路径不检查输出，或者把 `calibration_ms` 算进 `replay_ms`。
- C5 的 shuffle 偏移或 width 写错；C6/C7 漏写 residual；C7 用 FP16 累加平方和。
- R2 返回静态缓冲区的 view，或者没有清零补齐行。
- R4 忽略显存预算，或者把超过最大尺寸的 batch 当成零成本。

性能项看测量方法和分析，不要求 DP 总是胜过网格，也不按报告里最大的加速比打分。漏算校准、捕获、路由或回放成本的分析，对应项不得分。不允许为了通过测试而修改 oracle。
