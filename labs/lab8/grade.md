# Lab 8 验收与评分（100 分）

[任务书](README.md) · [参考答案导读](../../solutions/lab8/README.md) · [来源](resources.md)

| 项                          | 分值 | 自动或人工证据                                                                                        |
| --------------------------- | ---: | ----------------------------------------------------------------------------------------------------- |
| A1/A2 长度历史              |    4 | `a_length_history`：窗口淘汰、类别隔离、上中位数、冷启动                                              |
| A3 list scheduling / LPT    |    5 | `a_assign`：未排序输入下两种模式的精确队列、ID tie-break、60 个随机用例对照测试端调度器               |
| A4–A6 优势、直方图、采样    |    6 | `a_advantage`（两种 std）、`a_histogram`（ceil 秩、闭上界）、`a_sample`（动作 0 的 behavior 概率）    |
| B1/B2 分段账本与完整 batch  |    6 | `b_reassemble`（10 类损坏）、`b_batch_gate`（只训练快组的反例）                                       |
| B3 执行器                   |    7 | `b_scheduler`：工作守恒、可重组、手算时间线、偷队尾                                                   |
| B4 先完成者入选             |    4 | `b_keep_first_n`：并列规则、顺序统计量精确 oracle（$5/3$、$14/27$）                                   |
| C1/C2 有界组队列            |    5 | `c_queue_admission`、`c_queue_drain`：按组计容量、背压、超龄整组驱逐                                  |
| C3/C4 裁剪目标与梯度        |    9 | `c_objective`（独立闭式）、`c_gradient`（对你的目标做有限差分；$\rho\ne1$ 与越界活跃用例）            |
| C5/C6 版本界与解耦比值      |    6 | `c_staleness_bound`（模拟器中最大陈旧度恰为 $\eta$、不死锁）、`c_partial_ratios`（逐 token 的 $w,u$） |
| C7 学习闭环                 |    3 | `c_learning_closed_loop`：3 个种子、精确的单波计数                                                    |
| R1–R3 真实 trace 回放与预测 |   15 | `r_predictor`、`r_replay_synthetic`、`r_replay_reference`；bench R 的三张表                           |
| 实验与报告                  |   30 | 见下                                                                                                  |
| 环境证据                    |    — | 附 `bash tools/check_env.sh --lab 8` 的完整输出；缺失则 R 阶段不给分                                  |

自动测试共 19 个，每项检查打印一行 PASS/FAIL；一个测试里任何一项失败，这个测试的分值全扣，直到修好。

**实验与报告（30 分）**：

- R 阶段必交第 1–3 项，每行同时有吞吐和分布偏移，口径写成 `modeled from measured gen_ms`（10 分）。
- 你自己采的 trace（第 4 项），附头部信息与 `nvidia-smi` 时钟；没有 GPU 时说明原因并只交参考 trace（4 分）。
- A/B/C 的图：预测误差、窃取与 quantum 配对、`max_lag` 0/4 学习曲线与版本时间线（8 分）。
- 一个负结果，区分有观测支持的结论与猜测；固定样本数、固定更新数、固定墙钟三种口径不混用（5 分）。
- TODO 完成表与实际用时（3 分）。
- GUIDE.md 的 TRL 试跑是选做，可以抵扣以上任一项的失分，最多 5 分。

## 必交命令

```bash
bash tools/check_env.sh --lab 8
python tools/run.py --lab 8 --impl student --mode test --part all
python tools/run.py --lab 8 --impl student --mode bench --part all
python tools/run.py --lab 8 --impl student --mode bench --part R --device cuda   # WSL，有 GPU 时
```

## 必查错误

代码审查额外检查以下几点，发现即扣对应 TODO 的分：

- 调度决策读取了 `actual` 或本轮的真实长度（A3、B3、R1）；预测器观测了被 abort 的样本（R1）。
- 用当前策略概率覆盖 `behavior_p`，把 off-policy 差异“消掉”（A6、C3）。
- 梯度在 $\rho$ 越界时一律置 0，或漏乘 $\rho$（C4）；测试之外换一组数值仍应正确。
- 版本界写成 `<` 或多加 1（C5）；partial rollout 用整条轨迹一个版本做分母（C6）。
- 报告 `keep_first_n` 的速度却不报告保留样本的长度与奖励（R1 与报告）。
- 用预设的奖励曲线代替真实梯度更新（C7）。
