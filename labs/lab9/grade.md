# Lab 9 验收与评分（100 分）

[任务说明](README.md) · [资源与改编说明](resources.md) · [参考导读](../../solutions/lab9/README.md)

以 [README](README.md) 的必修任务为准。自动测试是得分的必要条件，不是充分条件；改测试预期、调用 solutions 或只交编译成功的截图都不得分。允许与参考实现不同的正确算法，但测试里的独立 oracle 必须全部通过。

| 项                           | 分值 | 自动或人工证据                                                                                                                |
| ---------------------------- | ---: | ----------------------------------------------------------------------------------------------------------------------------- |
| A1/A2 FLOPs、字节与 roofline |    8 | `A1 *`：tiny 模型逐元素暴力计数、Qwen3-1.7B 手算值；`A2 *`：四个手算时间                                                      |
| A3 加权最小二乘拟合 $\eta$   |    7 | `A3 noise-free *` 恢复已知系数（含需要重新标注的样本）、`A3 weighted-LS stationarity *` 有限差分                              |
| A4/A5 显存与 held-out        |    5 | `A4 exact fit *` 边界、`A4 * weights` 暴力计数；`A5 *` 偶数中位数与"只用训练集"                                               |
| B1–B3 建图、Dinic、最小割    |   12 | `B1 40 random placements`、`B2 max-flow == enumerated min-cut`、`B2 cancellation graph`、`B3 cut capacity *`                  |
| B4/B5 层放置                 |   10 | `B4 DP bottleneck == composition enumeration`、`B5 * >= 90% of the exhaustive optimum`、`B5 no single-node move *`            |
| B6/B7 流分解与 IWRR          |    5 | `B6 30 decompositions *`、`B7 prefix deviation *`、`B7 weights (5,1,1) *`                                                     |
| C1/C2 SLO 容量与 B&B         |   10 | `C1 *` 闭式暴力扫描；`C2 hand: co-located template chosen`、`C2 B&B optimum == brute force *`                                 |
| C3/C4 仿真与基线             |   10 | `C3 hand *`、`C3 M/D/1 mean wait *`、`C3 Little's law *`、`C3 completed req/s <= *`；`C4 * independent recomputation`         |
| R 真实系统                   |   15 | R1–R4：`profile test` 的 `SUMMARY part R-gpu` 全部通过（WSL 记录）；R5/R6：`SUMMARY part R`；实测 profile CSV 与 bench R 记录 |
| 实验与报告                   |   18 | README "报告与验收" 六项：表格完整、measured 与 modeled 分列、至少一个负结果、与三个基线同口径对比、局限讨论                  |
| 环境证据                     |    — | 附 `bash tools/check_env.sh --lab 9` 的完整输出；缺失则 R 阶段不给分                                                          |

报告 18 分的构成：结果完整性 6 分，因果解释与负结果 6 分（例如"成本最低的共置方案在仿真里 SLO 达成率最低"要给出排队论解释），measured/modeled 区分与局限 6 分。

**R 阶段的两种交法**：有 GPU 的同学交自己在 WSL 下的 `results/lab9/rtx4060_profile.csv`、`--part R --device cuda` 的 test 与 bench 记录，满分 15。没有 GPU 的同学用随附 profile 完成 R5/R6，并在报告里写明"使用作者的参考 profile"，R 阶段最多 8 分（R1–R4 的 7 分不能用编译成功或参考数据替代）。

## 必交命令

```bash
bash tools/check_env.sh --lab 9
python tools/run.py --lab 9 --impl student --mode test --part all
python tools/run.py --lab 9 --impl student --mode test --part R --device cuda     # WSL/Linux
python tools/run.py --lab 9 --impl student --mode bench --part all
python tools/run.py --lab 9 --impl student --mode bench --part R --device cuda    # WSL/Linux; CPU-only: drop --device cuda
```

提交每条命令打印的 `RECORD:` JSON 和 `results/lab9/rtx4060_profile.csv`。GPU 计时要说明当时的电源模式（插电与否）和环境温度大致情况；同一台笔记本前后两次测量差 10% 以上时，报告两次并解释。

## 必查错误

- decode 的字节里漏掉 KV 读；attention FLOPs 用了 $H_{kv}$；因果掩码少数一个位置。
- 拟合只做一轮标注，或者用绝对误差最小二乘；在测试集上拟合。
- 把 GPU 容量放在入边上而不是拆点的内部边上；Dinic 没有反向残量；在原图而不是残量图上找割。
- DP 的层数循环少了 $k=\min(K_j,\ell)$；局部搜索越过显存上限。
- IWRR 的权重写反；流分解把链路边也算成一级 pipeline。
- C1 忘了除以份额，或忘了 prefill 打断 decode；B&B 的下界对模型求和；仿真的面积不含排队中的请求。
- 用 16 MiB 以下的拷贝测"显存带宽"（测到的是 L2）；分位数用最近秩；把 warmup 计入统计。
- 把规格表乘 $\eta$ 得到的其他 GPU 吞吐写成"实测"。
