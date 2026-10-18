# Lab 4 参考答案与解题导读

[任务书](../../labs/lab4/README.md) · [评分](../../labs/lab4/grade.md) · [来源](../../labs/lab4/resources.md)

参考答案在 [policy_student.rs](src/policy_student.rs)、[engine_student.rs](src/engine_student.rs)、[replay_student.rs](src/replay_student.rs)、[real/accept_student.py](real/accept_student.py) 和 [real/profile_student.py](real/profile_student.py)。这些文件与学生文件同名，BEGIN/END 标记一一对应，覆盖 19 个 TODO。测试、benchmark、引擎和采集脚本都只在学生侧，`run.py --impl reference` 覆盖答案时不会替换它们。

## A 为什么这样算

A1 按首次拒绝的位置分配概率质量：前缀生存概率乘当前位置的拒绝概率，最后一个桶是全部接受。A2 从这个分布求一、二阶矩，所以 $a_j$ 随位置怎么变都适用。A3 固定用 nearest-rank，小样本 p95 的定义才可重现，R2 的 Python 版用同一个定义，Rust 的 `calibrate` 与 Python 的汇总才对得上。A4 区间外夹取，A5 缺列报错，不让“没测”变成“免费”。

测试的 oracle 是枚举全部 $2^k$ 个接受/拒绝向量，包括没观察到的后缀，再按首次拒绝的位置聚合。它和“前缀乘积”的实现是两条独立路径，能抓到把后缀也当成拒绝，或者漏掉最后一个桶的错误。

## B 删失、保守分数与预算

**B1 删失。** 计数里不放先验，先验在读数时加。每轮折扣作用到所有位置，包括 $k=0$ 的轮次，否则长时间不推测时旧证据永远不会变旧。只有已接受的前缀和首次拒绝的那个位置被观察到；后缀记失败就是把 $S_j=P(A\ge j)$ 当成 $a_j$。

**B2 Wilson 下界的推导。** 正态近似的置信条件是 $\lvert\hat p-x\rvert\le z\sqrt{x(1-x)/n}$。把它反过来看成关于真值 $x$ 的不等式，两边平方：

$$
(\hat p-x)^2\le\frac{z^2}{n}x(1-x)
\iff
\left(1+\frac{z^2}{n}\right)x^2-\left(2\hat p+\frac{z^2}{n}\right)x+\hat p^2\le0.
$$

二次方程的两个根就是区间的两端：

$$
x_\pm=\frac{\hat p+\frac{z^2}{2n}\pm z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}{1+\frac{z^2}{n}}.
$$

B2 取 $x_-$，R4 取 $x_+$。$\hat p=1$ 时 $x_-=n/(n+z^2)$，1/1 在 $z=1$ 时是 0.5，100/100 是 0.990，所以下界能区分小样本和大样本，而均值做不到。计数经过折扣时 $n$ 不是整数，公式照样能算，但覆盖率保证没有了，所以叫“保守分数”。测试不复用这个闭式，而是在 $[0,\hat p]$ 或 $[\hat p,1]$ 上二分求 $(\hat p-x)^2-z^2x(1-x)/n=0$ 的根，再和你的数比较。少写分母、把 $z^2$ 写成 $z$ 这类错误只能被数值 oracle 抓到，排序检查抓不到。

**B3。** 候选只有个位数，穷举比任何推导都可靠。并列选短 $k$；返回 `None` 表示没有可行项，由调用方记账。

**B4。** Page-Hinkley 统计量用**旧**均值更新：$m_t=\max(0,m_{t-1}+\bar x_{t-1}-x_t-\delta)$。先更新均值再算统计量，均值会被新值拉低，检测变慢，触发位置晚一步。测试用阈值 2.1 的手算轨迹把这一步差别暴露出来（旧均值在第 22 步得 2.139，新均值只有 2.035）。

**B5 与 B6：复杂度和并列规则。** 多请求共享预算是多选背包。

- B5 贪心每次扫描所有请求的所有升级，一次 $O(nK)$，最多升级 $nK$ 次，总共 $O(n^2K^2)$ 次 `committed_moments` 调用。并列规则：比值差在 $10^{-12}$ 内视为相等，保留先遇到的，也就是编号小的请求、较短的 $k$。它不保证最优，测试只检查它与契约一致、可行、不超过最优、停下时极大。
- B6 把成本离散成 $w_{ik}=\lceil T_i(k)/\delta-10^{-9}\rceil$ 个刻度，容量 $C=\lfloor B/\delta+10^{-9}\rfloor$。状态 $V_i(b)$ 是请求 $i..n-1$ 在剩余 $b$ 个刻度时的最大总价值，$V_n(b)=0$，

$$
V_i(b)=\max_{k:\,w_{ik}\le b}\left\{\mathbb{E}[N^{(i)}_k]+V_{i+1}(b-w_{ik})\right\},
$$

时间 $O(nKC)$，空间 $O(nC)$。重建时从 $i=0$ 开始，选第一个满足 $\mathbb{E}[N^{(i)}_k]+V_{i+1}(b-w_{ik})\ge V_i(b)-10^{-9}$ 的 $k$，得到字典序最小的最优向量。向上取整只会高估成本，所以 DP 的解在原始实数预算下一定可行；代价是量子太粗时可能错过真正的最优，测试的实例成本都是 0.1 的整数倍，量子取 0.1 时离散化是精确的。$10^{-9}$ 的容差吸收 $2.6/0.1=26.000000000000004$ 这类浮点误差，否则反例里的最优解会被误判为超预算。

**反例为什么成立。** 基础分配后剩 1.6。A：增益 0.6、成本 1.0，比值 0.6；B：增益 0.9、成本 1.6，比值 0.5625。贪心先取 A，剩 0.6 放不下 B。AdaServe 在树上按路径概率贪心是对的，因为它的每个节点成本相同，按价值排序取前 $B$ 个就是最优；B5 的升级成本各不相同，问题才变成背包。

## C 时间和守恒

引擎只在验证完成时提交。一轮中间到达的请求先入队，正在运行的请求随后回队，所以手算时间线是 r0=[1,3]、r1=[2]。就绪队列空时跳到下一个到达；超过截止时间的请求照样排空，显式记 infeasible，不删除。

`Fixed(k)` 不看预算，但引擎在执行前统一比较 $T(k)$ 和预算，所以所有策略的 infeasible 口径相同；原版只给返回 `None` 的策略计数，固定 $k$ 永远显示 0 次违约，对比是不公平的。

C 的环境按深度给条件接受率。测试 `c_per_position_beats_pooled_iid` 构造“每轮恰好接受 2 个”的负载：按位置估计很快学到 $a_3=0$，停在 $k=2$；池化 i.i.d. 估计在 $k=2$ 时从不见失败，$\hat\alpha$ 偏高，一直付更长的 $k$。影子重放测试 `c_mean_policy_shadow_replay` 和 `c_drift_reset_replay` 在测试里独立实现均值控制器（折扣计数、Beta 均值、穷举选 $k$、CRN 抽样、Page-Hinkley），按引擎的轮日志逐轮比对，任何一个环节错位都会在某一轮暴露。

## R 真实 trace

**为什么手写循环。** HF assisted generation 只支持 batch 1、用 DynamicCache、每轮在 Python 里调用草稿的 `generate()`，wall clock 里混着大量框架开销（参考 trace 里 HF 跑一个 prompt 6–8 s，手写循环 2 s 左右）；它也不暴露每轮的接受数，默认的 0.4 置信度阈值还会提前停草稿。手写循环用 StaticCache 加每个宽度一个 CUDA graph，把 $c$ 从 eager 的 0.96 降到 0.50。

**R1 对齐。** `draft_argmax[i]` 是草稿看到 `sequence[:i+1]` 后的输出，猜的是 `sequence[i+1]`。`generated[i+1]` 在 `sequence` 里的下标是 `prompt_len+i+1`，所以对应的猜测是 `draft_argmax[prompt_len+i]`。`generated[0]` 来自 prefill，第一次推测从它之后开始，所以标记有 `len(generated)-1` 个。

**R1 耦合采样。** 三个均匀数分别决定草稿 token、接受测试和残差 token。草稿 token 的边缘分布是 $q$，接受概率是 $\min(1,p(x)/q(x))$，拒绝时从残差抽，这就是 Leviathan 的算法，所以输出的边缘分布是 $p$。随机数只由 (seed, prompt, 位置) 决定，接受与否就和策略无关，$A(k)=\min(L,k)$ 对采样 trace 也成立。

**R2 测量。** 计时前先让 GPU 用同一个调用连续忙 300 ms，把笔记本 GPU 从低 P-state 拉起来；warmup 吸收首次调用的分配和 cuBLAS 选算法；`stable`（$p_{75}\le1.5\,p_{25}$）标出仍然双峰的行，采集脚本会重测；计时区间只包住被测的一次前向，`stop()` 在结束事件上同步。中位数和 IQR 用 nearest-rank，和 Rust A3 一致。

**R3 回放精确性。** 贪心 trace 上 $L(e)$ 与策略无关，所以 $\min(L(e),k)$ 对任何策略都是精确的，前提是推测循环的数值和 teacher forcing 完全一样。bf16 下这个前提只近似成立：目标模型在宽度 $k+1$ 和宽度 1 下舍入不同，会让循环的输出在近乎并列的位置离开贪心路径（参考 trace 里分叉处 top-2 logit 差全部 $\le0.25$）；草稿在一轮全部接受后要一次补喂 2 个 token，宽度 2 的舍入也会翻转近乎并列的猜测。我们试过让草稿只用宽度 1，回放能逐位精确，但每次全部接受后要多一步草稿，$k=1$ 的实测加速从 1.06 掉到 0.91，于是保留宽度 2，用“分叉前至少 95% 的轮次相同”验收（参考 98.4%）。

**R4 为什么乐观有用。** 没观测过的深度取上端 1，B3 会选更长的 $k$ 去探测；一轮长 $k$ 同时给所有更浅的深度提供一个样本，乐观值被迅速修正。在全拒绝的 trace 上，第一个位置的上端会随着失败累积降下来，控制器回到 $k=0$；折扣让证据慢慢变旧，上端回升，于是隔一段时间再探测一次。

**参考 trace 上的结果**（2026-10-18，RTX 4060 Laptop，驱动 566.24）：graph 模式 $T_d=9.0$ ms、$T_v(0)=18.1$ ms、$T_v(1..8)=19.1$–19.3 ms，60 个计时行全部满足 $p_{75}\le1.5\,p_{25}$；$S_1..S_6$ 从 0.81 降到 0.40，$a_1..a_6$ 从 0.81 升到 0.90；固定 $k=1,2,3$ 的实测加速 1.06、1.04、0.96，回放预测的耗时比实测少 2%–6%；控制器里最好的是 `k2`，`censored_ucb` 慢 2.3%，`arm_ucb` 慢 7.1%，`hf_heuristic` 慢 10.0%，均值类控制器停在 $k=0$、慢 19%。

## 运行与对照

在 `labs/` 目录运行参考实现：

```bash
python tools/run.py --lab 4 --impl reference --mode test --part all
python tools/run.py --lab 4 --impl reference --mode bench --part all
flock /tmp/mlsys-gpu.lock python tools/run.py --lab 4 --impl reference --mode test --part R --device cuda
flock /tmp/mlsys-gpu.lock python tools/run.py --lab 4 --impl reference --mode bench --part R --device cuda
python tools/fault_inject.py --lab 4                 # CPU faults
python tools/fault_inject.py --lab 4 --device cuda   # GPU fault（自动加 GPU 锁）
```

A/B/C/R 分别有 4/6/8/8 个 Rust 测试，共 26 个；R 阶段另有 Python 检查。完整 benchmark 输出 A 60 行、B 120+3 行、C 180 行，R 每份 trace 输出成本表、$S_j$/$a_j$、固定 $k$ 对照和 24 行控制器对比。

## oracle 与 fault 对照表

由 `python tools/fault_inject.py --lab 4 --json ...` 的输出整理。所有 fault 都在预期的 part 失败（CPU 42 个、CUDA 1 个，均为 CAUGHT）。

| TODO | 独立 oracle                       | fault id                      | 注入的错误               | 在哪个 part 失败 | 状态   |
| ---- | --------------------------------- | ----------------------------- | ------------------------ | ---------------- | ------ |
| A1   | $2^k$ 枚举，按首次拒绝聚合        | `a1-survival-not-updated`     | 生存概率不更新           | A                | CAUGHT |
| A1   | 同上                              | `a1-last-bucket-dropped`      | 丢掉全部接受的桶         | A                | CAUGHT |
| A2   | 手算均值 1.625、方差 0.484375     | `a2-variance-uses-mean`       | 二阶矩写成一阶矩         | A                | CAUGHT |
| A3   | nearest-rank 手算                 | `a3-rank-off-by-one`          | floor+1 代替 ceil        | A                | CAUGHT |
| A4   | context 150、200 的手算插值       | `a4-interp-wrong-weight`      | 插值权重反向             | A                | CAUGHT |
| A5   | 手算分组中位数                    | `a5-mean-not-median`          | 均值代替中位数           | A                | CAUGHT |
| A5   | 缺列必须报错                      | `a5-missing-k-as-zero`        | 缺列填 0                 | A                | CAUGHT |
| B1   | 逐步手算计数                      | `b1-suffix-as-failure`        | 后缀记失败               | B                | CAUGHT |
| B1   | $k=0$ 仍衰减                      | `b1-no-global-decay`          | 只衰减观测到的位置       | B                | CAUGHT |
| B2   | 二分法解 Wilson 二次方程          | `b2-wilson-center-z`          | 中心项用 $z$ 代替 $z^2$  | B                | CAUGHT |
| B2   | 同上                              | `b2-wilson-no-denominator`    | 丢掉 $1+z^2/n$           | B                | CAUGHT |
| B2   | Beta(1,1) 手算                    | `b2-prior-not-applied`        | 用裸 MLE                 | B                | CAUGHT |
| B3   | 900 例穷举 + 精确并列             | `b3-ties-prefer-longer`       | 并列选长 $k$             | B                | CAUGHT |
| B3   | 同上                              | `b3-remaining-not-capped`     | 允许 $k=r$               | B                | CAUGHT |
| B4   | 手算轨迹 + 独立检测器             | `b4-new-mean`                 | 先更新均值               | B                | CAUGHT |
| B4   | 同上                              | `b4-no-window-reset`          | 触发后不清空累计量       | B                | CAUGHT |
| B5   | 测试端契约贪心、穷举最优、极大性  | `b5-ignore-budget`            | 不检查预算               | B                | CAUGHT |
| B5   | 同上 + “最好比值优先”例子         | `b5-first-not-best`           | 取第一个可行升级         | B                | CAUGHT |
| B6   | 穷举最优与字典序                  | `b6-floor-ticks`              | 成本向下取整             | B                | CAUGHT |
| B6   | 同上                              | `b6-tie-last`                 | 并列取最大 $k$           | B                | CAUGHT |
| C1   | 手算提交数与时间戳                | `c1-commit-accepted-only`     | 漏掉 bonus token         | C                | CAUGHT |
| C1   | 同上                              | `c1-first-overwritten`        | 首 token 时间被覆盖      | C                | CAUGHT |
| C2   | 同时刻到达与 arrival = now        | `c2-arrival-strict`           | `>` 写成 `>=`            | C                | CAUGHT |
| C3   | 三种 infeasible 手算              | `c3-fixed-not-infeasible`     | 固定 $k$ 不计 infeasible | C                | CAUGHT |
| C3   | 影子均值控制器逐轮重放            | `c3-observe-kmax`             | 观测的提出数写错         | C                | CAUGHT |
| C3   | 影子重放 + Page-Hinkley           | `c3-no-policy-reset`          | 检测触发但不重置         | C                | CAUGHT |
| C3   | 手算两请求时间线                  | `c3-requeue-before-arrivals`  | 先回队再收到达           | C                | CAUGHT |
| C4   | 手算 p95 与 goodput               | `c4-tpot-divides-tokens`      | TPOT 除以 token 数       | C                | CAUGHT |
| C4   | 同上                              | `c4-goodput-ms`               | 忘了 ms 到 s             | C                | CAUGHT |
| R1   | 手算与随机对齐                    | `r1-flags-off-by-one`         | 草稿猜测错一位           | R                | CAUGHT |
| R1   | 真实循环逐轮比对（CUDA）          | `r1-flags-off-by-one-cuda`    | 与被条件的 token 比较    | R（cuda）        | CAUGHT |
| R1   | vLLM 例子 $[1,0.6,0.2]$           | `r1-survival-as-hazard`       | 把 $P(A\ge j)$ 当 $a_j$  | R                | CAUGHT |
| R1   | 暴力 $a_j$、i.i.d. 平坦           | `r1-hazard-over-all-starts`   | hazard 用生存分母        | R                | CAUGHT |
| R1   | Monte Carlo 输出分布 $=p$         | `r1-accept-test-swapped`      | 接受比值写反             | R                | CAUGHT |
| R1   | 同上                              | `r1-residual-is-p`            | 从 $p$ 而非残差重采样    | R                | CAUGHT |
| R2   | 假时钟                            | `r2-warmup-timed`             | 样本含 warmup            | R                | CAUGHT |
| R2   | nearest-rank 手算                 | `r2-median-upper`             | 上中位数                 | R                | CAUGHT |
| R3   | 逐 $(e,k)$ 删失检查               | `r3-replay-ignores-censoring` | 回放返回 $L(e)$          | R                | CAUGHT |
| R3   | 暴力回放 + 实测轮日志             | `r3-schedule-off-by-one`      | 下一轮起点差一位         | R                | CAUGHT |
| R3   | 暴力 run length                   | `r3-runs-no-restart`          | 拒绝后不清零             | R                | CAUGHT |
| R3   | 暴力回放                          | `r3-propose-past-end`         | 提出不存在的步           | R                | CAUGHT |
| R4   | 行为检查（全拒绝 trace 仍会探测） | `r4-upper-unexplored-zero`    | 未观测位置取 0           | R                | CAUGHT |
| R4   | 二分法求上根                      | `r4-upper-uses-minus`         | 返回下端                 | R                | CAUGHT |

参考答案没有实现树验证、batch 验证、每轮停止，也没有把采样 trace 接入 Rust 回放；这些是挑战题。
