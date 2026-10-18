# Lab 8 参考答案导读

[任务书](../../labs/lab8/README.md) · [评分](../../labs/lab8/grade.md) · [来源](../../labs/lab8/resources.md)

参考实现在 [`src/rollout_student.rs`](src/rollout_student.rs)（A1–A6、B1–B4、C1–C7）和 [`src/replay_student.rs`](src/replay_student.rs)（R1–R3），共 20 个 TODO、425 行有效代码，只用 Rust 标准库。错误注入清单在 [`faults.json`](faults.json)，共 53 个。本页按阶段说明实现要点、为什么这样做和常见错误，再给出推导、运行命令和 oracle 与 fault 的对照表。

## A：信息边界与约定

**A1/A2。** 历史只接收已完成的长度，窗口按类别独立淘汰。上中位数 `sorted[len/2]` 是约定；测试用偶数长度的历史区分上下中位数。

**A3。** list scheduling 的两个 tie-break 都要写明：任务按 `(估计降序, id 升序)`，worker 按 `(负载, 编号)`。旧测试的输入 $[3,3,2,2,2]$ 已经降序，忽略 `lpt` 标志也能通过；现在用 $[2,3,2,3,2]$ 并断言精确队列。注意 LPT 在这个实例上更差（7 对 6），测试检查的是算法行为，不是最优性。

**A4。** 两种 std 都实现。样本 std 在 $G=1$ 时分母为 0，按常数组处理返回 0。不加 epsilon 是为了让测试能精确比对 $\pm1/\sqrt2$；verl 加 `1e-6`、TRL 加 `1e-4`，接入真实框架时要对齐。

**A6。** `behavior_p` 记的是动作 1 的概率，计算比值时动作 0 要取 $1-p$。若改成“所选动作的概率”，C3 里动作 0 会再取一次补，比值就错了；测试在 logit 2（$p\ne0.5$）下检查动作 0 的样本。

## B：账本、执行器与选择

**B1/B2。** 只做账本检查：排序后逐段比对起点、身份和终止标志。B2 的关键是比集合而不是比人数。

**B3。** 每轮先让所有空闲 worker 按编号取活，再跳到最早完成时刻，同时完成的 worker 按编号结算。这个顺序决定了确定性：在“偷队尾”的测试里，$t=5$ 时 w0 先于 w1 选活，于是 w0 从 w1 的队列里偷走 job 2，共两次窃取。偷队尾而不是队首，是为了让被偷者保留它马上要做的任务（局部性），也是 Cilk 式窃取的惯例。

**B4。** 返回的是位置，按 `(finish, sample_idx)` 排序。保留顺序统计量的推导见下文。

## C：语义与梯度

**C1/C2。** 容量按组计，避免部分组占满容量造成死锁；超龄的判断用饱和减法，未来版本单独拒绝。

**C3/C4。** `objective` 是纯函数，`update` 先算梯度再一次性更新。测试对学生自己的 `objective` 做差分，所以两者必须一致：若 `objective` 写错、`update` 按正确公式写，差分检查会失败；若两者错得一样，独立闭式 `c_objective` 和手算梯度会失败。

**C5。** 只有一行：`generated / batch <= version + eta`。难点在推导和测试，见下文。

**C6。** 每个 token 自己的 `behavior_p` 就是采样时该版本的概率，不需要知道版本号本身。实现里不要按 `version` 去查别的策略。

**C7。** 和旧版相同，只是改用 `expected_reward` 与新的 `update`。单波测试的推导写在测试注释里。

## R：回放的几个决定

- **调度单位是组。** 参考 trace 是 batch = 组采的，单样本的 `gen_ms` 只在组内有意义。所以一个 worker 一次跑一组，时长取保留样本的最大 `gen_ms`。`keep_first_n` 节省的是“等最后几条”的时间，被 abort 的槽位在静态 batch 里并没有被别人用上。
- **先预测、后观测。** 每轮先对所有组 `predict`，本轮结束后才 `observe`，而且只观测保留样本。被 abort 的样本长度在真实系统里是未知的；把它们喂给预测器是信息泄漏。
- **`LptPredicted` 第一轮为什么差。** 冷启动时所有预测都是 256，LPT 退化为按 ID 轮转的静态分配，比共享队列拉取差。`LptPull` 保留了拉取的自适应性，只用预测决定顺序。
- **同时报告两套统计。** 平均长度和平均奖励都给出“保留样本”和“全体样本”两个值。`first_idx_n` 作为同样本数的无偏对照，用来把“样本变少”和“样本被选择”分开：优势全零的组数在两种子集里都是 36（全体 28），说明这一项来自组变小，不是选择偏差。

## 推导

### 裁剪目标的四种越界情形

单样本 $J=\min(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A)$。

- $A>0$：$J=A\min(\rho,\operatorname{clip}\rho)$。$\rho<1-\epsilon$ 时 $\operatorname{clip}\rho=1-\epsilon>\rho$，取 $\rho$，**活跃**（情形 3）；$\rho>1+\epsilon$ 时取 $1+\epsilon$，**零梯度**（情形 5）。
- $A<0$：$J=A\max(\rho,\operatorname{clip}\rho)$。$\rho<1-\epsilon$ 时取 $1-\epsilon$，**零梯度**（情形 4）；$\rho>1+\epsilon$ 时 $\operatorname{clip}\rho=1+\epsilon<\rho$，取 $\rho$，**活跃**（情形 6），且 $\lvert\partial J/\partial\rho\rvert=\lvert A\rvert$ 不随 $\rho$ 衰减，$\rho\nabla\log\pi$ 随 $\rho$ 线性增长。

所以零梯度的条件是 $(A>0\wedge\rho>1+\epsilon)\vee(A<0\wedge\rho<1-\epsilon)$，也就是 C4 里 `inactive` 的写法。对 Bernoulli 策略 $\partial\log\pi(a)/\partial z=a-p$，活跃时单样本梯度为 $\rho A(a-p)$。

测试的两个数值用例（behavior $p=0.5$，动作 $[1,0]$）：

- $z=0.3$、$A=[+1,-1]$：$\rho=[2p,2(1-p)]$，平均梯度 $\frac12[2p(1-p)+2(1-p)p]=2p(1-p)=0.48892$；漏乘 $\rho$ 得 $\frac12[(1-p)+p]=0.5$。
- $z=3$、$A=[-1,+1]$：情形 6 与情形 3，平均梯度 $\frac12[-2p(1-p)-2(1-p)p]=-2p(1-p)=-0.09035$；“越界即零”得 0。

### 版本界

trainer 按提交顺序消费，第 $k$ 批（第 $kB$ 到 $(k+1)B-1$ 条）在版本 $k$ 训练，所以第 $g$ 条在版本 $\lfloor g/B\rfloor$ 训练。它在版本 $v$ 被生成，陈旧度 $s_g=\lfloor g/B\rfloor-v$。要求 $s_g\le\eta$，即 $\lfloor g/B\rfloor\le v+\eta$。

界是**紧**的：生产者足够快时，它会一直提交到 $\lfloor g/B\rfloor=v+\eta$，这一条恰好陈旧 $\eta$。测试的模拟器检查最大陈旧度等于 $\eta$，因此三种 off-by-one 都会暴露：

| 写法                                     | 后果                                       |
| ---------------------------------------- | ------------------------------------------ |
| `generated / batch < version + eta`      | $\eta=0$ 时一条也不能提交，死锁            |
| `generated / batch <= version + eta + 1` | 最大陈旧度 $\eta+1$                        |
| `generated <= (version + eta) * batch`   | $\eta=0$ 时只能提交 1 条，凑不满一批，死锁 |

### 先完成者入选

$m$ 条独立同分布的长度，完成时间随长度单调，保留 $n$ 条即保留前 $n$ 个顺序统计量。$m=3$、长度在 $\{1,2,3\}$ 上均匀、$n=2$ 时：

$$
\mathbb E[\max]=\sum_{k=1}^{3}\Pr(\max\ge k)=1+\frac{26}{27}+\frac{19}{27}=\frac83,\qquad
\mathbb E[\bar\ell_{\text{kept}}]=\frac{6-8/3}{2}=\frac53.
$$

奖励 $[\ell\ge2]$：全体期望命中 $3\cdot\frac23=2$，被丢弃的那条（最长的）命中概率 $1-\frac1{27}$，保留平均奖励 $\frac{2-26/27}{2}=\frac{14}{27}$。测试枚举全部 27 种长度组合，独立算出同样的值。

## 运行命令

从 `labs/` 执行：

```bash
python tools/run.py --lab 8 --impl reference --mode test --part all
python tools/run.py --lab 8 --impl reference --mode bench --part all
python tools/run.py --lab 8 --impl reference --mode bench --part R --device cuda   # WSL + GPU
python tools/fault_inject.py --lab 8 --require-coverage
```

## oracle 与 fault 对照表

每个 TODO 至少一个 fault，核心 TODO 至少两个。下表由 `python tools/fault_inject.py --lab 8 --json results/faults-lab8.json` 的结果整理，全部在 CPU 上运行（本 lab 没有 CUDA 部分）。

| TODO | 独立 oracle                                                                          | fault id                           | 注入的错误                                                                                                                                        | 在哪个 part 失败 | 状态   |
| ---- | ------------------------------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ------ |
| A1   | 手算窗口淘汰（$[5,1,9,7]$，窗口 3）                                                  | `a1-window-off-by-one`             | history keeps window+1 values; [5,1,9,7] with window 3 then predicts from 4 values                                                                | A                | CAUGHT |
| A1   | 手算窗口淘汰（$[5,1,9,7]$，窗口 3）                                                  | `a1-no-eviction`                   | window never evicts; retained-length and median checks reject it                                                                                  | A                | CAUGHT |
| A2   | 偶数历史的上中位数、冷启动 32                                                        | `a2-lower-median`                  | lower median: even history [1,2,3,4] must predict 3, not 2                                                                                        | A                | CAUGHT |
| A2   | 偶数历史的上中位数、冷启动 32                                                        | `a2-cold-start-zero`               | cold start must be 32                                                                                                                             | A                | CAUGHT |
| A3   | 未排序输入的精确队列 + 测试端独立调度器（60 例）                                     | `a3-lpt-ignores-flag`              | LPT applied even when lpt=false; the unsorted D8-2 input exposes the input-order queues [[0,2,4],[1,3]]                                           | A                | CAUGHT |
| A3   | 未排序输入的精确队列 + 测试端独立调度器（60 例）                                     | `a3-never-sorts`                   | lpt=true without sorting; exact LPT queues [[1,0,4],[3,2]] on unsorted input reject it                                                            | A                | CAUGHT |
| A3   | 未排序输入的精确队列 + 测试端独立调度器（60 例）                                     | `a3-tie-break-reversed`            | equal estimates with shuffled ids must be ordered by ascending id                                                                                 | A                | CAUGHT |
| A3   | 未排序输入的精确队列 + 测试端独立调度器（60 例）                                     | `a3-worker-tie-highest`            | equal loads must go to the lower worker index; exact queues differ                                                                                | A                | CAUGHT |
| A4   | 手算 $\pm1$、$\pm1/\sqrt2$、$\pm\sqrt3/2$；均值 0 方差 1                             | `a4-always-population`             | ignores the bessel flag: sample std of [0,1] gives +-0.707, not +-1                                                                               | A                | CAUGHT |
| A4   | 手算 $\pm1$、$\pm1/\sqrt2$、$\pm\sqrt3/2$；均值 0 方差 1                             | `a4-sample-std-always`             | Bessel correction applied to the population convention                                                                                            | A                | CAUGHT |
| A4   | 手算 $\pm1$、$\pm1/\sqrt2$、$\pm\sqrt3/2$；均值 0 方差 1                             | `a4-no-constant-guard`             | constant groups divide by zero and return NaN instead of zeros                                                                                    | A                | CAUGHT |
| A5   | 手算 ceil 秩与闭上界                                                                 | `a5-floor-rank`                    | q=0.3 on 5 values must return sorted[2]=5, floor gives 4                                                                                          | A                | CAUGHT |
| A5   | 手算 ceil 秩与闭上界                                                                 | `a5-exclusive-bound`               | bucket upper bounds are inclusive                                                                                                                 | A                | CAUGHT |
| A6   | 动作 1 的概率 $p \neq 0.5$ 时逐样本核对 behavior 概率；`u` 恰等于 $p$ 时必须选动作 0 | `a6-behavior-of-taken-action`      | behavior_p is defined as P(action 1); storing P(taken action) breaks ratios for action 0                                                          | A                | CAUGHT |
| A6   | 动作 1 的概率 $p \neq 0.5$ 时逐样本核对 behavior 概率；`u` 恰等于 $p$ 时必须选动作 0 | `a6-le-instead-of-lt`              | u == p must give action 0                                                                                                                         | A                | CAUGHT |
| B1   | 10 类损坏输入                                                                        | `b1-no-gap-check`                  | gap and overlap accepted                                                                                                                          | B                | CAUGHT |
| B1   | 10 类损坏输入                                                                        | `b1-ignores-version`               | mixed-version trajectory accepted                                                                                                                 | B                | CAUGHT |
| B2   | 只训练快组、版本不符的反例                                                           | `b2-group-count-only`              | only the fast group is trained; group sizes still look complete                                                                                   | B                | CAUGHT |
| B2   | 只训练快组、版本不符的反例                                                           | `b2-ignores-version`               | a batch sampled by another policy version passes the gate                                                                                         | B                | CAUGHT |
| B3   | 手算时间线、偷队尾的完成顺序、工作守恒                                               | `b3-steal-head`                    | stealing the victim's head changes the hand timeline [6,1,1,1]                                                                                    | B                | CAUGHT |
| B3   | 手算时间线、偷队尾的完成顺序、工作守恒                                               | `b3-no-requeue`                    | unfinished jobs are dropped after one quantum: work conservation and reassembly fail                                                              | B                | CAUGHT |
| B3   | 手算时间线、偷队尾的完成顺序、工作守恒                                               | `b3-steal-not-counted`             | steal count must be reported (hand timeline expects one steal)                                                                                    | B                | CAUGHT |
| B4   | 27 种组合枚举的顺序统计量（$5/3$、$14/27$）                                          | `b4-keeps-slowest`                 | keeps the slowest instead of the fastest; exact order-statistic oracle (5/3, 14/27) rejects it                                                    | B                | CAUGHT |
| B4   | 27 种组合枚举的顺序统计量（$5/3$、$14/27$）                                          | `b4-tie-by-position`               | ties must be broken by sample_idx, not by position in the input                                                                                   | B                | CAUGHT |
| C1   | 逐条手算准入结果与计数                                                               | `c1-capacity-per-sample`           | capacity counted in samples deadlocks a partial group                                                                                             | C                | CAUGHT |
| C1   | 逐条手算准入结果与计数                                                               | `c1-lag-off-by-one`                | lag exactly max_lag must still be admitted                                                                                                        | C                | CAUGHT |
| C1   | 逐条手算准入结果与计数                                                               | `c1-no-mismatch-check`             | id/group mismatch admitted                                                                                                                        | C                | CAUGHT |
| C2   | 手算驱逐与出队                                                                       | `c2-evicts-nothing`                | stale group is trained instead of evicted                                                                                                         | C                | CAUGHT |
| C2   | 手算驱逐与出队                                                                       | `c2-partial-group`                 | returns a partial group                                                                                                                           | C                | CAUGHT |
| C3   | 测试端独立闭式；手算 $-0.905148$                                                     | `c3-current-as-behavior`           | recomputing behavior from the current policy forces rho=1; the logit-3 hand value -0.905 rejects it                                               | C                | CAUGHT |
| C3   | 测试端独立闭式；手算 $-0.905148$                                                     | `c3-max-instead-of-min`            | pessimistic min replaced by max; closed-form oracle rejects it                                                                                    | C                | CAUGHT |
| C4   | 对学生 `objective` 的中心差分 + 手算 $2p(1-p)$、$-2p(1-p)$                           | `c4-missing-ratio-factor`          | forgot the ratio factor: at logit 0.3 the gradient is 0.5 instead of 2p(1-p)=0.4889 (passes at rho=1)                                             | C                | CAUGHT |
| C4   | 对学生 `objective` 的中心差分 + 手算 $2p(1-p)$、$-2p(1-p)$                           | `c4-zero-when-out-of-range`        | zero gradient whenever rho leaves [1-eps,1+eps]: the active (A<0, rho>1+eps) and (A>0, rho<1-eps) case must give -2p(1-p)                         | C                | CAUGHT |
| C4   | 对学生 `objective` 的中心差分 + 手算 $2p(1-p)$、$-2p(1-p)$                           | `c4-wrong-clip-branch`             | clip branches swapped by sign of A: active and flat regions exchanged                                                                             | C                | CAUGHT |
| C4   | 对学生 `objective` 的中心差分 + 手算 $2p(1-p)$、$-2p(1-p)$                           | `c4-no-group-mean`                 | gradient summed instead of averaged over the group; finite difference of the mean objective rejects it                                            | C                | CAUGHT |
| C4   | 对学生 `objective` 的中心差分 + 手算 $2p(1-p)$、$-2p(1-p)$                           | `c4-returns-after`                 | must return the objective before the step                                                                                                         | C                | CAUGHT |
| C5   | 生产者-训练者模拟器：最大陈旧度恰为 $\eta$、不死锁                                   | `c5-staleness-off-by-one`          | admits eta+1-stale data; the simulator sees max staleness eta+1                                                                                   | C                | CAUGHT |
| C5   | 生产者-训练者模拟器：最大陈旧度恰为 $\eta$、不死锁                                   | `c5-strict-less`                   | eta=0 never submits (deadlock); the simulator reports no progress                                                                                 | C                | CAUGHT |
| C5   | 生产者-训练者模拟器：最大陈旧度恰为 $\eta$、不死锁                                   | `c5-per-sample-bound`              | bound applied per trajectory instead of per batch: with eta=0, B=4 only one trajectory is admitted and the trainer never fills a batch (deadlock) | C                | CAUGHT |
| C6   | 测试端逐 token 计算 $w,u$                                                            | `c6-no-behavior-correction`        | drops the off-policy correction: resumed tokens from older versions must carry pi_prox/pi_behav                                                   | C                | CAUGHT |
| C6   | 测试端逐 token 计算 $w,u$                                                            | `c6-ratio-to-behavior`             | trust-region ratio centred on behavior instead of prox (AReaL's decoupling undone)                                                                | C                | CAUGHT |
| C7   | 单波精确计数 $(1,24,1)$；3 个种子的学习验收                                          | `c7-no-version-sync`               | queue never learns the new version, so no stale group is dropped under max_lag 0                                                                  | C                | CAUGHT |
| C7   | 单波精确计数 $(1,24,1)$；3 个种子的学习验收                                          | `c7-no-retry`                      | a backpressured group is lost instead of retried; update count drops below 400                                                                    | C                | CAUGHT |
| R1   | 手算 12 行 trace；测试端拉取模拟、下界与预测来源                                     | `r1-predictor-leaks-future`        | predictor sees this round's true lengths before dispatch; recorded predictions must be cold in round 0                                            | R                | CAUGHT |
| R1   | 手算 12 行 trace；测试端拉取模拟、下界与预测来源                                     | `r1-keep-first-shift-not-reported` | reports the all-sample mean as the kept mean, hiding keep-first-n's length bias                                                                   | R                | CAUGHT |
| R1   | 手算 12 行 trace；测试端拉取模拟、下界与预测来源                                     | `r1-observes-aborted`              | aborted samples never finished; observing them leaks lengths the system cannot know                                                               | R                | CAUGHT |
| R1   | 手算 12 行 trace；测试端拉取模拟、下界与预测来源                                     | `r1-duration-sum`                  | a batched group occupies its worker for max(gen_ms), not the sum                                                                                  | R                | CAUGHT |
| R1   | 手算 12 行 trace；测试端拉取模拟、下界与预测来源                                     | `r1-pull-ignores-prediction`       | lpt_pull without sorting equals fifo; hand makespan 3800 for round 1 rejects it                                                                   | R                | CAUGHT |
| R1   | 手算 12 行 trace；测试端拉取模拟、下界与预测来源                                     | `r1-zero-adv-uses-all`             | zero-advantage groups must be counted on the kept subset                                                                                          | R                | CAUGHT |
| R2   | 手算窗口 + 随机用例对照测试端历史                                                    | `r2-no-window`                     | per-prompt history never evicted                                                                                                                  | R                | CAUGHT |
| R2   | 手算窗口 + 随机用例对照测试端历史                                                    | `r2-global-unbounded`              | global history must keep only 16*window values                                                                                                    | R                | CAUGHT |
| R3   | 测试端最近秩分位数（随机用例）                                                       | `r3-rank-floor`                    | nearest rank uses ceil; the randomized cases and window checks differ                                                                             | R                | CAUGHT |
| R3   | 测试端最近秩分位数（随机用例）                                                       | `r3-no-global-fallback`            | unseen prompt must fall back to the global history                                                                                                | R                | CAUGHT |

结果：53 个 fault 全部 CAUGHT，0 MISSED，0 STALE（WSL Ubuntu 24.04，2026-10-18）。注入错误的说明保留英文原文，与 `faults.json` 一致。
