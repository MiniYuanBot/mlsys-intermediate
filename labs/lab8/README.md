---
title: Lab 8 — rollout 调度、陈旧度与样本语义：从合成事件到真实长度 trace
aliases: [Lab8, rollout scheduling lab]
tags: [mlsys, lab, reinforcement-learning, rust, scheduling]
---

# Lab 8：rollout 调度、陈旧度与样本语义，从合成事件到真实长度 trace

[讲义 D8](../../course/modules/08-rollout-scheduling.md) · [评分](grade.md) · [公开来源与借鉴边界](resources.md) · [参考答案导读](../../solutions/lab8/README.md)

> [!important] 你要交付什么
> 用 Rust 实现 RL rollout 一侧的调度与语义检查：长度历史与 LPT、组优势、分段账本、带窃取的执行器、“保留先完成的 $n$ 条”选择（A/B）；按组计容量的有界队列、PPO 裁剪目标及其梯度、AReaL 式版本界、partial rollout 的解耦重要性项，以及一个真实更新参数的 contextual bandit 闭环（C）。R 阶段把调度器放到一份**真实采样**的长度 trace 上回放（Qwen3-0.6B，RTX 4060 Laptop），同时报告吞吐和分布偏移，并实现一个按 prompt 的分位数长度预测器。A/B/C 的时间是合成 token 单位；R 的时长由实测的单样本完成时间建模得到，标为 `modeled from measured gen_ms`，不是实测的系统吞吐。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。在 `labs/` 下先安装、再检查：

```bash
bash tools/setup_env.sh --lab 8 && bash tools/check_env.sh --lab 8
```

Windows 原生（PowerShell）：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 8
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 8
```

| 阶段 | 需要                      | WSL2 / Linux | Windows 原生 | macOS |
| ---- | ------------------------- | ------------ | ------------ | ----- |
| A    | Rust                      | 可做         | 可做         | 可做  |
| B    | Rust                      | 可做         | 可做         | 可做  |
| C    | Rust                      | 可做         | 可做         | 可做  |
| R    | Rust；可选模型 Qwen3-0.6B | 可做         | 可做         | 可做  |

A/B/C/R 的测试和回放都只需要 Rust，不需要 GPU，Rust 标准库之外没有依赖，离线编译。R 阶段有两项可选内容：一是 `--mode bench --part R --device cuda` 会用 `real/sample_trace.py` 重新采一份小 trace（WSL/Linux，需要 NVIDIA GPU、torch、transformers 和 `~/models/Qwen3-0.6B`，约 2 分钟），这些依赖由其他 lab 安装，lab8 的安装脚本不装；二是 [real/GUIDE.md](real/GUIDE.md) 讲怎样在单独的 `~/venvs/mlsys-rl` 里用 TRL 跑 GRPO 并导出训练中的 trace，安装脚本不碰它，指南本身也没有在本机验证过。缺少依赖时，runner 只打印一行 `MISSING ...` 和对应的 `fix:` 命令，退出码为 3；在 Windows 上请求 `--device cuda` 的采样，`fix:` 给出一条能直接复制的 WSL 命令。各平台细节见[环境配置](../setup.md)。

## 文件和实施顺序

| 文件                                          | 学生负责            | 已提供                                                                           |
| --------------------------------------------- | ------------------- | -------------------------------------------------------------------------------- |
| `src/rollout_student.rs`                      | A1–A6、B1–B4、C1–C7 | 数据结构、`Policy::probability/action_probability`、`uniform`、`expected_reward` |
| `src/replay_student.rs`                       | R1–R3               | JSON 读取、`RolloutTrace::parse`、`Strategy`、`ReplayReport`、`all_equal`        |
| `tests/contracts.rs`                          | —                   | 19 个测试，每项检查打印 `PASS`/`FAIL`，跑完再判失败                              |
| `src/main.rs`、`exercise.py`                  | —                   | benchmark 与阶段入口                                                             |
| `real/sample_trace.py`                        | —                   | 采样 trace 的脚本（不是 TODO）                                                   |
| `real/prompts.jsonl`                          | —                   | 32 道课程自编的算术应用题（附标准答案、难度标签，随本仓库 MIT 许可证发布）       |
| `real/data/qwen3-0.6b-arith-g8-rtx4060.jsonl` | —                   | 参考 trace，112 KB，见 R 节                                                      |
| `real/GUIDE.md`                               | —                   | TRL GRPO 桥接指南（未在本机验证）                                                |

以下命令都在 **`mlsys-intermediate/labs/`** 运行：

```bash
python tools/run.py --lab 8 --impl student --mode test --part A
python tools/run.py --lab 8 --impl student --mode test --part B
python tools/run.py --lab 8 --impl student --mode test --part C
python tools/run.py --lab 8 --impl student --mode test --part R
python tools/run.py --lab 8 --impl student --mode bench --part all
python tools/run.py --lab 8 --impl student --mode bench --part R --device cuda   # WSL：先采一份新 trace 再一起回放
```

直接调试一个测试：`cargo test --offline --release --manifest-path lab8/Cargo.toml --test contracts c_gradient -- --nocapture`。请保留所有 BEGIN/END TODO 标记；标记前的注释写了契约、边界和对应测试。各阶段的测试只调用本阶段和更早阶段的函数，所以要按 A→B→C→R 的顺序完成。

| 阶段         |   TODO | 参考实现有效行 | 预计时间 |
| ------------ | -----: | -------------: | -------: |
| A 基础       |      6 |             62 |       5h |
| B 核心       |      4 |            101 |       7h |
| C 前沿       |      7 |            149 |       9h |
| R 真实 trace |      3 |            113 |       6h |
| 报告         |      — |              — |       3h |
| **合计**     | **20** |        **425** |  **30h** |

有效行由 `python tools/workload_audit.py` 统计，只计参考答案 TODO 区域内的代码，不要求写得一样长。估时包括调试和小实验，讲义 5h 另计；初学 Rust 的借用与集合可能再加 3–6h。这份估时尚未经过学生试做，请在报告里记录每个 TODO 的实际用时。

## A 基础：长度历史、LPT、组优势与采样（5h）

### A1/A2 `LengthModel::observe` / `predict`（0.75h）

每个 prompt 类别保留最近 `max(window,1)` 个**已完成**长度；预测取上中位数 `sorted[len/2]`，没有历史时返回 32。`predict` 不能修改历史。测试 `a_length_history`：窗口淘汰、类别隔离、偶数长度取上中位数、`window=0` 按 1 处理。

### A3 `assign`（1h）

list scheduling：`lpt=true` 时按预计长度降序访问任务（相同时 ID 小的在前），`lpt=false` 保持输入顺序；每个任务交给预计负载最小的 worker（相同时编号小的优先）。返回每个 worker 的 ID 队列。测试 `a_assign` 用讲义 D8-2 的**未排序**输入 $[2,3,2,3,2]$，断言两种模式的精确队列（input order 为 $[[0,2,4],[1,3]]$，LPT 为 $[[1,0,4],[3,2]]$），再用“估计全相等、ID 乱序”检查 tie-break，最后和测试端的独立调度器比对 60 个随机用例。

### A4 `advantages`（1h）

组内 $(r_i-\bar r)/\text{std}$。`bessel=false` 用总体 std（除以 $G$），`bessel=true` 用样本 std（除以 $G-1$，verl/TRL/CS336 的约定）。空输入返回空；std 小于 $10^{-10}$ 时（常数组，或 $G=1$ 且 `bessel=true`）返回全零。分母不加 epsilon。测试 `a_advantage`：$[0,1]$ 在两种约定下为 $\pm1$ 与 $\pm1/\sqrt2$，$[0,0,1,1]$ 为 $\pm1$ 与 $\pm\sqrt3/2$，以及总体约定下均值 0、方差 1。

### A5 `histogram`（0.5h）

桶为 $(-\infty,b_0],(b_0,b_1],\ldots,(b_{\text{last}},\infty)$，分位数取 `sorted[ceil((N-1)q)]`。测试 `a_histogram` 用 $q=0.3$ 区分 ceil 与 floor，并检查桶的上界是闭的。

### A6 `Policy::sample`（0.75h）

从**这个**快照采样一组：`u < p` 时选动作 1；动作等于 `context % 2` 时奖励为 1；`id = group * len + offset`；`behavior_p` 记的是选**动作 1**的概率（不是所选动作的概率）。测试 `a_sample` 在 logit 2 下检查动作 0 的样本也记录 $p$，并检查 `u == p` 时选动作 0。

### A 实验（1h）

`--mode bench --part A` 给出窗口 4/16/64 在分布突变下的 MAE 与 p50/p95。画一张逐事件的预测与实际长度图，说明预测信息何时可用；不得让预测器读取尚未完成的长度。

## B 核心：分段账本、完整 batch、执行器与先完成者入选（7h）

### B1 `reassemble`（1h）

乱序分段按 `start` 排序后：从 0 开始、无缺口、无重叠、无空段、id/group/version 一致、只有最后一段是终止段；不满足就返回 `Err`，不得返回部分轨迹。测试 `b_reassemble` 覆盖 10 类损坏。它只验证账本，不验证 KV/RNG 的恢复（讲义 §6）。

### B2 `training_ready`（0.75h）

完成集合必须与预选 ID 集合完全一致（无重复、无多余、无缺失），版本一致，每组人数等于 `group_size`。测试 `b_batch_gate` 的关键反例是“只训练先完成的那组”：人数看上去都对，batch 已经变了。

### B3 `schedule`（3h）

事件驱动的 quantum 执行器：

1. 用 A3 的 LPT（预计长度）初始化每个 worker 的队列。
2. 每个空闲 worker 按编号顺序取本地队首；本地为空且允许窃取时，从**最长**的其他队列**尾部**取一个（相同时编号小的为受害者），计一次 steal。
3. 一个任务运行 `min(quantum, actual - progress)` 个单位。只有执行器读 `actual`。
4. 跳到最早的完成时刻，按 worker 编号结算所有同时完成的 worker，各输出一个 `Segment`（token 值为 `id*10000+位置`），未完成的任务放回本 worker 队尾。

测试 `b_scheduler`：token 工作量守恒、每条轨迹可由 B1 重组、手算时间线 $[6,1,1,1]$（不窃取 7、窃取 6），以及一个两层深队列上的窃取顺序（必须偷队尾，分段完成顺序为 $[0,3,1,2]$）。

### B4 `keep_first_n`（1h）

DARTS 代码里的选择规则：返回按 `(finish, sample_idx)` 最小的 `min(n,len)` 个样本在输入中的**位置**。测试 `b_keep_first_n` 用位置顺序与 `sample_idx` 顺序相反的并列来检查 tie-break，并用一个精确的顺序统计量 oracle：$m=3$，长度在 $\{1,2,3\}$ 上均匀，奖励为 $[\ell\ge2]$，保留 2 条，平均长度必须是 $5/3$（全体为 2），平均奖励必须是 $14/27$（全体为 $2/3$）。推导见讲义 §7.2。

### B 实验（1.25h）

`--mode bench --part B` 给出 64 个任务在 workers 2/4/8 $\times$ quantum 16/64 $\times$ 窃取开关下的 makespan、利用率、分段数和窃取数（12 行），以及合成组上“先完成者入选”的长度与奖励偏移（4 行）。画窃取开关的配对图和 quantum 与分段数的关系。这个模型不计迁移和恢复成本，报告里写明这一局限；只凭这些结果不能声称真实系统会变快。

## C 前沿：队列、裁剪梯度、版本界、解耦比值与闭环（9h）

### C1/C2 `GroupQueue::push` / `pop`（1.5h）

容量按 **group** 计：新组占一个槽，已占槽组的成员总能进入。拒绝：未来版本或超过 `max_lag` 的版本（计入 `stale`）、id 与 group 不匹配、已消费的组、重复 id、与组内已有成员版本不同。队列满时返回 false（计入 `backpressure`），由生产者重试。`pop` 先把整组超龄的组驱逐，再按组号返回最小的完整组。测试 `c_queue_admission`、`c_queue_drain`。

### C3 `Policy::objective`（1h）

$J=\frac1G\sum_i\min(\rho_iA_i,\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i)$，$\rho_i=\pi(a_i)/\mu(a_i)$，$A$ 用总体 std。$\pi$ 用当前 logit，$\mu$ 用记录的 `behavior_p`（动作 0 时为 $1-$`behavior_p`）。纯函数。测试 `c_objective` 用独立闭式比对四种情形，另有手算值：logit 3、奖励 $[0,1]$ 时 $J=-0.905148$。

### C4 `Policy::update`（1.5h）

在更新**前**的 logit 上对整组求梯度，一次上升：`logits[c] += lr * dJ/dz_c`。活跃分支上单样本梯度为 $\rho A(a-p)$；只有 $A>0,\rho>1+\epsilon$ 或 $A<0,\rho<1-\epsilon$ 时为 0。返回更新前的目标值，版本加一。

测试 `c_gradient` 对**你自己的** `objective` 做中心差分，与步长比较，四个用例：

| 用例              | logit | 奖励    | 两个样本所在情形 |           平均梯度 |
| ----------------- | ----: | ------- | ---------------- | -----------------: |
| $\rho=1$          |     0 | $[1,0]$ | 1/2              |                0.5 |
| $\rho\ne1$ 未裁剪 |   0.3 | $[1,0]$ | 1/2              |   $2p(1-p)=0.4889$ |
| 越界但活跃        |     3 | $[0,1]$ | 6 与 3           | $-2p(1-p)=-0.0903$ |
| 越界且裁剪        |     3 | $[1,0]$ | 5 与 4           |                  0 |

漏乘 $\rho$ 的实现在第二行得到 0.5；“$\rho$ 越界即零梯度”的实现在第三行得到 0。推导见讲义 §3。

### C5 `may_submit`（0.5h）

AReaL 的生成端准入：已提交 `generated` 条、持有版本 `version` 时，下一条可以提交当且仅当 $\lfloor\text{generated}/B\rfloor\le\text{version}+\eta$。测试 `c_staleness_bound` 先查几组直接值，再跑一个生产者-训练者模拟器：对 $B\in\{1,4\}$、$\eta\in\{0,1,2,3\}$，最大陈旧度必须**恰好**等于 $\eta$，而且不死锁。推导见讲义 §5.1。

### C6 `decoupled_terms`（1h）

一条分段轨迹的每个 token 返回 $(w,u)$：$w=\pi_{\text{prox}}(a)/\pi_{\text{behav}}(a)$，$u=\pi_\theta(a)/\pi_{\text{prox}}(a)$，behav 是**实际采样该 token 的版本**（用该 token 记录的 `behavior_p`），不是整条轨迹的一个版本。测试 `c_partial_ratios`：前两个 token 由 $v_5$ 采样、后两个由 $v_6$ 续写、$\pi_{\text{prox}}=v_7$，逐 token 比对，并检查 $wu=\pi_\theta/\pi_{\text{behav}}$。

### C7 `learn`（2h）

每波从同一快照采 4 组 $\times$ 8 条，经 B3 执行合成 token 工作量，按组的最后终止段排序送入队列；每次推入前把 `queue.version` 设为当前策略版本，超龄组整组丢弃；没有空槽时出队一组、更新（学习率 0.3、$\epsilon=0.2$），再重试被阻塞的组；波末清空并更新。每波记录精确的期望奖励和累计计数。测试 `c_learning_closed_loop`：`max_lag=4` 时 3 个种子 100 波后期望奖励 $>0.85$、恰好 400 次更新、无丢弃；单波 `capacity=2`、`max_lag=0` 时恰好更新 1 次、丢弃 24 条、背压 1 次（推导写在测试注释里）。

### C 实验（0.5h）

`--mode bench --part C` 扫 3 个种子 $\times$ 容量 1/2/4 $\times$ `max_lag` 0/1/4，输出第 0/9/49/99 波（108 行）。计数是**累计**快照。把 `max_lag=0` 和 4 的曲线配对，解释严格新鲜度为什么只用上了每波的一部分组，以及哪个 context 得到的更新更多。

## R 真实 trace：回放、长度预测与分布偏移（6h）

**trace 是怎么来的。** `real/sample_trace.py` 用 Qwen3-0.6B（bf16，`enable_thinking=False`）对 `real/prompts.jsonl` 的 32 道题各采 2 轮、每轮 8 条，温度 1.0、不截 top-k/top-p，最多 512 token，一组一次 `generate`（batch = 组）。文件头写着 `"label": "sampled with Qwen3-0.6B, not from GRPO training"`，以及设备、驱动、日期和库版本：RTX 4060 Laptop，驱动 566.24，CUDA 12.6，torch 2.14.1+cu126，transformers 5.18.0，2026-10-18。权重没有更新过，所有行 `policy_version=0`；`round` 只表示第几次采样，回放时把它当作一步 RL。

- `gen_ms` 是该样本出 EOS（或到 512）那一步的墙钟时间，从这组的 `generate` 调用开始计，包含 prefill。它是 batch = 组时单条样本的完成时刻，不是单独生成这一条的延迟。每个 decode 步由一个 `StoppingCriteria` 打时间戳（`time.perf_counter`）。采样前用同一个 `generate` 让 GPU 连续忙至少 300 ms，并在头尾记录了 `nvidia-smi` 的时钟；整次运行持有 `/tmp/mlsys-gpu.lock`，用时 8.1 分钟。
- 奖励按规则判：取最后一个 `####` 之后的整数与标准答案比较；没有 `####` 时退回全文最后一个数字（`fallback=true`，512 条里有 172 条），这是一个宽松的判分，报告里要说明。
- 实测摘要：长度 p50 208.5、p90 312、最大 512（5 条截断）；正确率 82%；decode 每步中位数 26.2 ms；`gen_ms` 对长度的线性拟合约 $27.3\,\text{ms/token}+36\,\text{ms}$。

schema（每行一个 JSON 对象；`type` 为 `header`/`group`/`footer` 的行是元数据，没有 `type` 或 `type` 为 `sample` 的行是样本）：

| 字段                               | 含义                             |
| ---------------------------------- | -------------------------------- |
| `prompt_id`、`group`、`sample_idx` | 题号、组号（全局唯一）、组内序号 |
| `round`                            | 第几步（可省，默认 0）           |
| `policy_version`                   | 生成该样本的策略版本             |
| `response_tokens`                  | 回答长度（到 EOS 为止）          |
| `gen_ms`                           | 完成时刻，ms                     |
| `reward`                           | 规则奖励                         |
| `truncated`                        | 是否到上限（可省）               |

**回放模型**（modeled，不是 measured）：每个 `round` 是一步同步 RL；该步的组在 `workers` 个同速 worker 上执行，一个 worker 一次跑一组（一次批量 `generate`），占用时间为它**保留**的样本里最大的 `gen_ms`；下一步在最慢的 worker 结束后开始。

### R2/R3 `QuantilePredictor::observe` / `predict`（1h）

按 prompt 保存最近 `window` 个已完成长度，同时维护最近 `16*window` 个的全局历史。预测取最近秩分位数 `sorted[max(ceil(q*n),1)-1]`：有本 prompt 的历史就用它，否则用全局历史，都没有时返回 `default`。`predict` 是纯函数。测试 `r_predictor`：手算的窗口与回退、全局窗口上限，以及 40 个随机用例与测试端独立实现的比对。

### R1 `replay`（3h）

逐轮：

1. 对本轮**每一组**先调用 `predictor.predict`，记录 `(group, 预测, 该组真实最长长度)`。本轮的任何长度都还不能被观测。
2. 选保留样本：全部；或 `KeepFirstN(n)` 用 B4 按 `(gen_ms, sample_idx)` 选；或 `FirstIdx(n)` 取 `sample_idx < n`（无偏子集）。组时长为保留样本 `gen_ms` 最大值四舍五入到毫秒。
3. 按策略求本轮 makespan：`Fifo`/`KeepFirstN`/`FirstIdx` 是共享队列拉取（等价于 `assign(时长, lpt=false)`）；`LptPredicted` 是按预测的静态 LPT；`LptPull` 是按预测降序排好再拉取；`WorkStealing` 是按预测 LPT 初始化后调用 B3（`quantum = u64::MAX`，不抢占）；`LptOracle` 是按真实时长的静态 LPT，只作诊断。
4. 本轮结束后才 `observe`，并且只观测**保留**样本的长度：被 abort 的样本从未完成，系统不知道它的长度。

报告保留样本和全体样本两套统计：平均长度、平均奖励、优势全为 0 的组数。测试 `r_replay_synthetic` 用一份 12 行的手算 trace（行故意乱序）逐策略核对每轮 makespan、窃取数、预测值和偏移；`r_replay_reference` 在参考 trace 上用测试端独立计算的拉取模拟、下界、保留样本统计与“第二轮预测只来自第一轮”核对你的结果。

### R 实验与必交结果（2h）

`--mode bench --part R` 对参考 trace（以及 `results/lab8-real/` 下的所有 trace）输出 workers 2/4/8 $\times$ 16 种策略配置的表。参考答案在 4 个 worker 上的结果（两轮总时长，下界 122.1 s）：

| 策略                     |  总时长 | 保留样本 | 保留平均长度 |  长度偏移 |
| ------------------------ | ------: | -------: | -----------: | --------: |
| `fifo`                   | 132.2 s |      512 |        218.5 |         0 |
| `lpt_predicted`，$q=1$   | 132.4 s |      512 |        218.5 |         0 |
| `lpt_pull`，$q=1$        | 127.2 s |      512 |        218.5 |         0 |
| `work_stealing`，$q=0.5$ | 125.6 s |      512 |        218.5 |         0 |
| `lpt_oracle`             | 122.6 s |      512 |        218.5 |         0 |
| `keep_first_4`           |  99.4 s |      256 |        190.3 | $-12.9\%$ |
| `first_idx_4`            | 122.0 s |      256 |        221.1 |  $+1.2\%$ |

必交：

1. 三张表（workers 2/4/8），每行同时列吞吐和分布偏移。说明为什么 `lpt_predicted` 第一轮不如 `fifo`（冷启动时所有预测相同）。
2. 预测器的 $2\times2$：{$q=0.5$, $q=1$} $\times$ {`lpt_pull`, `work_stealing`}，固定 worker 数，说明收益来自哪一项。
3. 同样本数下 `keep_first_4` 与 `first_idx_4` 的对照：时长差、长度偏移、奖励偏移、优势全零组数。写一句带口径的结论（讲义 D8-7 第 2 题）。
4. 用 `--device cuda` 采一份你自己的 trace（8 题 $\times$ 2 轮，约 2 分钟），附上它的头部信息，比较它与参考 trace 的长度分布和策略排序。没有 GPU 时说明原因，只交参考 trace 的结果。
5. （选做，计报告分）按 [GUIDE.md](real/GUIDE.md) 在 TRL 里跑 20 步 GRPO，导出训练 trace 并回放；凡指南标为“未验证”的项写明实测结果。

## 报告与验收（3h）

提交学生实现文件、全部测试日志（`results/` 下的 RECORD）、A/B/C/R 的原始 CSV、你采集的 trace，以及不超过 5 页的报告。报告至少包含：A 阶段的预测误差图；B 阶段窃取与 quantum 的配对图、先完成者入选的偏移表；C 阶段 `max_lag` 0/4 的学习曲线与版本时间线；R 阶段第 1–4 项；一个负结果（例如“窃取在这份 trace 上的收益小于预测误差带来的损失”）并区分哪些有观测支持、哪些仍是猜测；TODO 完成表和实际用时。评分见 [grade.md](grade.md)。

## 可选挑战

以下挑战不计入 30h，选一项继续：

- 给 B3 的每个分段加恢复成本（re-prefill 与上下文长度成正比），重新扫描 quantum，找出收益消失的临界点。
- 把 R1 的调度单位从“组”换成“单条样本”，模拟连续批处理下 abort 后立即释放槽位，比较 `keep_first_n` 的时间收益。
- 实现 RollPacker 式 tail batching：每轮起跑 $1.25B$ 个组，先完成的 $B$ 个进入本轮，其余进入 tail 队列，报告按长度推迟带来的 prompt 顺序变化。
- 用 TRL 训练 trace 测“旧策略长度预测新策略”的误差随训练步的漂移（讲义研究问题 (a)）。
- 在 C6 的基础上实现 verl 的 TIS 截断（`rollout_is_threshold`），用 C7 的闭环比较有无截断时的学习曲线。

扩展时保留现有测试和 oracle，确认改动没有破坏原来的契约。

## 资料汇总

| 资料                                                                                                                                                    | 类型 | 指定观察                                         | 时间       |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------ | ---------- |
| [讲义 D8](../../course/modules/08-rollout-scheduling.md) §3、§5、§7                                                                                     | 讲义 | 六种裁剪情形；$\eta$ 界的推导；顺序统计量的偏差  | 已计入讲义 |
| [AReaL](https://arxiv.org/abs/2505.24298)                                                                                                               | 论文 | §5.1 的 $\eta$ 条件、§5.2 的 Eq. 5，对照 C5/C6   | 20min      |
| [DARTS `ray_trainer_darts.py`](https://github.com/PKU-DAIR/DARTS/blob/e07825aee39612e0659ce7e5a38f8ae02a2250b2/darts/recipe/darts/ray_trainer_darts.py) | 代码 | `dynamic_batch_reorder` 与冗余分配，对照 A3/B4   | 15min      |
| [verl fully_async 文档](https://verl.readthedocs.io/en/latest/advance/fully_async.html)                                                                 | 文档 | `staleness_threshold`、四种模式图，对照 C1/C2/C7 | 10min      |
| [HF Deep RL Course Unit 8](https://huggingface.co/learn/deep-rl-course/unit8/visualize)                                                                 | 图解 | 六情形表，对照 C4                                | 5min       |
| [CS336 Assignment 5](https://github.com/stanford-cs336/assignment5-alignment)                                                                           | 课程 | GRPO adapter 的接口与测试，对照 A4/C3            | 选读       |
| [逐任务改编与上游实现](resources.md)                                                                                                                    | 来源 | 每个 TODO 借鉴了什么、简化到哪里                 | 15min      |
| [real/GUIDE.md](real/GUIDE.md)                                                                                                                          | 指南 | TRL GRPO 的显存估算与 trace 导出                 | 15min      |
| [评分标准](grade.md)、[参考答案导读](../../solutions/lab8/README.md)                                                                                    | 验收 | 完成后核对，不替代报告                           | —          |
