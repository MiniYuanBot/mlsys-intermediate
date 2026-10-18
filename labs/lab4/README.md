# Lab 4：推测解码的 SLO 控制，从条件接受率到真实 trace 回放

[讲义 D4](../../course/modules/04-speculative-slo.md) · [评分](grade.md) · [公开来源与借鉴边界](resources.md) · [参考答案导读](../../solutions/lab4/README.md)

> [!important] 你要交付什么
> 用 Rust 实现接受量与成本的计算、带删失的估计器、单请求与多请求的预算选择，以及有到达、排队和完成事件的推测服务模拟器（A/B/C）。R 阶段在 RTX 4060 上用 PyTorch 跑 Qwen3-0.6B（草稿）和 Qwen3-1.7B（目标），采集可回放的逐位置接受 trace 和 cudaEvent 成本表，再把它们放回 Rust 模拟器比较控制器。A/B/C 的输出标为 `modeled`；只有 R 阶段在本机测得的成本和加速标为 `measured`。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。在 `labs/` 下先安装、再检查：

```bash
bash tools/setup_env.sh --lab 4 && bash tools/check_env.sh --lab 4
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 4
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 4
```

| 阶段 | 需要                                                                                | WSL2 / Linux | Windows 原生 | macOS       |
| ---- | ----------------------------------------------------------------------------------- | ------------ | ------------ | ----------- |
| A    | Rust                                                                                | 可做         | 可做         | 可做        |
| B    | Rust                                                                                | 可做         | 可做         | 可做        |
| C    | Rust                                                                                | 可做         | 可做         | 可做        |
| R    | Rust + PyTorch；GPU 部分另需 NVIDIA GPU + transformers；模型 Qwen3-0.6B、Qwen3-1.7B | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |

A/B/C 只需要 Rust，不需要 GPU。R 阶段分两半：CPU 一半用仓库里提交的参考 trace 做 Rust 回放，并用 PyTorch（CPU 版即可）跑 R1/R2 的单元检查；GPU 一半加载两个 Qwen3 模型（bf16 合计约 5.5 GB 磁盘、4.6 GB 显存），重新采集 trace。模型默认放在 `~/models/Qwen3-0.6B` 和 `~/models/Qwen3-1.7B`，可以用 `MLSYS_MODELS` 改目录。暂时不想下载模型就加 `--no-models`（PowerShell 是 `-NoModels`），这时 R 阶段只能做 CPU 一半。缺少依赖时 runner 只打印一行 `MISSING ...` 和对应的 `fix:` 命令，退出码为 3；在 Windows 上请求 `--device cuda`，`fix:` 给出的是一条能直接复制的 WSL 命令。各平台的细节见[环境配置](../setup.md)。

## 文件和实施顺序

| 文件                                      | 学生负责     | 已提供                                                                                              |
| ----------------------------------------- | ------------ | --------------------------------------------------------------------------------------------------- |
| `src/policy_student.rs`                   | A1–A5、B1–B6 | `ProfilePoint::round_cost`、`Estimator::pooled`                                                     |
| `src/engine_student.rs`                   | C1–C4        | `Policy`/`Acceptance` 接口、`Fixed`、`PerPosition`、`Pooled`、`Synthetic`、`draw`                   |
| `src/replay_student.rs`                   | R3、R4       | JSON 读取、`Trace`、`TraceEnv`、`compare_before_divergence`、`schedule_ms`、`ArmUcb`、`HfHeuristic` |
| `real/accept_student.py`                  | R1           | —                                                                                                   |
| `real/profile_student.py`                 | R2           | —                                                                                                   |
| `real/engine.py`                          | —            | 静态 KV cache、每个宽度一个 CUDA graph、普通贪心与手写推测循环                                      |
| `real/collect.py`                         | —            | 采集驱动：成本表、贪心 trace、真实推测运行、耦合采样、HF 交叉检查                                   |
| `real/prompts.jsonl`                      | —            | 36 个课程自编 prompt（6 类任务，随本仓库许可证发布）                                                |
| `real/data/*.jsonl`                       | —            | 参考 trace（RTX 4060 Laptop，驱动 566.24，2026-10-18，73 KB）                                       |
| `tests/exercises.rs`、`real/test_real.py` | —            | 26 个 Rust 测试、R 阶段 Python 检查；每项打印 `PASS`/`FAIL`                                         |
| `src/main.rs`、`exercise.py`              | —            | benchmark 与阶段入口                                                                                |

以下命令都在 **`mlsys-intermediate/labs/`** 运行。按 A、B、C、R 的顺序完成；每个阶段的测试只调用本阶段和更早阶段的函数。

```bash
python tools/run.py --lab 4 --impl student --mode test --part A
python tools/run.py --lab 4 --impl student --mode test --part B
python tools/run.py --lab 4 --impl student --mode test --part C
python tools/run.py --lab 4 --impl student --mode test --part R                 # CPU：回放 + R1/R2 单元检查
python tools/run.py --lab 4 --impl student --mode test --part R --device cuda   # WSL：再加真实模型检查
python tools/run.py --lab 4 --impl student --mode bench --part all
python tools/run.py --lab 4 --impl student --mode bench --part R --device cuda  # 采集新 trace 并回放
```

直接调试一个测试：`cargo test --offline --release --manifest-path lab4/Cargo.toml --test exercises c_hand_timeline -- --nocapture`。每个测试会把每项检查打印成一行 `PASS 测试名/检查名` 或 `FAIL 测试名/检查名: 期望 vs 实际`，跑完所有检查后才判失败，一次就能看到全部问题。请保留所有 BEGIN/END TODO 标记；标记前的注释写了契约、边界和对应测试。可以加 helper、测试和 benchmark，不要把参考答案导入学生代码。

| 阶段       |   TODO | 参考实现有效行 | 预计时间 |
| ---------- | -----: | -------------: | -------: |
| A 基础     |      5 |             75 |       4h |
| B 核心     |      6 |            113 |       7h |
| C 前沿     |      4 |            116 |       7h |
| R 真实系统 |      4 |            118 |       7h |
| 报告       |      — |              — |       3h |
| **合计**   | **19** |        **422** |  **28h** |

有效行由 `python tools/workload_audit.py` 统计，只计参考答案 TODO 区域内的代码行，不要求你写得一样长。估时包括调试和小实验，讲义 5h 另计；R 阶段含一次约 5 分钟的 GPU 采集。尚未经过学生试做，请在报告里记录实际用时。

## A 基础：把“接受率”和“成本”变成可检验对象（4h）

### A1 `committed_distribution`（1h）

输入条件接受率 $a_1,\ldots,a_K$（代码里 0 起编号）和 $k\le K$，输出 $\Pr(N=n)$，$n=1..k+1$。第 $n$ 个桶是“前 $n-1$ 个接受、第 $n$ 个拒绝”，最后一个桶是全部接受，同样计入 bonus token。不变量：长度 $k+1$，和为 1，每项非负。边界：$k=0$ 返回 `[1.0]`。测试 `a_distribution_oracle` 枚举全部 $2^k$ 个接受/拒绝向量，按首次拒绝的位置聚合，不调用你的函数。

### A2 `committed_moments`（0.5h）

从 A1 的分布算均值和方差，不要用常数接受率的捷径。`[0.5,0.25]` 应得均值 1.625、方差 0.484375；讲义例子 `[0.8,0.5,0.25]` 应得 2.3 和 0.81。

### A3 `percentile`（0.5h）

nearest-rank：$\mathrm{rank}=\max(1,\lceil qn\rceil)$，返回排序后第 rank 个。输入可以乱序，不能修改调用方的数组；空输入返回 `None`。`[1,2,3,4]` 的 p50 是 2、p75 是 3。R2 的 Python 版必须用同一个定义，否则 Rust 端的校准和 Python 端的汇总会对不上。

### A4 `interpolate`（1h）

`ProfilePoint` 的 `verify[k]` 是验证 $k$ 个候选的耗时，`draft` 是草稿一步的耗时，单位 ms。按上下文线性插值，区间外夹到边界值。手算 context 100 与 300 之间的 150 和 200。报告里要说明区间外是夹取，不能把它写成该区间的实测结果。

### A5 `calibrate`（1h）

把重复测量按 `(context, k)` 分组，取 A3 的中位数，context 升序；任何 context 缺了某个 $k$ 就报错。缺失值不能填 0，否则验证会被当成免费，算出无限大的收益。R 阶段把 4060 上的真实 cudaEvent 样本喂给这个函数。

**A 小实验**：`--mode bench --part A` 输出两张成本表（`optimistic` 即 $c=0.08$，`rtx4060_graph` 即实测量级 $c\approx0.5$）下各 $\alpha$、各 $k$ 的期望提交量、方差、每 ms 产出和最优 $k$。画出两张表下 $R(k)$ 随 $k$ 的曲线，解释为什么同一个 $\alpha$ 在两种成本下的结论不同。

## B 核心：删失、保守分数、预算（7h）

### B1 `Estimator::observe`（1h）

计数从 0 开始。每轮先把**所有**位置乘以 decay（包括 $k=0$ 的轮次），再给已接受的前缀加成功、给首次拒绝的位置加失败；没走到的后缀不动。把后缀记成失败等于把 vLLM 的 $P(A\ge j)$ 当成 $a_j$，见讲义 §1.3。测试 `b_censored_observation_and_decay` 逐步检查计数。

### B2 `Estimator::probabilities`（1h）

$z=0$ 返回 Beta(1,1) 均值 $(s+1)/(n+2)$；$z>0$ 返回 Wilson score 区间的下端，$n=0$ 时返回 0。先自己推导下端公式（讲义 §4.2），再实现。测试 `b_wilson_numeric_and_ranking` 用二分法解 Wilson 的二次方程得到数值答案，在 $z=1$ 和 $z=1.96$、8 组计数上逐个比较，误差容限 $10^{-9}$；只做到“1/1 低于 100/100”这种排序是过不了的。

### B3 `choose_k`（1h）

在 $T(k)\le B$、$k\le r-1$ 中最大化 $\mathbb{E}[N_k]/T(k)$，速率差在 $10^{-12}$ 内视为并列，选较短的 $k$；没有可行项或 $r=0$ 时返回 `None`。测试对 25 组概率、6 种 remaining、6 种预算共 900 个情形做穷举对比，另有精确并列和严格最优的手算例子。

### B4 `Drift::update`（1h）

单侧 Page-Hinkley 检测：用**旧**均值更新累计量，再更新均值；超过阈值时计数、清空窗口、把均值设成当前值。测试 `b_drift_exact_events` 先用一段手算轨迹（阈值 2.1 时恰好在第 22 步触发一次），再用测试里独立写的检测器在随机交替的数据上逐步比对触发位置，最后检查单调上升不触发。

### B5 `allocate_round`（1.5h）

多个请求共享一轮预算。先给每个请求 $k=0$，基础都放不下就返回 `None`；然后反复选“边际增益 / 额外成本”最大、放得下、增益为正的单个升级；并列时选编号小的请求，再选较短的 $k$。这是启发式。测试检查：手算轨迹、并列规则、“最好比值优先而非第一个”的例子，以及 300 个随机实例上 (1) 与测试端独立实现的同一契约一致，(2) 不超预算，(3) 不超过穷举最优，(4) 停下时没有放得下的正收益升级。测试**不要求**贪心最优。

### B6 `allocate_round_exact`（1.5h）

把成本按量子离散成整数刻度，做多选背包的动态规划，复杂度 $O(nKB/\delta)$。刻度取 $\lceil T/\delta-10^{-9}\rceil$，容量取 $\lfloor B/\delta+10^{-9}\rfloor$；容差是为了吸收 $2.6/0.1=26.000000000000004$ 这类二进制浮点误差。价值差在 $10^{-9}$ 内视为相等，返回字典序最小的最优向量。测试在 300 个随机实例上和穷举比最优值和并列规则。

**贪心反例**（写进报告）：两个请求，基础分配后剩 1.6。A 的升级增益 0.6、成本 1.0；B 的升级增益 0.9、成本 1.6。贪心按比值选 A（0.6 对 0.5625），剩下的 0.6 放不下 B，总增益 0.6；DP 只选 B，总增益 0.9。测试 `b_exact_matches_exhaustive` 检查 DP 选 B、贪心的值不超过 DP。

**B 对照实验**：`--mode bench --part B` 输出两张表。第一张是 high/low/recovery 三段、每段 40 步的固定观测回放（每步都提出 4 个），比较均值控制与保守控制在相同证据下的决策；它不能证明在线恢复能力。第二张是 2/4/8 个请求各 200 个随机实例上贪心相对 DP 的次优比例和最坏比值。

## C 前沿：从 round 比值到完整请求服务（7h）

### C1 `State::commit`（1h）

在验证完成时一次性提交 $\min(A+1,\text{remaining})$ 个 token，返回提交数；每个 token 记一个可见时刻（同一轮的时刻相同），首 token 和完成时刻各只记一次。

### C2 `enqueue`（0.5h）

把到达时间不晚于 `now` 的请求按顺序放进就绪队列，每个只放一次；同一时刻的多个到达都要推进 cursor。

### C3 `simulate`（4h）

单服务台、轮级 round-robin 的离散事件引擎。契约写在函数注释里，要点：

- 就绪队列空时跳到下一个到达；每轮的预算是 $\min(\text{round bound},\ \text{deadline}-\text{now})$。
- 控制器通过 `Policy` 接口只看到成本、预算、剩余长度和它自己的历史观测，读不到 `Request::accept`。
- **所有**策略（包括 `Fixed(k)`）在 $T(k)$ 超过预算时都记一次 infeasible；策略返回 `None` 时执行 $k=0$ 并记一次。不能删掉迟到的请求来提高 goodput。
- 每轮把 accepted/k 喂给 B4 的检测器，触发时调用 `policy.reset()`。
- 本轮期间到达的请求先入队，正在运行的请求随后回队。每轮追加一条 `RoundRecord`。

环境的接受由 `Acceptance` 接口给出。合成环境 `Synthetic` 按 `(seed, request, emitted, depth)` 派生随机数（common random numbers），并且**按深度给条件接受率**：每个请求带一个 `accept` 向量，C 的 benchmark 用了随深度衰减、平坦、随深度上升三种形状，所以按位置估计和池化 i.i.d. 估计会给出不同的决策。R 阶段用同一个接口把真实 trace 接进来。

测试：`c_hand_timeline`（手算两请求时间线和空闲跳转）；`c_infeasible_counts_all_policies`（`Fixed(2)` 超预算、截止时间逼近、预算低于 $T(0)$ 三种情况的 infeasible 计数和逐轮标记）；`c_mean_policy_shadow_replay`（测试端独立写一个均值控制器，按你的轮日志逐轮重放，检查每轮的 $k$、接受数、时长、infeasible 标记、提交数和重置）；`c_drift_reset_replay`（同样的影子重放，用一段从全接受变为全拒绝的负载触发 Page-Hinkley 重置）；`c_per_position_beats_pooled_iid`（每轮恰好接受 2 个的非 i.i.d. 负载上，按位置估计稳定在 $k=2$，池化估计更慢）；`c_conservation_empty_and_overload`（守恒、空输入、过载、no-probe 与 $k=0$ 时间线相同）。

### C4 `summarize`（1.5h）

$\mathrm{TTFT}=t_{\mathrm{first}}-t_{\mathrm{arrival}}$，$\mathrm{E2E}=t_{\mathrm{finish}}-t_{\mathrm{arrival}}$，平均 $\mathrm{TPOT}=(t_{\mathrm{finish}}-t_{\mathrm{first}})/(n-1)$（输出 1 个 token 的请求不计），三者都报 nearest-rank p95。达标 = 完成时刻不晚于绝对截止时间；goodput = 达标数 /（最后完成时刻减首次到达时刻），输入是 ms，输出是请求/秒。逐个请求检查 token 守恒。空输入全部返回 0。

**C 对照实验**：`--mode bench --part C` 在两张成本表、3 个到达间隔（以 $T_v(0)$ 为单位 8、14、30）、5 个 seed、6 个策略的全部组合上输出 180 行（`k0`、`k2`、`k4`、`mean`、`pooled_iid`、`no_probe_z1`）。至少画出 p95 TPOT、goodput 和 infeasible 随到达间隔的变化，报告 5 个 seed 的均值和离散程度。要回答：为什么同一个 `mean` 控制器在 $c=0.08$ 的表下会探索、在 $c\approx0.5$ 的表下停在 $k=0$；这在 $c\approx0.5$ 下算不算错误决策。再附一个过载反例：每轮都满足 round bound，排队仍让请求错过截止时间。

## R 真实系统：Qwen3-0.6B → 1.7B 的接受 trace 与成本（7h）

R 阶段的主工具是手写循环（`real/engine.py`），HF assisted generation 只做交叉检查。原因有两个：HF 不暴露每轮的接受数，而且它默认 `assistant_confidence_threshold=0.4`，会提前停止草稿，不是固定 $k$。

**引擎（已提供，先读懂）**：两个模型都用 `StaticCache`，目标模型对宽度 1..9 各捕获一个 CUDA graph，草稿对宽度 1、2 各捕获一个。eager 模式下草稿一步约 27.8 ms、目标约 29.0 ms（$c\approx0.96$），推测永远不划算；graph 模式下约 9.0 ms 和 18.1 ms（$c\approx0.50$）。transformers 5.18 的 `StaticLayer` 不看 `cache_position`，而是在设备端的 `cumulative_length` 后面追加，所以引擎每次调用前（graph 里也一样）把这个计数器改写成起始位置。推测循环每轮只同步一次 host；一轮全部接受后，草稿要补喂 2 个 token（宽度 2）。

**数据与许可**：`real/prompts.jsonl` 的 36 个 prompt 是课程自己写的（解释、写作、代码、数学、翻译、抽取各 6 个），跟随本仓库的 MIT 许可证。Qwen3 的 chat template 默认开启思考模式，引擎用 `enable_thinking=False`，因为思考 trace 会改变接受率。每个 prompt 最多生成 128 个 token。

### R1 `accept_student.py`：从模型输出得到接受数据（2h）

一个 TODO 区域里有四个纯函数：

- `greedy_flags(draft_argmax, prompt_len, generated)`：`draft_argmax[i]` 是草稿看到 `sequence[:i+1]` 后的猜测，即对 `sequence[i+1]` 的猜测。返回长度 `len(generated)-1` 的 0/1 列表，第 $i$ 位表示草稿对 `generated[i+1]` 猜对。`generated[0]` 来自 prefill，不参与推测。这一步最容易差一位。
- `position_rates(flags, kmax)`：对每个 $j$，只在第 $j$ 个深度可观测的起点（$i+j\le n$）上算 $S_j=P(L\ge j)$ 和 $a_j=P(L\ge j\mid L\ge j-1)$。
- `hazard_from_survival(survival)`：把 vLLM 风格的 $S_j$ 换成 $a_j$，测试用 vLLM PR #17010 的例子 $[1,0.6,0.2]\to[1,0.6,0.333]$。
- `coupled_step(p, q, u_draft, u_accept, u_resid)`：一次推测采样决策，用给定的三个均匀数：逆 CDF 从 $q$ 抽草稿，`u_accept * q[x] < p[x]` 时接受，否则逆 CDF 从残差 $\max(p-q,0)$ 抽纠正；同时返回 $\beta=\sum\min(p,q)$。

测试（`real/test_real.py`）：手算对齐例子和 200 个随机对齐；与测试端暴力计算的 $S_j$、$a_j$ 逐项比较；i.i.d. 标记下 $S_j$ 按 $\alpha^j$ 衰减而 $a_j$ 保持平坦；`coupled_step` 用 2 万次 Monte Carlo 检查输出分布等于 $p$、接受率等于 $\sum\min(p,q)$（4 倍标准差容限），再检查几个逆 CDF 的手算点。`--device cuda` 时还会加载两个模型，用你的 `greedy_flags` 生成标记，和手写循环 $k=2,4$ 的真实轮日志在分叉之前逐轮对比（至少 95% 相同）。

### R2 `profile_student.py`：测量方法（0.5h）

- `measure(fn, timer, warmup, reps)`：先不计时地调用 `warmup` 次，再计时 `reps` 次，只返回计时样本。`collect.py` 传入的 timer 用 `cudaEventRecord` 包住被测调用，`stop()` 在结束事件上同步；调用 `measure` 之前，`collect.py` 先让 GPU 用同一个调用连续忙至少 300 ms，避免笔记本 GPU 停在低 P-state（显存频率 810 MHz 对 8000 MHz）。
- `summarize(samples)`：nearest-rank 的中位数、p25、p75、样本数，以及 `stable`（$p_{75}\le1.5\,p_{25}$），定义与 A3 相同。`collect.py` 对不稳定的行最多重测两次，仍不稳定就保留 `stable: false`；Rust 测试要求参考 trace 的 graph 行全部稳定。

测试用一个假时钟：前 3 次调用各耗 100、50、40 ms（模拟首次调用的分配和频率爬升），之后是正常值。把 warmup 也算进样本的实现会被抓到。另有双峰样本（一半在低 P-state）检查 `stable` 为假。

### R3 `replay_student.rs`：删失回放（2.5h）

一个 TODO 区域里有三个函数：

- `run_lengths(flags)`：从每个位置起连续 1 的个数，倒序一遍 $O(n)$。
- `censored_accept(runs, emitted, k)`：$\min(L(e),k)$。控制器只能知道它付过钱的那一段，$k=0$ 什么都不揭示。
- `fixed_k_schedule(runs, k)`：固定 $k$ 的闭式回放，提出数 $\min(k,n-e-1)$，下一轮从 $e+A+1$ 开始。

已提供的 `TraceEnv` 用 `censored_accept` 实现 `Acceptance`，所以 C3 的引擎可以原样跑真实 trace。测试：暴力算 run length；对每个 $(e,k)$ 检查删失；固定 $k$ 的回放同时与测试端暴力和你的 C3 引擎比对；在参考 trace 上，回放的逐轮 (提出, 接受) 在分叉前与真实循环的记录至少 95% 相同（参考答案 98.4%，即 808 轮中 795 轮）。不到 100% 的原因是 bf16：目标模型在宽度 $k+1$ 和宽度 1 下舍入不同，草稿在补喂 2 个 token 时也一样。参考 trace 里 8 个 prompt 各跑 5 个 $k$，共 40 次运行，只有 5 次全程等于普通贪心；分叉处目标 top-2 logit 的差全部不超过 0.25（1–2 个 bf16 ulp）。

### R4 `CensoredUcb::upper`：用删失结构做乐观探索（1h）

每个位置返回 Wilson 区间的上端，没观测过的位置返回 1。已提供的 glue 把它交给 B3 选 $k$，观测仍走 B1。为什么这样能探索：乐观值只在走到那个位置时被修正，而一轮长 $k$ 会同时修正所有更浅的位置（讲义 §4.4）。测试用二分法解 Wilson 二次方程的上根逐个比对，再检查行为：全接受的 trace 上用满 $k$；全拒绝的 trace 上探测稀少，但证据衰减后仍会继续探测；在容易的 trace 上比 no-probe 下界快。

### R 实验与必交结果（1h，含一次约 5 分钟的 GPU 采集）

1. 在 WSL 里跑 `--mode bench --part R --device cuda`。它先用 `real/collect.py` 采集一份新 trace 到 `results/lab4-real/trace-<时间>.jsonl`（含 graph 和 eager 两种成本表、36 个 prompt 的贪心标记、8 个 prompt 各跑 5 个 $k$ 的真实推测运行、6 个 prompt 的耦合采样 trace、2 个 prompt 的 HF 交叉检查），再同时回放参考 trace 和你的 trace。没有 GPU 时，`--mode bench --part R` 只回放参考 trace。
2. **成本表**（`measured`）：两种执行方式下 $T_v(L,k)$、$T_d$、$c$。
3. **$S_j$ 与 $a_j$**（`measured`）：全体和按任务。回答 $a_j$ 是平的、衰减还是上升，并解释选择效应。
4. **固定 $k$ 的实测加速与模型**：实测加速、回放 + 成本表的预测、i.i.d. 公式的预测，以及预测耗时与实测耗时之比。参考 trace 上实测 $k=1,2,3$ 分别为 1.06、1.04、0.96，预测耗时比实测少 2%–6%。说明差在哪里。
5. **控制器对比**（`modeled`，输入是 measured 的 trace 和成本表）：`k0..k6`、`mean`、`pooled_iid`、`no_probe_z1`、`hf_heuristic`、`arm_ucb`、`censored_ucb` 在 sequential 和 loaded 两种到达下的总时长、p95 TPOT、达标数和相对最好固定 $k$ 的 regret。参考 trace 上最好的是 `k2`，`censored_ucb` 慢 2.3%，`arm_ucb` 慢 7.1%，均值类控制器停在 $k=0$、慢 19%。你的结果可能不同，照实报告。
6. 两份 trace 的头部信息（设备、驱动、日期、torch 版本）和 `check_env` 输出一起附在报告里。

## 报告与验收（3h）

提交学生实现文件、全部测试日志（`results/` 下的 RECORD）、A/B/C/R 的原始 CSV、你采集的 trace，以及不超过 5 页的报告。报告至少包含：A 阶段两张成本表下的 $R(k)$ 曲线；B 阶段贪心反例和 DP 的对照；C 阶段 p95 TPOT/goodput/infeasible 三张图和一个过载反例；R 阶段第 2–5 项的表或图；一个负结果（例如“在这个 trace 上没有自适应策略赢过最好的固定 $k$”）以及你对原因的判断，区分哪些有观测支持、哪些仍是猜测；TODO 完成表和实际用时。评分见 [grade.md](grade.md)。

## 可选挑战

以下挑战不计入 28h，选一项继续：

- 给 `CensoredUcb` 加上代价感知的探测计划（例如只在估计的探测成本低于预期收益时才拉长 $k$），在按任务切换顺序排列的 trace 上和 `ArmUcb` 比较切换后 200 轮的累计 regret。
- 用 R 阶段记录的草稿 top-1 概率（trace 里的 `draft_p`）做每轮停止，和 HF 的 0.4 阈值、SpecDec++ 的阈值策略比较。
- 把耦合采样 trace（`type: sample`）接进 Rust 回放，比较贪心与 $T=0.6$ 采样下的 $a_j$ 和最优 $k$。
- 把单服务台扩展成 batch 验证：同一轮多个请求共享一次目标前向，成本按 $\text{batch}\times(k+1)$ 个位置重新测量，再让 C3 调用 B6。

扩展时保留现有测试和 oracle，确认改动没有破坏原来的契约。

## 资料汇总

| 资料                                                                                            | 类型 | 指定观察                                                      | 时间       |
| ----------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------------- | ---------- |
| [讲义 D4](../../course/modules/04-speculative-slo.md) §1、§4、§6                                | 讲义 | 三种接受率的换算；删失结构 $A(k)=\min(L,k)$；回放精确性的条件 | 已计入讲义 |
| [Leviathan 等](https://arxiv.org/pdf/2211.17192)                                                | 论文 | Eq. 1 与 Figure 3，对照 A 阶段的两张成本表                    | 15min      |
| [HF assisted decoding 文档](https://huggingface.co/docs/transformers/main/en/assisted_decoding) | 文档 | 不支持 batch；`assistant_confidence_threshold` 的默认行为     | 10min      |
| [AdaServe 论文](https://arxiv.org/abs/2501.12162)                                               | 论文 | goodput 与 TPOT SLO 的定义，对照 C4                           | 15min      |
| [BanditSpec 代码](https://github.com/sail-sg/BanditSpec)                                        | 代码 | README 中臂的定义，对照 `ArmUcb`                              | 10min      |
| [逐任务改编与上游实现](resources.md)                                                            | 来源 | 每个 TODO 借鉴了什么、简化到哪里                              | 15min      |
| [评分标准](grade.md)、[参考答案导读](../../solutions/lab4/README.md)                            | 验收 | 完成后核对，不替代报告                                        | —          |
