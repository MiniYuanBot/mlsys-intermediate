# Lab 3：动态形状、编译摊销与 CUDA graph 分桶

> [!abstract] 产出
> 完成 **21 个 TODO**：A1–A5 是 trace 与计时统计，B1–B5 是有预算的分桶 DP 和运行时状态，C1–C4 把计划接到真实的 `torch.compile`，C5–C7 是带 warp shuffle 和向量化访存的融合 residual-add + RMSNorm CUDA kernel，R1–R4 在 4060 上为一个 Qwen3-0.6B 形状的 decode block 挑选 CUDA graph 捕获尺寸。每个检查打印一行 PASS 或 FAIL，跑完一个阶段才汇总。

[课程 D3](../../course/modules/03-dynamic-compilation.md) · [资料与借鉴](resources.md) · [评分](grade.md) · [参考解答](../../solutions/lab3/README.md)。命令都从 `labs/` 执行。

## 0. 环境

本 lab 的依赖与其他 lab 互不累加。在 `labs/` 下先安装、再检查：

```bash
bash tools/setup_env.sh --lab 3 && bash tools/check_env.sh --lab 3
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 3
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 3
```

| 阶段 | 需要                                    | WSL2 / Linux | Windows 原生 | macOS       |
| ---- | --------------------------------------- | ------------ | ------------ | ----------- |
| A    | Python 标准库                           | 可做         | 可做         | 可做        |
| B    | Python 标准库                           | 可做         | 可做         | 可做        |
| C    | PyTorch；GPU 部分另需 nvcc + NVIDIA GPU | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |
| R    | PyTorch；GPU 部分另需 NVIDIA GPU        | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |

- C1–C4 在 CPU 上调用 Inductor 的 C++ 后端。Windows 原生上它要用 MSVC 的 `cl` 编译生成的 C++，找不到 `cl` 时 runner 打印 `MISSING cl` 并以退出码 3 结束，这时请到 WSL 里做 C。GPU 上的 Inductor 和 C5–C7 只在 WSL/Linux 下跑。
- C5–C7 的正确性测试 `test.cu` 不依赖 torch，只有 `inductor_bench.py` 的对比需要 torch。
- R 在 CPU 上测试选择、补零和规划逻辑（用 `emulate` 后端代替真实捕获），bench 用仓库里提交的 RTX 4060 实测 profile 做规划，输出标成 modeled。R 的 GPU 部分捕获真实的 `torch.cuda.CUDAGraph`。
- 缺依赖时 runner 只打印一行 `MISSING ...` 加 `fix:`，退出码为 3。在 Windows 上请求 CUDA，`fix:` 给出的是一条能直接复制的 WSL 命令。细节见[环境配置](../setup.md)。

## 文件和实施顺序

| 文件                                    | 学生负责                      | 已提供                                                                                             |
| --------------------------------------- | ----------------------------- | -------------------------------------------------------------------------------------------------- |
| `src/pipeline_student.py`               | A1–A5、B1–B5                  | `Sample`、`ArtifactCache` 骨架                                                                     |
| `src/execution_student.py`              | C1–C4                         | `operation`、`request_tensors`、`synchronize`                                                      |
| `src/fusion_student.cuh`                | C5 `block_reduce_sum`、C6、C7 | launch 包装                                                                                        |
| `src/graphs_student.py`                 | R1–R4                         | `GraphRunner` 的捕获部分、`plan_cost`、`plan_memory`                                               |
| `pipeline_test.py`、`execution_test.py` | —                             | A/B 与 C1–C4 的独立 oracle（穷举、Fraction、FP64）                                                 |
| `test.cu`、`fusion_common.cuh`          | —                             | C5–C7 的 FP64 oracle、两 kernel 基线（不依赖 torch）                                               |
| `bench.cu`、`inductor_bench.py`         | —                             | 融合 kernel、两 kernel 基线和 Inductor kernel 的同口径计时                                         |
| `bench_scenarios.py`                    | —                             | C 的 stable/shift/tail 三组真实 trace 对比                                                         |
| `real/decode_block.py`                  | —                             | Qwen3-0.6B 宽度的 decode block、合成 continuous batching trace                                     |
| `test_r.py`、`bench_r.py`               | —                             | R 的测试与 profile、规划、回放                                                                     |
| `real/data/profile_rtx4060.json`        | —                             | 2026-10-18 在 RTX 4060 Laptop 上实测的 profile                                                     |
| `torch_bench.py`                        | —                             | 旧的独立诊断，[PyTorch 桥接指南](../../course/guides/pytorch-bridge.md)仍在用，不属于本 lab 的验收 |

按 A → B → C → R 的顺序做，后面的阶段复用前面的函数：B1 调用 A3，C3/C4 调用 B，R4 调用 B1。A5 的测试临时替换成测试里自带的 `route`，所以做 A 时不需要先完成 B2。

```bash
python tools/run.py --lab 3 --impl student --mode test --device cpu --part A
python tools/run.py --lab 3 --impl student --mode test --device cpu --part B
python tools/run.py --lab 3 --impl student --mode test --device cpu --part C    # C1-C4
python tools/run.py --lab 3 --impl student --mode test --device cuda --part C   # + C5-C7
python tools/run.py --lab 3 --impl student --mode test --device cpu --part R
python tools/run.py --lab 3 --impl student --mode test --device cuda --part R
python tools/run.py --lab 3 --impl student --mode bench --device cuda --part all
```

**数据契约。** trace 是按到达顺序排列的 `[length, ...]`。C 阶段只改变行数，宽度和 dtype 固定。RMSNorm 每一行独立归一化，所以**可以补行，不能随意补宽度**。每个候选上界的 profile 包含 `steady_ms`、`eager_ms`、`first_premium_ms`、`mad_ms`，其中 $\text{premium}=\max(\text{first}-\text{median(steady)},0)$ 是当前进程和当前缓存状态下的附加成本，不能叫作冷编译时间。

## A 基础：先把时间和工作量算对（7h）

### A1 `split_trace`（1h）

- 输入：trace、训练条数 $k$。输出：前 $k$ 条请求的直方图（按长度排序的 `(length, count)`）和其余请求组成的 held-out 列表（保持原顺序）。
- 不变量：先按时间切分，再统计；直方图计数之和等于 $k$。
- 边界：长度必须是正整数；训练段和 held-out 段都不能为空，否则抛 `ValueError`。
- 测试：手算的 `[8,4,8,99,99]`，以及用 `Counter(trace[:k])` 独立重算的第二条 trace。

### A2 `summarize`（1.5h）

- 输入：`Sample(bound, first_ms, steady_ms, eager_ms)` 列表。输出：`{bound: {steady_ms, eager_ms, first_premium_ms, mad_ms}}`。
- steady 和 eager 取中位数，MAD 为相对 steady 中位数的绝对偏差的中位数，premium 小于 0 时取 0。
- 边界：重复的 bound、空样本、NaN 或负数都抛 `ValueError`。
- 测试：`[1,1,200]` 的手算例子、偶数个样本的中位数、first 小于 steady 的情况。

### A3 `segment_costs`（1.5h）

- 输出 `costs[i, j]`：区间 $[i,j]$ 内的长度都补到 $n_j$，成本为一次 premium 加上 $\sum f\cdot t(n_j)$。查实测表，不做线性假设。
- 边界：直方图未排序或有重复长度时抛 `ValueError`。
- 测试：三个长度的 6 个区间都在测试里逐一手算。

### A4 `trace_shift`（1h）

- 输出训练分布 $p$ 与最近窗口经验分布 $q$ 的总变差 $\text{TV}=\tfrac12\sum_n\lvert p(n)-q(n)\rvert$，在两者支撑集的并集上求和。
- 边界：两边都空返回 0，只有一边空返回 1；长度或计数不为正时抛 `ValueError`。
- 测试：0.25、0.5 等分数用例和随机用例，与测试里用 `Fraction` 精确计算的 oracle 比较。只测 0 和 1 抓不住忘记除以 2 的错误。

### A5 `padding_account`（1h）

- 输出按频次加权的 `real_rows`、`executed_rows`、`padding_fraction`、`fallback_calls`。fallback 请求按真实行数执行，不补齐。
- 测试：手算的 77/88/2，以及用测试自带的 first-fit 重算的第二组直方图。

**A 的手算（写进报告）：** ① `[8,4,8,99,99]` 前 3 条切分，若先排序再切分，泄漏了什么？② `[1,1,200]` ms 的均值、中位数、MAD。③ 长度 $(4,3),(8,2),(16,1)$，bound 16 的 premium 为 7 ms、warm 为 4 ms，为什么整段成本是 31 ms？④ 训练 $\{2{:}2,4{:}2\}$、最近 $[2,2,2,4]$ 的 TV。⑤ `[(3,10),(7,1),(20,2)]` 路由到 `[4,8]` 时的 padding 比例。

## B 核心：有限资源下的规划和执行（9h）

### B1 `plan`（3h）

- 输入：直方图、profile、容量 $K$。输出：升序的上界列表，最后一个等于最大长度，长度不超过 $K$。
- 递推 $D_{k,j}=\min_{i<j}\{D_{k-1,i}+c(i+1,j)\}$，答案取 $\min_{k\le K}D_{k,m}$，然后回溯。
- 并列规则：bucket 数少的优先；同样多时，最后一段起点最早的优先。
- 测试：对 4 种 premium、5 种容量以及随机的非线性 profile 枚举全部划分，与 DP 结果比较；另有两个专门构造的并列用例。

### B2 `route`（1h）

- 返回不小于长度的最小上界。超过所有上界，或者 $\text{bound}/\text{length}>\text{max\_padding}$ 时返回 `None`（走 eager）。比值**恰好等于** `max_padding` 时允许补齐。
- 测试：比值恰好为 2 的边界、长度正好等于某个上界、1 到 34 的全部长度与线性 first-fit 逐个比较。

### B3 `ArtifactCache.put`（2h）

- 事务式 LRU：已有的键刷新值并移到最近使用；新键在满时淘汰最久未用且**没被 pin** 的项；找不到可淘汰项就返回 `False`，键、值和顺序都不变；被 pin 的键不能被替换。
- 测试：`put(a),put(b),lease(a),put(c),lease(c),put(d)` 的完整追踪，以及刷新后 LRU 顺序的检查。

### B4 `ArtifactCache.lease`（1h）

- `lease(key)` 让 pin 计数加 1 并移到最近使用，键不存在时返回 `None`；`lease(key, release=True)` 让计数减 1，没有 lease 时抛 `ValueError`。
- 测试：两次 lease、一次 release 后仍然被 pin；lease 会改变淘汰顺序。

### B5 `admit`（1h）

- 当且仅当 $n\,(t_e-t_c-m\cdot\text{MAD})>P$ 且每次的保守收益为正时准入，等号拒绝。$n$ 只能来自历史预测。
- 测试：10 次（恰好回本，拒绝）与 11 次（准入）、$m=4$（永不准入）、编译不更快的情况。

**B 的分析（写进报告）：** DP 只在给定成本表和“连续区间”约束下最优。guard 会把 padding 过多的请求送去 fallback，admission 可能拒绝编译，cache 也会淘汰，所以真实运行成本和 DP 的预估不同。选一个实测 profile，画出所有划分的成本和 DP 选中的那个。

## C 前沿：让计划真正控制编译和执行（12h）

### C1 `measure`（1h）

- 第一次调用单独计时，然后做 `warmup` 次不计时的调用，再计时 `repetitions` 次。每次计时调用前后都要同步设备。
- 测试：调用次数、慢的第一次调用不能混进 steady 样本；CUDA 上一个 20M 周期的 GPU sleep 必须出现在样本里（检查是否同步）。

### C2 `calibrate`（1.5h）

- 每个候选上界都真正执行 `torch.compile(operation, backend="inductor", dynamic=False, fullgraph=True)`，检查结果等于 eager，用 C1 计时 compiled 和 eager。
- `calibration_ms` 是整个候选循环的 wall time，包括最后不会被选中的候选。
- 测试：artifact 必须是 Dynamo 包装后的函数；每个 artifact 都与 FP64 oracle 比较；`calibration_ms` 不小于所有 premium 之和。

### C3 `execute`（2h）

- route → admit → 用 `(device, dtype, width, bound)` 作为键 lease 或 put → 同时补齐 x 和 residual → 调用 → 切回真实行数 → 在 `finally` 里 release。不能 lease 时走 eager。
- 测试：5 个请求的路径（补齐、过大、超出 padding 比例、正好等于上界）、每个输出都与 FP64 oracle 比较、执行后没有残留的 pin、admission 拒绝的路径、artifact 抛异常时也要 release。

### C4 `workflow`（2.5h）

- 先切分，只校准训练长度，按 horizon 倍数预测，规划，用 `request_tensors(n, width, device, HOLDOUT_SEED + i)` 生成 held-out，计时回放（`replay_ms` 不包含校准），再计时一次 eager 回放。
- 返回 `bounds, profiles, events, outputs, calibration_ms, replay_ms, eager_replay_median_ms, eager_first_ms, horizon_multiplier`。
- 测试：每个 held-out 输出都独立重建输入，与 FP64 oracle 比较；profile 只覆盖训练长度；`replay_ms` 与 `calibration_ms` 分开；事件顺序与 held-out 一致。

### C5 `block_reduce_sum`（1.5h）

- 先在 warp 内用 `__shfl_xor_sync(0xffffffff, v, offset)` 做 butterfly（offset 为 16、8、4、2、1），再由每个 warp 的 lane 0 写入 shared memory，最后再做一次 warp 归约。所有线程都拿到总和；连续调用两次也要正确（写 shared 之前先同步）。
- 测试：blockDim 为 32、64、96、256、544、1024，输入是小整数，因此浮点和是精确的；两次连续调用，每个线程拿到的结果都要检查。

### C6 `fused_add_rmsnorm_f32`（2h）

- 第一遍：$z=x+r$，**写回 residual**，累加 $z^2$。第二遍：读回 residual，写 $y=z\cdot\text{rsqrt}(\text{sum}/D+\epsilon)\cdot\gamma$。$D$ 是 4 的倍数时用 `float4`，否则用标量。
- 测试：rows 取 1、3、17、64，$D$ 取 1、3、31、127、128、513、1024、2048、4095、4096；residual 必须逐位等于 $x+r$，$y$ 与 FP64 oracle 比较；输出先填成 NaN，漏写的位置一定会失败。

### C7 `fused_add_rmsnorm_f16`（1.5h）

- FP16 存储，FP32 计算；$D$ 为偶数时用 `__half2`。平方和必须用 FP32 累加。
- 测试：奇数和偶数宽度；取值范围为 $\pm100$ 的行，每个线程的平方和都超过 FP16 的最大值 65504。

**C 的实验：**

- `bench_scenarios.py` 在 stable、shift、tail 三组 trace 上比较 eager、static、dynamic、fixed-bucket 和 measured plan（horizon 为 1 和 10）。每个策略运行前调用 `torch.compiler.reset()`，磁盘缓存保留，所以 first call 不是冷编译。
- `bench.cu` 与 `inductor_bench.py` 用同一套规则计时：每种方法录成一张 20 次调用的 CUDA graph，先连续回放至少 300 ms，再取 25 次，报告中位数、p25、p75 和 `stable` 列。Inductor 生成的 Triton 源码写到 `build/inductor_<dtype>_<width>.py`，报告里要对照它和你的 C6 解释差异。

## R 真实系统：CUDA graph 分桶（6h）

服务框架为 decode 在若干 batch size 上捕获 CUDA graph，运行时把 batch 补到下一个已捕获的尺寸。vLLM 和 SGLang 用的是手工网格（讲义 §5.5 给出了带 commit 的源码位置）。本阶段测出每个尺寸的真实成本，再用 B1 的 DP 在显存预算下选尺寸。

### R1 `select_graph`（0.5h）

- 返回 `sizes` 中不小于 batch 的最小元素，超过最大尺寸返回 `None`。只能往上补，不能截断。与 vLLM 的 `_compute_bs_to_padded_graph_size` 和 SGLang 的 `_pad_to_bucket` 规则相同。
- 测试：1 到 40 的每个 batch 都与线性 first-fit 比较。

### R2 `GraphRunner.run`（1.5h）

- 把 x 拷入 `static_in[:batch]`，把 `static_in[batch:size]` 清零，回放，返回输出的**拷贝**。直接返回 view 会与静态缓冲区共用内存，下一次回放就会覆盖它。
- 测试：CPU 上用 `emulate` 后端，CUDA 上用真实 graph；batch 序列先大后小，检查补齐行为零、每一步都与 eager 一致、之前返回的结果在之后的回放中没有被改写。

### R3 `cost_model`（1h）

- 把原始测量变成 B1 需要的 profile：`steady_ms` 取 replay 的中位数，`first_premium_ms` 取捕获时间，同时保留 `pool_mib` 和 MAD。
- 测试：手算中位数和 MAD；拒绝重复 batch、空样本、NaN 和负值。

### R4 `plan_capture_sizes`（2h）

- 在 $\lvert S\rvert\le K$、$\text{pool}(\max S)+o\lvert S\rvert\le M$ 的约束下最小化 $J(S)$（讲义 §5.6）。枚举最大尺寸 $t$，前缀交给 `plan`，超过 $t$ 的 batch 付 eager 成本；也要考虑 $S$ 为空。
- 测试：4 组随机 profile，$K$ 取 0 到 3，4 种显存预算，2 种单图开销，与所有子集的暴力枚举比较。

### R 的实验（1h）

`bench_r.py --device cuda` 在每个 batch size（1 到 64）上测 eager 延迟、graph 回放延迟、捕获时间和私有池大小（每个配置都先连续运行 300 ms），然后在训练 trace 上规划，在 stable 和 shift 两条 held-out trace 上与 vLLM 默认网格、vLLM 网格的前 $K$ 个尺寸、SGLang 默认网格、均匀 $K$ 段和全 eager 比较，报告 modeled 的 ms/step 和相对 hindsight 的 regret。显存不设限时，几种策略还会把 held-out trace 用真实 graph 回放一遍，输出 measured 的 ms/step。`--device cpu` 直接读取提交的 profile，只输出 modeled 结果。

## 报告与验收（4h）

- A 的五个手算、B 的划分成本图和 DP 选中的解、cache 的 pin 状态追踪。
- C：三组 trace 的真实结果（五次独立进程），`calibration_ms` 单列，至少一个“不获益”的结果；C6/C7 与两 kernel 基线、Inductor kernel 的对比表（只用 `stable=yes` 的行），并解释至少一个 fused 不胜出的尺寸。
- R：你自己的 profile（`build/profile_measured.json`）、modeled 与 measured 两列的对比、各策略的 regret；说明 modeled 排序和 measured 排序是否一致，以及为什么。
- 附 `bash tools/check_env.sh --lab 3` 的完整输出、`results/` 里的 RECORD 文件路径和实际工时。

工时按做完 foundations 的学生估算，还没有经过试学验证：A 7h、B 9h、C 12h、R 6h、报告 4h，共约 38h（等编译的时间单独记录）。

## 可选挑战

- 用 `ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum` 解释 f32 $1024\times4096$ 时两 kernel 基线为什么比 fused 快。
- 用 `torch.library` 把 C6 注册成自定义算子，让 C2 的候选里同时有 Inductor kernel 和你的 kernel。
- 把 R 的 decode block 改成 28 层，检验讲义“研究问题”里的假设；或者在 R4 的显存模型里加入 piecewise graph。
- 实现 A4 + B5 的在线准入：只用已经到达的请求检测漂移，触发重新校准，并保留冻结计划作为对照。

## 资料汇总

| 资料                                                                                                                                                                                        | 类型   | 指定观察                                                        | 时间  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------- | ----- |
| [课程 D3](../../course/modules/03-dynamic-compilation.md)                                                                                                                                   | 讲义   | §2 的 recompile 计数、§5.6 的 $J(S)$、§6.2 的归约不变量         | —     |
| [torch.compile 排错](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)                                                                    | 文档   | 用 `TORCH_LOGS=recompiles` 解释 C 阶段 dynamic 基线的重编译次数 | 20min |
| [vLLM `cudagraph_dispatcher.py`](https://github.com/vllm-project/vllm/blob/371745ebd6c12f314402c6c319de2e7ef82c2981/vllm/v1/cudagraph_dispatcher.py)                                        | 代码   | `_compute_bs_to_padded_graph_size`，与 R1 对照                  | 15min |
| [SGLang `base_cuda_graph_runner.py`](https://github.com/sgl-project/sglang/blob/57db5089c207ae4ce189cc092a89a2027c8aea29/python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py) | 代码   | `_pad_to_bucket` 与静态缓冲区的拷入，与 R2 对照                 | 15min |
| [Triton layer norm 教程](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)                                                                                         | 教程   | 与 `build/inductor_f32_2048.py` 对照分块方式                    | 20min |
| [CUDA 编程指南：warp shuffle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)                                                                      | 文档   | `__shfl_xor_sync` 的 mask 和 width 参数，与 C5 对照             | 15min |
| [资源与简化边界](resources.md)、[参考实现导读](../../solutions/lab3/README.md)                                                                                                              | 本仓库 | 完成后对照 oracle 与 fault 表                                   | —     |
