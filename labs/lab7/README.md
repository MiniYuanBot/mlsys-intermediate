---
title: Lab 7 — 路由约定、专家缓存与真实的 copy/compute 重叠
aliases: [Lab7, MoE offload lab]
tags: [mlsys, lab, moe, cuda, offload]
---

# Lab 7 — 路由约定、专家缓存与真实的 copy/compute 重叠

[课程 D7](../../course/modules/07-moe-offload.md) · [实验索引](../index.md) · [评分标准](grade.md) · [资源与改编说明](resources.md) · [参考导读](../../solutions/lab7/README.md)

> [!important] 完成定义
> A/B/C 用 C++17 在 CPU 上完成：三种已发布路由归一化约定的精确实现、按字节记账的专家缓存（LRU/LFU/预取/Belady 下界）、真实 INT8 专家 MLP 与 residual block。C 的 GPU 部分和 R 在 WSL 的 RTX 4060 上完成：C3/C4/C6 验证槽复用的事件依赖图是否正确，R 用 4–32 MiB 的专家、pinned 池、预分配显存槽和分段 cudaEvent 测出真实的 H2D 带宽与重叠比 $\rho$，并用一条实测驱动的判定式决定每个专家该走 PCIe 还是走 CPU。每个 TODO 都有独立 oracle 和至少一个错误注入，核心 TODO 至少两个。

需要先掌握 foundations 的 C++、CUDA stream/event、pinned memory 和基准测量。**下列命令都从 `mlsys-intermediate/labs/` 执行。** 预计 **30h**：A 6h、B 7h、C 7h、R 7h、报告 3h，另加 D7 讲义 5h。估时按已具备前置知识估计，首次调试多流 CUDA 可能再加 4–8h。

> [!important] 三种证据分开报告
> A 验证数据布局与数值；B 是事件模型，它的 us 由设定的带宽和计算成本算出，标 **modeled**；C 的 ms 是主机墙钟，包含分配、pinning 和启动开销；R 是 cudaEvent 分段计时，标 **measured**。四者不可相除，也不能拼成一个 tok/s。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。在 `labs/` 下先安装、再检查：

```bash
bash tools/setup_env.sh --lab 7 && bash tools/check_env.sh --lab 7
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 7
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 7
```

| 阶段 | 需要                                  | WSL2 / Linux | Windows 原生   | macOS       |
| ---- | ------------------------------------- | ------------ | -------------- | ----------- |
| A    | C++17                                 | 可做         | 可做           | 可做        |
| B    | C++17                                 | 可做         | 可做           | 可做        |
| C    | C++17；GPU 部分另需 nvcc + NVIDIA GPU | 可做         | 仅 CPU 部分    | 仅 CPU 部分 |
| R    | C++17 + nvcc + NVIDIA GPU             | 可做         | 不支持，转 WSL | 不支持      |

A、B 和 C 的 CPU 部分（C1/C2/C5）只要 C++17。C3/C4/C6 和整个 R 需要 nvcc 与 GPU，只能在 WSL2 或 Linux 上做；`setup_env` 会装 CUDA Toolkit，缺少时 `check_env` 只报 `[WARN]`，不挡 A/B/C。缺依赖时 runner 只打印一行 `MISSING ...` 和对应的 `fix:`，退出码 3；在 Windows 上请求 CUDA 阶段，`fix:` 给出一条可以直接复制的 WSL 命令。各平台细节见[环境配置](../setup.md)。

R 阶段会申请一块 pinned 主机内存（最大约 200 MiB）和若干显存槽。WSL2 对 pinned 内存有未公开的上限，`build_pool` 在 `cudaMallocHost` 失败时抛出带 `pinned` 字样的错误，bench 会跳过该档并打印原因。显存需求远低于 1 GB，不下载模型。

## 文件和实施顺序

| 文件                                                 | 学生负责                             | 已提供                                                      |
| ---------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------- |
| [src/moe_student.hpp](src/moe_student.hpp)           | A1–A6、B1–B8、C1/C2/C5 共 17 个 TODO | 数据结构、`RouterSpec`/`Policy` 枚举、校验分支              |
| [src/pipeline_student.cuh](src/pipeline_student.cuh) | C3、C4、C6                           | 缓冲区分配、stream/event 创建、D2H 与清理                   |
| [src/r_pipeline.cuh](src/r_pipeline.cuh)             | R1–R5                                | `Pool`/`Segment`/`CostModel` 结构、`destroy_pool`、计时骨架 |
| [test.cpp](test.cpp)                                 | 只读                                 | A/B/C 的 CPU 检查与独立 oracle                              |
| [bench.cpp](bench.cpp)                               | 只读                                 | A/B/C 的 CPU benchmark                                      |
| [cuda_exercise.cu](cuda_exercise.cu)                 | 只读                                 | C 的 GPU 检查（含槽复用压力用例）                           |
| [r_fixture.cuh](r_fixture.cuh)                       | 只读                                 | 专家 payload pattern、验证 kernel、spin 预热、设备信息      |
| [r_test.cu](r_test.cu) / [r_bench.cu](r_bench.cu)    | 只读                                 | R 的分组检查与测量                                          |

只替换成对 `BEGIN TODO` / `END TODO` 标记之间的占位代码；可以添加辅助函数和自己的测试，但不要改预期值来让测试通过。

```bash
python tools/run.py --lab 7 --part A --impl student --mode test
python tools/run.py --lab 7 --part B --impl student --mode test
python tools/run.py --lab 7 --part C --impl student --mode test --device cuda
python tools/run.py --lab 7 --part R --impl student --mode test --device cuda
python tools/run.py --lab 7 --part all --impl student --mode bench
python tools/run.py --lab 7 --part R --impl student --mode bench --device cuda
```

用 `--impl reference` 独立检验参考答案（两者 RECORD 的 `implementation` 字段不同，参考 run 不能充当学生 run）。CPU 机器上 `--part C --device cpu` 只跑 C1/C2/C5，runner 会明确提醒 C3/C4/C6 未执行。`--device cuda` 会自动获取 `/tmp/mlsys-gpu.lock`，RECORD 里的 `gpu_lock` 字段写明是 `acquired` 还是 `inherited`。

每个检查都打印一行 `PASS <名称>` 或 `FAIL <名称>: <期望> vs <实际>`，遇到未完成的 TODO 也会继续往下跑，一次就能看到所有没过的检查。R 阶段的六组（R1/R2/race/R3/R4/R5）各自在独立进程里运行，一组崩掉不影响其余。

## A 基础：选择和加权是两件事（6h）

### A0 手算路由与显存账（0.5h）

按讲义 §1.2，对 logits $[2,1,0,-1]$、$k=2$ 分别算出三种约定下的权重：先 top-k 再 softmax、softmax 全部后不重归一化、sigmoid 加 selection-only bias 后重归一化并乘 2.5。再按 §2.1 算 OLMoE-1B-7B 的单专家字节数（$3\times 2048\times 1024\times 2$）、routed 专家总字节数和 12.6 GB/s 下的单专家传输时间。最后用 $D=33$、$H=67$ 手算一条 INT8 专家记录的字节偏移，C6 会用到。

### A1 `route_topk`（1.25h）

输入 `logits` 是行主序 $[\mathrm{tokens},\mathrm{experts}]$，加上一个 `RouterSpec{norm, bias, scale}`。输出每个 token 的 $k$ 条 route，按选择分数降序；分数相同优先编号小的专家。

- **选择分数**：`SigmoidRenorm` 用 $\mathrm{sigmoid}(z_e)+b_e$，其余两种用 $z_e+b_e$。`bias` 为空或长度等于 experts，其他长度抛 `std::invalid_argument`。
- **权重**：`TopkSoftmax` 对选中的 $k$ 个 logit 做 softmax；`SoftmaxNoRenorm` 对全部专家做 softmax 后直接取选中位置（权重和小于 1）；`SigmoidRenorm` 用选中专家的**无偏** sigmoid 除以它们自己的和。三者最后都乘 `spec.scale`。
- **不变量**：bias 绝不进入权重；`exp` 前必须减去最大值，否则 $\{1000,1000,0\}$ 会溢出成 nan；并列必须确定性。
- 对应测试：`a_router`（三种约定各自比对 long double oracle 与手算值、并列、溢出、bias、非法 shape、零 token）。

### A2 `dispatch`（0.5h）

counting sort 成 expert-major：第一遍校验索引并计数，第二遍做**排他**前缀和，第三遍按原 route 顺序 scatter。`offsets` 有 $E+1$ 项、非递减、首项 0、末项等于 route 总数；空专家区间长度为 0；桶内保持输入顺序。非法 token/expert 索引抛 `std::invalid_argument`。对应 `a_dispatch`（独立的 $O(RE)$ 布局 oracle、空输入、非法索引、桶内稳定性）。

### A3 `quantize`（0.5h）

逐行对称 INT8：$s_r=\max_c\lvert w_{rc}\rvert/127$，全零行 $s_r=1$，四舍五入后钳位到 $[-127,127]$。对应 `a_quant`（行 scale 等于 rowmax/127、全零行、误差不超过半个 scale、峰值饱和到 $-127$、非法 shape）。

### A4 `combine`（0.5h）

按 ordered 顺序把 $[\mathrm{routes},\mathrm{width}]$ 的专家输出带权**累加**回 token。重复的 token/expert 必须相加，不能覆盖；没有 route 的 token 保持零行；输出缓冲区大小不符抛 `std::invalid_argument`。对应 `a_combine`（手算 $[2,4,6,8]\to[3,14]$、空 dispatch、未路由 token、非法 shape）。

### A5 `pack`（1h）

按专家把 route 补齐到 tile 的整数倍。只有 padding 行的 `route_index` 为 $-1$，真实 token 即使激活全零也有合法下标；每条原 route 都必须出现一次。对应 `a_packing`（空专家、重复 route、不满的末尾 tile、刚好满的桶）。

### A6 `router_stats`（1h）

统计的是**路由分配数**，不是不同 token 数。熵按经验专家频率 $p_e=n_e/R$ 计算，imbalance 是 $\max_e n_e/(R/E)$，`padded` 与 A5 的打包行数一致，`overflow` 只是诊断——本 lab 采用 dropless 语义，**绝不能为了降低计算量真的丢 route**。对应 `a_stats`（手算熵 0.5623351446、route 数不等于专家数的 1.0114042647、imbalance、padding、overflow、空输入）。

### A 实验与观察（0.75h）

跑 `--mode bench --part A`：tokens 13/64/127 $\times$ tile 1/4/16 共 9 行，画 `padded_rows/routes`，解释 13 token、top-2、8 专家、tile 16 为什么浪费大量空间。再看 `router_mass` 的 6 行：同一批 logits 下三种约定的权重和均值与最小值，说明“不重归一化”保留了什么信息。附一张原值、量化整数、scale、还原值的 2$\times$3 表。

## B 核心：缓存、替换策略与字节账（7h）

### B0 先画状态机（0.75h）

四态 absent→loading→ready→(pinned)→ready。loading 已占显存但不能被消费或逐出；pin 是引用计数。单个串行 copy engine 下 $\mathrm{ready}=\max(\mathrm{now},\mathrm{link\_free})+\mathrm{bytes}/B$。

手算：容量 20 B、带宽 2 B/us，0 us 请求 12 B 再请求 8 B（分别 6 us、10 us 完成），6 us pin 第一个，10 us 请求 10 B 必须 `Blocked`，而且**不能先逐出 8 B 再失败**。

### B1–B8 编码（5h）

| TODO                | 关键不变量                                                                                                                       | 测试                      |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| B1 `Cache::request` | 字节预算；串行链路排队；先规划全部受害者再提交；`Blocked` 不改状态；预取触碰不提升已驻留 entry；逐出未用预取时计入 `wasted`      | `b_admission`、`b_policy` |
| B2 `advance`        | 单调时间；只报告 $(\mathrm{now},\mathrm{time}]$ 内完成的加载，按完成时刻再按 ID 排序                                             | `b_lifetime`              |
| B3 `pin`            | 只 pin 已完成加载的 entry，并刷新 recency                                                                                        | `b_lifetime`              |
| B4 `release`        | 不允许释放不存在或引用数为 0 的 lease                                                                                            | `b_lifetime`              |
| B5 `observe`        | 先把**整张**转移表乘 0.9，再累加 previous→expert，最后更新 previous                                                              | `b_predictor`             |
| B6 `predict`        | 行内归一化求置信度；按置信度降序、ID 升序；放不下的候选跳过而不是终止扫描                                                        | `b_predictor`             |
| B7 `replay`         | demand 等到 ready 才计算；预取只在 observe 之后发起，预算为 $\mathrm{compute}\times B$；预取与 demand 共用同一条链路和同一个缓存 | `b_replay`                |
| B8 `oracle_misses`  | 等大小 Belady；永不再用视为无穷远；是隔离的 oracle，不能喂给在线策略                                                             | `b_oracle`                |

实现顺序建议 B2→B3→B4→B1→B5→B6→B7→B8。

B1 的替换策略由 `Policy` 选择：LRU 看 `stamp`，LFU 看驻留期间的 demand 使用次数、并列再看 `stamp`。`b_policy` 用一条**必须让两者给出不同答案**的轨迹检查（3 槽填满后依次 demand `0,0,1,2`，LRU 淘汰 0，LFU 淘汰 1）。

B7 返回的三个字节口径要分清：`wasted` 是被 demand 碰到之前就被逐出的预取字节，`stranded` 是结束时仍驻留但从未被 demand 碰过的预取字节，二者之和不超过 `prefetched`。

### B 受控实验（1.25h）

跑 `--mode bench --part B`：三条 trace（`phase` 相变、`uniform` 无局部性、`skewed` Zipf）$\times$ 容量 2048/4096/6144/8192 B（1–4 槽）$\times$ {LRU, LFU} $\times$ 预取开关，共 48 行。每行给出 `finish_us, transferred_bytes, prefetched_bytes, wasted_bytes, stranded_bytes, misses, prefetch_hits, blocked, oracle_misses`。

必须回答：

1. 6144 B（3 槽）这一档在 `phase` 上发生了什么？为什么它比 4096 和 8192 都更能说明问题？
2. 在 `uniform` 上预取把 misses 从 86 降到 85，字节数却从 176 KB 涨到 324 KB。按讲义 §4.4 的账算出这笔交易，并说明你会不会在真实系统里开它。
3. LRU 与 LFU 在哪条 trace、哪个容量下分出了胜负？差异有多大，够不够支撑一个结论？
4. 开预取后 `misses` 低于 `oracle_misses` 的行，解释两个计数的口径差异（见讲义 §4.4），**不要**写成“突破了 Belady 下界”。

## C 前沿：有限显存槽里的真实专家执行（7h）

### C0 对照主源（0.75h）

按 [resources.md](resources.md) 读 MegaBlocks 的 `indices_and_bins` / `permute_and_compute` 和 mixtral-offloading 的逐层 LRU，提交一张“主源机制→本实验函数→省略内容”表。本 lab 用 rowwise INT8 MLP 和 1/2 个槽，没有复现 block-sparse GEMM、磁盘卸载或多 GPU。

### C1/C2/C5：CPU 数值与完整 block（3h）

C1 实现 `down(ReLU(up(x)))`，INT8 矩阵在内积中反量化；C2 用 A5 的打包布局执行专家再经 A4 合并，空专家不得启动任何工作；C5 实现 dense gate（布局是 $[\mathrm{experts},\mathrm{width}]$）→ top-k → 专家 → residual。

对应 `c_expert`（FP64 反量化 oracle、手算 $[1,4]$、ReLU 截断、非法宽度）、`c_forward`（逐 route 的 FP64 oracle，奇数维度、5 个专家、空 dispatch）、`c_block`（手算 $[3,5]$、非方阵 gate 的 $[6,3]$、top-2 权重和为 1）。

### C3/C6/C4：CUDA 执行与事件链（3.25h）

顺序 C3 → C6 → C4。C3 每个线程算一条 route 的一个输出通道；C6 把 up/down 权重与 row scale 序列化进连续的 pinned 记录（**注意 `qn` 数的是 INT8 字节，`sn` 数的是 float 元素**）；C4 只允许 1 或 2 个权重槽，必须建立两个方向的依赖：

```mermaid
flowchart LR
  prev[previous compute on this slot] --> cp[copy stream writes the new expert]
  cp --> ready[ready event]
  ready --> up[compute stream: up + ReLU]
  up --> down[down]
  down --> done[done event]
  done --> next[next copy into the same slot]
```

host 的提交顺序**不构成**跨 stream 的依赖；循环内禁止 `cudaDeviceSynchronize`。

验收用 `c_cuda_forward`（3 seed $\times$ $D=7/33$ $\times$ 1/2 槽）、`c_cuda_slot_reuse`（512 token、8 专家、重复 8 次，让槽复用的竞态暴露成数值错误）和空 dispatch 用例，容差 `max_abs_error <= 2e-4`。

> [!note] C 阶段不测重叠
> 这里的专家记录只有 17–67 KB，在 12.6 GB/s 下是 1.3–5.3 µs，低于 WSL2 的 kernel launch 延迟；而且墙钟包含 pinning、`cudaMalloc`、stream/event 创建和 CPU combine。**C 只验证依赖图的正确性，定量的重叠测量在 R 阶段。** 不要用 C 的 1 槽 vs 2 槽墙钟差声称观察到了重叠。

## R 真实系统：MB 级专家的 pinned 流水线（7h）

### R0 读与量（0.5h）

先跑一次 `--part R --mode bench`，记下本机的 `asyncEngineCount`、pinned 与 pageable H2D 带宽，以及 `nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv` 的输出。对照 [NVIDIA 传输重叠教程](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)的 **C1060 面板**（单 copy engine，与本机同构）。

### R1 `build_pool`（1.5h）

一次性建好**所有**计时区间会用到的资源：`cudaMallocHost` 的 pinned 专家池（用 `rfix::fill_pool` 填充）、每槽一块显存并 memset 成 0、一条 copy stream 加每槽一条 compute stream（都用 `cudaStreamNonBlocking`）、每步 4 个**启用计时**的事件加 `wall_a`/`wall_b`、零初始化的 `device_sum`/`device_bad`。

- 为什么必须 pinned：pageable 内存上的 `cudaMemcpyAsync` 会**静默**退化为同步的 staging 拷贝，不报错，只是不重叠。
- 为什么槽要清零：这样“计算跑在拷贝前面”就会把零求和，被校验和抓住。
- 为什么事件不能用 `cudaEventDisableTiming`：这种事件不能传给 `cudaEventElapsedTime`。
- `cudaMallocHost` 失败时抛出带 `pinned` 字样的 `std::runtime_error`（WSL2 的 pinned 上限未公开）。

对应 `r_pool`（`cudaPointerGetAttributes` 确认 page-locked、每个专家的 payload、槽互不相同且清零、stream 与事件数量、销毁）。

### R2 `run_pipeline`（2h，核心）

把 schedule 轮转过各个槽，step $i$ 用槽 $i \bmod \mathrm{slots}$，建立三条依赖：

1. step $i$ 的**拷贝**等 step $i-\mathrm{slots}$ 的 `comp_end`（写后读：别覆盖还在被读的槽）；
2. step $i$ 的**计算**等自己的 `copy_end`（读后写：别读没写完的槽）；
3. 循环结束后让 copy stream 等每个槽最后一次 `comp_end`——compute stream 是 `cudaStreamNonBlocking`，**不会**自动与默认流汇合，少了这一步读事件会拿到 `cudaErrorNotReady`。

拷贝全部走 `pool.copy_stream`（单 copy engine），计算走各槽自己的 stream，这样 step $i+1$ 的拷贝才能和 step $i$ 的计算重叠。循环内不分配、不同步。

对应 `r_pipeline`（1/2/3 槽下每步的校验和等于 host 端按 `rfix::pattern` 算的值、`bad` 为 0、分段事件有计时、墙钟不超过串行和、槽轮转正确）和 `r_race`（把计算用自旋拉长到几十 ms，缺了依赖 1 必然在 kernel 执行期间被覆盖，device 端重读检查会报出来）。

### R3 `overlap_ratio`（0.5h）

$$
\rho=\frac{\sum T_{\mathrm{copy}}+\sum T_{\mathrm{compute}}-T_{\mathrm{wall}}}{\min\left(\sum T_{\mathrm{copy}},\ \sum T_{\mathrm{compute}}\right)},
$$

截断到 $[0,1]$，任一侧为 0 时返回 0。分母**不能**用 $T_{\mathrm{wall}}$。对应 `r_overlap`（完全串行 0、完全隐藏 1、手算 0.5、空输入、时钟噪声下的截断）和 `r_overlap_measured`（16 MiB、2 槽、平衡的计算量下中位数 $\rho>0.2$；单 stream 或循环内同步会掉到约 0）。

### R4 `measure_h2d`（1h）

对一次 `bytes` 的 H2D 传输测带宽：pinned 用 `cudaMallocHost`、pageable 用 `new`；用 `rfix::spin_warmup` 让 GPU 以**同一个拷贝**连续忙至少 300 ms（笔记本 GPU 空闲后会掉到低 P-state），再计时 $\ge 9$ 次取中位数，按十进制 GB 报告 $\mathrm{GB/s}=\mathrm{bytes}/(t_{\mathrm{median}}\times 10^{-3})/10^{9}$。`is_pinned` 必须来自 `cudaPointerGetAttributes`，不能照抄传进来的布尔值。对应 `r_bandwidth`（与测试自己计时的同一个拷贝相差 35% 以内、单位口径、pinned 标志、pageable 不快于 pinned）。

### R5 `hybrid_plan`（1h）

Fiddler 式判定，`slots` 个 GPU 槽按 LRU：已驻留则走 GPU（刷新 recency、不搬字节）；否则比较 $\mathrm{bytes}/(B\cdot 10^{9})\cdot 10^{3}+T_{\mathrm{gpu}}$ 与 $T_{\mathrm{cpu}}$，取小者，**并列走 GPU**（那次拷贝顺便暖了缓存）。CPU 步不搬权重，所以**不得**让专家变成 resident，也不得淘汰任何东西。对应 `r_hybrid`（CPU 极慢、CPU 极快、并列、全 CPU 的总时间、以及一次独立的 LRU 重算）。

### R 测量与报告（0.5h 以上）

跑 `--part R --mode bench`，输出三张表：

1. **pinned vs pageable H2D**：1/4/16/32/64 MiB $\times$ 两种内存，每行带 `stable`。
2. **流水线重叠**：专家 4/8/16/32 MiB $\times$ 槽 1/2/3 $\times$ `reads` 2/16，12 步、9 次重复，给出 wall 的中位数与 p25/p75、$\sum T_{\mathrm{copy}}$、$\sum T_{\mathrm{compute}}$、串行和、$\rho$、H2D GB/s 和 `stable`。
3. **CPU/GPU crossover**：$T_{\mathrm{cpu}}$（实测标量内核）、$T_{\mathrm{gpu}}$、$T_{\mathrm{fetch}}$，以及 `cpu_speedup` 1/4/16/64 下的 CPU 步数、GPU 步数、命中数、搬运字节和总时间。**`cpu_speedup` 不是 1 的行是 modeled，不是实测。**

必须回答：单槽的 $\rho$ 为什么精确等于 0？第三个槽为什么几乎没有额外收益？把 `reads` 从 2 提到 16 让计算时间涨了约 12 ms，墙钟为什么只涨了约 1.5 ms？在这台机器上标量 CPU 专家从来不划算，需要多大的向量化加速比才会翻转？`stable` 为 `NO` 的行你是怎么处理的？

## 报告与验收（3h）

必交：代码；A/B/C/R 四个阶段的完整日志与 RECORD JSON；A 的 padding 图、B 的 48 行表、C 的数值表、R 的三张表；一张槽复用的双向依赖图；至少两个你新增的反例；`bash tools/check_env.sh --lab 7` 的完整输出。

报告回答：哪条 trace 上预取有效、哪条无效，为什么？事件模型（B）预测的 finish 和真实流水线（R）的墙钟差在哪？换成 OLMoE bf16 时，权重、KV、hidden 和主机 RAM 哪个先超预算？

## 可选挑战

以下不计入 30h。每项都要保证数值结果、route 集合和字节预算口径一致。

- 用 transformers 的 `output_router_logits=True` 录一条真实的 OLMoE-1B-7B 路由 trace，替换 B 的合成 trace 重跑 48 行矩阵，看结论是否还成立。模型不在 `~/models` 时不要下载（13.8 GB），先只写采集脚本和显存/内存预算估算。
- 把 B 的准入与预测接到 R 的真实流水线上，让专家跨调用驻留，比较事件模型与实测墙钟。
- 实现 SpecMD 的 Least-Stale 或 Euro-Par 的 LLRU，在同一套 replay harness 上与 LRU/LFU/Belady 对比。
- 给 R 的 CPU 专家写一个 AVX2 版本，实测 `cpu_speedup` 到底是多少，替换 bench 里 modeled 的那一列。
- 用 `nsys profile --trace=cuda -o r.nsys-rep` 抓一次 R2，在 Windows 的 GUI 里确认 CUDA HW 轨道上 memcpy 与 kernel 的方块确实重叠（WSL2 上 Nsight Systems 是 CLI-only preview）。

## 资料汇总

| 资料                                                                                                                          | 类型     | 指定观察                                                      | 时间  |
| ----------------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------------------------------- | ----- |
| [D7 讲义](../../course/modules/07-moe-offload.md)                                                                             | 讲义     | §1.2 的五种路由约定表、§5.1 的 $\rho$ 定义、§5.3 的依赖图     | —     |
| [resources.md](resources.md)                                                                                                  | 来源映射 | 主源实现、固定入口与本 lab 的简化边界                         | 20min |
| [NVIDIA, How to Overlap Data Transfers](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)                | 实测图解 | 只看 C1060 面板（单 copy engine），圈出两种提交顺序的临界路径 | 20min |
| [NVIDIA, How to Optimize Data Transfers](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)              | 实测图解 | pageable→pinned 的 staging 图，对照自己实测的 GB/s            | 15min |
| [CUDA Programming Guide, 异步执行](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html) | 文档     | “pageable 内存上 `cudaMemcpyAsync` 退化为同步“那一句          | 10min |
| [CUDA on WSL User Guide §5.1–5.2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)                                     | 文档     | Unified Memory 不支持、pinned 受限、nsys 为 preview           | 10min |
| [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)                                            | 文档     | 怎么看 CUDA API / H2D / kernel 三条轨道                       | 选读  |
| [评分标准](grade.md) · [参考导读](../../solutions/lab7/README.md)                                                             | 验收     | 分值、必交证据、oracle 与 fault 对照表                        | —     |
