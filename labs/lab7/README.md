---
title: Lab 7 — 从路由布局到真正执行专家的卸载流水线
aliases: [Lab7, MoE offload lab]
tags: [mlsys, lab, moe, cuda, offload]
---

# Lab 7 — 从路由布局到真正执行专家的卸载流水线

[课程讲义](../../course/modules/07-moe-offload.md) · [资源与源码借鉴](resources.md) · [评分标准](grade.md) · [参考解题导读](../../solutions/lab7/README.md)

本实验要求完成一个能计算数值结果的小型 MoE 执行器：先实现 gate→top-k→分段/打包→量化专家 MLP→加权合并，再把专家参数放到 pinned host memory，通过一个或两个显存槽分批载入。你需要实现路由、状态机和 CUDA 依赖关系；仅完成数据传输还不足以验证专家卸载。

> [!important] 三种证据分开报告
> A 验证数据布局和数值结果，B 使用有明确假设的事件模型，C 则在 CPU/CUDA 上实际计算并计时。B 的 us 由设定的带宽和计算成本得出；C 的 ms 来自主机墙钟，包含分配、传输、启动与回收。二者不可直接相除。

## 0. 任务量、环境和产物

已完成 foundations 的 CUDA、计时和 C++ 基础即可开始。CPU 路径可在 Mac/Linux/WSL 编译；CUDA 必修部分在 RTX 4060 或其他 NVIDIA 机器完成，预计显存远低于 1GB，不下载模型。只需要 C++17、Python 标准库与 CUDA toolkit。

参考答案的 **20 个 TODO 约 253 行有效实现**：A 72 行、B 106 行、C 75 行。计数排除空行、注释和只有括号的行，不包括测试、驱动、缓冲区分配或报告。行数用于核对工作量，不要求你的答案具有相同行数。

| 部分   | 实际要完成的机制                                   | 编码/调试 | 手算/分析 | 合计    |
| ------ | -------------------------------------------------- | --------- | --------- | ------- |
| A 基础 | top-k、稳定分段、INT8、加权还原、tile、负载统计    | 4h        | 2h        | 6h      |
| B 核心 | 字节容量缓存、加载事件、pin、因果预取、oracle 对照 | 5h        | 2h        | 7h      |
| C 前沿 | 真正 MLP、路由 block、有限槽异步 GPU 执行          | 6h        | 3h        | 9h      |
| 汇总   | 整理图表、失败反例、源码阅读卡与报告               | —         | 4h        | 4h      |
| 合计   | 不含可选挑战                                       |           |           | **26h** |

上述时间按已具备前置知识估计，尚无学生试学统计。首次调试 CUDA 可能增加 4–8h。如果单个 TODO 连续 90 分钟没有进展，先缩小到 1 token、1 expert、奇数维度的输入，保留日志后查看提示，暂缓大矩阵测试。

产物：20 个 TODO 的实现；三个阶段日志；布局、流量、真实执行三张结果表；一张手绘/绘制 stream 依赖时间线；3–5 页报告。报告放在 `reports/lab7/`，以 `labs/` 为工作目录。

## 1. 执行约定

以下命令均在 **`mlsys-intermediate/labs/`** 执行。公共 runner 在临时副本运行，保留源文件哈希与输出，不覆盖你的答案。

```bash
python3 tools/run.py --lab 7 --part A --impl student --mode test
python3 tools/run.py --lab 7 --part B --impl student --mode test
python3 tools/run.py --lab 7 --part C --impl student --mode test --device cuda
python3 tools/run.py --lab 7 --part all --impl student --mode bench --device cuda
```

用 `--impl reference` 独立检验参考答案。CPU 机器可先用 `--device cpu --part C` 完成 C1/C2/C5；runner 会明确提醒 C3/C4/C6 尚未执行，不能据此声称完整通过。直接调试入口是 `python3 lab7/exercise.py --mode test --part A`，但正式提交使用公共 runner 的日志。

核心文件是 [moe_student.hpp](src/moe_student.hpp) 与 [pipeline_student.cuh](src/pipeline_student.cuh)。只替换成对 `BEGIN TODO`/`END TODO` 标记内的失败占位代码；也可以添加辅助函数和自己的测试。给定的 [test.cpp](test.cpp)、[fixture.hpp](fixture.hpp)、[cuda_exercise.cu](cuda_exercise.cu) 提供契约和独立检查，不要通过修改预期值来让测试通过。

## 2. A 基础：数据到底怎样流过 MoE

### A0 手算路由与显存账（40min）

给 logits `[[1000,1000,0],[-10,-9,-8]]`，top-k=2。写出稳定 tie-break、选中后归一化的权重，以及为何不能直接 `exp(1000)`。再把 routes `[(0,2,.5),(0,0,.5),(1,2,1),(1,2,1)]` 按 expert 分桶，写出 offsets 与逆向合并式。

一个专家包含 up `[H,D]` 和 down `[D,H]`：INT8 权重的字节数为 `2HD`，FP32 row scales 为 `4(H+D)`。双槽仅说明参数工作区，输入、hidden、route 输出、allocator 等仍需另外计入显存。用 D=33、H=67 手算记录偏移，随后用于 C6。

### A1–A6 编码（4h）

| TODO              | 有效行参考 | 必须思考的决策                                     | 对应测试     |
| ----------------- | ---------: | -------------------------------------------------- | ------------ |
| A1 `route_topk`   |         16 | 在 top-k 内 softmax；相同 logit 优先较小 expert ID | `a_router`   |
| A2 `dispatch`     |          9 | counting→prefix sum→稳定散射；空桶仍有 offsets     | `a_dispatch` |
| A3 `quantize`     |         11 | 对称 INT8、常数零行、每输出行 scale、饱和舍入      | `a_quant`    |
| A4 `combine`      |          7 | 重复 route 必须加和，不能用赋值覆盖                | `a_combine`  |
| A5 `pack`         |         14 | 按专家补齐 tile；mask 与真实零向量分开             | `a_packing`  |
| A6 `router_stats` |         15 | assignment 熵、max/mean、padding、容量溢出诊断     | `a_stats`    |

先实现 A2/A4，再 A1/A3/A5/A6。A5 的 `route_index=-1` 只表示 padding，不得把低分 token 当作 padding。A6 中 overflow 是诊断值，本实验采用 dropless 语义，不能为降低计算量直接删掉这些 route。

### A 实验与观察（1h20min）

运行 A benchmark，扫 tokens=13/64/127 和 tile=1/4/16，共 9 行。画 `padded_rows/routes`。解释为什么 13 tokens、top-2、8 experts、tile=16 可能浪费大量空间，而均匀的 64 tokens 较平整。再提交一组极偏路由，打印熵、imbalance、overflow；均衡统计不能替代实际 latency。

独立检查量化误差是否不大于各行 `scale/2`（考虑浮点容差），不能只检查结果是否与参考实现逐元素相等。附原值、量化整数、scale 与还原值的 2×3 表。

## 3. B 核心：缓存、加载和未来预测必须分清

### B0 先画状态机（45min）

状态为 absent→loading→ready→pinned→ready。loading 已占显存但不能被消费或逐出；pin 是引用计数。copy engine 串行，因此两个大小不同的载入需要预约 `max(now,link_free)+bytes/bandwidth`。cache-hit 的 ready 判据不能用“map 中存在”代替。

手算容量20B、带宽2B/us：0us 请求12B，再请求8B；6us 第一个完成并 pin，10us 第二个完成。此时请求10B必须 blocked，且不能为了“尝试腾空间”先逐出8B然后失败。

### B1–B8 编码（5h）

| TODO                | 有效行参考 | 关键不变量                                       | 测试          |
| ------------------- | ---------: | ------------------------------------------------ | ------------- |
| B1 `Cache::request` |         29 | 字节预算；先规划全部受害者再提交；blocked 原子性 | `b_admission` |
| B2 `advance`        |          9 | 单调时间；按完成时间输出事件                     | `b_lifetime`  |
| B3 `pin`            |          4 | 只 pin 已完成载入的 entry                        | `b_lifetime`  |
| B4 `release`        |          3 | 不允许释放不存在的 lease                         | `b_lifetime`  |
| B5 `observe`        |          4 | 仅历史转移、全局指数遗忘                         | `b_predictor` |
| B6 `predict`        |         14 | 置信度与字节预算共同准入                         | `b_predictor` |
| B7 `replay`         |         25 | demand/prefetch 共用缓存和链路；真实事件推进     | `b_replay`    |
| B8 `oracle_misses`  |         18 | 等大小 Belady 仅作未来信息下界                   | `b_oracle`    |

B7 不能将 prefetch 的耗时记为0后又忽略其带宽占用。prefetch 可以与 compute 重叠，但下一次 demand 也可能排在它后面。B8 的未来信息必须隔离于在线策略；它适用于本实验等大小专家，不能把结果直接当作变大小专家的最优解。

### B 受控实验（1h15min）

给定轨迹前60步在3个专家间循环，后60步换成4个新专家。容量4096/8192/12288B、专家2048B、带宽128B/us、compute20us，prefetch on/off，共6行。保留 `finish_us, transferred_bytes, wasted_bytes, misses, blocked, oracle_misses`。

这里的 `wasted_bytes` 只统计尚未消费就被逐出的预取，不含实验结束时仍驻留但未使用的预取，请在报告中列明这一定义。观察小容量下的收益，以及容量充足时无收益的情况；再构造一次相位突变，观察预测造成的流量浪费。miss 减少不一定意味着耗时减少。

## 4. C 前沿：有限显存槽里的真实专家执行

### C0 对照主源（45min）

按 [resources.md](resources.md) 阅读 MegaBlocks 的 `indices_and_bins` / `permute_and_compute`、MoE-Infinity 的 overlap-window byte admission。提交一张“主源机制→本实验函数→省略内容”表。这里使用 rowwise INT8 MLP 和两个槽，没有复现 MoE-Infinity 的完整多层引擎、磁盘卸载或多 GPU。

### C1/C2/C5：CPU 数值 oracle 与完整 block（2h）

C1 实现 `down(ReLU(up(x)))`，量化矩阵在内积中还原；C2 使用 A5 的打包布局执行专家，再 A4 合并；C5 实现实际 dense gate、top-k、专家执行与 residual add。这条路径必须完成实际计算，不能只返回合成的 checksum。

`c_expert` 有手算结果，`c_forward` 使用独立 double accumulation，`c_block` 检查完整 block。CPU benchmark 三个 seed × 两个 width，共6行，计时包括 gate 和布局，保留 checksum。它不是 BLAS 优化性能竞赛。

### C3/C6/C4：CUDA 执行与事件链（4h）

先 C3，再 C6，最后 C4。C3 每线程计算一个 route 的一个输出通道，循环 INT8 内积；C6 将 up/down 权重及 scales 放入连续 pinned 记录；C4 只允许1/2个权重槽并实现依赖：

```mermaid
flowchart LR
  old[该槽上次 compute done] --> cp[copy stream 写入新专家]
  cp --> ready[ready event]
  ready --> up[compute stream up + ReLU]
  up --> down[down]
  down --> done[done event]
  done --> next[下次复用该槽]
```

host 的提交顺序不足以保证数据依赖。CUDA stream 内的操作有序，跨 stream 则需要通过事件建立依赖。循环内禁止 `cudaDeviceSynchronize`；每个槽的最终 compute 完成后，才将输出拷回 CPU 合并。输入在 GPU 常驻，参数只保留有限槽。

| TODO        | 有效行参考 | 验收                                       |
| ----------- | ---------: | ------------------------------------------ |
| C1/C2/C5    |   13/10/12 | `c_expert`、`c_forward`、`c_block`         |
| C3 kernel   |         10 | `c_cuda_forward`，包括奇数维度和非满 block |
| C6 layout   |          7 | 同上，错误字节/元素偏移会导致数值错误      |
| C4 pipeline |         23 | 同上，多次复用槽；1/2 slots 都输出正确     |

### C 真实测量与反例（2h15min）

CUDA test：3 seeds × D=7/33 × slots=1/2，共12行。CUDA bench：3 seeds × D=64/128 × slots=1/2，共12行。测试 `max_abs_error≤2e-4`；这是当前小输入数值容差，不是所有模型的误差界。每配置先预热1次，再计时5次取中位数；比较1/2槽墙钟，报告中明确计入分配、pinned 拷贝准备、CUDA启动等开销。

验收不要求双槽一定更快：本实验的计算规模很小，分配和启动成本可能占主导。若要单独测量稳态性能，可在挑战项中将初始化移出循环并预热，但仍须保留端到端测量值。使用 Nsight Systems 检查 copy/compute 是否发生重叠；未安装 profiler 时，可以先提交依赖图和正确性结果，但不能据此声称观察到了重叠。

## 5. 交付报告

必交：代码、A/B/C完整日志、三张表、一张依赖图、一个失败反例、源码阅读卡。报告回答：哪个容量下预测无效？真实计算与事件模型哪些成本不同？若换大模型，权重、KV、hidden 与host RAM哪个先超预算？

## 可选挑战

以下任务不计入26h。选择一项深入即可；每项都要确保数值结果、route 集合和实际字节预算一致。

- 让 GPU 执行器跨 token 保留专家缓存，记录复用与重新载入次数。
- 采集真实小 MoE 的路由，检查合成轨迹上的预测收益是否仍成立。
- 将 B 的 admission 接入 C，把分配移出循环后测稳态性能，同时保留端到端墙钟。
- 增加 cuBLAS/grouped GEMM 基线，区分专家计算与卸载带来的开销。

## 资料汇总

- [资源映射](resources.md)：主源实现、固定入口及本 lab 的简化范围。
- [D7 讲义与读图自测](../../course/modules/07-moe-offload.md)、[实验评价规范](../../course/guides/experiments.md)、[评分标准](grade.md)。
- [NVIDIA 传输重叠图解](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)与 [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)：用于读懂 C 阶段的依赖与时间线。
- [transfer_bench.cu](transfer_bench.cu) 仅作可选传输校准；正式 C 阶段运行 [cuda_exercise.cu](cuda_exercise.cu) 的真实专家计算。
