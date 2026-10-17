# Lab 3：从 shape trace 到真实编译执行闭环

> [!abstract] 产出
> 完成 **15 个 TODO、264行有效代码**：A78行，B73行，C113行。分桶依据需要从理想延迟推进到实际测量，完整实现 **历史切分→真实校准→有限特化规划→有容量的artifact缓存→guard/fallback→held-out真实执行**，再编写 CUDA residual RMSNorm，验证局部融合。

[课程 D3](../../course/modules/03-dynamic-compilation.md) · [资料与借鉴](resources.md) · [评分](grade.md) · [参考解答](../../solutions/lab3/README.md)。命令均从 `labs/` 执行。

## 0. 工作量和环境

| 阶段                                | 阅读/手算 | 实现/调试 | 实验/分析 |    合计 |
| ----------------------------------- | --------: | --------: | --------: | ------: |
| A 基础：trace、计时统计和padding    |      1.5h |      3.5h |        1h |      6h |
| B 核心：DP、guard、cache和admission |        1h |        5h |        2h |      8h |
| C 前沿：真实torch.compile与CUDA融合 |      1.5h |        5h |      3.5h |     10h |
| 总报告                              |         0 |         0 |        2h |      2h |
| **必修合计**                        |           |           |           | **26h** |

时间按已完成 foundations 的学生估算，尚未经过试学验证。请记录实际用时，工具链首次构建和编译缓存冷启动单独计时。26h lab + 5h 课程 + 5h 论文/源码阅读约占一周36h，挑战另计6–12h。等待编译的时间应与实际编码时间分开记录。

A/B 只需 Python。C1–C4 使用 PyTorch，实际调用 `torch.compile(backend="inductor")`；Linux CPU 可完成，使用 CUDA 时需要相应环境。C5 必须在 NVIDIA GPU 上运行。Mac 的编译后端与本地 Linux 不同，可以先做 A/B，再转到 WSL 或服务器完成 C。CPU all 会明确报告 C5 未运行，不能据此认定全部 GPU 要求已通过。

```bash
python tools/run.py --lab 3 --impl student --mode test --device cpu --part A
python tools/run.py --lab 3 --impl student --mode test --device cpu --part B
python tools/run.py --lab 3 --impl student --mode test --device cuda --part C
python tools/run.py --lab 3 --impl student --mode bench --device cuda --part all
```

先完成前置部分：B 复用 A 的表，C 复用 A/B。A5 的阶段测试会临时注入直接上界 oracle，因此做 A 时不必提前完成 B2。

## 1. 文件和数据契约

| 文件                                 | 作用                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| `src/pipeline_student.py`            | A1–A5、B1–B5：离线分析、有限DP与运行时状态                   |
| `src/execution_student.py`           | C1–C4：实际计时、编译、零填充执行、校准闭环                  |
| `src/fusion_student.cuh`             | C5：CUDA fused residual RMSNorm                              |
| `pipeline_test.py`                   | 手算、独立穷举、缓存状态反例                                 |
| `execution_test.py`                  | 真正Inductor编译、两条执行路径、held-out oracle              |
| `test.cu` / `bench.cu`               | CUDA与double CPU数值比较、融合实测                           |
| `policy_student.py`                  | 旧线性成本API的兼容适配器；复用B1/B2，不再留另一套未覆盖TODO |
| `torch_bench.py` / `policy_bench.py` | 旧版独立诊断，保留作对照，不是新闭环的替代                   |

Trace 按到达顺序记录为 `[length,...]`。实验只改变行数，width 和 dtype 固定。RMSNorm 在每一行内部归约，因此**补行合法、随意补width不合法**。输入是一对形状相同的`x,residual`，输出保持相同形状。

每个候选bound的profile包含 `steady_ms`、`eager_ms`、`first_premium_ms`、`mad_ms`。first premium取`max(first-median_steady,0)`，是当前进程/缓存制度下的首次调用附加开销估计，不能称为精确编译时间。

## A. 基础：先把时间和工作量算对

| TODO / API           | 要补充的算法                                      | 有效行 | 测试                          |
| -------------------- | ------------------------------------------------- | -----: | ----------------------------- |
| A1 `split_trace`     | 先按时间切分，再统计训练shape；保留hold-out顺序   |     10 | `chronological_split`         |
| A2 `summarize`       | median/MAD、first premium、非法数据/重复bound拒绝 |     18 | `robust_profiles`             |
| A3 `segment_costs`   | 前缀频次和、真实lookup区间成本表                  |     17 | `interval_cost`               |
| A4 `trace_shift`     | 联合support上的总变差距离，处理新shape            |     13 | `distribution_shift`          |
| A5 `padding_account` | 按频次加权真实/填充rows，分别统计fallback         |     17 | `weighted_padding_accounting` |

### A.1 必须先做的手算

1. `[8,4,8,99,99]`在前三次后切分：写训练直方图和未来trace。若先排序再切分，哪一个未来信息已泄漏？
2. 三个warm测量`[1,1,200]ms`：均值、median、MAD分别是多少？只保留median能否隐藏设备抖动？因此原始样本和MAD都要留下。
3. shapes `(4,3次),(8,2次),(16,1次)`，bound16的premium=7ms、warm=4ms：整个区间成本为何是31ms？premium为何只付一次？
4. 长度始终2与长度1/3交替均值相同。A4应给多大距离？仅监控平均长度会漏掉什么？
5. `[(3,10),(7,1),(20,2)]`映射到bounds `[4,8]`：先加权算真实rows与执行rows，再算padding比例；不能平均三个shape的百分比。

`Sample` 表示已采集的时间序列，其中的性能数据并非学生编造。A 中的固定数字只用于算法单测，C 会生成真实测量结果。

## B. 核心：有限资源下规划和执行生命周期

| TODO / API               | 核心问题                                           | 有效行 | 测试                       |
| ------------------------ | -------------------------------------------------- | -----: | -------------------------- |
| B1 `plan`                | prefix×bucket_count的DP、容量上限、回溯与确定性tie |     26 | `dp_exhaustive`            |
| B2 `route`               | 上界二分、padding guard、未知大shape fallback      |     11 | `guarded_route`            |
| B3 `ArtifactCache.put`   | transactional LRU admission，只淘汰未pin项         |     16 | `pinned_transactional_lru` |
| B4 `ArtifactCache.lease` | acquire/release计数、LRU touch、非法release        |     10 | `lease_lifetime`           |
| B5 `admit`               | 预测调用量摊销first premium，加入MAD保守裕量       |      9 | `admission_break_even`     |

### B.1 DP必须求什么最优解

候选边界取训练shape；bucket[i,j]的成本为该最大shape实测warm×该区间频次，再加一次first premium。令`D[k,j]`覆盖前j个shape、恰用k个bucket：

$$D[k,j]=\min_{i<j}\{D[k-1,i]+Cost(i,j-1)\}.$$

最多capacity个bucket，在最终j=n列选择最小值并回溯。**这个DP在给定成本表与连续区间约束下最优，不代表真实运行时最优**：guard可能使高padding请求fallback、admission可能拒绝编译、缓存可能淘汰，都会使真实成本不同。报告必须解释这层差异。

测试独立枚举四个shape的全部分割、四种premium和capacity=1/2/3/4。验证时不能再调用被测 DP 生成预期答案。选择一个实测 profile，画出候选分割与所选解的成本。

### B.2 缓存不是一个dict就足够

手工追踪容量2的 `put(a),put(b),lease(a),put(c),lease(c),put(d)`。d插入失败时，键、值、LRU顺序都应不变。尝试替换仍被pin的a也必须失败。连续两次lease与一次release后，不能提前淘汰。

这里管理的是**应用层artifact引用**，不控制 PyTorch 内部的 Dynamo 或磁盘编译缓存。容量按 artifact 个数计算，不按字节计算；限制3个entry并不意味着“限制编译内存为3MB”。实际 artifact 内存和编译并发可以作为后续研究的问题。

### B.3 admission的反例

eager=3ms、compiled=1ms、premium=10ms、MAD=.5ms、margin=2，则保守每次收益1ms。预计10次调用恰好回本，策略拒绝；11次才允许。若margin=4，任何调用数都无法让保守收益为正。

预测调用数只能根据历史记录和已声明的 horizon 倍率计算，不能统计 held-out 后再回填预测。比较一倍与十倍预测时，也要保留预测错误带来的负结果。

## C. 前沿：让计划真正控制编译和执行

| TODO / API                   | 必须形成的实际路径                                         | 有效行 | 验收                        |
| ---------------------------- | ---------------------------------------------------------- | -----: | --------------------------- |
| C1 `measure`                 | 第一次独立计时、warmup、同步后的逐次wall时间               |     17 | `timed_samples`             |
| C2 `calibrate`               | 实际Inductor编译、正确性、first/warm/eager与总校准成本     |     18 | `real_calibration`          |
| C3 `execute`                 | admission→cache lease→双输入补行→真实编译调用→切回→release |     28 | `executor_padding_fallback` |
| C4 `workflow`                | 历史切分、实测、规划、预测、真实held-out、eager逐请求核对  |     30 | `closed_loop`               |
| C5 CUDA `fused_residual_rms` | 一block/row、shared reduction、真实width、gamma            |     15 | `test.cu`，15组不规则shape  |

### C.1 编译候选不是免费的

C2 需要编译多个候选再做选择，其中一部分最终可能不会使用。`calibration_ms`必须单列，不能只报告选中方案的 warm 时间。候选编译对象可以复用 PyTorch 已有的磁盘或进程 cache，因此 first-call 不代表“全冷编译”。需要冷启动对照时，使用自己的独立 cache 目录，记录目录策略并保留日志，不要删除全局缓存。

C2的真实测试执行 `torch.compile(... backend="inductor", dynamic=False, fullgraph=True)`，不是mock。C3测试为确定性地覆盖编译与eager两条分支，使用固定admission profile，但**调用的compiled artifact仍然是真实C2结果**。性能结论只能依据 C4 的真实 profile，分支测试中的1ms/3ms不能作为实测结果写入报告。

### C.2 必交矩阵和失败分析

先用默认的小 trace 跑通流程。bench 模式随后会自动运行`bench_scenarios.py`中的稳定、突变、长尾三组场景，每组包含36个训练请求和60个held-out请求，训练部分只含4/8/16三种候选bound。必修对照已有可执行驱动和参考实现，也可以增加自己的 trace。

| 组别 | 训练与held-out关系                    | 重点观察                               |
| ---- | ------------------------------------- | -------------------------------------- |
| 稳定 | 类似频率，held-out包含bucket内部shape | 填充、admission和回本                  |
| 突变 | 常见shape交换频率，加入未见中型shape  | A4漂移、缓存/guard、预测失配           |
| 长尾 | 少量大于最大bound的请求               | eager fallback而非错误截断或无限重编译 |

每组自动比较 eager、static、dynamic、fixed-bucket 与 measured plan（horizon=1/10）。驱动会在运行各策略前调用`torch.compiler.reset()`，清理 Dynamo 进程状态，避免不同策略累计触发 recompile limit；磁盘 cache 仍保留，因此不能称为全冷编译。`bench_scenarios.py`与 C4 使用相同 seed，生成相同输入、dtype 和 shape 顺序，并逐一与 eager 核对输出。旧`torch_bench.py`仅用于独立诊断，其默认 trace 不同，不能把结果放进配对曲线。measured plan 的 replay_ms 是首次完整路由的 wall time，基线同时提供 first/warm，比较时须选择相应口径；calibration_ms 单独列出，并计入整个生命周期的总成本。

必交表：bound计划、预估成本、实测calibration_ms、实测replay_ms、eager replay时间、compiled/eager次数、padding比例、TV shift、horizon倍率。重复五个独立进程，画累计总成本与生命周期的关系。当前replay包含Python列表/路由/padding等开销；这是闭环wall time，CUDA单kernel时间在C5另记，不混为同一指标。

### C.3 CUDA局部融合与闭环关系

C5将residual add和RMSNorm合为一个kernel，CPU oracle用double累加；gamma不是全1，epsilon固定1e-5。width=1/31/128/513/1024、rows=1/17/64，分别验证与双kernel基线一致。

当前 PyTorch C2–C4 由 Inductor 选择内核，C5 是手写融合的独立实测对照，**没有将C5注入torch.compile图**。这里已提供两者的完整答案与测量。通过 torch.library 注册自定义算子并将其纳入选择，属于挑战任务；现有两条路径不能描述为一个统一的 megakernel。

## 报告与验收

报告必须含所有TODO测试、A的手算、DP穷举示例、pin状态变化、C的三组真实trace、至少一个不获益结果、工时表。评分见 [grade.md](grade.md)。完整C应跑CUDA；只有CPU时明确C5未验收。

## 可选挑战

选做6–12h：把C5经自定义算子注册加入编译候选；或实现“候选预估→少量试编译→更新成本”的主动校准，以减少C2全候选成本。挑战也需要强基线和 held-out 验证，新增选项的数量不能说明研究价值。

## 资料汇总

对照 [resources.md](resources.md) 阅读 CS336 的系统测量要求、PyTorch 的 guard/recompile 行为和 Mirage 的搜索/验证思想。写出本课「给定 shape 边界的 DP」与 Mirage「tensor 程序候选搜索」各自搜索空间和等价性保证的区别。

| 资料                                                                                                                               | 本实验中怎样使用                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [课程 D3](../../course/modules/03-dynamic-compilation.md)                                                                          | 完成 padding、回本、事务缓存的自测；按讲义观察首次编译与稳态时间线。                  |
| [PyTorch 编译 profiling](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_profiling_torch_compile.html) | 查首次调用与 graph break 的 trace 例子，事件名称结合安装版本核对。                    |
| [Perfetto UI 指南](https://perfetto.dev/docs/visualization/perfetto-ui)                                                            | 查看 `torch.profiler` 导出的兼容 Chrome trace；runner 运行记录 JSON 不是 trace 格式。 |
| [资源、定位与简化边界](resources.md)、[参考实现](../../solutions/lab3/README.md)                                                   | 对齐测量、DP、实际执行与独立手写融合；避免把 C5 描述成已注入 Inductor 的算子。        |
