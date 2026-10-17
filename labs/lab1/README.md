---
tags: [mlsys, lab, kv-cache, quantization, retrieval]
---

# Lab 1：从 INT4 格式到流式缓存和 PQ 检索

[课程 D1](../../course/modules/01-kv-cache.md) · [实验索引](../index.md) · [评分标准](grade.md) · [资源与改编说明](resources.md) · [参考导读](../../solutions/lab1/README.md)

> [!important] 完成定义
> 本实验使用 C++17，可在 CPU 上完成。你需要实现量化格式、增量缓存和乘积量化检索三个子系统，并用它们完成数值计算。A/B/C 均提供待补全的算法、完整参考答案、独立 oracle 和测量程序；移植论文原系统另列为挑战。

需要先掌握 foundations 中的 C++、内存层次、attention 和基准测量。Mac mini 与 Windows WSL 均可完成必修，无需下载大模型。**下列命令从 `mlsys-intermediate/labs/` 执行**。预计 **24h**：A 5h、B 7h、C 9h、报告 3h，另加 D1 课程 5h。估时包含独立编写代码和调试。运行参考答案通常只需几秒，不能据此判断学习是否完成；后续计划应根据第一个实验的实际用时调整。

## 文件和实施顺序

| 文件                    | 学生负责                         | 已提供                                |
| ----------------------- | -------------------------------- | ------------------------------------- |
| `src/kv_student.hpp`    | A1/A2 编解码、B1 选择            | 分组索引、双精度累加 attention oracle |
| `src/cache_student.hpp` | A3/A4 收集与误差、B2–B5 增量缓存 | 状态结构和接口                        |
| `src/pq_student.hpp`    | C1–C4 训练、编码、查询、取回     | 码本布局与结果类型                    |
| `test.cpp`              | 可追加自己的反例                 | A/B/C 独立测试入口、手算与性质 oracle |
| `bench.cpp`             | 可增加实验变量                   | 三 seed、两长度、三预算的完整测量     |
| `exercise.py`           | 不需要修改                       | C++ 编译、阶段选择                    |

每个 `BEGIN TODO` 到 `END TODO` 之间都是待实现区域，前面的注释说明输入、输出、不变量和实现提示。参考实现位于对应的 `solutions/lab1/src/`。不要删除原测试以通过验收。有效代码量见[自动统计](../docs/workload.md)，只计需要学生补全的区域。可以对照源码核实哪些辅助代码已经提供。

```bash
python tools/run.py --lab 1 --impl student --mode test --part A
python tools/run.py --lab 1 --impl student --mode test --part B
python tools/run.py --lab 1 --impl student --mode test --part C
python tools/run.py --lab 1 --impl student --mode bench
```

初始骨架会抛出明确异常。先完成 A，再做 B/C。分阶段运行时只执行相应测试，但后续阶段仍依赖前面的实现。全部测试用 `--part all`（默认）。runner 使用临时副本，不覆盖你的答案，并记录编译命令、源代码哈希与输出。

## A 基础：能独立解码的低位宽表示（5h）

### A0 手算和布局检查（0.5h）

画出 `T=5,D=7,g=4` 的 token 分组与 channel 分组；写出 `(0,6),(1,0),(4,6)` 的组号、byte offset 和 nibble shift。区分 `ceil(TD/2)` 与 `T*ceil(D/2)`：本格式连续打包，行之间没有补齐 byte。

计算两种布局的 metadata 大小。scale/offset 都是 FP32，每组 8 bytes。小组带来更低表示误差，也可能使总空间超过原表示；不能只报告“每元素 4 bit”。

### A1 `quantize`（1.5h）

输入是连续行优先 FP32，codes 为无符号 0–15。先验证 shape/group 和有限性，再统计每组真实元素的 min/max，得到 scale/offset，最后打包。尾组不引入零 padding 参与范围统计。

常量组使用有限非零 scale，要求精确恢复常量。偶数线性索引使用低 nibble；奇数尾元素的无效高 nibble 必须清零。提示：先写三遍循环的正确版本，再考虑融合统计。

验收会检查两轴 × 四种 group 是否满足半 scale 误差界，仅有 roundtrip 输出还不够。正确性回归包含有限 FP32 极值和 subnormal；性能输入保持常规尺度。用 double 做范围差值与量化算术，并对下溢 scale 保留最小正值。

### A2 `dequantize`（0.75h）

只能读取 Packed 的 codes 和元数据，不能保存或访问原浮点输入。码值 8–15 不是负数。写出 `offset + code*scale` 与组号之间的对应关系，检查 odd tail 和 group 大于维度。

接口接收由合法 `quantize` 产生的 Packed；挑战可以加入损坏元数据的校验。不能把测试中的原 x 放在结构体里绕过解码。

### A3 `gather_rows`（1h）

行收集需要返回一个新的连续数组，不能只返回 index。ID 必须严格递增且在范围内；允许空选择。先校验全部 ID，再开始拷贝，避免部分输出掩盖错误。

手算输入 `[1,2;3,4;5,6]` 取行 0/2 的输出。解释为什么调用方仍持有旧 K/V 时，这个函数会增加瞬时内存；要减少逻辑持有量，调用方还需释放原表示。

### A4 `errors` 与数值诊断（1.25h）

实现 relative L2、max absolute error、cosine，累计用 double。约定两零向量 cosine=1，仅一个为零时为 0；relative L2 的参考范数分母下限为 `1e-20`。

新增一个手算非零向量测试和一个 Inf 拒绝测试。回答：接近零的 attention 输出为何会让 relative L2 很大？为什么还要同时看 max_abs 和任务指标？这些指标是数值诊断，不能直接替代文本质量。

## B 核心：有状态的增量压缩缓存（7h）

### B1 `select_tokens`（1h）

先形成 sink/recent 的并集，再按分数降序填补预算，同分取小 ID，最后按原位置排序。不可行预算显式拒绝；重叠 mandatory 不重复计数。

该函数不规定 score 来源。若用于历史驱逐，不能偷看未来 query；若用于 C 的当前 query 检索，必须把 query 计算与候选扫描计入成本。这是两种不同生命周期。

### B2 `ResidualCache.append`（2h）

状态包含已压缩 chunks 和高精度 recent K/V。每次 append 一行，当 recent 行数达到 `residual+block` 时，把最早一个完整 block 压缩，K 沿 token 轴分组，V 沿 channel 轴分组。

需要维护：

1. `chunks.size()*block + recent_rows == tokens`；
2. append 返回后 `recent_rows < residual+block`；
3. 已压缩 block 的量化元数据不因新 token 改变；
4. 非法输入在修改状态前拒绝。

本 lab 为便于建立状态机，对 K/V 使用同一个 block flush 条件。KIVI 原实现中，K/V 的压缩时机不同，见[改编说明](resources.md)。报告中应说明这一简化，不能称为原论文的逐行复现。

先在 residual=0/block=4 上画 1–9 次 append，再调试 residual=3 和 8。空 residual 仍要支持查询，不能假定 recent 一定非空。

### B3 `materialize`（1h）

这是调试 oracle：按时间顺序解码每个 chunk 并连接 recent。允许分配完整密集缓存；生产查询路径 B5 不允许调用它。用 token 数、输出长度和 residual 原值验证顺序没有颠倒。

### B4 `bytes`（0.5h）

统计压缩 codes、scale、offset 和 FP32 recent 的**逻辑 payload**。它不包含 vector capacity、对象头、分配器、查询临时张量，也不是 RSS。

在报告同时列出 FP32 全缓存 `8*T*D` 和 FP16 理论基线 `4*T*D`。本实现实际存储的 residual 是 FP32，不得把它按 FP16 算。

### B5 `attend` 与 online softmax（2.5h）

维护最大 score `m`、归一化分母 `z` 和加权 V 累积。新 score 到来时，用新的最大值同时缩放旧分母和旧输出。每次只解码一个 block；最后处理 recent。

提供的测试在三种 residual 设置下检查 111 个 append 边界，将流式输出与密集解码 oracle 比较，并用大 score 检查 softmax overflow。容差 `max_abs < 2e-5` 用于 FP32 输出与不同累加顺序，不能放大到隐藏索引错误。

记录查询额外空间的阶数：块解码临时数组约为 `O(block*D)`，状态为 `O(D)`；量化 payload 仍需持有全部历史。CPU 实验不证明 GPU decode 加速。

## C 前沿：PQCache 启发的可执行检索机制（9h）

### C0 阅读与问题拆解（1h）

读 [PQCache](https://arxiv.org/abs/2407.12820) 的码本、检索、CPU/GPU 数据流，再读[源码映射](resources.md)。注意区分“用低位宽表示所有值”和“用压缩索引选出要取回的精确值”。本阶段实现 PQ 与精确 KV 收集，不实现原系统的多头异步 offload。

### C1 `ProductIndex.fit`（2.5h）

把维度均分为 M 个子空间，每个子空间独立做 Lloyd 聚类。确定性初始化使用第 `c*N/C` 行；最近中心同分取最小 ID；空簇保留旧中心。每轮记录更新中心后的总 SSE。

依次分配码本和 assignment/count/sum，寻找最近中心、更新中心，再记录 SSE。再次 fit 时必须清空旧 codes，避免用新码本解码旧 ID。

验收会检查 SSE 不增、空簇数值有限，并用完整 codebook 的手算例子核对结果。需要自己实现训练过程，不能用某个库的 `fit()` 调用代替，以便理解码本训练如何影响检索误差和准备成本。

### C2 `encode`（1.5h）

对冻结码本按子空间取最近中心，保存 uint16 ID。先验证所有 keys，构造临时 codes，再发布结果；编码不重新训练。此处最多 65536 个中心，实际基准用 16。

记录索引大小：`M*C*(D/M)*4 + T*M*2`。训练时工作数组、原精确 KV 和查询 lookup 都不在这个公式里；总系统内存需要另行相加。

### C3 `scores`（1.5h）

为当前 q 构造 `lookup[p,c]=dot(q_part,center)`，再按 codes 求和。attention ranking 使用内积，不能误写成 query 的欧氏最近邻。

测试使用从未进入 fit 的 16 个 query，与重构中心后直接 dot 的 oracle 逐项比较；确认 query 没有修改 book/codes。报告复杂度 `O(CD + TM)`，同时分析小 T 时建表开销为何可能不划算。

### C4 `retrieve` 和完整实验（2.5h）

依次计算 ADC scores，按 mandatory 约束选择预算内的候选，物理收集原 K/V，再计算精确 attention。输出 IDs、attention 和本次收集 bytes；预算覆盖全部时应还原完整 attention。

默认基准已完成 3 seeds × 2 lengths × 3 budgets 的测量，共 18 行，每行包括 append、流式查询、PQ fit+encode、PQ 查询+选择+取回+attention、recall、误差和分项字节。查询时间是 20 次平均，fit 是一次准备成本，均为 CPU wall time。

recall 的 oracle 使用同样 mandatory 约束下的精确内积 top-k，不能把它解释成自然语言任务准确率。`fetch_bytes` 是一次物理收集的 FP32 字节，不等于 PCIe 实测流量。精确 K/V 本实验仍在 CPU 内存。

```bash
python tools/run.py --lab 1 --impl reference --mode test
python tools/run.py --lab 1 --impl reference --mode bench --output results/lab1/reference.json
python tools/extract.py results/lab1/reference.json
python tools/extract.py results/lab1/reference.json --table 0 --out results/lab1/reference.csv
```

## 报告与验收（3h）

提交实现、自己的至少两个反例、运行 JSON/CSV 和 `reports/lab1.md`。画三张图：预算→recall/误差；预算→查询时间/取回 bytes；长度→准备成本和查询成本。图中区分三 seed，保留负结果。

回答：何时压缩元数据使小缓存不划算？PQ 索引 bytes 是否算了原 KV？在线 softmax 与密集 oracle 为何允许微小差异？若 query 分布改变，哪些保证仍成立？申请 GPU 后要验证的是哪一个尚未测量的瓶颈？

自动测试检查算法性质，报告用于评估测量结果的解释是否合理；验收不设固定的加速倍数门槛。完整评分见 [grade.md](grade.md)，参考试跑见[验证记录](../docs/validation.md)。

## 可选挑战

挑战另计 8–16h：从[本地模型采集工具](../tools/capture_model.py)读取真实 K/V；比较层/head 间码本泛化；将 selective gather 迁移到 pinned host + CUDA stream 并测真实传输；为 decode 新 token 加增量索引与重建策略。先保留至少一个失败案例，并建立公平的 baseline，再判断是否值得继续研究。

可先用本实验的 [prompts.txt](prompts.txt) 跑通采集流程。它包含四条中英文示例，适合检查输入、生成与 K/V 导出；正式研究需另选能代表目标负载的文本，并记录来源和长度分布。

在 `labs/` 目录执行，替换模型路径，并使用尚不存在的输出目录：

```bash
python tools/capture_model.py --model /path/to/Qwen3-0.6B \
  --prompts lab1/prompts.txt --output results/lab1/model-probe-01 \
  --max-input 128 --max-new 32 --capture-kv
```

Windows PowerShell 中将上述命令合为一行；模型路径加引号。Mac 或无 CUDA 时加 `--device cpu`。环境依赖与计时边界见[模型采集说明](../setup.md#真实模型采集)。

## 资料汇总

| 资料                                                                                       | 本实验中怎样使用                                                                   |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| [课程 D1](../../course/modules/01-kv-cache.md)                                             | 完成分组地址、flush 状态与 PQ 字节账的自测；讲义内有缓存动画和量化曲线的观察任务。 |
| [KIVI](https://github.com/jy-yuan/KIVI)、[PQCache](https://github.com/HugoZHL/PQCache)     | 分别对应非对称缓存量化与 PQ 检索；不把必修局部机制称为完整论文复现。               |
| [资源、定位与简化边界](resources.md)                                                       | 查具体上游文件、借鉴内容和未覆盖的系统部分。                                       |
| [参考实现](../../solutions/lab1/README.md)、[实验规范](../../course/guides/experiments.md) | 独立作答后核对不变量，按规范记录测量和负结果。                                     |
