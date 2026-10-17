# Lab 7 参考解题导读

[任务书](../../labs/lab7/README.md) · [验收标准](../../labs/lab7/grade.md) · [来源映射](../../labs/lab7/resources.md)

参考答案覆盖全部 20 个必修 TODO，文件与学生侧使用相同的相对路径：[`src/moe_student.hpp`](src/moe_student.hpp) 是A/B与CPU C，[`src/pipeline_student.cuh`](src/pipeline_student.cuh) 是CUDA C。公共 runner 只在临时副本中叠加这些文件；不要将 solutions 整体覆盖到自己的练习目录。

## 1. A：先确保路线和数值守恒

A1先稳定排序，再对选中的logits减最大值并归一化。A2用counts→prefix→cursor散射，与排序结果具有相同expert分段但保留桶内原始顺序。A3每输出行使用最大绝对值/127，零行scale=1；这允许CPU/CUDA采用相同INT8布局。

A4是加法scatter，重复route刻意保留。A5把每个expert补到tile整数倍，额外行的route_index=-1；C2跳过这些行，绝不能把输出散射到负索引。A6计算的overflow不驱动丢token，因此结果与dropless oracle可比较。

手算例：routes `[(0,2,.5),(0,0,.5),(1,2,1),(1,2,1)]` 的offsets为 `[0,1,1,4,4]`。expert顺序的scalar输出 `[2,4,6,8]` 合并后是 `[3,14]`。这个例子能同时发现重复 route 被覆盖和顺序错位两类 bug。

## 2. B：把时间和所有权作为状态

B1先确定所有可逐出的ready且unpinned entry，再判断总空闲字节；失败时不逐出任何entry。新增载入的完成时间同时依赖请求时间和link_free，立即占用capacity。B2按时间推进完成事件；B3/B4的引用计数确保同一专家被多个消费者使用时不会提前逐出。

B5对历史转移统一衰减，B6按置信度排序并逐个检查字节预算；预测函数看不到完整trace。B7把demand、prefetch、compute和完成事件放在同一时间线上：miss时等待真实ready，pin后才compute，结束再release。模型虽然简单，带宽/显存/预测污染三者会互相制约。

B8 单独计算 Belady 诊断下界，仅适用于等大小专家，并允许使用未来信息。它不参与 B7，也不能将 trace 传给 Predictor 来改善在线结果。

## 3. C：验证的确计算了正确专家

C1实现量化两层MLP，C2使用packed route布局执行与还原，C5串起实际gate与residual block。CPU测试有独立double accumulation，不以调用同一函数作为oracle。

C6将up/down和各自scale放入连续pinned记录；C3用相同offset读GPU槽。C4每次选择一个槽，在copy stream等待该槽上次done，再复制参数，记录ready；compute stream等待ready，连续执行up/down，最后记录done。不同槽可独立推进；相同槽不能在上次compute读完前覆盖。

每个 route 独立写入输出，最后由 CPU 执行 weighted combine，因此 GPU 端无需 atomic。这个设计便于验证，不追求完整 MoE 引擎的最优吞吐。教学核由一个 thread 处理一个输出通道，尚未实现 shared-memory tiling 和高效 GEMM。

## 4. 如何复核

在`labs/`执行：

```bash
python3 tools/run.py --lab 7 --impl reference --part A --mode test
python3 tools/run.py --lab 7 --impl reference --part B --mode test
python3 tools/run.py --lab 7 --impl reference --part C --mode test --device cuda
python3 tools/run.py --lab 7 --impl reference --part all --mode bench --device cuda
```

CPU 路径包含 14 个命名检查，CUDA 路径另检查 12 组配置。两条路径分别运行，保存 runner 输出的记录。B 的模拟时间不能用于证明 C 的加速，CUDA benchmark 也不要求双槽一定更快。

对照答案时，建议只查看当前卡住的 TODO。先写出不变量、画出最小反例，再看实现。除了掌握 map/stream 的调用顺序，还要能解释为什么 blocked 必须满足事务语义，以及 copy 为什么需要等待 done。
