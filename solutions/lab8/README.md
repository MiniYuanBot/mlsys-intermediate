# Lab 8 参考解题导读

[任务书](../../labs/lab8/README.md) · [验收标准](../../labs/lab8/grade.md) · [主源映射](../../labs/lab8/resources.md)

全部13个TODO的参考答案位于 [`src/rollout_student.rs`](src/rollout_student.rs)，覆盖历史预测、事件调度、队列、梯度与真实小策略学习。实现不依赖第三方 Rust crate，学习曲线来自实际运行。

## 1. A：明确已知信息与采样分布

A1/A2只保留已完成的每类长度窗口，以经验上中位数预测；冷启动固定32。A3按预测长度的降序排任务，再放入最小预计负载worker，tie-break保证可复现。A5的分位数使用`ceil((N−1)q)`的明确约定，桶包含其上界，最后一个是overflow。

A4 在每个 group 内减去均值，再除以总体标准差；常数组返回零。A6在固定logit快照上把uniform转换为二动作，记录当时的行为概率和版本。采样之后不允许用训练后的参数覆盖behavior概率。

## 2. B：队列与时间事件是两个不同顺序

B1 按 start 重排收到的 segments，再检查连续性及 terminal；接收顺序可以不同于轨迹顺序，但重组后必须恢复正确顺序。B2 要求 expected ID 集合完全一致且无重复，因此也能发现整个慢 group 缺失的情况。

B3先把job ID放进LPT队列；空闲worker可从最长队列末尾偷工作。executor决定quantum的实际长度并安排完成事件；主循环直接跳到最早事件，结算同一时间全部完成worker，再重新调度。暂停的job保留version与精确progress，输出可由B1重建。

手算 `[6,1,1,1]`、两个worker、所有estimate=2、quantum=3：无steal的finish为7；允许steal后，空闲worker在t=2取走另一个队列中的短任务，finish=6、steals=1。测试同时检查这个明确时间表与更大样本的token守恒。

## 3. C：背压、完整group与梯度的结合

C1容量按group预留，允许已预留group继续补齐，避免partial group堵死队列。C2整组剔除超龄样本，然后按group ID取完整组。consumed集合防止重复训练；由于实验有限，未实现生产系统的集合回收。

C3对clipped importance surrogate显式求导。动作1的log概率导数是1−p，动作0是−p；乘ratio和group advantage，截断生效的分支梯度为0。更新前冻结全部当前logit，group结束后统一更新并推进version。`c_gradient`用有限差分而非只检查某个预设loss。

C4从快照采样每波4×8个真实动作，经B3执行合成token工作量，再按group完成顺序进入有界队列。队列满时训练者消费一组并更新策略；生产者保留被阻塞组重新检查freshness。每波保存reward和累计工作量快照，使图表每行都有一致时间语义。

max_lag=4 时，四类 context 都能持续参与训练，给定的 3 个 seed 在 100 波后达到测试要求。max_lag=0 会拒绝仍来自旧快照的组；学习效果变差是本实验要观察的反例，应保留相应结果。所有结论仅适用于当前 contextual bandit，不能外推到 LLM 质量。

## 4. 运行与验证

从`labs/`执行：

```bash
python3 tools/run.py --lab 8 --impl reference --part A --mode test
python3 tools/run.py --lab 8 --impl reference --part B --mode test
python3 tools/run.py --lab 8 --impl reference --part C --mode test
python3 tools/run.py --lab 8 --impl reference --part all --mode bench
```

测试包含 12 项契约检查，完整 benchmark 输出 A3/B12/C108 行；运行时保留命令、环境和结果。B 的时间单位来自事件模型，C 的曲线来自实际更新的 policy。答案不包含真实 KV/RNG 保存、多 GPU rollout、语言模型 reward 或 DARTS 冗余选样；这些内容需要后续模型实验验证。
