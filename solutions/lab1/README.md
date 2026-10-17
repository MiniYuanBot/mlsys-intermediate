# Lab 1 参考答案导读

[任务与全部接口](../../labs/lab1/README.md) · [评分](../../labs/lab1/grade.md) · [来源](../../labs/lab1/resources.md)

完整答案覆盖 13 个 TODO：A1–A4、B1–B5、C1–C4。对应文件是 [kv_student.hpp](src/kv_student.hpp)、[cache_student.hpp](src/cache_student.hpp)、[pq_student.hpp](src/pq_student.hpp)。测试、benchmark 与 driver 在 labs 中，参考模式临时覆盖同名源码。

A 先构造组 min/max，再打包无符号 nibble；常量 scale=1，code=0。gather 强制有序唯一 ID，保证下游位置与原顺序一致。误差函数给零向量明确约定，避免图里产生无法解释的 NaN。

B 的 flush 仅消费最老完整块。材料化函数提供密集 oracle，查询则逐块执行在线 softmax：遇到新的最大值时，同步缩放分母与加权 V，从而支持跨量化块归一化。payload bytes 不包含容器 capacity 或临时解码数组。当前 append 对输入错误保证不修改状态，不承诺分配失败时的强异常保证。

C 的 Lloyd 实现使用确定性初始化，空簇保持旧中心，避免除零；重训练会清空旧 codes。ADC 的表是内积而非 L2，查询不训练码本。最终取回原始 K/V 后计算 attention，因此压缩检索误差来自候选遗漏，而不是对 V 做 PQ。

运行（cwd 为 `labs/`）：

```bash
python tools/run.py --lab 1 --impl reference --mode test
python tools/run.py --lab 1 --impl reference --mode bench
```

参考测试覆盖 111 个增量边界、两种量化轴、精确小码本、SSE 单调和 16 个 held-out query 的独立 dot oracle。基准产生 18 行，三 seed 两长度三预算。具体实测与源代码哈希见[验证记录](../../labs/docs/validation.md)。这些数值结果供核对使用，无须抄写；你需要自己解释预算曲线和负结果。

可尝试的错误注入：移除 softmax 旧累积缩放；交换 nibble；重 fit 后保留旧 codes；将内积表改成距离；让 gather 接受重复 ID。逐一确认目标测试失败后恢复正确实现。通过这些错误注入，检查测试能否发现不变量被破坏。
