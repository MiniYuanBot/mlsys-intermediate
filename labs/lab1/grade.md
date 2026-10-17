# Lab 1 评分与试跑依据

[任务](README.md) · [参考导读](../../solutions/lab1/README.md) · [实际运行记录](../docs/validation.md)

满分 100。A/B/C 都是必修，开放挑战不补偿缺失的阶段。参考已按默认测试/benchmark 实际执行，以下数值容差来自 oracle 的计算精度；没有根据参考机器速度设硬性性能门槛。

| 项               | 分值 | 自动或人工证据                                                            |
| ---------------- | ---: | ------------------------------------------------------------------------- |
| A1/A2 格式       |   12 | 两轴、尾组、常量、nibble、半 scale 误差；`test_a`                         |
| A3/A4 数据与误差 |   10 | 有序唯一 gather、非法 ID、零向量约定；`test_a`                            |
| A 推理           |    3 | 手算组号与 metadata；新增手算测试                                         |
| B1/B2 状态       |   12 | mandatory 并集、同分规则；3 residual×37 append，非法输入不改变状态        |
| B3/B4 还原与账本 |    6 | 时间顺序、payload 实际计数，报告不称 RSS                                  |
| B5 查询          |   12 | 与 materialized oracle 的 max_abs < 2e-5；代码检查不调用 materialize      |
| C1/C2 PQ         |   15 | exact book、SSE 单调、空簇有限、refit 清码；编码不修改 book               |
| C3/C4 查询闭环   |   15 | held-out lookup 对照 < 2e-6、full-budget 精确恢复、mandatory、fetch bytes |
| 实验与报告       |   15 | 18 行默认矩阵、三张图、至少两个新增反例、准备/查询摊销与简化边界          |

每项可按子性质部分给分；编译失败不导致手算部分自动归零，但相关实现项未完成。测试通过只覆盖声明的数据范围；需要保留原始 JSON 和源码版本才能认定可复现。

必交命令：`--part A/B/C --mode test` 各通过，`--part all --mode bench` 输出 18 条数据。A 阶段可以在 B/C 尚未填写时单独运行。参考 run 不能充当学生 run：二者 record 中 implementation 字段不同。

必查错误：将索引 bytes 当总系统内存；把 logical bytes 当 allocator RSS；fit 使用未来 query；只减少索引而未物理 gather；只看 recall 不看 attention 数值；将 CPU 微基准称为论文 GPU 吞吐复现。上述错误扣对应解释项，先修结论再进入研究扩展。
