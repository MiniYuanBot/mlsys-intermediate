# Lab 2 验收与评分（100 分）

每一行对应真正执行的测试，不能用通过旧版 `test.cu` 代替新必修。计时不设固定硬件速度阈值。参考实现试运行范围与日志由 [项目验证记录](../docs/validation.md) 汇总；本文的正确性门槛来自独立 oracle 和明确格式契约。

| 部分         | 分值 | 可重复验收                                                          |
| ------------ | ---: | ------------------------------------------------------------------- |
| A1/A2        |    8 | 零组/尾组尺度、15 个合法 signed code、奇数行 stride                 |
| A3/A4        |    8 | artifact 非法长度/非有限值拒绝；多 batch 对独立 dense double oracle |
| A5/A6        |    8 | 实际字节计数、未 clipping 半步长误差界、clipping 穷举最优           |
| A 手算       |    6 | 2×5/group3 的逐 byte 布局；clipping 与激活误差的反例                |
| B1/B2/B3     |   10 | warp 与 pair，特别 group=3 的跨组 byte；所有边界形状                |
| B4           |    8 | batch=1/3/4/7，尾 token 不越界，解码结果跨 token 复用               |
| B5/B6        |    8 | split-K 长度17的尾分区、独立合并、double oracle                     |
| B 性能分析   |    9 | 至少5次独立记录、四路径曲线、一个负结果、计时边界                   |
| C1/C2        |    6 | alpha=0恒等、浮点变换等价、不修改校准原输入                         |
| C3/C4        |   10 | 穷举候选最小化、冻结 artifact held-out；真实 CUDA调用核对           |
| C5           |    7 | joint group/alpha 硬字节预算；不可行预算显式失败                    |
| C 对照和报告 |    8 | calibration/held-out分开、alpha0/clipping-only对照、预算曲线        |
| 可复现与工时 |    4 | 环境、seed、命令、原日志、真实工时，说明未完成内容                  |

```bash
python tools/run.py --lab 2 --impl student --mode test --device cpu --part A
python tools/run.py --lab 2 --impl student --mode test --device cuda --part all
python tools/run.py --lab 2 --impl student --mode bench --device cuda --part all
```

`quant_test.cpp` 覆盖 A1–A6、C1–C5；`pipeline.cu` 覆盖 B1–B6 和 C 的冻结 artifact→CUDA 集成。测试使用的函数名和断言见源文件。若数值不正确，相应内核的性能项不得以加速获分；若只有 CPU，则保留 A 与 C 的主机项成绩，B 和 C GPU集成明确未验收。A6 clipping 不适用 A5 的未 clipping 半步长上界，不把它误判为失败。

自动测试能覆盖明确反例，不能证明实现没有任何错误；报告中的新 workload 和曲线必须由本人运行，参考答案不是可提交的个人实验结果。
