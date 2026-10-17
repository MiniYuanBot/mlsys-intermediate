# Lab 6 评分标准

[任务](README.md) · [参考解释](../../solutions/lab6/README.md) · [实际试跑](../docs/validation.md)

满分 100。评分对应已提供的 14 个测试方法及 24 行测量矩阵。默认计算 FP32；梯度对照 rtol=1e-4、atol=1e-6，形状/状态/字节使用精确检查。机器性能不设固定门槛。

| 项目            | 分值 | 验收证据                                                  |
| --------------- | ---: | --------------------------------------------------------- |
| A1 forward/梯度 |    6 | 冻结 W、非零 B 的输入梯度、bias                           |
| A2/A3 merge     |    6 | train 拒绝、幂等、三轮精确恢复                            |
| A4 注入         |    6 | 嵌套目标、冻结、bias/device 保留、坏目标无修改            |
| A5/A6 state     |    9 | cloned CPU、名称/shape/finite/scale 验证、失败不部分更新  |
| A7 账本         |    3 | trainable/frozen 实际 bytes 与范围解释                    |
| B1 checkpoint   |    6 | dropout 条件下逐参数梯度一致                              |
| B2/B4 规划      |   10 | 独立穷举最优解、预算、不可行与非法输入                    |
| B3/B5 测量      |    9 | storage view 去重/排除参数；真实 fw+bw 校准及单位         |
| C1/C2 offload   |    8 | payload 所有权、重复 unpack、CUDA 真拷贝、计数            |
| C3 执行         |    7 | 四策略输入和参数梯度；缺 owner 拒绝                       |
| C4 训练         |    8 | 2+3 微批与完整 batch 的 optimizer 更新一致；clip 在累积后 |
| C5 完整测量     |    7 | warmup、真实 optimizer、CUDA 同步、24 行与 loss 一致      |
| 报告            |   15 | 两张图、至少两个独立反例、预测/实际差距、准备成本和边界   |

CPU 可完成数学/训练正确性；C1/C2 的 CUDA 实际拷贝以及 C5 的 GPU 峰值证据要在 4060/服务器补齐。CPU 的 null 峰值和零拷贝应保留，不给它虚构 CUDA 分。可先获得其余部分并明确未验项目。

阶段命令 `--part A/B/C --mode test`，整体 `--mode bench`。参考 record 验证题目可执行，学生自己的 run 才验证本人实现。没有要求达到“offload 一定更快/更省”的方向性结论；出现慢或峰值更高需解释机制。

逐 TODO 的测试关联见 test 方法名称与 README。故意破坏保存 RNG、微批权重或参数排除应能触发相应失败；将阈值调大到吞掉错误、关掉原 dropout 测试或直接返回参考张量不能获得实现分。
