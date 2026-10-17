# Lab 8 验收与评分（100分）

[任务书](README.md) · [参考解题导读](../../solutions/lab8/README.md)

| 部分                  | 分值 | 实际验收                                                 |
| --------------------- | ---: | -------------------------------------------------------- |
| A1/A2 历史预测        |    6 | `a_length_history`：窗口/冷启动/class隔离                |
| A3/A5 调度输入与分布  |   10 | `a_assign,a_histogram`：守恒、已知划分、overflow         |
| A4/A6 优势与采样账本  |    9 | `a_advantage,a_sample`：常数组、快照行为概率             |
| B1/B2 分段与完整batch |   12 | `b_reassemble,b_batch_gate`：身份/版本/gap/重复/慢组     |
| B3 动态调度           |   18 | `b_scheduler`：token工作守恒、恢复、确定时序、无未来偷看 |
| C1/C2 有界group队列   |   10 | `c_queue_admission,c_queue_drain`：背压、完整组、超龄    |
| C3 梯度               |   10 | `c_gradient`：有限差分、零优势、clipped分支              |
| C4 实际学习闭环       |   10 | `c_learning_closed_loop`：3个seed、真实参数变化、lag反例 |
| 实验与报告            |   15 | A3/B12/C108行；三张图、主源映射、模拟/学习边界           |

参考实现已在CPU离线编译并通过12项Rust契约测试及全部benchmark；GPU不是本lab依赖。准确运行环境、命令与哈希见 [项目验证记录](../docs/validation.md)。主执行代理更新最终统一日志前，请勿用旧版本实验记录替代本版。

报告分为数据完整性5分、机制解释与错误反例5分、来源/工作量/学习目标边界5分。B不设“steal必须更快”门槛；C的reward>0.85仅适用于给定宽松lag toy测试。严格lag下较差的学习结果应保留并解释。

代码审查额外检查：调度决策不得读取actual长度；训练group不得由快样本拼凑；不能把当前概率重新写回behavior概率以消除off-policy差异；不能用预设reward曲线替代梯度训练。每个TODO都可通过阶段测试定位，自动通过不替代报告解释。
