# 参考实现

[课程索引](../course/index.md) · [实验索引](../labs/index.md) · [验证状态](../labs/docs/validation.md)

目录布局与 `labs/labN/` 对应，仅覆盖 TODO 文件。测试、benchmark 和提供的 driver 留在 labs。参考实现用于解释机制，不是各论文原系统或性能最优实现。

以下命令从 **`labs/` 目录** 执行。A/B/C全部必修TODO均有答案和测试；学生个人反例、报告和开放研究挑战仍需独立完成。对照答案前，先记录自己的最小失败输入与预期输出。

```bash
python tools/run.py --lab 4 --impl reference --mode test
python tools/run.py --lab 2 --impl reference --mode bench --device cuda
# 需要手动调试时，创建新副本；目标已存在会拒绝。
python tools/materialize.py --lab 2 --impl reference --to /tmp/mlsys-lab2-reference
```

`run.py` 每次复制整个 lab 到临时目录后覆盖参考 TODO，结束后清理副本；结果记录始终保留。不会修改学生实现，无需先 git commit 或创建 worktree。

| Lab | 完整参考导读               | 主要实现范围                                             |
| --- | -------------------------- | -------------------------------------------------------- |
| 1   | [13个TODO](lab1/README.md) | INT4、流式残余缓存、online softmax、PQ训练/ADC/gather    |
| 2   | [17个TODO](lab2/README.md) | 量化校准、PTX、warp/pair/batch/split-K、激活感知搜索     |
| 3   | [15个TODO](lab3/README.md) | trace/profile、DP、缓存lease、真实Inductor执行、CUDA融合 |
| 4   | [14个TODO](lab4/README.md) | 概率统计、置信控制、离散事件请求队列和SLO指标            |
| 5   | [14个TODO](lab5/README.md) | 祖先前缀、事务pin/lease、准入、公平调度和并发回放        |
| 6   | [17个TODO](lab6/README.md) | LoRA生命周期、激活规划、实际offload、微批训练            |
| 7   | [20个TODO](lab7/README.md) | 路由/INT8、字节缓存、事件模拟、实际CPU/CUDA专家执行      |
| 8   | [13个TODO](lab8/README.md) | 预测、分段恢复/窃取、版本队列、真实小型策略梯度闭环      |

建议先独立尝试一个 TODO 至少 45 分钟，记录最小失败例子再对照。读答案后关闭文件重新实现，并添加自己的反例测试；“参考测试通过”不是个人实验完成证明。
