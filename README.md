# MLSys Intermediate

接续 [mlsys-foundations](https://github.com/MiniYuanBot/mlsys-foundations) 的自学课程，包含 KV 缓存、低比特算子、动态编译、推测解码调度、前缀复用、低显存训练、MoE 卸载与 rollout 调度八个专题。每个专题配有讲义、自测题和基础、核心、前沿三个阶段的实验，可以按兴趣选择一个方向深入。

| 目录                              | 内容                                                   |
| --------------------------------- | ------------------------------------------------------ |
| [course/](course/README.md)       | 课程纲要、讲义、阅读资料、研究路线和笔记模板           |
| [labs/](labs/README.md)           | lab1–lab8 的任务、学生代码、测试、benchmark 和实验工具 |
| [solutions/](solutions/README.md) | 各 lab 的参考实现与解题导读                            |

## 开始

1. 阅读[选题与硬件地图](course/guides/directions.md)，按[每周 36h 路线](course/guides/roadmap.md)选择主方向。每讲约 5h，配套实验约 23–26h，包含实现、调试、测量和报告；可根据实际进度调整。
2. 从[课程索引](course/index.md)找到讲义，按[环境配置](labs/setup.md)准备对应实验的工具链。可以复用 foundations 的环境，再补装所选 lab 的依赖。
3. 按讲义小节学习，先做自测，再核对答案。进入对应 lab 后，依次完成 A/B/C，并按 `grade.md` 自查。开放挑战另计时间，无需做完全部实验才开始研究。

实验文档中的命令默认在 `labs/` 目录执行。以 lab1 的基础阶段为例：

```bash
cd labs
python3 tools/doctor.py
python3 tools/run.py --lab 1 --impl student --mode test --part A
```

初始代码的 TODO 尚未实现，第一次测试失败是预期行为。完成相应任务后重跑测试，再按实验要求运行 benchmark。原始输出保存在 `labs/results/`，个人报告放在 `labs/reports/`。

各实验的 CPU/CUDA 范围见[实验索引](labs/index.md)。CUDA 部分需要 NVIDIA GPU；Windows 原生、Linux、WSL2 和 macOS 的环境与命令见[平台配置](labs/setup.md)，Mac 使用 CPU 路线。参考答案的运行和对照方法见 [solutions/](solutions/index.md)，建议独立尝试后再看；参考运行使用临时副本，不覆盖学生源码。

## 阅读与开发

用 VS Code 打开仓库根目录，即可同时编辑讲义、实验和参考答案；新终端默认进入 `labs/`。也可以把根目录作为 Obsidian vault 打开，从 [course/index.md](course/index.md) 开始阅读。

> 延伸阅读：[公开课程与 lab](course/readings/courses-and-labs.md) · [论文与代码](course/readings/resources.md) · [可视化资料](course/readings/visualizations.md)。
>
> 实验时可查阅：[公共工具用法](labs/tools/README.md) · [实验记录与分析](course/guides/experiments.md) · [报告模板](labs/templates/report.md) · [代码格式化与编辑器配置](labs/docs/formatting.md)。
