# 外部课程与开放 lab：按问题选用

[资料索引](resources.md)

以下链接来自大学或作者，可以直接访问。按问题选读即可，无需重新修读整套课程。本仓库的教学实现为自行编写，外部作业用于参考实验组织方式和对照机制。阅读时使用固定 commit，避免主分支切换作业年份后内容变化。

| 官方资源                                                                                                                                    | 具体内容/入口                                                                             | 对应模块       | 8GB 环境的取用方式                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------- |
| [Stanford CS336 Systems](https://github.com/stanford-cs336/assignment2-systems)                                                             | 仓库 handout、tests/adapters、benchmark/profiling 与系统任务；[固定手册链接](code-map.md) | D2/D3/D6       | 抽取 profiling、单算子与内存分析；不照搬多 B200 的原规模             |
| [Stanford CS336 Alignment](https://github.com/stanford-cs336/assignment5-alignment)                                                         | 2026 alignment handout、`tests/test_grpo.py`                                              | D8             | 先做数学单测和小模型闭环；正式训练另算显存与时间                     |
| [CMU Deep Learning Systems](https://dlsyscourse.org/)                                                                                       | HW3 NDArray CPU/CUDA backend；autograd 与训练系统讲义                                     | D2/D3/D6       | 已完成 foundations 则只对照 backend/autograd 接口，不重写整个 Needle |
| [CMU 15-442/642 MLSys](https://mlsyscourse.org/)                                                                                            | LLM serving、编译、系统论文专题与课程材料                                                 | D1/D3/D4/D5/D7 | 作为专题阅读入口；不是声称每方向都有可直接下载的完整作业             |
| [MIT 6.5940 2024](https://hanlab.mit.edu/courses/2024-fall-65940)                                                                           | 官方课程页的 Lab 2、Lab 4 notebooks 与量化课程                                            | D1/D2          | 数值量化/部署子任务；本课程补充真实 CUDA 融合与强 baseline           |
| [LCPU Infra Seminars](https://infra.seminars.lcpu.dev/) / [仓库](https://github.com/lcpu-club/wmhpc-training-camp-x-lcpu-ai-infra-seminars) | 日程、资料与作业入口，按站点当前条目选读                                                  | 全局衔接       | 用来查缺补漏和获得本地实践语境；不重复 foundations 已掌握部分        |
| [GPU-Puzzles](https://github.com/srush/GPU-Puzzles)                                                                                         | 小尺寸 GPU indexing/reduction 可视化谜题                                                  | D2/D3          | lane/warp 思考卡住时回查；不当作研究 benchmark                       |
| [Triton-Puzzles](https://github.com/srush/Triton-Puzzles)                                                                                   | 小算子与 mask/reduction 练习                                                              | D1/D2/D3       | 可作为另一种实现语言对照，不代替本 lab 的 CUDA/PTX 部分              |
| [learn-cuda](https://github.com/gau-nernst/learn-cuda)                                                                                      | `11_gemv`、`12_megakernel` 等作者实验                                                     | D2/D3          | 挑支持本机架构的目录；sm100/sm120 路径不是 4060 代码                 |

## MIT 官方 notebook 的直接入口

下列链接来自 MIT 2024 官方课程页，Colab 是否允许打开/复制受其分享权限和运行环境影响：

- [Lab 2](https://colab.research.google.com/drive/11IBla1q1McoZ2oCANCGHns8VtzG5nCMP)：配合量化主题，先做表示/误差，随后完成本仓库 D1/D2 的系统成本实验。
- [Lab 4](https://colab.research.google.com/drive/16H9RvSg4XIF35X3fLGQUVwAE9ccvDj14)：配合 LLM 部署主题；模型和 runtime 以 notebook 当前内容为准。

## 每方向的最短外部补充

| 方向 | 先看什么                                  | 最多投入 | 之后回到哪里                 |
| ---- | ----------------------------------------- | -------- | ---------------------------- |
| D1   | MIT 量化数值任务 + KIVI 分组代码          | 3h       | lab1 的误差/预算矩阵         |
| D2   | GPU-Puzzles 一个 reduction + AWQ GEMV     | 3h       | lab2 PTX 与三 baseline       |
| D3   | CS336 profiling + torch.compile tutorial  | 3h       | lab3 冷启动/形状实验         |
| D4   | CMU serving 专题 + AdaServe artifact 配置 | 3h       | lab4 非线性成本与估计偏差    |
| D5   | SGLang radix cache + 作者图解             | 3h       | lab5 pin/rollback/热点 trace |
| D6   | CMU autograd + checkpoint 文档            | 3h       | lab6 梯度与峰值测量          |
| D7   | MoE-Infinity expert I/O benchmark         | 3h       | lab7 总流量与 overlap        |
| D8   | CS336 GRPO 测试 + DARTS recipe            | 4h       | lab8 完整组/版本与长度 trace |

专家缓存、多 SLO 推测等专题的 lab，是结合公开论文与代码设计的教学任务，并非某所大学的现成作业。使用上游学生作业时，按各自说明处理许可与课程政策。
