# MLSys Intermediate

从 [课程总索引](index.md) 开始。本课程接续 `mlsys-foundations`，包含八个研究专题。每个专题围绕一个具体的系统问题展开：先推导机制，再实现和测量，最后判断候选研究问题是否值得继续。

项目内容分为三个目录：`course/` 放课程、阅读资源、研究路线与可视化；`labs/` 放实验、测试、公共工具和本地实验产物；`solutions/` 放参考实现。把项目目录作为 Obsidian vault 打开，入口是 `course/index.md`；标准相对 Markdown 链接也可在 GitHub 阅读。项目总入口、共享编辑器设置和格式配置位于[仓库根目录](../README.md)。

## 如何使用

1. 用 [选题地图](guides/directions.md) 了解八个具体方向的硬件要求与研究切口。
2. 按 [36h/周路线](guides/roadmap.md) 用两周初步了解各个方向：只读指定片段，完成一个小问题，无需做完八个 lab。
3. 选择一个主方向，完整学习约 5h 讲义：按小节读图、做随堂自测，再展开核对要点；最后用文末清单检查掌握情况。随后完成对应 lab 的分项任务。实验标出的 23–26h 包含学生新增测试、对照、数据分析和报告，并非运行现成脚本的耗时。
4. 用失败案例和原始数据支持下一步假设，并与最接近的相关工作比较。准备论文或申请服务器时，要说明一个可以复核的问题。

课时按已完成 foundations 的学习者估算，尚未经过学生试学验证。若实际用时超出计划，记录卡在哪里，减少当周任务；尚未完成的内容不算作已掌握。具体任务与时间安排见 [实验索引](../labs/index.md) 和各 lab 的 README/grade。

每讲结尾分别列出“资料汇总”和“自测清单”。[可视化阅读索引](readings/visualizations.md)按模块列出入口和观察任务，区分教学图解、公式演示与实测分析工具。lab 中的“可选挑战”另列，先完成必修部分，再决定是否深入。

## 开始实验

命令与 foundations 一致，统一在 `labs/` 目录执行：

```bash
cd labs
python3 tools/doctor.py
python3 tools/run.py --lab 1 --impl student --mode test
# TODO 未实现时失败是预期行为；先按 lab1 的指导实现，再重跑。
python3 tools/run.py --lab 1 --impl reference --mode test
```

使用 [环境说明](../labs/setup.md) 复用现有环境；运行产物保存在 `labs/results/`，个人报告写入 `labs/reports/`。参考实现通过临时副本运行，不覆盖学生源码。参考答案覆盖已提供的核心接口；新增实验与开放挑战的覆盖范围以各 lab 清单为准。

## 资源与质量记录

- [大学课程与开放 lab](readings/courses-and-labs.md)、[论文和源码索引](readings/resources.md)、[固定源码阅读入口](readings/code-map.md)。
- [离线交互可视化](visuals/index.html)：KV 预算、推测成本、编译摊销和激活预算；演示值不当作实测。
- [验证范围](../labs/docs/validation.md)、[独立评审记录](../labs/docs/editorial-review.md)、[统一格式规范](../labs/docs/formatting.md)。

资料核验和本次设计修订日期为 2026-10-17（Asia/Shanghai）。这八个方向供你开始探索，候选假设还需查新并与导师讨论。完成教学实验不等于完成论文。
