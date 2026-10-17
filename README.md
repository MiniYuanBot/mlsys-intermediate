# MLSys Intermediate

本课程接续 [mlsys-foundations](https://github.com/MiniYuanBot/mlsys-foundations)，包含八个研究专题：KV 缓存、低比特算子、动态编译、推测服务、前缀复用、低显存训练、MoE 卸载与 rollout 调度。每个专题配有讲义、分为基础、核心、前沿三个阶段的实验，以及参考实现。

| 目录                              | 内容                                                        |
| --------------------------------- | ----------------------------------------------------------- |
| [course/](course/README.md)       | 课程纲要、模块讲义、论文与源码、研究路线、可视化            |
| [labs/](labs/README.md)           | lab1–lab8 的任务、学生代码、测试、benchmark、评分与公共工具 |
| [solutions/](solutions/README.md) | 与学生代码对应的参考实现及解题导读                          |

## 开始

1. 阅读[选题与硬件地图](course/guides/directions.md)，按[每周 36h 路线](course/guides/roadmap.md)选择一个主方向。
2. 查看[课程索引](course/index.md)与[实验索引](labs/index.md)，按[环境说明](labs/setup.md)复用 foundations 的工具链。
3. 从所选模块的讲义与 lab README 开始，依次完成 A/B/C，用各实验的 `grade.md` 验收。

实验命令默认在 `labs/` 目录执行：

```bash
cd labs
python3 tools/doctor.py
python3 tools/run.py --lab 1 --impl student --mode test --part A
```

初始代码中的 TODO 尚未实现，测试会按预期失败。参考答案在临时副本中运行，不覆盖学生源码；用法见 [solutions/README.md](solutions/README.md)。CUDA 实验使用 NVIDIA GPU；各模块的 CPU/CUDA 范围见实验索引，Windows 建议沿用 WSL2 环境。

每讲预计需要约 5h，完成一个实验约需 23–26h。八个实验共 123 个 TODO、约 2038 行有效参考实现；统计口径见[工作量审计](labs/docs/workload.md)。不要求先完成全部实验再开展研究。

## 阅读与开发

用 VS Code 打开仓库根目录，可使用共享的 [.vscode 设置](.vscode/settings.json)，新终端默认进入 `labs/`。根目录也可直接作为 Obsidian vault 打开，以 [course/index.md](course/index.md) 为笔记入口；[.obsidian/](.obsidian/core-plugins.json)保留基础共享设置，个人工作区状态由 Git 忽略。

格式配置位于根目录，编辑器和命令行工具使用同一份规则：

- `.editorconfig`、`.gitattributes`：缩进、UTF-8、LF 与文本规范。
- `.clang-format`、`.gersemirc`、`.prettierrc.json`、`pyproject.toml`：沿用 foundations 的格式参数。
- `.clang-format-ignore`、`.prettierignore`、`.gitignore`：排除构建产物、依赖、缓存与本地实验数据。
- [formatter-versions.env](labs/tools/formatter-versions.env)：格式工具版本；安装和使用见[格式说明](labs/docs/formatting.md)。

`.envrc` 沿用 foundations 的共享虚拟环境约定，可在使用 direnv 时启用；也可以直接激活已配置的 Python 环境。

## 资源与验证

- [公开课程与 lab](course/readings/courses-and-labs.md)、[论文与代码](course/readings/resources.md)、[交互可视化](course/visuals/index.html)。
- [实际验证记录](labs/docs/validation.md)、[交叉审核](labs/docs/editorial-review.md)。
- 原始运行输出放在 `labs/results/`，个人报告放在 `labs/reports/`；精选验证证据保存在 `labs/docs/evidence/`。
