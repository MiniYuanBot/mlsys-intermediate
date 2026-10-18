# MLSys Intermediate

接续 [mlsys-foundations](https://github.com/MiniYuanBot/mlsys-foundations) 的 MLSys 自学课程，共九个研究方向，每个方向一讲讲义加一个分 A/B/C/R 四阶段的实验。方向列表与工时见[课程索引](course/index.md)。

| 目录                              | 内容                         |
| --------------------------------- | ---------------------------- |
| [course/](course/README.md)       | 讲义、选题、路线图、阅读资料 |
| [labs/](labs/README.md)           | 实验、测试与公共工具         |
| [solutions/](solutions/README.md) | 参考实现与错误注入表         |

## 使用方法

1. 读[选题地图](course/guides/directions.md)，按[路线图](course/guides/roadmap.md)扫描两周后选一个主方向。
2. 配置环境：WSL2/Linux/macOS 在 `labs/` 下运行

   ```bash
   bash tools/setup_env.sh --lab 1 && bash tools/check_env.sh --lab 1
   ```

   Windows 原生用 PowerShell 版脚本，只能做 CPU 部分，详见[环境配置](labs/setup.md)。

3. 学习讲义（从[课程索引](course/index.md)进入），然后做对应实验：

   ```bash
   cd labs
   python tools/run.py --lab 1 --impl student --mode test --part A
   ```

   TODO 未实现时测试失败是预期行为。各 lab 的步骤与验收标准在 `labs/labN/README.md` 和 `grade.md`。

4. 做完 R 阶段后，按[从 lab 到真实系统](course/guides/real-systems.md)继续。

Obsidian 用户把仓库根目录作为 vault 打开，入口是 [course/index.md](course/index.md)。

## 参考

[实验索引](labs/index.md) · [公共工具](labs/tools/README.md) · [验证记录](labs/docs/validation.md) · [格式规范](labs/docs/formatting.md) · [许可证](LICENSE)
