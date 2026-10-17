# 阅读资料

[课程索引](../index.md) · [36h/周路线](../guides/roadmap.md)

从当前专题对应的资料开始，先完成讲义指定的阅读任务，再按需要查阅论文全文和源码。

| 文件                                       | 用途                                                 |
| ------------------------------------------ | ---------------------------------------------------- |
| [reading.md](reading.md)                   | 论文筛选、精读、机制复现和查新方法                   |
| [paper-card.md](paper-card.md)             | 记录研究问题、证据、源码入口和下一步动作的论文卡模板 |
| [resources.md](resources.md)               | 论文、项目、课程和官方文档总表                       |
| [courses-and-labs.md](courses-and-labs.md) | 外部课程与开放 lab 的选读任务                        |
| [code-map.md](code-map.md)                 | 固定版本的源码入口与阅读重点                         |
| [visualizations.md](visualizations.md)     | 图解、交互资源与性能分析工具                         |
| `results/upstream-snapshots.json`          | 可选的本地上游 commit 与文件路径快照，由 Git 忽略    |

## 检查外部链接

[audit_external_links.py](audit_external_links.py) 检查仓库 Markdown 文档中的 HTTP/HTTPS 链接，只需要 Python 标准库。从仓库根目录执行：

```bash
python3 course/readings/audit_external_links.py
```

Windows 可用 `py -3` 或已激活环境中的 `python` 代替 `python3`，命令可直接在 PowerShell 执行。

结果保存在本目录的 `results/external-links.json`，该目录由 Git 忽略。每个链接记录来源文档、HTTP 状态及重定向后的地址；检查不会修改讲义和资料表。也可以指定输出路径：

```bash
python3 course/readings/audit_external_links.py --output course/readings/results/custom-links.json
```

在 `labs/` 目录中运行时，脚本路径写为 `../course/readings/audit_external_links.py`；默认输出位置不随当前目录改变。

HTTP 状态只反映检查时的可达性。403/429 可能来自访问限制或限流，需人工核对；页面可访问也不表示论文机制已复现。更新阅读资料时，另行核实论文版本、代码 commit 和实验条件。
