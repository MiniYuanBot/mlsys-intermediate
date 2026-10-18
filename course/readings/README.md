# 阅读资料

[课程索引](../index.md) · [路线图](../guides/roadmap.md) · [选题地图](../guides/directions.md)

## 使用方法

1. 先读 [reading.md](reading.md) 的“第一轮必读”（九个方向各 3 篇）和 S/D/R 阅读法。
2. 读每篇论文时填一张[论文卡](paper-card.md)。
3. 深入某个方向时，从下表找种子论文、源码入口、外部课程和图解。

| 文件                                       | 内容                                |
| ------------------------------------------ | ----------------------------------- |
| [reading.md](reading.md)                   | 阅读方法、第一轮必读、查新流程      |
| [paper-card.md](paper-card.md)             | 论文卡模板                          |
| [resources.md](resources.md)               | 各方向种子论文与待核实清单          |
| [code-map.md](code-map.md)                 | 上游仓库的源码入口与固定 commit     |
| [courses-and-labs.md](courses-and-labs.md) | 外部课程、开放 lab 与 LCPU 主题映射 |
| [visualizations.md](visualizations.md)     | 各方向的图解与工具，附观察任务      |

## 检查外部链接

从仓库根目录运行，结果写入本目录下被 Git 忽略的 `results/external-links.json`：

```bash
python course/readings/audit_external_links.py [--output PATH]
```

HTTP 状态只反映检查时能否访问；403/429 需人工核对。
