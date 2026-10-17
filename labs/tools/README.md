# 公共工具

[实验索引](../index.md) · [环境](../setup.md) · [格式规范](../docs/formatting.md)

所有示例从 `labs/` 执行。工具不自动下载模型，不把参考答案复制到学生目录；实验记录的默认位置是 `labs/results/`。

| 工具                                 | 用途                                     | 最小调用                                                                             |
| ------------------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------ |
| [doctor.py](doctor.py)               | 查看解释器、编译器和 GPU                 | `python tools/doctor.py`                                                             |
| [run.py](run.py)                     | 临时副本中编译/运行，记录源码和日志      | `python tools/run.py --lab 1 --impl student --mode test`                             |
| [materialize.py](materialize.py)     | 导出单个 lab 调试副本；目标须不存在      | `python tools/materialize.py --lab 2 --impl reference --to /tmp/my-lab2`             |
| [record.py](record.py)               | 记录学生新增实验；子进程 cwd 固定为 labs | `python tools/record.py --label my-exp -- python lab3/torch_bench.py --mode eager`   |
| [extract.py](extract.py)             | 从 stdout 中解析数据表，先列编号再导出   | `python tools/extract.py results/run.json`                                           |
| [plot.py](plot.py)                   | 筛选相同配置后画均值±样本标准差          | 见 [CSV 导出与绘图示例](../results/README.md)                                        |
| [capture_model.py](capture_model.py) | 已有本地模型的串行生成/KV 采集           | 见 [模型采集说明](../setup.md)                                                       |
| [analyze_kv.py](analyze_kv.py)       | K/V 热图与两轴量化误差                   | `python tools/analyze_kv.py results/probe/prefill-kv-0.pt --out results/kv-analysis` |
| [format_check.py](format_check.py)   | 按 foundations 规则统一格式/检查         | `python tools/format_check.py`                                                       |
| [check_links.py](check_links.py)     | 检查项目内 Markdown 链接                 | `python tools/check_links.py`                                                        |

## 记录不是结论

`run.py` 的 `passed` 仅表示指定命令的退出码为 0，不代表人工报告、学生新增实验或真实系统挑战已通过。用 `--part A/B/C/all` 选择阶段，默认运行 all；后续阶段可能依赖前面的实现。lab3 在 CPU 上实际运行 Inductor C1–C4，并明确提示 C5 CUDA 融合尚未运行。lab2 的 CPU 路径只支持 A/C 主机算法，B 需要 CUDA；lab7 的 CPU 路径不验证 CUDA 卸载流水线。

每次执行都会生成唯一时间戳。使用 `--output` 指定路径时，注意不要覆盖旧记录。日志保留失败命令、stdout/stderr、源码哈希、环境和状态。比较不同进程的结果前，先核对 GPU、形状、数据、计时范围和编译缓存状态。

CSV 解析示例：

```bash
python tools/extract.py results/run.json
# 按上一步显示的编号选择表。
python tools/extract.py results/run.json --table 0 --out results/table.csv
```

这里的 `run.json` 是占位文件名，替换为 runner 实际打印的 RECORD 路径。数据表字段依 lab 而变，lab2 CUDA 表的耗时列是 `device_us`，不能把同一绘图命令直接用于其他表。

## 维护课程时的检查

`python tools/workload_audit.py --output results/workload.json` 检查每个 lab 的学生代码与答案：BEGIN/END TODO 标记是否配对，编号是否唯一，三个阶段是否都有任务，参考实现中是否还有未实现的占位代码。统计只计参考 TODO 区域的有效物理行，排除注释、空行和纯括号行，不要求学生写出相同的行数。统计口径与时间安排见[工作量说明](../docs/workload.md)。

阅读资料的链接维护工具见 [course/readings/](../../course/readings/README.md)。
