# 公共工具

[实验索引](../index.md) · [环境](../setup.md) · [格式规范](../docs/formatting.md)

所有示例从 `labs/` 执行。工具不自动下载模型，不把参考答案复制到学生目录；实验记录的默认位置是 `labs/results/`。

| 工具                                   | 用途                                                   | 最小调用                                                                             |
| -------------------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| [setup_env.sh](setup_env.sh)           | 安装基础依赖和单个 lab 的依赖（锁定版本）              | `bash tools/setup_env.sh --lab 1`（Windows：`setup_env.ps1 -Lab 1`）                 |
| [check_env.sh](check_env.sh)           | 分 lab 的环境判定，失败项附 `fix:`                     | `bash tools/check_env.sh --lab 1`（Windows：`check_env.ps1 -Lab 1`）                 |
| [lab_meta.py](lab_meta.py)             | lab、阶段、依赖和平台可行性的注册表                    | `python tools/lab_meta.py --markdown 2`                                              |
| [verify_all.sh](verify_all.sh)         | 维护者在 WSL2/Linux 上的全量验证                       | `bash tools/verify_all.sh --quick`                                                   |
| [fault_inject.py](fault_inject.py)     | 把已知错误逐个注入参考实现，检查测试能否抓到           | `python tools/fault_inject.py --lab 1 --device cpu`                                  |
| [math_check.py](math_check.py)         | 检查 Markdown 中的 LaTeX、Unicode 数学字符和图中的中文 | `python tools/math_check.py --paths ../course/modules/01-kv-cache.md`                |
| [workload_audit.py](workload_audit.py) | TODO 配对与各阶段有效行统计                            | `python tools/workload_audit.py --output results/workload.json`                      |
| [test_platform.py](test_platform.py)   | 跨平台工具的回归测试                                   | `python tools/test_platform.py`                                                      |
| [doctor.py](doctor.py)                 | 解释器、编译器类别、GPU 与 Inductor 探针               | `python tools/doctor.py --compile-probe`                                             |
| [run.py](run.py)                       | 临时副本中编译/运行，记录源码和日志                    | `python tools/run.py --lab 1 --impl student --mode test`                             |
| [materialize.py](materialize.py)       | 导出单个 lab 调试副本；目标须不存在                    | `python tools/materialize.py --lab 2 --impl reference --to results/my-lab2-debug`    |
| [record.py](record.py)                 | 记录学生新增实验；子进程 cwd 固定为 labs               | `python tools/record.py --label my-exp -- python lab3/torch_bench.py --mode eager`   |
| [extract.py](extract.py)               | 从 stdout 中解析数据表，先列编号再导出                 | `python tools/extract.py results/run.json`                                           |
| [plot.py](plot.py)                     | 筛选相同配置后画均值与样本标准差误差条                 | 见 [CSV 导出与绘图示例](../results/README.md)                                        |
| [capture_model.py](capture_model.py)   | 已有本地模型的串行生成/KV 采集                         | 见 [模型采集说明](../setup.md)                                                       |
| [analyze_kv.py](analyze_kv.py)         | K/V 热图与两轴量化误差                                 | `python tools/analyze_kv.py results/probe/prefill-kv-0.pt --out results/kv-analysis` |
| [format_check.py](format_check.py)     | 按 foundations 规则统一格式/检查                       | `python tools/format_check.py`                                                       |
| [check_links.py](check_links.py)       | 检查项目内 Markdown 链接                               | `python tools/check_links.py`                                                        |

## 记录不是结论

`run.py` 的 `passed` 仅表示指定命令的退出码为 0，不代表人工报告、学生新增实验或真实系统挑战已通过。缺依赖时状态记为 `env-missing`，退出码为 3。用 `--part A/B/C/R/all` 选择阶段，默认运行 all；后续阶段可能依赖前面的实现。九个 lab 都已实现 R 阶段；如果某个 lab 的某个阶段没有可运行的部分，runner 打印 `NOT RUN R` 并说明原因。阶段在当前平台上不可做时（例如原生 Windows 上的 CUDA 部分），runner 以退出码 3 结束并给出 WSL 命令。lab3 在 CPU 上实际运行 Inductor C1–C4，并提示 C5–C7 的 CUDA 融合没有运行。lab2 的 CPU 路径只支持 A/C 主机算法，B 和 R 需要 CUDA；lab7 的 CPU 路径不验证 CUDA 卸载流水线。CUDA 运行会自动获取 `/tmp/mlsys-gpu.lock`，记录中的 `gpu_lock` 字段写明是 `acquired` 还是 `inherited`。

每次执行都会生成唯一时间戳。使用 `--output` 指定路径时，注意不要覆盖旧记录。日志保留失败命令、stdout/stderr、源码哈希、环境和状态。比较不同进程的结果前，先核对 GPU、形状、数据、计时范围和编译缓存状态。

CSV 解析示例：

```bash
python tools/extract.py results/run.json
# 按上一步显示的编号选择表。
python tools/extract.py results/run.json --table 0 --out results/table.csv
```

这里的 `run.json` 是占位文件名，替换为 runner 实际打印的 RECORD 路径。数据表字段依 lab 而变，lab2 CUDA 表的耗时列是 `device_us`，不能把同一绘图命令直接用于其他表。

## 维护者

全量验证、错误注入、公式检查和工作量审计的用法见[编写规范 §7](../docs/authoring.md#7-维护检查)；格式工具用 `python tools/setup_format.py` 安装，见[格式规范](../docs/formatting.md)；阅读资料的链接检查见 [course/readings/](../../course/readings/README.md)。
