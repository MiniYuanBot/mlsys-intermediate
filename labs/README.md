# Labs

[实验索引](index.md) · [仓库首页](../README.md)

## 使用方法

所有命令在 `labs/` 目录运行。

```bash
bash tools/setup_env.sh --lab N && bash tools/check_env.sh --lab N    # 安装并检查 lab N 的依赖
python tools/run.py --lab N --impl student --mode test --part A       # 测试你的实现（A/B/C/R/all）
python tools/run.py --lab N --impl student --mode bench --part all    # 运行 benchmark
python tools/run.py --lab N --impl student --mode test --device cuda  # CUDA 部分（WSL2/Linux）
```

Windows 原生改用 `tools\setup_env.ps1 -Lab N` 与 `tools\check_env.ps1 -Lab N`，只做 CPU 部分。运行记录保存在 `results/`，报告写在 `reports/`。每个 lab 的任务、顺序与验收标准见 `labN/README.md` 和 `labN/grade.md`。

## 参考

[环境配置](setup.md) · [公共工具](tools/README.md) · [工作量](docs/workload.md) · [验证记录](docs/validation.md) · [编写规范](docs/authoring.md) · [报告模板](templates/report.md) · [格式规范](docs/formatting.md) · [参考实现](../solutions/index.md)
