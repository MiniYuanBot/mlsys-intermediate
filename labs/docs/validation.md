# 参考实现验证说明

本页保留 2026-10-17 的试运行范围摘要。旧版逐次日志已清理；下表用于说明已检查的行为和边界，不提供当前源码的原始运行记录。复查时请执行文末命令，以新生成的日志为准。

## 已验证范围

| Lab | 正确性检查                                                  | benchmark 输出                  | 测量边界                                                      |
| --- | ----------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------- |
| 1   | C++ 两轴量化、极值、111 个增量边界、PQ 独立 oracle          | 18 行                           | CPU 缓存与检索，未测 GPU 论文系统                             |
| 2   | CPU A/C；CUDA 各 kernel 及校准接线                          | 主机校准 3 行、CUDA 40 行       | 局部 W4A32 线性层，未复现完整 AWQ/Marlin                      |
| 3   | CPU A/B/C1–C4 的真实 Inductor 执行；CUDA 全阶段含融合       | 19 条 JSON 记录、120 行融合数据 | 包含真实编译与执行，需另行说明进程和磁盘缓存状态              |
| 4   | Rust 14 项测试，含无探索退化反例                            | A20/B120/C60 行                 | 到达队列和 SLO 离散事件模型，时间不是实测服务延迟             |
| 5   | Rust 15 项测试，含不同祖先后块反例                          | A12/B8/C36 行                   | 实际缓存状态、lease 和并发 trace；服务时间来自模型            |
| 6   | CPU 与 CUDA 各 14 项测试，含 dtype 事务和梯度               | CPU/CUDA 各 24 行               | 微型网络 optimizer step；CUDA 路径包含实际 offload 和峰值测量 |
| 7   | CPU 14 项检查；CUDA 12 组单双槽专家配置                     | A9/B6/CPU C6/CUDA C12 行        | INT8 专家执行与传输；B 的模拟和 C 的测量分别报告              |
| 8   | Rust 12 项契约测试，含有限差分、裁剪分支和 3 个 seed 的学习 | A3/B12/C108 行                  | 实际小型策略梯度训练；token/worker 耗时来自事件模型           |

上述参考 test 与 benchmark 均曾运行通过。八个初始学生骨架的 A 阶段也曾检查，均在未完成的 TODO 处失败。参考实现可运行不代表学生已完成任务；评分仍按各 lab 的 `grade.md`，个人反例、报告和开放挑战单独验收。

## 环境与限制

试运行环境为 WSL2/Linux、RTX 4060 Laptop 8GB、CUDA Toolkit 12.6、PyTorch 2.14.1+cu126、Python 3.12.3，使用 C++17 与 Rust 标准库。GPU benchmark 串行执行，未排除其他桌面负载。未实机验证 macOS、原生 Windows/MSVC、多 GPU 或学校 HPC。

Lab2 的 B 和 C 的 GPU 集成、Lab3 C5、Lab6 的 GPU 搬运与峰值、Lab7 C3/C4/C6 需要 NVIDIA GPU。CPU 测试不能替代这些部分的验收。Lab7 的单双槽计时未附 profiler 时间线，不能仅据此认定传输与计算发生了重叠；Lab8 的学习曲线只适用于给定的小型环境。

## 重新运行

以下命令从 `labs/` 执行，使用已配置所需工具链和 PyTorch 的解释器。`run.py` 将命令、环境、源码 SHA256、stdout/stderr 和退出状态保存到本地 `results/`，并打印 `RECORD` 路径。

```bash
# CPU 可完成的实验。
for lab in 1 4 5 8; do
    python3 tools/run.py --lab "$lab" --impl reference --mode test
    python3 tools/run.py --lab "$lab" --impl reference --mode bench
done

# 需要 CUDA 验证的完整路径；依次运行，避免互相争用 GPU。
for lab in 2 3 6 7; do
    python3 tools/run.py --lab "$lab" --impl reference --mode test --device cuda
    python3 tools/run.py --lab "$lab" --impl reference --mode bench --device cuda
done

# 补充 CPU 路径。
python3 tools/run.py --lab 2 --impl reference --part A --mode test
python3 tools/run.py --lab 2 --impl reference --part C --mode test
for lab in 3 6 7; do
    python3 tools/run.py --lab "$lab" --impl reference --mode test
done
```

没有 GPU 时，可先按[实验索引](../index.md)运行 CPU 范围，并在报告中标注尚未执行的部分。数据表导出与绘图用法见[公共工具](../tools/README.md)。

维护课程时，可另行运行以下检查：

```bash
python3 tools/workload_audit.py --output results/workload.json
python3 tools/check_links.py
python3 tools/format_check.py
```

这些检查分别覆盖 TODO 配对与规模、内部 Markdown 文件链接和格式，不能替代算法测试与实验分析。

## 跨平台工具回归

编译驱动已统一选择 MSVC/GCC/Clang、处理 Windows 的 `.exe` 后缀，并将公共编译辅助代码复制到独立 lab 副本。Python 文档、CSV 和日志读写显式使用 UTF-8；格式工具提供跨平台 Python 安装入口。

本机 WSL2 的执行结果：

- `python tools/test_platform.py` 的 7 项回归通过，包括中文/空格路径下的真实编译和仓库外独立 lab 运行。MSVC 命令、macOS CUDA 拒绝、Node 包入口和下载文件名使用分支测试；这些分支测试不代表对应系统的实机运行。
- CPU：lab1/4/5/6/7/8 全阶段、lab2 A/C、lab3 A/B 测试通过。
- CUDA：lab2 全阶段、lab3 C、lab7 C 使用 `native` 架构编译并执行通过。
- lab1/7 CPU benchmark、lab2 CUDA benchmark 试跑通过；显式 `sm_89` 参数和 CPU/CUDA Inductor 探测通过。
- 默认 ASCII locale 下的 Markdown 链接检查通过；TODO 配对与有效代码量不变。

[三平台 CI](../../.github/workflows/portability.yml)在 Windows、Linux、macOS 分别检查 CPU 编译、独立副本、Rust 实验和 PyTorch CPU 训练。该工作流需要推送后由 GitHub 执行；本次本地验证没有原生 Windows/macOS runner，也未执行远程 CI。CUDA 仍以本机 WSL2 结果为准。
