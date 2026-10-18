# 实验索引

[课程索引](../course/index.md) · [环境](setup.md) · [参考答案](../solutions/index.md) · [工作量](docs/workload.md) · [验证状态](docs/validation.md)

所有命令从 `labs/` 目录执行。九个 lab 与九讲一一对应，每个 lab 都有四个阶段：

- **A 基础**：小问题与基本接口，先把量算对。
- **B 核心**：方向的核心机制。
- **C 前沿**：论文启发的缩小版机制与测量。
- **R 真实系统**：在 RTX 4060 上测真实 kernel、真实模型或真实 trace，再回到前面阶段的模型里对照。

每个阶段都有注释契约、待填代码、完整参考实现和独立 oracle 测试；报告与开放挑战另列。九个 lab 都实现了 R 阶段，只有 lab8 中用 TRL 跑 GRPO 的部分是指南，不计入必修。完成全部九个 lab 不是开始科研的前置条件，推荐只深入一个主方向，见[方向选择](../course/guides/directions.md)。

## 九个 lab

工时含实现、独立反例、调试、测量和报告，按 A/B/C/R/报告拆分的明细见[工作量](docs/workload.md)。平台范围与 [lab_meta.py](tools/lab_meta.py) 一致（`python tools/lab_meta.py --markdown N` 打印第 N 个 lab 的阶段与平台表）。

| Lab                 | 方向             | 四个阶段的重点（A / B / C / R）                                                                                 | 语言                       | 平台范围                                                                  | 工时 |
| ------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------- | ---: |
| [1](lab1/README.md) | D1 KV 缓存       | int4 格式 / 带残余窗口的流式缓存 / PQ 检索 / int4 KV 融合 decode kernel                                         | C++17 + CUDA               | A/B/C 全平台；R 仅 WSL/Linux                                              |  30h |
| [2](lab2/README.md) | D2 低比特 kernel | 可逐字节核对的 int4 格式 / 四条 W4A32 kernel / AWQ 式通道缩放 / sm_89 FP8 权重 GEMV（对照 cuBLAS）              | C++17 + CUDA/PTX           | A/C 全平台；B、R 与 C 的 GPU 部分仅 WSL/Linux                             |  35h |
| [3](lab3/README.md) | D3 动态编译      | trace 与计时 / 有预算的分桶 DP / 真实 `torch.compile` 与融合 RMSNorm kernel / CUDA graph 捕获尺寸               | Python + CUDA C++          | A/B 全平台；C1–C4 CPU/GPU；C5–C7 与 R 的 graph 捕获仅 WSL；R 规划可在 CPU |  38h |
| [4](lab4/README.md) | D4 推测解码      | 接受量与成本 / 删失估计与预算 / 推测服务模拟 / Qwen3-0.6B 到 1.7B 的真实接受 trace                              | Rust + Python/torch（R）   | A/B/C 全平台；R 回放全平台，GPU 采集仅 WSL                                |  28h |
| [5](lab5/README.md) | D5 前缀缓存      | 缓存身份 / 事务与所有权 / 准入、全命中规则与缓存感知路由 / 实测 prefill 代价与 Mooncake 格式 trace 回放         | Rust + Python（R）         | A/B/C 全平台；R 的 CPU 回放全平台，GPU 测量仅 WSL                         |  31h |
| [6](lab6/README.md) | D6 低显存训练    | LoRA 生命周期 / checkpoint 与预算规划 / saved-tensor offload 与训练步 / 比 eager 存得更少的 `autograd.Function` | Python/PyTorch             | 全部正确性 CPU 可做；CUDA 峰值与计时仅 WSL                                |  30h |
| [7](lab7/README.md) | D7 MoE 卸载      | 路由约定 / 字节缓存与替换策略 / INT8 专家执行与槽依赖 / MB 级 pinned 重叠流水线                                 | C++17 + CUDA               | A/B 与 C 的 CPU 部分全平台；C 的 GPU 部分与 R 仅 WSL/Linux                |  30h |
| [8](lab8/README.md) | D8 rollout 调度  | 长度历史与组优势 / 分段账本与先完成者入选 / 队列、裁剪梯度与版本界 / 真实长度 trace 回放                        | Rust（R 可选 Python 采样） | 全平台 CPU；可选采样需 WSL GPU                                            |  30h |
| [9](lab9/README.md) | D9 异构 serving  | roofline 代价模型 / Helix 式 max-flow 放置 / SLO 下的多模型共置 / 在 4060 上实测并重新拟合 $\eta$               | C++17 + CUDA               | A/B/C 全平台；R1–R4 仅 WSL/Linux；R5/R6 CPU                               |  25h |

九个 lab 合计 **277h**，讲义每讲 5h 共 45h，总计 **322h**。按 36h/周约 9 周，但不需要全部做完。

**最重的两个 lab。** lab3（38h）和 lab2（35h）第一遍可以先放下一部分，等主方向定下来再补：

- lab3：先跳过 C7（f16 版融合 kernel，约 1.5h）和 R4 `plan_capture_sizes`（捕获尺寸规划，约 2h），第一遍约 34.5h。C6 的 f32 版已经覆盖 warp shuffle 和向量化访存；R1–R3 已经能跑通 graph 捕获与选择。
- lab2：先跳过 B5/B6 的 split-K（约 2h）和 C5 `search_budget`（约 1h），第一遍约 32h。R 阶段整段保留：R4 `fp8_tiled_batch` 给出 FP8 与 FP16 交叉 batch 的实测点，R5 `weight_bytes` 是基准计算带宽和检查 L2 冲刷要用的。

这只是安排建议，不改变各 lab 的 `grade.md`；跳过的部分在报告里写明。

## 开始一个 lab

先准备环境：`bash tools/setup_env.sh --lab N && bash tools/check_env.sh --lab N`。Windows 原生用 PowerShell 版 `tools\setup_env.ps1 -Lab N` 与 `tools\check_env.ps1 -Lab N`。各 lab 的依赖互不累加；WSL2 Ubuntu 24.04 是主环境，可以做所有阶段，原生 Windows 与 macOS 只做 CPU 部分，详见[环境配置](setup.md)。

默认测试所有阶段；分阶段运行例如 `python tools/run.py --lab 1 --part A --mode test`，R 阶段用 `--part R`，GPU 部分加 `--device cuda`。缺依赖时 runner 打印 `MISSING ...` 和 `fix:`，以退出码 3 结束，记录为 `env-missing`；在原生 Windows 上请求 CUDA 也是退出码 3，并给出可以直接粘贴的 WSL 命令。`--impl reference` 在临时副本里叠加完整答案，不改学生文件。按上表确认每个阶段的 CPU/GPU 边界：命令退出 0 不代表没运行的硬件部分也通过了。

**GPU 计时的两条规则。** 一是一张卡同时只跑一个计时实验：`run.py --device cuda` 和 `fault_inject.py` 会自动拿 `/tmp/mlsys-gpu.lock`，运行记录里的 `gpu_lock` 字段写明是 `acquired` 还是 `inherited`；手动跑 benchmark 时用 `flock /tmp/mlsys-gpu.lock <命令>`。二是计时前先把 GPU 预热到稳定频率：笔记本 4060 一空闲就会降到低 P-state，实测显存频率会从 8000 MHz 掉到 810 MHz，访存受限的 kernel 会偶尔慢 8 倍。各 lab 的 benchmark 在每组计时前让 GPU 连续忙至少 300 ms，并输出 `stable` 列（$p_{75} \le 1.5\, p_{25}$），不稳定的行要重跑后再用。规则全文见 [authoring.md](docs/authoring.md) §5。

## 共用工具

| 工具                                                                              | 用途                                                                                  |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [setup_env.sh](tools/setup_env.sh) / [setup_env.ps1](tools/setup_env.ps1)         | 按 lab 安装锁定版本的依赖                                                             |
| [check_env.sh](tools/check_env.sh) / [check_env.ps1](tools/check_env.ps1)         | 分 lab 的环境判定，失败项附 `fix:`                                                    |
| [lab_meta.py](tools/lab_meta.py)                                                  | lab、阶段、依赖与平台可行性的唯一注册表                                               |
| [run.py](tools/run.py)                                                            | 学生/参考实现的测试与 benchmark；临时副本；保存完整日志与哈希；CUDA 运行自动加 GPU 锁 |
| [fault_inject.py](tools/fault_inject.py)                                          | 把 `solutions/labN/faults.json` 里的已知错误逐个注入参考实现副本，检查测试能否抓到    |
| [verify_all.sh](tools/verify_all.sh)                                              | 维护者在 WSL2/Linux 上的全量验证（参考实现、学生骨架、错误注入、仓库检查）            |
| [math_check.py](tools/math_check.py)                                              | 检查 Markdown 里的 LaTeX 公式、Unicode 数学字符和图中的中文                           |
| [workload_audit.py](tools/workload_audit.py)                                      | 校验学生/答案 TODO 配对，统计各阶段填空区域的有效代码行                               |
| [check_links.py](tools/check_links.py)                                            | 仓库内 Markdown 链接检查                                                              |
| [format_check.py](tools/format_check.py)                                          | 用 foundations 的格式参数与固定版本检查源码和文档                                     |
| [test_platform.py](tools/test_platform.py)                                        | 跨平台工具的回归测试                                                                  |
| [doctor.py](tools/doctor.py)                                                      | Python 侧探针：解释器、编译器类别、GPU；`--compile-probe` 实际编译一次 Inductor       |
| [extract.py](tools/extract.py) / [plot.py](tools/plot.py)                         | 从运行记录导出 CSV；画均值与样本标准差                                                |
| [capture_model.py](tools/capture_model.py) / [analyze_kv.py](tools/analyze_kv.py) | 从**已有本地模型**采集 decode 与 K/V（不自动下载）；离线分析 K/V 分布                 |
| [record.py](tools/record.py)                                                      | 为额外的编译或真实模型实验保存命令与日志                                              |
| [materialize.py](tools/materialize.py)                                            | 导出独立的 lab 副本用于调试或 profiler，不覆盖已有目录                                |

命令与输出说明见 [tools/README.md](tools/README.md)。工具运行成功只说明它覆盖的代码通过；人工报告、学生新增的测试和测量、开放挑战另行验收。

报告写在 `reports/`，原始数据放在 `results/`，不要用图片代替原始数值。验收标准见每个 lab 的 `grade.md`，整体验证状态见 [validation](docs/validation.md)。
