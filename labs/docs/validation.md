# 参考实现验证说明

本页记录 **2026-10-18** 在本机对九个 lab 的验证结果，2026-10-19 用完整的 `verify_all.sh` 复验。数字取自各方向在 WSL 中的运行记录。原始日志在本地 `labs/results/`（由 Git 忽略），复查时请重新运行文末命令，以新生成的日志为准。

**最终状态（2026-10-19）**：完整的 `bash tools/verify_all.sh`（含 benchmark、CPU 与 CUDA 两轮错误注入）70 项检查全部通过；其中 `format_check` 在最后一次 README 精简后重新格式化并单独复查通过。

## 验证环境

| 项目              | 版本或型号                                                                                |
| ----------------- | ----------------------------------------------------------------------------------------- |
| 系统              | Windows 11 上的 WSL2 Ubuntu 24.04（内核 6.6.114.1-microsoft-standard-WSL2）               |
| GPU               | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB，sm_89，24 SM；PCIe 4.0 x8；`asyncEngineCount=1` |
| 驱动              | Windows 驱动 566.24（最高支持 CUDA 12.7）                                                 |
| CUDA Toolkit      | 12.6（nvcc V12.6.85），`-arch=native`                                                     |
| Python 与 PyTorch | Python 3.12.3；torch 2.14.1+cu126；triton 3.8.0；transformers 5.18.0                      |
| Rust              | cargo / rustc 1.99.0                                                                      |
| C++               | g++ 13.3（Ubuntu 13.3.0），C++17                                                          |
| 模型              | Qwen3-0.6B、Qwen3-1.7B（本地 `~/models`，未下载 MoE 模型）                                |

环境由 `bash tools/check_env.sh --lab all` 检查，所有必需项为 `[ OK ]`。

## 各 lab 结果

“参考测试”是 `run.py --impl reference --mode test` 打印的检查数；“错误注入”是 `fault_inject.py` 抓到的错误数与 `solutions/labN/faults.json` 中错误总数之比，括号内是其中只能在 CUDA 上验证的个数。九个 lab 的学生骨架在 A 阶段都能编译，并在第一个 TODO 处失败（退出码 1）。

| Lab | 参考测试                                                                   | benchmark                                               | 错误注入                                                 | 关键实测结果                                                                                                                                                                                                                                                                                                                                                                     |
| --- | -------------------------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | CPU 167 项检查；CUDA R 4/4 组                                              | CPU 18 行；R 完成                                       | 49/49（CUDA 11）                                         | 4K 上下文：融合 kernel 70 µs、FP16 106 µs、先反量化再算 131 µs；32K：382 / 737 / 1549 µs。融合 kernel 达到 copy 峰值的 0.48–0.58，copy 峰值 195–223 GB/s                                                                                                                                                                                                                         |
| 2   | CPU A/C；CUDA 33 项主机检查 + 11 组 GPU（21 项）                           | CUDA 全部 `stable`，copy 峰值 223 GB/s                  | 63/63（CUDA 31）                                         | up_proj、batch 1：INT4 B3 83 µs（峰值的 0.36）、FP8 R3 87 µs（0.67）、FP16 cuBLAS 139 µs、cuBLASLt W8A8 77 µs。FP16 cuBLAS 到 batch 64 都是平的；FP8 SIMT 与 FP16 的交叉点在 batch 2–4（8192 方阵）和 4–8（up_proj），远低于模型预测的 23                                                                                                                                        |
| 3   | CPU A 43、B 124、C1–C4 38、R 463；CUDA C1–C4 39、C5 6、C6 40、C7 32、R 485 | CPU 与 CUDA 通过                                        | 61/61（CUDA 12）                                         | 代价模型下 DP 的 regret 为 1–3%，固定网格为 4–10%；但逐步同步的真实回放中，DP 1.409 ms/step 对 vLLM 默认 1.411（平稳），1.887 对 1.864（漂移），代价模型与实测的差距如实报告。融合 kernel、Inductor 都约 233 GB/s；f32 1024x4096 上两个 kernel 反而快于融合（推测是 L2，留作 ncu 挑战）；每个捕获 stream 多占约 8 MiB cuBLAS workspace                                           |
| 4   | CPU 26 项 Rust 测试 + Python R 检查；R CUDA 通过                           | CPU 与 R CUDA 通过                                      | 43/43（CUDA 1）                                          | 草稿相对成本 $c$：eager 0.96，CUDA graph 0.50。固定 $k=1,2,3,4,6$ 的实测加速 1.06 / 1.04 / 0.96 / 0.88 / 0.73，回放 1.09 / 1.04 / 0.97 / 0.88 / 0.75，i.i.d. 公式 1.17 / 1.20 / 1.18 / 1.12 / 1.00。$S_j$ 从 0.81 降到 0.40，而 $a_j$ 从 0.81 升到 0.90。没有自适应策略胜过最好的固定 $k=2$（censored_ucb 低 2.3%）；回放与实测在 98.4% 的轮次上一致（差异来自 bf16 的近似并列） |
| 5   | CPU 21 项 Rust 测试 + R1 Python；CUDA 通过                                 | A 12、B 8、C 16、R 16 行                                | 64/64（CUDA 2）                                          | 4060 上 Qwen3-0.6B prefill 拟合：底噪 22.1 ms，线性项 66.3 µs/token，二次项 15.3 ns/token$^2$；复制与排队的交叉点约 4.3k token（仓库内参考拟合 4.1k，FLOP 估算 3.8k）                                                                                                                                                                                                            |
| 6   | CPU A 50、B 46、C 48、R 49；CUDA A 50、B 46、C 49、R 53                    | CUDA 训练 24 行、预算 12 行、R 21 行；CPU 通过          | 73/73（CUDA 5；3 个 CPU 专属的别名错误在 CUDA 上不适用） | 8192 token、4 层（1024/3072/16）：7 种写法的保存字节与手算一致；峰值增量 eager 1188 MiB、LoRA-FA 804、融合 1125、checkpoint 577。seed 重生成 dropout 反而是 1413 对 eager 1348（反传时重新生成了 float mask），作为一条教训写进讲义                                                                                                                                              |
| 7   | CPU 92 项；CUDA C 15 项；R 6 组 50 项（含设备端回读的竞态检测）            | CPU A 15、B 48、C 6 行；CUDA C 12 行；R 10 + 24 + 30 行 | 62/62（CUDA 24）                                         | H2D pinned 12.6–12.8 GB/s，pageable 11.1–11.6 GB/s。单槽时重叠比 $\rho$ 在 24 行里全部为 0，两槽及以上 0.75–0.91；24 行中 3 行 `stable=NO`。混合执行：$T_{cpu}$ 7.74 ms，对比 $T_{fetch}+T_{gpu}$ 1.51 ms，标量 CPU 专家不划算（`cpu_speedup` 扫描是模型值）                                                                                                                     |
| 8   | A/B/C/R 共 19 项测试                                                       | 全部通过；R CUDA 采集新 trace（28 KB）                  | 53/53（无 CUDA 专属）                                    | 真实 trace：Qwen3-0.6B，32 道自编题，每题 2 轮、每轮 $G=8$ 条，共 512 条样本，112 KB，在 GPU 锁下采集 8.1 min（采样所得，不是 GRPO 训练中的 trace）。4 worker 回放：keep-first-4 用 99.4 s，无偏 4-of-8 用 122.0 s，长度偏移 -12.9%（190.3 对 218.5 token）；FIFO 132.2 s，LPT-oracle 122.6 s，下界 122.1 s                                                                      |
| 9   | CPU A 79、B 78、C 45、R-CPU 22；CUDA R-GPU 37                              | 全部通过                                                | 58/58（CUDA 8）                                          | copy 227–231 GB/s，GEMV 235–247 GB/s，WMMA GEMM $2048^3$ 22.1 TFLOPS（cuBLAS 25.4）；拟合 $\eta_c = 0.67$、$\eta_m = 0.91$、$t_0 = 4.3$ µs，GEMV 留出集误差 9.5%。$\eta$ 乘 0.75 会让排名第一的方案改变（Kendall 距离 3）；最便宜的共置方案在 Poisson 到达下 SLO 达标率 73%                                                                                                      |

九个 lab 的错误注入合计 526/526 被抓到。D2–D9 各自报告 `math_check.py` 在本方向讲义与 lab 文档上为 0 错误（D1 的记录没有单列这一项）。

## 全量复跑记录

- **2026-10-18，`--quick`**：45 项中 43 项通过。失败的两项是共享文档里的 13 个公式错误（Phase 3 中修正），以及尚未应用格式化工具。
- **2026-10-19，完整版**：第一次运行有 8 项失败。原因有三：格式化改变了代码换行，使 10 个错误注入的匹配串失效（STALE）；lab7 R2 的“墙钟不超过串行和”检查在笔记本时钟波动下偶发失败（单槽时实测超出 16%，容差改为 25% + 1 ms）；`solutions/lab8/README.md` 的表格里有 4 处公式含未转义的竖线。修正后从空的 `labs/results/` 重跑，70 项全部通过。

## 测量注意事项

- **笔记本的频率会掉。** 4060 Laptop 一空闲就降到低 P-state，实测显存频率会从 8000 MHz 掉到 810 MHz，访存受限的 kernel 会偶尔慢 8 倍（lab1 的 split 扫描在 32K 时出现过约 8 倍的双峰，原因尚未完全查清，作为问题留给学生）。各 benchmark 在每组计时前让 GPU 连续忙至少 300 ms，并输出 `stable` 列（$p_{75} \le 1.5\, p_{25}$）；`stable=NO` 的行要重跑后再用。
- **一张卡同时只跑一个计时实验。** `run.py --device cuda` 与 `fault_inject.py` 自动获取 `/tmp/mlsys-gpu.lock`。后台有其他负载时，lab1 的结果曾整体慢 2 倍，即使看起来很稳定也不能用。
- **只有一个 copy engine。** WDDM 下 `asyncEngineCount=1`，H2D 与 D2H 不能同时进行。lab7 的重叠比以此为前提，不能外推到有两个 copy engine 的数据中心卡。
- **没有 Nsight Systems 时间线。** WSL2 下 nsys 只有 CLI 预览，本次没有出时间线图。重叠是用分段 cudaEvent 和设备端回读间接证明的。
- **模型值与实测值要分开。** lab7 B 是事件模型，C 是主机 wall-clock，R 是 cudaEvent 实测；lab9 中 4060 以外的 GPU 全部是“规格表乘 4060 拟合的 $\eta$”，输出标注 modeled, not measured；lab3 的代价模型与真实回放有差距；lab4 的 i.i.d. 公式明显高估了实测加速。报告里必须写明每个数属于哪一类。

## 尚未验证的部分

- **原生 Windows 与 macOS。** 各方向的记录中，只有 lab1 和 lab9 写明在原生 Windows 上手动跑过 CPU 部分（通过；R 阶段按预期以退出码 3 结束，并提示 WSL 命令）；其余 lab 的原生 Windows CPU 部分和全部 macOS 运行只靠[三平台 CI](../../.github/workflows/portability.yml) 覆盖，本次整合没有确认远程 CI 的结果（待核实）。
- **学校 HPC 与多 GPU。** 没有在 HPC 或多卡机器上运行。
- **lab8 的 TRL GRPO 指南。** 只是指南：版本、4–7 GB 的峰值显存和单步时间都没有实测；vLLM 能否在驱动 566.24 上运行也未验证。
- **lab7 的真实 MoE 路由 trace。** 没有下载 MoE 模型；用 OLMoE 采集真实 trace 是可选挑战（约需 13.8 GB 磁盘）。
- **其他。** lab1 中 BitDecoding 的文件路径；lab2 中 INT4 慢于 FP8 的原因尚未用 SASS 证实；lab3 中 vLLM/SGLang 的默认值没有和已安装版本核对（本机未安装二者）；lab5 中 4060 的实际 PCIe 有效带宽。

参考实现可运行不代表学生已完成任务。评分按各 lab 的 `grade.md`；个人反例、报告和开放挑战单独验收。

## 重新运行

以下命令从 `labs/` 执行。先确认环境（见[环境配置](../setup.md)）：

```bash
bash tools/check_env.sh --lab all
```

**WSL2 / Linux + NVIDIA GPU：全量验证。**

```bash
bash tools/verify_all.sh            # 参考 test + bench（CPU 与 CUDA）、学生骨架、错误注入、仓库检查
bash tools/verify_all.sh --quick    # 只跑 test，不跑 benchmark
```

日志写入 `results/verify-<时间戳>/`。从 Windows 一侧调用时，`bash -lc` 不读 `~/.bashrc`，要先激活虚拟环境：

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/__Coding__/github/mlsys-intermediate/labs && source ~/venvs/mlsys/bin/activate && bash tools/verify_all.sh --quick"
```

**拆开的等价命令。**

```bash
python tools/run.py --lab N --impl reference --mode test                  # CPU
python tools/run.py --lab N --impl reference --mode test --device cuda    # CUDA，自动加 GPU 锁
python tools/run.py --lab N --impl reference --mode bench --device cuda
python tools/fault_inject.py --lab N --device cpu
python tools/fault_inject.py --lab N --device cuda
python tools/workload_audit.py --output results/workload.json
python tools/check_links.py
python tools/math_check.py
python tools/format_check.py
python tools/test_platform.py
```

**原生 Windows（PowerShell，只做 CPU 部分）。** 先执行 `mlsys` 激活环境：

```powershell
foreach ($n in 1,3,4,5,6,7,8,9) { python tools\run.py --lab $n --impl reference --mode test }
python tools\run.py --lab 2 --impl reference --part A --mode test
python tools\run.py --lab 2 --impl reference --part C --mode test
python tools\fault_inject.py --lab 1 --device cpu
```

lab2 的 B 阶段与各 lab 的 CUDA 部分在原生 Windows 上以退出码 3 结束，并给出可以直接粘贴的 WSL 命令。lab3 的 C1–C4 需要 MSVC 的 `cl`，找不到时同样退出码 3。没有 GPU 时，按[实验索引](../index.md)运行 CPU 范围，并在报告里标明哪些部分没有执行。数据表导出与绘图见[公共工具](../tools/README.md)。
