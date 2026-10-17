# 第三次修订实际验证记录

日期：2026-10-17。本轮重构了八个实验的基础A、核心B、前沿C和全部参考答案。有效记录见[revision3索引](evidence/revision3/README.md)，机器可读清单见[summary.json](evidence/revision3/summary.json)。每份记录包含命令、环境、stdout/stderr、返回码和源码SHA256。

## 实际执行范围

| Lab | 实际正确性验证                              | 实际benchmark                   | 能说明什么                                        |
| --- | ------------------------------------------- | ------------------------------- | ------------------------------------------------- |
| 1   | C++两轴/极值量化、111增量边界、PQ独立oracle | 18行（三seed/两长度/三预算）    | 实际CPU缓存与检索数值，非GPU论文系统              |
| 2   | CPU A/C；CUDA所有kernel及校准接线           | 主机校准3行、CUDA40行           | 实际C++/CUDA W4A32，不是完整AWQ/Marlin系统        |
| 3   | CPU A/B/C1–C4真实Inductor；CUDA全阶段含融合 | 19条真实JSON记录、120行融合数据 | 真编译/执行；缓存制度有说明，不当作冷编译         |
| 4   | Rust 14测试，含无探索退化反例               | A20/B120/C60行                  | 到达队列与SLO离散事件模型，时间非实测服务延迟     |
| 5   | Rust 15测试，含不同祖先后块反例             | A12/B8/C36行                    | 实际缓存状态/lease/并发trace，服务ms为模型        |
| 6   | CPU和CUDA各14测试，含dtype事务和梯度        | CPU/CUDA各24行                  | 实际微型网络optimizer step；CUDA实际offload与峰值 |
| 7   | CPU数值/事件；CUDA12组形状的单双槽专家      | A9/B6/CPU C6/CUDA C12行         | 实际INT8专家执行+传输；B模拟与C测量分列           |
| 8   | Rust 12契约测试、确定性时序反例             | A3/B12/C108行                   | 实际小型策略梯度学习；token/worker耗时是事件模型  |

八个lab的参考test和bench均已通过。补充执行Lab2 CPU A/C、Lab3 CPU、Lab6 CPU；八个初始学生骨架的A阶段均按预期在TODO处失败，未静默导入答案。参考运行仅证明题目可执行，不是学生个人完成证明。

评分依据见各lab的`grade.md`：逐算法性质/状态检查、数值oracle与报告方法分开，没有要求固定加速倍数。个人报告和开放研究挑战未被自动测试代替。

## 环境与测量边界

WSL2/Linux，RTX4060 Laptop 8GB；CUDA Toolkit12.6，PyTorch2.14.1+cu126，Python3.12.3；C++17与Rust标准库。不同记录保留实际版本。GPU基准串行运行，避免本轮多个GPU实验互相占用；不能保证系统没有其他桌面负载。

未实机验证Mac、原生Windows/MSVC、多GPU或学校HPC。Lab2完整B/C、Lab3 C5、Lab6实际搬运/峰值、Lab7 C3/C4/C6需要NVIDIA GPU；其余CPU数值或事件系统不伪装为GPU性能。

## 输出和工作量检查

- [workload](workload.md)：123个成对TODO，2038行有效参考实现；[原始统计](evidence/revision3/workload.json)。
- [格式检查](evidence/revision3/format-check.json)：使用foundations配置，C/C++/CUDA、Python、Rust、Shell、Markdown/JSON/HTML统一检查。
- [交付检查](evidence/revision3/quality-check.json)：目录结构、局部链接、Python语法、TODO配对、所有正式bench表导出。
- [交叉审核](editorial-review.md)：算法/文稿缺陷、具体修复和复测范围。

Lab3第一次benchmark退出0，但stdout交错破坏了JSONL；该记录仅作诊断，正式结果使用`lab3-reference-bench-cuda-clean.json`。Lab7早期表头缺列及单次计时结果由`lab7-reference-bench-cuda-final2.json`替代。清单不会把这些旧输出当作最终可导出证据。

## 复查

以下命令cwd为`labs/`，使用已安装PyTorch的解释器：

```bash
python tools/workload_audit.py
python tools/check_links.py
python tools/format_check.py
python tools/run.py --lab 1 --part A --impl reference --mode test
python tools/run.py --lab 2 --impl reference --mode test --device cuda
python tools/run.py --lab 6 --impl reference --mode bench --device cuda
```

[第一版原始证据](evidence/README.md)和[第二版证据](evidence/revision2/README.md)保留历史数值；本版源码和评分以当前任务书及revision3记录为准。

## D7/D8 讲义与实验对照复核（2026-10-17）

本次对照学生接口、参考实现、契约测试和 benchmark，修正了两篇讲义保留的旧版实验说明。D7 更新缓存的四种返回值、加载完成时刻、pin 引用计数、预测预算及实际 CUDA 执行范围；D8 更新优势标准化公式、组身份检查、裁剪梯度及版本阈值对学习的影响。实验实现未改动。

| 复核项          | 本次执行结果                                                                  | 原始记录                                            |
| --------------- | ----------------------------------------------------------------------------- | --------------------------------------------------- |
| Lab 7 正确性    | 14 项 CPU 检查、12 组 CUDA 配置通过；CUDA 最大绝对误差为 `2.23517e-08`        | [test](evidence/lecture-sync-lab7-test-cuda.json)   |
| Lab 7 benchmark | A9/B6/CPU C6/CUDA C12 行；CUDA 最大绝对误差为 `2.08616e-07`，低于 `2e-4` 容差 | [bench](evidence/lecture-sync-lab7-bench-cuda.json) |
| Lab 8 正确性    | 12 项 Rust 契约测试通过，包括有限差分、裁剪分支和 3 个 seed 的实际学习        | [test](evidence/lecture-sync-lab8-test-cpu.json)    |
| Lab 8 benchmark | A3/B12/C108 行；27 次训练均完成，保留严格版本阈值下的较差结果                 | [bench](evidence/lecture-sync-lab8-bench-cpu.json)  |

Lab 7 在 RTX 4060 Laptop 8GB 上运行：B 的 4096B 容量下，预取将 demand 等待未命中从 120 降为 9，总接纳字节却从 245760B 增至 249856B。讲义据此区分 demand 等待、总加载量与 Belady 换入下界。此次没有采集 profiler 时间线，不能仅凭单双槽墙钟差异声称观察到了重叠。

Lab 8 的 100 波实验中，max_lag=4 时最终 expected reward 为 0.9823–0.9846，更新 400 次；max_lag=0 时为 0.6203–0.6209，更新 100 次并丢弃 2400 条超龄样本。这是当前 bandit 与事件模型的结果，比较固定的是波数，不是更新次数或墙钟预算。

复现本次运行，在 `labs/` 下执行：

```bash
python3 tools/run.py --lab 7 --impl reference --part all --mode test --device cuda
python3 tools/run.py --lab 7 --impl reference --part all --mode bench --device cuda
python3 tools/run.py --lab 8 --impl reference --part all --mode test
python3 tools/run.py --lab 8 --impl reference --part all --mode bench
```
