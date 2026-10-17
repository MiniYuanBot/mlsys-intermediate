---
title: 从教学实验到可信的研究证据
tags: [measurement, reproducibility]
---

# 从教学实验到可信的研究证据

[课程索引](../index.md) · [实验报告模板](../../labs/templates/report.md)

## 先声明证据层级

| 层级            | 本仓库例子                               | 能说明什么                               | 不能说明什么                             |
| --------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| 数学/离散模型   | D3 bucket 成本、D4 推测控制、D8 调度     | 在明确假设下策略是否正确、是否有潜在收益 | GPU 实际加速、在线 tail latency、RL 收敛 |
| 合成 microbench | D1 attention、D2 GEMV、D3 fusion、D7 H2D | 特定 shape/数据/设备上的成本与数值行为   | 完整 LLM 吞吐、任务质量、多卡可扩展性    |
| 微型真实训练    | D6 LoRA block 网络                       | 当前网络的梯度与显存—重计算权衡          | 大模型微调质量、FlexLLM 共服务效果       |
| 真实模型/trace  | `capture_model.py`、自己接入引擎         | 被测模型和负载范围的端到端行为           | 超出硬件、模型、数据范围的结论           |
| 完整论文复现    | 另行跑原作者 artifact                    | 论文特定实验能否复现                     | 自动证明新颖性或你的优化普遍更好         |

## 每次实验的最低规范

1. **锁变量**：模型 revision、tokenizer/chat template、精度、batch、输入/输出分布、随机 seed、最大 token、计时区域。参数搜索用校准集；最终表用独立测试集。
2. **先正确性**：exact 方法对齐输出/梯度；approx 方法对比原始精度并报告质量变化。位宽正确不等于模型质量正确。
3. **计时区域写清**：CUDA Event 测设备队列区间；端到端 wall clock 包括 CPU 调度和必要传输。异步 GPU 计时前后同步。编译、模型加载、tokenize、预热分别记录，不能选择性遗漏。
4. **公平 baseline**：同输入/精度/缓存预算/SLO；包括优化过的库和生产框架。仅与朴素实现对照可以验证机制，但不足以说明新方法的价值。
5. **重复与随机化**：同一 kernel 至少 5 个独立测量批次，变换方法顺序；工作负载至少 3 seeds。默认教学 sweep 不等于充分统计功效；训练收敛结论需更多独立 seed 与置信区间。
6. **记录硬件状态**：显卡型号、驱动、Toolkit、功率/温度、其他进程。笔记本功率和 WSL 调度会产生波动；小于噪声的改善不能作为贡献。
7. **保留负结果**：OOM、超时、不满足 SLO、fallback 都计入；不要只保留跑通的配置。报告记录入口见 [run.py](../../labs/tools/run.py)。

## 指标定义

- **TTFT**：请求到达服务端至第一个输出 token；含排队。只测 prefill CUDA kernel 的时间不可称线上 TTFT。
- **ITL**：相邻输出 token 的发出间隔。一次推测提交多个 token 时，需要报告突发提交的情况和用户可见时间，不能将整轮耗时均匀分摊后当作 tail ITL。
- **TPOT**：常见请求平均 `(完成时间−首 token 时间)/(输出数−1)`；输出数 <2 单列，注明定义。
- **Goodput**：观测窗口内**满足预先定义 SLO 的请求数**除以窗口长度；请求级与 token 级不要混用。窗口内未完成的请求属于 censored，必须说明处理方式。
- **显存**：至少区分已分配 tensor、allocator reserved 和进程占用。KV payload 的数学估计不是进程峰值；CPU offload 还需报告 host RAM、pinned bytes、PCIe 流量。
- **质量**：重构误差 → 层输出误差 → 固定文本 NLL/任务正确率 → 下游学习曲线，是不同证据。随机 attention 误差小不能替代长上下文任务质量。
- **RL 效率**：既报告生成/训练吞吐，也报告相同数据预算和相同 wall time 的 reward/accuracy；减少难题训练量后，不能将结果称为等价加速。

## 一个完整研究实验矩阵

至少包含：2–3 个模型或明确解释的替代架构、短/长/混合 workload、3 个容量或负载点、强 baseline、组件消融、控制开销、错误预测/极端输入、OOM/fallback、一个 held-out workload。先通过最小实验矩阵判断是否能得到有用信息，再申请扩大规模，无需一开始就运行所有配置的笛卡尔积。

报告提升幅度时，给出分母、测量次数和区间。选择统计方法时需考虑相关性：同一请求中的 token 并不独立；可对独立请求或 trace seed 做 bootstrap，训练结果则按 seed 汇总。100 次 GPU loop 不能当作 100 次独立系统实验。

## 从单卡升级

单卡实验可以验证某个 kernel 或策略机制是否有效；多 GPU 网络、NVLink 和跨节点一致性的结论仍需实测。HPC 申请前先写资源假设，不预设某个分区、管理员权限或计费。迁移后重测 baseline，避免将不同卡上的绝对时间直接相比。

## 资料汇总

[PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) · [NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) · [Perfetto](https://perfetto.dev/) · [预注册模板](../templates/proposal.md)
