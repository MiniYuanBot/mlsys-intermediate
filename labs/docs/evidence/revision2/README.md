# 重写与目录调整后的验证

[验证说明](../../validation.md) · [18 次调用汇总](summary.json) · [独立内容评审](../../editorial-review.md)

此目录保存 2026-10-17（Asia/Shanghai）课程修订时**重新执行**的核心参考测试和默认 benchmark 记录。运行时使用新的 `labs/` 工作目录，源码已按 foundations 的规范格式化。原有第一版证据在上级目录保持不变。

## 执行范围

| Lab | 正确性                                     | 默认 benchmark                   |
| --- | ------------------------------------------ | -------------------------------- |
| 1   | C++ CPU                                    | 18 行机制实验                    |
| 2   | CUDA                                       | 90 行三基线 sweep                |
| 3   | CPU bucket、CUDA fusion                    | 假设成本模型与 CUDA 120 行 sweep |
| 4   | Rust 7 tests                               | 4 策略×5 seeds×3 phases          |
| 5   | Rust 6 tests                               | 2 策略×4 容量×3 phases           |
| 6   | CPU/CUDA 各 5 tests，含新增输入梯度 oracle | 3 seeds×3 policies，真实微型训练 |
| 7   | C++ CPU                                    | CPU 缓存 replay 与 CUDA H2D 校准 |
| 8   | Rust 6 tests                               | 合成长度与预测质量调度           |

各 lab 的 test/bench 合计 16 次，加上 lab3、lab6 的额外 CPU test，共 18 次。每个 JSON 包含源文件哈希、实际 argv、stdout/stderr、退出状态和环境信息。

[统一格式检查](format-check.json) 和 [最终目录/链接/源码核对](quality-check.json) 独立于参考测试记录。后者核对最终源文件是否与实际运行记录中的哈希一致，避免只测试旧版本。

本次没有重新运行四种真实 `torch.compile` 模式、模型采集或 Compute Sanitizer，相关原始记录仍属于第一版。程序中仅涉及目录或格式调整的部分已通过静态检查或路径验证；旧运行时间仍作为历史数据保留。

## 如何解释结果

`passed` 是核心参考程序与已提供测试/benchmark 的运行状态。各 lab 新要求的学生扩展（新增 oracle、负载矩阵、消融、报告、完整系统挑战）需要单独验收，此目录的记录不能证明这些扩展已完成。参考模式替换 TODO 文件，实验新增代码应按各 lab 指定的位置保留。

报告中不能将历史与当前微基准之间的噪声差异解释为优化收益。本次主要验证目录迁移、格式统一和新增正确性检查是否影响程序的正常运行。
