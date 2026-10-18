# Lab 2 评分与试跑依据

[任务](README.md) · [参考导读](../../solutions/lab2/README.md) · [实际运行记录](../docs/validation.md)

满分 100。A/B/C/R 都是必修，可选挑战不补偿缺失的阶段。数值容差来自 oracle 的计算精度（FP32 累加对 FP64，容差按 $10^{-5}(1+\sum\lvert w\rvert\lvert x\rvert)$ 计）；没有按参考机的速度设性能门槛，也不要求超过 cuBLAS。

| 项                             | 分值 | 自动或人工证据                                                                                                                                                                                                                                                             |
| ------------------------------ | ---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1/A2 格式                     |   10 | `A1 group absmax, not row absmax ...`、`A1 tail group stops at cols ...`、`A1 per-group scales = FP64 oracle ...`、`A2 known bytes ...`、`A2 odd width ...`、`A2 every code = oracle rounding ...`：手算组 scale、ties-away、行步长、尾半字节                              |
| A3/A4 解码与线性层             |    6 | `A3 decodes a hand-built artifact ...`、`A3 rejects ...`、`A4 hand artifact ...`、`A4 = FP64 oracle decoder ...`：手写 artifact，测试自己的解码器                                                                                                                          |
| A5/A6 证书与 clipping          |    6 | `A5 ratio, MSE and bytes = in-test recomputation ...`、`A6 per-group ratio = exhaustive weight-SSE oracle ...`：设计的 $\pm0.5$ 组必须选到 0.9，码要重新量化                                                                                                               |
| B1/B2/B3 解包、warp、byte-pair |    9 | B1 全字节探针、B2 36 种形状、B3 $g=3$ 跨组字节：都对同一份字节的 FP64 解码                                                                                                                                                                                                 |
| B4/B5/B6 复用与 split-K        |    8 | B4 batch 1/3/4/7/8；B5/B6 chunk 17、$N<17$、partial 缓冲区逐段读回                                                                                                                                                                                                         |
| B 基准与分析                   |    6 | `--mode bench --part B` 的 RECORD：四个形状、四条 int4 路径与三条 cuBLAS 基线，`stable` 列与 `frac_peak`；至少一条 SASS 或 `ncu` 证据解释 int4 kernel 的带宽差距                                                                                                           |
| C1/C2 通道 scale 与变换        |    6 | `C1 hand 2x3 X ...`、`C1 clamp to 1e-4 BEFORE the power ...`、`C1 = FP64 oracle ...`、`C2 hand case ...`、`C2 X' W'^T = X W^T ...`                                                                                                                                         |
| C3/C4/C5 搜索、冻结执行、预算  |   10 | `C3 ... in-test exhaustive search`、`C3 outlier channels: the search must pick alpha > 0`、`C4 hand artifact ...`、`C5 channel-scale bytes count ...`；`--part C --device cuda` 的 GPU 集成组通过                                                                          |
| C 实验                         |    4 | 校准表：同分布与分布偏移 held-out、RTN 与 clipping-only 对照，解释为什么偏移后变差                                                                                                                                                                                         |
| R 真实系统                     |   20 | R1–R5 五组全部 PASS（9 分，其中 R1 的 oracle 与 GPU `cvt.rn.satfinite` 一致这一项必须为 PASS）；`--mode bench --part R` 的 RECORD JSON，含 copy 峰值、batch 1–64 扫描、cuBLASLt FP8 行（5 分）；交叉 batch 的实测值与讲义 §1 的 modeled 值对比，并说明差距来自哪里（6 分） |
| 实验与报告                     |   15 | 四张必交图、两个新增反例、一个负结果（例如 int4 不比 FP16 cuBLAS 快的形状）、同字节口径的对比、modeled 与 measured 分列                                                                                                                                                    |
| 环境证据                       |    — | 附 `bash tools/check_env.sh --lab 2` 的完整输出；缺失则 R 阶段不给分                                                                                                                                                                                                       |

每项可以按子性质部分给分。编译失败不让手算部分归零，但相关实现项算未完成。B 与 R3/R4 的测试输入由 `oracle.hpp` 打包，不依赖你的 A2/R2，所以 A 没做完也能拿 B 的分；反过来 B 的基准用你自己的 A2 打包，A2 错了基准会报 `WRONG_OUTPUT`。

**必交命令**：`--part A`、`--part C` 的 `--mode test` 通过；在 WSL 中 `--part B`、`--part C`、`--part R` 加 `--device cuda` 的 `--mode test` 通过；`--mode bench --part C`、`--part B --device cuda`、`--part R --device cuda` 完成且没有 `WRONG_OUTPUT`；`bash tools/check_env.sh --lab 2` 的输出。参考 run 不能充当学生 run，二者 RECORD 的 `implementation` 字段不同。

**必查错误**：把 L2 里的命中当成 DRAM 带宽（关掉冲刷、形状小于 L2）；把逻辑字节当成 profiler 实测字节；拿 W8A8 的速度和 W4A16/W8A32 比却不提激活也被量化；只报最快的一次而不报中位数和 IQR；`stable` 为 `NO` 的行直接进结论；把 modeled 的交叉 batch 当成 measured；校准集上的提升说成 held-out 的提升；把本 lab 的 W4A32 SIMT 结果说成 Marlin 这类 W4A16 Tensor Core kernel 的结果。出现这些错误扣相应的解释分，先改结论再做研究扩展。
