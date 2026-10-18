# Lab 1 评分与试跑依据

[任务](README.md) · [参考导读](../../solutions/lab1/README.md) · [实际运行记录](../docs/validation.md)

满分 100。A/B/C/R 都是必修，可选挑战不补偿缺失的阶段。数值容差来自 oracle 的计算精度和讲义 §4.2 推导的量化误差界；没有按参考机的速度设性能门槛。

| 项                  | 分值 | 自动或人工证据                                                                                                                                                                                                                                                              |
| ------------------- | ---: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A1/A2 格式          |   12 | `A1 ... per-group offset=min, scale=(max-min)/15`、`codes within half an oracle step`、`tail [4,4,4]`、`A2 decodes from codes+metadata only`：逐组 FP64 元数据、两轴五种组大小、尾组、半字节顺序                                                                            |
| A3/A4 收集与误差    |    6 | `A3 ...`、`A4 ...`：手算行收集、非法 ID、手算 relative L2 与 cosine、零向量约定                                                                                                                                                                                             |
| B1 选择             |    5 | `B1 boundary tie keeps the smaller id`、`B1 matches brute force on 300 random tie-heavy cases`                                                                                                                                                                              |
| B2/B3/B4 状态与账本 |   12 | `B2 chunk metadata equals FP64 oracle of the raw oldest block`、`B2 chunk K per-channel, V per-token`、`B3 ... in time order`、`B4 bytes = ...`：3 种 residual 乘 37 次追加，非法输入不改状态                                                                               |
| B5 查询             |    7 | `B5 online softmax = FP64 attention on decoded cache`、`B5 output vs raw-history FP64 within derived quantization bound`、溢出用例                                                                                                                                          |
| C1/C2 PQ 训练与编码 |   10 | `C1 codebook equals oracle Lloyd`、`C1 first update strictly lowers SSE`、`C2 held-out codes = nearest center`、等距并列                                                                                                                                                    |
| C3/C4 打分与取回    |   10 | `C3 scores = q . reconstructed key`、`C4 ids = mandatory + ADC top-k`、`C4 recall ... beats random by 3x`、取回字节、满预算还原                                                                                                                                             |
| R 真实系统          |   20 | R1/R2/R3/fused 四组全部 PASS（8 分）；`--mode bench --part R` 的 RECORD JSON，含实测 copy 峰值、三条路径的中位数和 IQR、split 扫描（6 分）；计时口径与字节口径写清楚（见 [authoring §5](../docs/authoring.md#5-测量规则)），融合路径的带宽差距至少有一条 `ncu` 证据（6 分） |
| 实验与报告          |   18 | CPU 三张图、GPU 两张图、至少两个新增反例、负结果、split 异常点的解释或排除、同字节口径的对比                                                                                                                                                                                |
| 环境证据            |    — | 附 `bash tools/check_env.sh --lab 1` 的完整输出；缺失则 R 阶段不给分                                                                                                                                                                                                        |

每项可以按子性质部分给分。编译失败不让手算部分归零，但相关实现项算未完成。测试通过只说明在声明的数据范围内成立；保留原始 JSON 和源码版本才算可复现。

**必交命令**：`--part A`、`--part B`、`--part C` 的 `--mode test` 各自通过；`--mode bench`（CPU）输出 18 行；在 WSL 中 `--part R --device cuda` 的 `--mode test` 通过、`--mode bench` 完成；`bash tools/check_env.sh --lab 1` 的输出。参考 run 不能充当学生 run，二者 RECORD 的 `implementation` 字段不同。

**必查错误**：把索引字节当成系统总内存；把逻辑字节当成分配器 RSS；fit 用了未来 query；只减少索引而没有物理收集；只看 recall 不看 attention 数值；把 CPU 微基准说成 GPU 吞吐；GPU 计时包含了 L2 冲刷或主机拷贝；把“先反量化再算”的显存节省说成带宽节省；把单次计时或单个 split 的结果当结论。出现这些错误扣相应的解释分，先改结论再做研究扩展。
