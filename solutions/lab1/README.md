# Lab 1 参考答案导读

[任务与全部接口](../../labs/lab1/README.md) · [评分](../../labs/lab1/grade.md) · [来源](../../labs/lab1/resources.md)

参考实现覆盖 16 个 TODO：A1–A4、B1–B5、C1–C4、R1–R3。文件是 [kv_student.hpp](src/kv_student.hpp)、[cache_student.hpp](src/cache_student.hpp)、[pq_student.hpp](src/pq_student.hpp)、[decode_student.cuh](src/decode_student.cuh)。测试、基准和 driver 在 `labs/lab1/`，参考模式运行时用这里的同名文件覆盖学生骨架。错误注入清单在 [faults.json](faults.json)。

## A：格式

**做法。** `quantize` 分三遍：第一遍按 `group_index` 统计每组的 min/max，第二遍写 scale 和 offset，第三遍算码并打包。分三遍是为了让每个不变量单独可查：min/max 只覆盖本组真实元素；编码和解码用同一组元数据；两个半字节写进同一个 byte 之前 byte 已清零。`dequantize` 只读码和元数据，用 double 还原后截到 FP32 有限范围。

**为什么测试要这样写。** 旧版测试只检查“每个元素离原值不超过半个 scale”，而这个 scale 是学生自己存的。两类错误因此能混过去：

- scale 用全局 min/max：scale 变大，误差界跟着变大，每个元素仍在“自己的”半步之内；
- 尾组补零：$[4,4,4]$ 补零后 offset 为 0、scale 为 $4/15$，三个 4 仍能恰好解出来。

新测试用 FP64 从原始输入独立算出每组的 $m$ 和 $(M-m)/15$，逐组比对元数据，再用**测试算的** scale 检查码。数据是非对称的（每个通道有自己的偏移，行内、列内都在变），所以换轴、全局统计、补零都会让元数据对不上。A2 的测试直接手写 `Packed`，解码器无从借用原始输入。

**常见错误。** 码 8 到 15 当成负数；奇数尾元素所在 byte 的高半字节没清零；常量组 scale 为 0 导致除零；相反符号的 FP32 极值在 float 里相减溢出。

## B：流式缓存

**做法。** `append` 先校验，再把新行接到 recent，行数达到 `residual+block` 时取最早 `block` 行，K 用 `Axis::Channel`、V 用 `Axis::Token` 量化成一个 chunk，从 recent 删掉这几行。`attend` 对每个 chunk 解码一块、在线更新 $(m,\ell,a)$，最后处理 recent。

**为什么测试要这样写。** 旧版 B 测试把 `attend` 和 `materialize` 互相对比，两者读的是同一份缓存；flush 时把块清零、量化最新块、K/V 轴互换，两条路径一起错，彼此仍然一致。新测试三件事都从**原始追加的 K/V** 出发：

1. chunk 数按 flush 规则闭式算出：$n\ge R+B$ 时为 $\lfloor(n-R)/B\rfloor$，否则为 0；
2. 每个 chunk 的元数据与“原始数据中第 $b$ 块”的 FP64 重算比对，并显式断言 K 是 Channel、V 是 Token；
3. `attend` 的输出与原始历史上的 FP64 attention 比较，误差必须落在讲义 §4.2 推导的界内：

$$
\lvert\Delta o_j\rvert\le\max_t\frac{s^V_t}{2}+\frac{R_j}{2}\big(e^{2\varepsilon}-1\big),\qquad
\varepsilon=\max_t\sum_c\frac{\lvert q_c\rvert s^K_{c}}{2\sqrt D}.
$$

推导的关键一步是 $\sum_t(p'_t-p_t)=0$，所以可以给 $v_{tj}$ 减去任意常数，取列中点后每项不超过 $R_j/2$；再用 $p'_t/p_t\in[e^{-2\varepsilon},e^{2\varepsilon}]$。这个界松，但它把“读的是另一段历史”与“量化误差”分开了。紧的检查（$2\times10^{-5}$）另在缓存的实际值上做。

B1 加了“预算边界上的三方并列”和 300 组暴力枚举。暴力枚举的目标是“mandatory 之外的分数和最大，并列时 ID 序列字典序最小”，这正好等价于“降序选、同分取小编号”，排序规则写反就会失败。

**常见错误。** 漏掉对加权 V 的重缩放；residual 为 0 时假定 recent 非空；flush 条件写成 `>`；非法输入在部分修改状态后才抛出。当前 `append` 对输入错误保证不改状态，不承诺分配失败时的强异常保证。

## C：PQ 检索

**做法。** Lloyd 用确定性初始化（第 $\lfloor cN/C\rfloor$ 行），同距离取小编号，空簇保留旧中心，每轮更新后记 SSE；refit 清空旧码。ADC 的表存内积，不存距离。`retrieve` 用 ADC 分数选 ID，再收集**原始** K/V 做精确 attention，所以检索误差只来自漏召回，不来自对 V 做 PQ。

**为什么测试要这样写。** 旧版只检查 SSE 不增，中心完全不动时 SSE 恒定，同样“不增”。新测试有一份规则相同的独立 Lloyd，码本和每轮 SSE 都要对上，随机数据上首轮更新必须严格下降。旧版对 `scores` 取反、C4 选最低分都能过，因为只检查了 ID 个数和首尾 mandatory；新测试检查返回的 ID 是否是 ADC 分数下合法的 top-k（容差 $10^{-4}$，并列取小编号），并要求在有簇结构的数据上 recall 至少是随机选择的 3 倍。

**常见错误。** 编码时所有子空间都用第 0 个子空间的码本；用欧氏距离排序；取回字节只算 K。

## R：融合 decode kernel

**做法。** 一个 warp 对应一个（KV head，split），lane $l$ 持有通道 $[4l,4l+4)$。

- **R1**：读一个 `uint16`（4 个码，偶数通道在低半字节）。K 的 scale/offset 用 `float4` 一次读 4 个通道，按（块，通道）索引；V 每个 lane 只需一组（`vgroup` 是 4 的倍数）。
- **R2**：每次发出 4 个 token 的 K/V 读取，再统一做点积、shuffle 规约和在线更新；两个 query head 共用一次 K/V 读取。int4 前缀和残余窗口分成两段循环，跨界的 split 也能处理。
- **R3**：先求全局最大值 $M$，再按 $e^{m_s-M}$ 加权合并分子和分母。

**合并公式为什么对。** $e^{m_s-M}a_s=\sum_{t\in s}e^{\sigma_t-M}v_t$，分母同理，分子分母同乘 $e^{M}$ 就是完整 softmax。减 $M$ 是数值上的必需：测试里局部最大值相差 100，$e^{130}$ 超过 FP32 上限。

**为什么测试要这样分组。** R3 不依赖 R1、R2，直接喂人造的部分状态；R2 用 FP16 稠密缓存，不经过 R1；R1 用探针 kernel 单独检查解包；`fused` 才把三者接起来。四组在独立进程里跑，`asm("trap;")` 只会让当前组失败，学生可以按 R3、R2、R1 的顺序逐个完成。GQA 映射错误（$h\bmod H_{\mathrm{kv}}$）在 R2 被抓到，因为每个 query head 的部分状态都和 FP64 逐一比对。

**一个性能上的发现。** 初版 R1 在函数里用 `if (t >= prefix)` 区分 int4 行和残余行。32K 时融合路径 690 µs，比 FP16（646 µs）还慢。把残余窗口拆成单独的循环、R1 保持无分支后降到 382 µs。我的解释是分支让编译器无法把相邻 token 的访存连续发出，这一点没有用 SASS 逐条核对，学生可以用 `cuobjdump -sass` 验证。另一个对照实验把 K 元数据读取换成常数，速度没有可测的变化，说明剩余差距不在元数据，更可能在每个 token 的 shuffle 规约和每 lane 2 B 的小粒度访存。

**参考机实测**（RTX 4060 Laptop，驱动 566.24，CUDA 12.6，一层、batch 1、残余窗口 64 行；cudaEvent 中位数，25 次，计时前冲 L2；2026-10-18）：

| 长度 | split | FP16 µs | 先反量化 µs | 融合 µs | 融合 GB/s | 占 copy 峰值 |
| ---: | ----: | ------: | ----------: | ------: | --------: | -----------: |
|   1K |    32 |    37.9 |        56.3 |    37.9 |        61 |         0.31 |
|   4K |    86 |   106.5 |       131.1 |    69.6 |        93 |         0.48 |
|  16K |   342 |   427.0 |       756.7 |   183.3 |       121 |         0.62 |
|  32K |   683 |   737.3 |      1549.3 |   382.0 |       113 |         0.58 |

copy 峰值在不同运行之间是 195 到 223 GB/s，笔记本 GPU 频率会漂移，比较要用同一次运行里的数。32K 的 split 扫描早期出现过 8 倍的双峰（split=256 约 3.6 ms，其他运行约 0.4 ms）。排查结果：GPU 上没有其他进程时仍会出现，加了后台负载反而消失；运行时采样显示 GPU 在 P0（显存 8000 MHz）和 P5（显存 810 MHz）之间切换，而原来 5 次预热不到 1 ms，升频来不及。计时前加 300 ms 的忙等预热后，5 次运行全部稳定：split=128–256 约 0.35–0.40 ms 最快，split=4096 约 0.97 ms（切分太少，SM 吃不满）。教训：笔记本 GPU 的时序实验要先预热到稳定频率，记录时钟，并检查 IQR；多个实验共用一张卡时还要加锁，否则结果会混入别人的负载。

**常见错误。** 半字节顺序反了；K 的 scale 当成 4 个通道共用；V 的 scale 用块首 token 的；split 结束位置少 1；合并时忘了减全局最大值。

## 运行命令

从 `labs/` 执行：

```bash
python tools/run.py --lab 1 --impl reference --mode test --part all
python tools/run.py --lab 1 --impl reference --mode bench
# WSL
python tools/run.py --lab 1 --impl reference --mode test  --part all --device cuda
python tools/run.py --lab 1 --impl reference --mode bench --part R   --device cuda
python tools/fault_inject.py --lab 1 --device cpu
python tools/fault_inject.py --lab 1 --device cuda
```

参考实现：CPU 167 项检查全部通过，R 29 项检查全部通过。具体记录和源码哈希见[验证记录](../../labs/docs/validation.md)。数值供核对，不必抄写；预算曲线、带宽差距和负结果要你自己解释。

## oracle 与 fault 对照表

49 个 fault：CPU 38 个、CUDA 11 个，每个 TODO 至少 2 个。2026-10-18 在 WSL 上运行 `fault_inject.py --lab 1 --device cpu` 和 `--device cuda`，全部 CAUGHT，没有 STALE。

| TODO | 独立 oracle                                                   | fault id                     | 注入的错误                                                                              | 在哪个 part 失败 | 状态   |
| ---- | ------------------------------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------------- | ---------------- | ------ |
| A1   | 逐组 FP64 min/max/scale；用 oracle scale 检查每个码在半步内   | `a1-scale-denominator-16`    | int4 has 16 codes but 15 steps                                                          | A (cpu)          | CAUGHT |
| A1   | 逐组 FP64 min/max/scale；用 oracle scale 检查每个码在半步内   | `a1-global-minmax`           | one min/max for the whole tensor                                                        | A (cpu)          | CAUGHT |
| A1   | 逐组 FP64 min/max/scale；用 oracle scale 检查每个码在半步内   | `a1-tail-zero-padding`       | a partial tail group is padded with zeros before min/max                                | A (cpu)          | CAUGHT |
| A1   | 逐组 FP64 min/max/scale；用 oracle scale 检查每个码在半步内   | `a1-nibble-order`            | even index written to the high nibble                                                   | A (cpu)          | CAUGHT |
| A1   | 逐组 FP64 min/max/scale；用 oracle scale 检查每个码在半步内   | `a1-truncate-not-round`      | truncation instead of round-to-nearest doubles the worst-case error                     | A (cpu)          | CAUGHT |
| A1   | 逐组 FP64 min/max/scale；用 oracle scale 检查每个码在半步内   | `a1-dirty-tail-byte`         | packed bytes not cleared before OR-ing nibbles                                          | A (cpu)          | CAUGHT |
| A2   | 手写 Packed，按 $o+s\,q$ 逐元素重算                           | `a2-signed-nibble`           | codes 8..15 read as negative two's complement                                           | A (cpu)          | CAUGHT |
| A2   | 手写 Packed，按 $o+s\,q$ 逐元素重算                           | `a2-high-nibble-first`       | decoder reads the nibbles of a byte in the wrong order                                  | A (cpu)          | CAUGHT |
| A2   | 手写 Packed，按 $o+s\,q$ 逐元素重算                           | `a2-scale-offset-swap`       | metadata fields swapped                                                                 | A (cpu)          | CAUGHT |
| A3   | 手算行收集与非法 ID                                           | `a3-accept-duplicates`       | duplicate ids pass validation and silently duplicate rows                               | A (cpu)          | CAUGHT |
| A3   | 手算行收集与非法 ID                                           | `a3-row-offset`              | row index not scaled by row width                                                       | A (cpu)          | CAUGHT |
| A4   | 手算 relative L2、cosine、max abs                             | `a4-zero-cosine`             | two zero vectors get cosine 0 instead of the convention 1                               | A (cpu)          | CAUGHT |
| A4   | 手算 relative L2、cosine、max abs                             | `a4-rel-normalized-by-b`     | relative L2 divided by the candidate norm                                               | A (cpu)          | CAUGHT |
| A4   | 手算 relative L2、cosine、max abs                             | `a4-signed-max`              | max error without abs                                                                   | A (cpu)          | CAUGHT |
| B1   | 手算边界并列 + 300 组暴力枚举                                 | `b1-tie-reversal`            | ties broken toward the larger id                                                        | B (cpu)          | CAUGHT |
| B1   | 手算边界并列 + 300 组暴力枚举                                 | `b1-recent-off-by-one`       | recent window one token short                                                           | B (cpu)          | CAUGHT |
| B1   | 手算边界并列 + 300 组暴力枚举                                 | `b1-unsorted-output`         | output left in mandatory-then-score order instead of position order                     | B (cpu)          | CAUGHT |
| B2   | 从原始追加 K/V 重算最早块的元数据；显式轴断言；flush 计划闭式 | `b2-flush-zeroing`           | flushed block quantizes zeros                                                           | B (cpu)          | CAUGHT |
| B2   | 从原始追加 K/V 重算最早块的元数据；显式轴断言；flush 计划闭式 | `b2-kv-axis-swap`            | K per-token and V per-channel                                                           | B (cpu)          | CAUGHT |
| B2   | 从原始追加 K/V 重算最早块的元数据；显式轴断言；flush 计划闭式 | `b2-flush-newest`            | quantizes the newest block but erases the oldest                                        | B (cpu)          | CAUGHT |
| B2   | 从原始追加 K/V 重算最早块的元数据；显式轴断言；flush 计划闭式 | `b2-late-flush`              | flushes one append late                                                                 | B (cpu)          | CAUGHT |
| B3   | 测试自带解码器拼接 chunk + recent；与原始历史对比顺序         | `b3-reverse-chunks`          | blocks concatenated newest first                                                        | B (cpu)          | CAUGHT |
| B3   | 测试自带解码器拼接 chunk + recent；与原始历史对比顺序         | `b3-omit-recent`             | materialize drops recent K rows                                                         | B (cpu)          | CAUGHT |
| B4   | 闭式字节公式                                                  | `b4-recent-fp16`             | FP32 recent rows counted as FP16                                                        | B (cpu)          | CAUGHT |
| B4   | 闭式字节公式                                                  | `b4-no-metadata`             | scale/offset bytes omitted from the payload                                             | B (cpu)          | CAUGHT |
| B5   | 解码缓存上的 FP64 attention；原始历史上的误差界               | `b5-no-acc-rescale`          | running weighted V sum not rescaled when the max changes                                | B (cpu)          | CAUGHT |
| B5   | 解码缓存上的 FP64 attention；原始历史上的误差界               | `b5-no-sqrt-d`               | logits not scaled by 1/sqrt(d)                                                          | B (cpu)          | CAUGHT |
| B5   | 解码缓存上的 FP64 attention；原始历史上的误差界               | `b5-skip-recent`             | recent window ignored by the query path                                                 | B (cpu)          | CAUGHT |
| C1   | 独立 Lloyd（同规则）；首轮 SSE 严格下降                       | `c1-lloyd-no-center-update`  | centers never move                                                                      | C (cpu)          | CAUGHT |
| C1   | 独立 Lloyd（同规则）；首轮 SSE 严格下降                       | `c1-wrong-seed-rows`         | seeds the first C rows instead of rows c*N/C                                            | C (cpu)          | CAUGHT |
| C1   | 独立 Lloyd（同规则）；首轮 SSE 严格下降                       | `c1-keep-old-codes`          | refit keeps codes produced by the previous book                                         | C (cpu)          | CAUGHT |
| C2   | 暴力最近中心；等距并列                                        | `c2-tie-largest`             | equidistant key goes to the larger center id                                            | C (cpu)          | CAUGHT |
| C2   | 暴力最近中心；等距并列                                        | `c2-wrong-subspace`          | every subspace is compared with subspace 0's codebook                                   | C (cpu)          | CAUGHT |
| C3   | 重建 key 的 FP64 点积                                         | `c3-negated-score`           | ADC score sign flipped                                                                  | C (cpu)          | CAUGHT |
| C3   | 重建 key 的 FP64 点积                                         | `c3-euclidean`               | negative squared distance instead of inner product                                      | C (cpu)          | CAUGHT |
| C4   | ADC top-k 合法性 + FP64 attention + recall 对随机             | `c4-select-lowest`           | retrieval ranks by negated scores (keeps the least similar keys)                        | C (cpu)          | CAUGHT |
| C4   | ADC top-k 合法性 + FP64 attention + recall 对随机             | `c4-v-from-k`                | attention reads K rows as values                                                        | C (cpu)          | CAUGHT |
| C4   | ADC top-k 合法性 + FP64 attention + recall 对随机             | `c4-bytes-k-only`            | fetch accounting forgets the V rows                                                     | C (cpu)          | CAUGHT |
| R1   | 主机端 FP64 解码器；原始 K/V 半步界                           | `r1-nibble-order`            | the two nibbles of each byte swapped (high nibble read as the even channel)             | R (cuda)         | CAUGHT |
| R1   | 主机端 FP64 解码器；原始 K/V 半步界                           | `r1-k-scale-not-per-channel` | K metadata applied per lane group like a per-token scale instead of per channel         | R (cuda)         | CAUGHT |
| R1   | 主机端 FP64 解码器；原始 K/V 半步界                           | `r1-v-scale-per-chunk`       | V scale taken from the chunk's first token (per-channel-per-chunk) instead of per token | R (cuda)         | CAUGHT |
| R2   | 每个（head，split）的 FP64 重算（FP16 稠密输入）              | `r2-no-sqrt-d`               | logits not scaled by 1/sqrt(D)                                                          | R (cuda)         | CAUGHT |
| R2   | 每个（head，split）的 FP64 重算（FP16 稠密输入）              | `r2-split-end-off-by-one`    | each full split drops its last token                                                    | R (cuda)         | CAUGHT |
| R2   | 每个（head，split）的 FP64 重算（FP16 稠密输入）              | `r2-gqa-head-mapping`        | query head h paired with KV head h % H_kv instead of h / G (HF repeat_kv)               | R (cuda)         | CAUGHT |
| R2   | 每个（head，split）的 FP64 重算（FP16 稠密输入）              | `r2-no-online-rescale`       | weighted V sum not rescaled when the running max grows                                  | R (cuda)         | CAUGHT |
| R2   | 每个（head，split）的 FP64 重算（FP16 稠密输入）              | `r2-skip-residual`           | residual (FP32) tokens never attended                                                   | R (cuda)         | CAUGHT |
| R3   | FP64 log-sum-exp 合并（最大值相差 100、空 split）             | `r3-no-max-rescale`          | partials added without the $e^{m_s-M}$ weights                                          | R (cuda)         | CAUGHT |
| R3   | FP64 log-sum-exp 合并（最大值相差 100、空 split）             | `r3-unweighted-normalizer`   | numerator rescaled but denominator not                                                  | R (cuda)         | CAUGHT |
| R3   | FP64 log-sum-exp 合并（最大值相差 100、空 split）             | `r3-no-max-subtraction`      | exponent taken without subtracting the global max                                       | R (cuda)         | CAUGHT |
