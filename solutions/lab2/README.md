# Lab 2 参考答案导读

[任务与全部接口](../../labs/lab2/README.md) · [评分](../../labs/lab2/grade.md) · [来源](../../labs/lab2/resources.md)

参考实现覆盖 22 个 TODO：A1–A6、B1–B6、C1–C5、R1–R5。文件是 [quant_student.hpp](src/quant_student.hpp)（A、C）、[gemv_student.cuh](src/gemv_student.cuh)（B）、[fp8_student.hpp](src/fp8_student.hpp)（R1、R2、R5）、[fp8_gemv_student.cuh](src/fp8_gemv_student.cuh)（R3、R4）。测试、基准和 driver 在 `labs/lab2/`，参考模式运行时用这里的同名文件覆盖学生骨架。错误注入清单在 [faults.json](faults.json)。

## 这一版改了什么，为什么

旧版有三处测试漏洞和一处测量问题，这一版都改了：

1. **A1 的 scale 从来没有被独立检查过。** 旧测试的每个 scale 检查都经过学生代码：零组、`group == cols` 的已知码（组就是整行）、对学生自己 `p.scales` 算的半步证书、和学生 `unpack` 比较的 A4。按整行取 absmax 的量化器能通过全部 A。新测试先用手算的例子 $[7,1,-1\mid0.7,0.1,0]$、$g=3$ 要求 scale 为 $[1,0.1]$，再在 36 种形状上和 FP64 的逐组 absmax 比对。A2–A6 用 `oracle.hpp` 自己打包的 artifact 或手写的 `Packed`，不再依赖学生的 A1/A2。
2. **C3 的 oracle 是循环的。** 旧测试调用学生的 `channel_scales`、`transform`、`pack`、`unpack` 去重算最优 $\alpha$，所以统计量写成最大值、不归一化、不 clamp 都能通过；C1 也只在 $\alpha=0$（全为 1）时检查过。新测试有：手算的 $2\times3$ 例子（均值 $[2,2,4]$，$\alpha=0.5$ 时 $2^{[-1/4,-1/4,1/4]}$）、先 clamp 再求幂的例子（$[0.1,10]$）、不调用任何学生函数的穷举搜索 `oracle::search`，以及一个必须选出 $\alpha>0$ 的离群通道例子。
3. **B 的 GPU 结果和学生 A4 比较。** 现在 kernel 的输入由 `oracle::pack` 打包，输出和同一份字节的 FP64 解码比较，B 不再继承 A 的错误。
4. **基准测的是 L2。** 旧基准是 $\le2048^2$ 的方阵，int4 只有 2 MiB，30 次连续调用都在 32 MiB 的 L2 里；没有任何强基线。新基准用 Qwen3-1.7B 的真实形状加 $8192^2$，每次计时前冲掉 64 MiB，300 ms 预热，报中位数和 IQR，并把 cuBLAS FP32/FP16、先解包再 cuBLAS、cuBLASLt FP8 放进同一张表。`bench_plan.hpp` 由 R5 测试检查：关掉冲刷，测试就失败。

旧的 `common.cuh`、`test.cu`、`bench.cu` 从来没有被 runner 编译过，讲义却在描述它们；`pipeline.cu` 拆成了 `gpu_test.cu`（测试）和 `gpu_bench.cu`（基准）。这四个文件都已删除。

## A：格式

**做法。** `calibrate` 两层循环：行、行内的组，组在 $\min(N,(j+1)g)$ 处截止。`pack` 先清零字节再按位或，奇数宽度行尾的高半字节因此自然为 0。`unpack` 和 `packed_linear` 各自独立解码（后者不调用前者），这样一个解码 bug 不会在两处互相掩盖。`clip_pack` 先用普通 absmax 打包，再逐组枚举比例，选中后把这一组的码按新 scale 重写，旧码先用掩码清掉。

**为什么测试要这样写。**

- 半步证书 $\lvert w-\hat w\rvert\le s/2$ 本身没有判别力：scale 越大，半步越大。必须用独立算出的 scale 去判断码，或者直接比对 scale。
- 舍入的平局要专门构造（$2.5$、$0.5$），随机数据几乎碰不到；`nearbyint` 与 `lround` 只在平局上不同。
- A6 的随机数据几乎从不选 $\rho<1$（我模拟了 200 组 32 个 Uniform 或 Gaussian 值，候选 $\{1,0.9,0.8,0.5\}$ 下胜者只有 1 和 0.9）。所以测试加了两个设计的组：31 个 $\pm0.5$ 加一个最大值，$\rho=1$ 时每个 $0.5$ 都在 3.5 步的平局上，$\rho=0.9$ 的平方误差和小一个数量级。
- 容差：A5 的比例在 FP32 scale 和 FP64 重算之间有 $10^{-6}$ 量级的相对差，检查用 $10^{-5}+10^{-4}\cdot\text{ratio}$；A2 的码检查对离舍入边界 $10^{-4}$ 以内的元素放行两种结果。

**常见错误。** 按行取 absmax；scale 除以 8；尾组读进下一行；全零组 scale 为 0；码存成 $q+7$；偶数列放高半字节；按 $\lfloor N/2\rfloor$ 算行步长；把半字节当有符号 int4 做符号扩展。

## B：四条 W4A32 kernel

**做法。** B2 一个 warp 一行，lane 跨步 32 列，五轮 `__shfl_down_sync`。B3 每个 lane 一个字节、两列各自查 scale。B4 的 4 个累加器放在寄存器数组里，循环上界是常数 4、用 `token + t < batch` 判断，编译器能把数组留在寄存器里。B5 的 partial 布局是 $[\mathrm{parts},\mathrm{rows}]$，这样 B6 一个线程读一列连续的内存段。

**为什么测试要这样写。** 每个 kernel 单独一个进程（`gpu_test B1`…`B5`），一个 `asm("trap;")` 只让自己那一组失败。输出缓冲区先填 `0xFF`（NaN），没写到的位置一定失败。B4 的 batch 里加了 8：tile 起点写成 `blockIdx.y * 3` 时，batch 7 的两个 tile 恰好仍能覆盖 0–6，只有 batch 8 会漏掉 token 7（fault `b4-tile-step-3` 最初就是这样漏网的）。B5/B6 额外把 partial 缓冲区读回逐段核对，区分“partial 写错”和“merge 漏段”。容差是 $10^{-5}(1+\sum_k\lvert w_k x_k\rvert)$：FP32 累加 $K$ 项的舍入误差与项的绝对值和成正比，而不是与结果成正比。

**性能上的发现**（measured，见下面的表）。四条 SIMT kernel 在 batch 1 都只用了 0.20–0.38 的带宽。B4 在 batch 1 比 B2 还慢（up_proj 154 对 102 µs）：它为 4 个 token 留了寄存器和分支，batch 1 时这些都是浪费。B5 在这些形状上从不划算，因为 $M\ge2048$ 时并行度本来就够。我没有用 SASS 逐条核对 INT4 kernel 慢的原因；最可能的是每个权重的 `bfe`、减 8、`I2F`、乘 scale 四条指令加上逐字节的小粒度加载，学生可以用 `cuobjdump -sass` 验证。

## C：AWQ 式通道缩放

**做法。** C1 先用 double 累加 $\lvert x\rvert$，再 clamp、求幂，最后除以几何均值 $\sqrt{\max s\cdot\min s}$。归一化让 $\alpha$ 变化时 scale 不会整体漂移；它不改变数学上的等价性，只影响 FP32 下的数值范围。C3 的参考输出用原始浮点 $W$ 算一次，所有候选都从原始 $W$、$X$ 重新构造，避免缩放累积。C5 先算 artifact 的完整字节（码、组 scale、通道 scale），不可行的组直接跳过，不做搜索。

**为什么测试要这样写。** 测试里的 `oracle::search` 在 FP32 里做 $W\cdot s$ 和 $X/s$（与契约一致），用 oracle 的打包器和 FP64 输出，所以它和参考实现在同一个 $\alpha$ 上得到完全相同的 MSE（相对差 $<10^{-4}$）。离群通道例子：8 行 32 列，两列激活放大 30 倍，组大小 32；oracle 的最优 $\alpha$ 大于 0，并且 MSE 比 $\alpha=0$ 低 20% 以上（测试同时断言这一点，fixture 失效时会报出来）。这个例子专门针对“搜索永远返回 0”和“参考输出用了量化后的 $W$”（那样 $\alpha=0$ 的 MSE 是 0，永远胜出）。

**校准表（measured，`--mode bench --part C`）。** 同分布的 held-out 在四个宽度上都比 RTN 好（2.8 倍到 12%）；放大通道换到另一列后，三个宽度上比 RTN 差 2.7–14 倍。原因：缩放把被放大的那一列的权重放大、把其他列的相对步长变大；激活的大通道一换，原来被“牺牲”的列变成了重要的列。这说明 C3 在校准集上不劣于 $\alpha=0$ 只是有限候选最小化的性质，不是泛化保证。clipping-only 在这组数据上和 RTN 几乎一样，因为均匀分布的组几乎不选 $\rho<1$。

## R：FP8 权重 GEMV

**R1 的做法。** 先处理 NaN 和饱和（`!(a < 448)` 同时接住 $\pm\infty$），再用 `frexp` 取指数，算出 $a$ 所在区间的网格步长 $2^{\max(e-1,-6)-3}$（$e-1$ 是二进制指数，$-6$ 是 subnormal 的下限），用 `nearbyint` 在网格单位上做 RNE，最后把结果拆回指数和尾数字段。舍入可能把值进到下一个 binade（例如 $1.9375\to2$），所以字段从舍入后的值重新取，而不是沿用舍入前的指数。$[448,464]$ 里的值舍入后仍是 448（464 是 448 与 480 的中点，平局取偶数尾数，落在 448），超过 464 的值在 RNE 下会进到 480，而 480 的位模式是 NaN。satfinite 把这两种情况都给 448，所以饱和判断放在舍入之前、按 $\ge448$ 一刀切是等价的。

**R1 的 oracle 为什么可信。** `oracle::e4m3_nearest` 对 127 个非负有限码暴力找最近值，平局取尾数最低位为 0 的码，完全不涉及位运算技巧。它本身也要过一关：`gpu_test R1` 的第一项把它和 GPU 上的 `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` 在约 2 万 1 千个输入上逐一比对（精确码、每对相邻码的中点和中点两侧、饱和区、subnormal 区、对数均匀的随机数），参考机上全部一致。只测平局才能区分 RNE 和 ties-away，只测饱和区才能抓到“480 变成 NaN”。

**R3/R4 的做法。** 每个 lane 每次一条 `uint4` 加载读 16 个码，`__nv_cvt_fp8x2_to_halfraw2` 两两转换，16 个乘积先累加再乘一次组 scale。`cols` 是 4 的倍数时用 `float4` 读 $x$。这一项改动让 $8192^2$ 上 R3 的带宽利用率从 0.55 升到 0.81、up_proj 从 0.52 升到 0.67：标量读 $x$ 时，每 16 个权重要 16 条 32 位加载，比读权重本身的指令还多。

**R5 与基准计划。** `weight_bytes` 是讲义 §1 的字节模型。`bench_plan.hpp` 列出形状、batch、冲刷大小，R5 测试对每个（形状，格式）检查“冲刷量至少两倍 L2，或权重本身至少两倍 L2”。fault `r5-stale-l2-bench` 关掉冲刷，Qwen3-1.7B 的全部 INT4/FP8 层都不满足，测试失败。

**为什么 B/R 的基准用学生的打包器。** 基准要测的是学生交上来的格式；若 A2 或 R2 是错的，`rel_err` 会超出界，整行标 `WRONG_OUTPUT`，基准以非零码退出。测试则用 oracle 打包，两者分工不同。

## 参考机实测

RTX 4060 Laptop，驱动 566.24，CUDA 12.6，2026-10-18，一次完整的 `--mode bench --part all --device cuda` 运行（copy 峰值 223 GB/s，L2 冲刷，300 ms 预热，25 次中位数，全部行 `stable=yes`）。µs 是 measured；`floor` 是权重读一遍除以 copy 峰值，modeled。

| 形状（$M\times K$）      | batch |         INT4 B3 | INT4 B4 | FP8 R3/R4 | FP16 cuBLAS | FP32 cuBLAS | 先解包再 FP16 | cuBLASLt W8A8 | INT4 floor | FP8 floor |
| ------------------------ | ----: | --------------: | ------: | --------: | ----------: | ----------: | ------------: | ------------: | ---------: | --------: |
| up_proj $6144\times2048$ |     1 |              83 |     141 |   87 / 88 |         139 |         288 |           374 |            77 |         30 |        58 |
| up_proj                  |     4 | 316（逐 token） |     154 |   — / 137 |         147 |         307 |           325 |            77 |          — |         — |
| up_proj                  |    64 |               — |    2219 |  — / 1873 |         153 |         353 |           359 |            80 |          — |         — |
| $8192\times8192$         |     1 |             418 |     762 | 384 / 390 |         574 |        1191 |          1956 |           391 |        159 |       309 |
| $8192\times8192$         |     4 |               — |     858 |   — / 698 |         640 |        1180 |          2269 |           372 |          — |         — |
| $8192\times8192$         |    64 |               — |   13756 |  — / 9268 |         677 |        1577 |          2158 |           376 |          — |         — |

读法见讲义 §5.3。两点补充：FP16 cuBLAS 在 $8192^2$ 上的 `frac_peak` 是 1.05，纯读比 copy（读写各半）略快，所以不要把 copy 峰值当成严格上界；cuBLASLt W8A8 在 batch 1 到 64 之间几乎不变，因为 FP8 Tensor Core 的脊点约 227 FLOP/B，batch 64 时强度才约 128（逐张量 scale，每权重 1 B）。

## 运行命令

从 `labs/` 执行：

```bash
python tools/run.py --lab 2 --impl reference --mode test  --part A
python tools/run.py --lab 2 --impl reference --mode test  --part C
python tools/run.py --lab 2 --impl reference --mode bench --part C
# WSL
python tools/run.py --lab 2 --impl reference --mode test  --part all --device cuda
python tools/run.py --lab 2 --impl reference --mode bench --part all --device cuda
python tools/fault_inject.py --lab 2 --device cpu
python tools/fault_inject.py --lab 2 --device cuda
```

参考实现：主机 33 项检查、GPU 11 组共 21 项检查全部通过。具体记录和源码哈希见[验证记录](../../labs/docs/validation.md)。数值供核对，不必抄写；交叉 batch、带宽差距和负结果要你自己解释。

## oracle 与 fault 对照表

63 个 fault：CPU 32 个、CUDA 31 个，每个 TODO 至少 2 个，A1、A2、R1 等核心 TODO 有 4–5 个。覆盖了改版要求的全部漏洞：按行 absmax（`a1-row-absmax`、`r2-row-absmax`）、零点差一（`a2-zero-point-off-by-one`、`a3-zero-point-off-by-one`）、半字节顺序（`a2-nibble-order`、`a4-nibble-order`）、符号扩展（`a3-sign-extension`、`b1-sign-extension`）、split-K 不归约（`b6-partial-not-reduced`）、AWQ 搜索恒返回 0（`c3-always-alpha-zero`）、激活端没做逆缩放（`c2-activation-not-inverted`、`c4-no-inverse-scaling`）、FP8 饱和错误（`r1-no-saturation`、`r2-scale-480`）、基准测 L2（`r5-stale-l2-bench`、`r5-small-flush`）。2026-10-18 在 WSL 上运行 `fault_inject.py --lab 2 --device cpu` 和 `--device cuda`，全部 CAUGHT，没有 STALE。注入错误一列取自 `faults.json` 的 `why` 字段。
