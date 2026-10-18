# Lab 6 参考实现导读

[任务](../../labs/lab6/README.md) · [评分](../../labs/lab6/grade.md) · [来源与简化](../../labs/lab6/resources.md) · [实现 A–C](src/train_student.py) · [实现 R](src/fused_student.py) · [错误注入](faults.json)

先独立完成再读这一页。下面每个阶段写三件事：实现要点、为什么这样做、常见错误；最后是运行命令和 oracle 与 fault 的对照表。

## A：参数的生命周期

**A1.** 基座和低秩分支都用 `F.linear`，权重布局 $[d_{\mathrm{out}},d_{\mathrm{in}}]$ 不需要手动转置；bias 只交给基座那次 `F.linear`。$W$ 冻结靠 `requires_grad=False`，不靠 `no_grad()`：后者会切断 $dX=GW+s(GB)A$ 的第一项。这类错误 loss 照样下降、$B$ 照样有梯度，所以测试必须令 $B\ne0$、给定上游 $G$，再逐项对 $dX$、$dA$、$dB$ 的稠密公式。

**A2/A3.** merge 前存 `weight.clone()`，unmerge 时 `copy_` 回去并把备份设为 `None`。用 $W+\Delta-\Delta$ 恢复，每个元素做了两次舍入；测试把 $B$ 的标准差设成 10，让 $\Delta$ 远大于 $W$ 的元素，三轮之后 `torch.equal` 必然失败。merge 要求 eval 模式：train 模式下 optimizer 还会更新 $A$、$B$，而 $W$ 里已经包含了旧的 $\Delta$。

**A4.** 两遍：第一遍只检查（名字存在、是 `nn.Linear`、无重复、非空），第二遍才构造和替换。新模块先 `.to(device, dtype)` 再拷权重，再 `train(old.training)`。冻结要对整个模型做，而不只是被替换的层，测试的第三个 Linear 就是为这一点放的。

**A5/A6.** 保存时 `detach().cpu().clone()`：CPU 上 `.cpu()` 返回同一个张量，不 clone 就会别名。加载分三步：全部校验、全部转换并检查有限性（float64 的 $10^{100}$ 转 float32 是 inf）、全部写入。只有这样，最后一项失败时前面的模块才能保持不变。

**A7.** `parameters()` 已经按对象去重（`named_parameters(remove_duplicate=True)`），所以“同一个 Parameter 被两个模块共享”不需要处理，原来的测试也因此测不出 `seen` 集合的作用。真正需要处理的是不同 Parameter 对象共享一块 storage（用 `.data` 或视图构造的 tied weight）。参考实现按 `storage_key` 去重、按 `untyped_storage().nbytes()` 计整块，任何一个别名可训练就算 trainable。

## B：重计算与规划

**B1.** 先完整校验再执行；`zip(strict=True)`。非重入 checkpoint 的区域在反传时整体重跑一遍，所以 forward pre-hook 在 checkpoint 块上触发两次，这是证明 checkpoint 真正发生的最直接证据。原测试只比较梯度，一个永远 retain 的实现也能通过。保存字节是第二个证据：三块全 checkpoint 时只剩三个块输入，每个 $2\times5\times16\times4=640$ 字节。用 `use_reentrant=True` 时输入不需要梯度，区域内的参数拿不到梯度，测试直接报错。

**B2/B4.** 从旧状态构造新状态，每块只扩展一次，天然满足 0/1 约束。B4 的 Pareto 剪枝：按单位升序扫，只保留代价严格下降的状态，因为单位更多而代价不更低的状态不可能出现在最优解里。返回时按（代价、单位）取最小。测试用穷举，不复用 DP。

**B3.** 按 `storage_key` 去重、记整块 storage 的字节：一个视图被保存，整块 storage 都活着。返回 `tensor.detach()`，不能返回原张量，否则 payload 带着 `grad_fn`，形成引用环。

**B5.** 第 $i$ 块的输入是前 $i$ 块在 `no_grad` 下的 detach 输出。测试第一块把宽度从 8 变到 16，忘了推进输入时第二块直接形状不符。每个模式先预热一次再取中位数；返回前清梯度。

## C：offload 与训练步

**C1/C2.** payload 是 `(copied, device, value)`。拷贝用 `torch.empty(..., device="cpu")` 加 `copy_`，保证新 storage：`tensor.to("cpu")` 对 CPU 张量是 no-op，这正是 `offload_devices=("cpu",)` 让 CPU 学生也能测到的别名错误。unpack 每次 `to(device, copy=True)` 返回新张量，所以修改第一次的结果不影响第二次。payload 里不能留设备张量的引用：那样正确性全过、显存一点不省，只有 CUDA 上“前向后 allocated 增量低于 retain 的 60%”这一项能抓住。

**C3.** 校验全部在执行之前，包括“有 offload 模式就必须有 owner”。原实现虽然检查了，但没有测试；fault `c3-no-owner-check` 删掉检查后，错误会在第二块（offload 块）才以 `AttributeError` 出现，第一块已经跑过、dropout 已经消耗了 RNG。测试用 forward hook 计数和 `torch.get_rng_state()` 比较来证明“什么都没发生”。

**C4.** 每个微批除以**总**元素数再 backward，等于全 batch 均值；裁剪在循环之后做一次。测试的关键是先确认全 batch 梯度范数 $\lVert g\rVert$ 远大于 $c=10^{-3}$：裁剪因子 $c/\lVert g\rVert$ 很小，“不裁剪”和“裁剪”差别巨大；“每微批裁剪”把每个微批的梯度都缩到 $c$，累积后的范数最多是 $2c$、方向也变了。原测试用 `max_grad_norm=1`，从没确认梯度范数超过 1，三种实现都能通过。owner 检查放在 `zero_grad` 之前，否则一次被拒绝的调用已经清掉了调用方的梯度。

**C5.** 预热 step 创建 AdamW 状态；CUDA 上再忙 300 ms，让笔记本 GPU 离开低频状态；然后重置计数和峰值，逐步用 CUDA event 计时，报告中位数、IQR 和 `stable`。`lr=0` 让各策略间权重不变，但 AdamW 的运算照常执行。

## R：比 eager 存得更少

**出发点。** 2026-10-18 在 torch 2.14.1 上的探针（讲义 §3.1）确认：$W$ 冻结时，eager LoRA 只存 $X$ 和 $XA^\top$；$A$ 冻结时只存 $XA^\top$。原计划里“只存 $X$ 和 $XA^\top$ 的 Function”因此没有任何收益，R 改为三个 eager 做不到、或者做到了但值得亲手证明的保存集。

**R1（LoRA-FA）。** 存 $Z$ 和参数；反传 $dB=s\,G^\top Z$，$dX=GW+s(GB)A$。和 eager LoRA-FA 的保存集完全相同，测试把这一点作为检查之一，让学生看到“写了 Function 不等于省了”。

**R2（融合 MLP）。** 存 $X,z_1,u,z_2$。反传先由 $u$ 重算 $g$，用完 `del g` 再算 $du$，避免 $g$ 和 $\mathrm{GELU}'(u)$ 同时存活。推导：

$$
dz_2=s_2\,GB_2,\quad dB_2=s_2\,G^\top z_2,\quad dA_2=dz_2^\top g,\quad du=(GW_2+dz_2A_2)\odot\big(\Phi(u)+u\,\varphi(u)\big)
$$

$$
dz_1=s_1\,du\,B_1,\quad dB_1=s_1\,du^\top z_1,\quad dA_1=dz_1^\top X,\quad dX=du\,W_1+dz_1A_1
$$

每 token 省 $f$ 个数。4060 上 4 层、$h=1024$、$f=3072$、2048 token 实测前向后少持有的字节与 $LNf\times4=100{,}663{,}296$ 相差不到 5%；fw+bw 峰值也更低，但降幅小于保存字节的降幅，因为峰值还包含反传时同时存活的梯度和重算出的 $g$。

**R3（种子 dropout）。** 前向和反传都调用 `dropout_mask(shape, p, seed, ...)`，它用私有 `torch.Generator`，不读也不改全局 RNG。保存 $X$ 和 $Z$；和 eager 相比省掉的是 mask（CUDA 上每元素 1 字节，CPU 上与输入同 dtype）。$X\odot m$ 和 $X$ 一样大，所以存哪个都行，这里存 $X$ 是为了反传时同时算 $dA$ 和 $dX$ 的 mask 部分。`seed=None` 时从全局 CPU RNG 抽一个整数，这样 `torch.manual_seed` 能复现整个训练。参考机上的实测提醒了一件事：反传里一次性生成的 float mask 比 eager 保存的 bool mask 大 4 倍，8192 token 时 R3 的峰值比 eager 高约 5%。这一项的收益只在保存量上，要在峰值上兑现，需要分块生成 mask（可选挑战 2 的方向）。

**R4（测量）。** 要点是把三个数分开：saved bytes（hook 数出来的清单）、前向后 allocated 增量（这些张量确实活在设备上）、fw+bw 峰值增量（相对前向前的 allocated，而不是绝对峰值）。测试的玩具在测量前留了一块 64 MiB 的常驻张量和一块已释放的 128 MiB，专门抓“不重置峰值”和“报绝对峰值”两种错误。

**测量注意。** 笔记本 4060 在两次配置之间会降到 810 MHz 显存频率，GPU 操作慢到 8 倍；`measure_variant` 和 `measure_policy` 在计时前用同一个 pass 让 GPU 忙至少 300 ms，并输出 `stable` 标志。峰值显存不受这个问题影响。

## 运行命令

```bash
# cwd: labs/
python tools/run.py --lab 6 --impl reference --mode test --part all --device cpu
python tools/run.py --lab 6 --impl reference --mode test --part all --device cuda     # WSL
python tools/run.py --lab 6 --impl reference --mode bench --part all --device cuda    # WSL
python tools/fault_inject.py --lab 6                     # CPU faults + CUDA-only faults on their device
python tools/fault_inject.py --lab 6 --device cuda       # CUDA-only faults
```

## oracle 与 fault 对照表

状态一栏来自 2026-10-18 在 WSL（RTX 4060 Laptop、torch 2.14.1+cu126）上的 `fault_inject` 运行：先在各 fault 自己的设备上跑（`--lab 6` 与 `--lab 6 --device cuda`，73/73 CAUGHT），再把全部 fault 改到 `cuda` 上重跑一遍。标 n/a 的三个 fault 只在 CPU 上是 bug：`.cpu()`、`.to("cpu")`、`.to(device)` 跨设备时本来就会拷贝，所以它们声明为 `device: cpu`，CPU 测试用 `offload_devices=("cpu",)` 专门覆盖这条路径。

| TODO | 独立 oracle                                                                    | fault id                       | 注入的错误                                                                                                               | 在哪个 part 失败 | 状态                         |
| ---- | ------------------------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ---------------- | ---------------------------- |
| A1   | dense float32 formulas for Y, dX, dA, dB (nonzero B, bias, 3-D input)          | `a1-base-under-no-grad`        | frozen W wrongly implemented as no_grad: dX loses the G W term; the dense dX = G W + s G B A oracle catches it           | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A1   | dense float32 formulas for Y, dX, dA, dB (nonzero B, bias, 3-D input)          | `a1-bias-twice`                | bias added in the base linear and again at the end; caught because the test sets a nonzero bias                          | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A1   | dense float32 formulas for Y, dX, dA, dB (nonzero B, bias, 3-D input)          | `a1-scale-dropped`             | alpha/r scale forgotten; the dense output/gradient oracle uses s explicitly (B != 0)                                     | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A1   | dense float32 formulas for Y, dX, dA, dB (nonzero B, bias, 3-D input)          | `a1-merged-still-adds-lora`    | merged layer adds the delta twice (once in W, once from the branch); merged-output oracle catches it                     | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A2   | test-computed W + s B A; train-mode rejection                                  | `a2-merge-allowed-in-train`    | merge in train mode would let the optimizer update A/B while W already contains the old delta                            | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A2   | test-computed W + s B A; train-mode rejection                                  | `a2-merge-not-idempotent`      | second merge adds the delta again and overwrites the backup with the merged weight                                       | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A3   | torch.equal with a pre-merge clone over 3 rounds, large delta                  | `a3-unmerge-by-subtraction`    | restore by W + d - d is not bitwise W; three-round torch.equal check with a large delta catches it                       | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A3   | torch.equal with a pre-merge clone over 3 rounds, large delta                  | `a3-backup-not-released`       | backup copy of W stays alive after unmerge (a full extra weight of memory)                                               | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A4   | original model output; exact trainable-name set; dtype/flag/bias checks        | `a4-mutate-while-validating`   | validation interleaved with mutation: a later bad target leaves earlier modules modified                                 | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A4   | original model output; exact trainable-name set; dtype/flag/bias checks        | `a4-original-not-frozen`       | base parameters of untouched layers stay trainable; the exact trainable-name set catches it                              | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A4   | original model output; exact trainable-name set; dtype/flag/bias checks        | `a4-dtype-lost`                | a float64 model gets float32 adapters; dtype check and float64 forward fail                                              | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A4   | original model output; exact trainable-name set; dtype/flag/bias checks        | `a4-train-flag-lost`           | an eval-mode target comes back in train mode (dropout/merge behave differently)                                          | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A5   | deepcopy snapshot vs later model update; exact field set                       | `a5-state-aliases-model`       | on CPU .cpu() is a no-op, so the saved state aliases the live parameter; later updates leak into it                      | A                | CAUGHT (cpu) / n/a (cuda)    |
| A5   | deepcopy snapshot vs later model update; exact field set                       | `a5-scale-missing`             | state without scale cannot be validated on load; the exact field set check catches it                                    | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A6   | bitwise snapshot of the first adapter after a failing load                     | `a6-copy-before-all-validated` | copy while staging: an overflow in the last entry leaves earlier modules already overwritten                             | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A6   | bitwise snapshot of the first adapter after a failing load                     | `a6-no-overflow-check`         | finite float64 1e100 becomes inf in float32 and is loaded silently                                                       | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A6   | bitwise snapshot of the first adapter after a failing load                     | `a6-no-shape-check`            | a [1] tensor broadcasts through copy_ into B instead of being rejected                                                   | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A7   | hand-counted storage bytes (tied object, 3 aliases of one storage, view)       | `a7-dedup-by-object-id`        | distinct Parameters aliasing one storage are counted three times; parameters() only dedups identical objects             | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A7   | hand-counted storage bytes (tied object, 3 aliases of one storage, view)       | `a7-last-alias-wins`           | bucket decided by the last alias seen; a storage with any trainable alias must count as trainable                        | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| A7   | hand-counted storage bytes (tied object, 3 aliases of one storage, view)       | `a7-tensor-not-storage-bytes`  | a view Parameter keeps its whole storage alive; counting view bytes undercounts                                          | A                | CAUGHT (cpu) / CAUGHT (cuda) |
| B1   | forward pre-hook call counts; test-side saved-tensor hook bytes; retain grads  | `b1-ignore-keep`               | every block retained: gradients still match, only the forward-hook recompute count and saved bytes reveal it             | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B1   | forward pre-hook call counts; test-side saved-tensor hook bytes; retain grads  | `b1-no-rng-preservation`       | recompute draws a new dropout mask; per-parameter gradients differ from retain                                           | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B1   | forward pre-hook call counts; test-side saved-tensor hook bytes; retain grads  | `b1-no-length-check`           | zip silently truncates: a 1-element policy runs a 1-block network                                                        | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B1   | forward pre-hook call counts; test-side saved-tensor hook bytes; retain grads  | `b1-reentrant`                 | reentrant checkpoint with an input that does not require grad produces no parameter gradients                            | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B2   | exhaustive 16-subset search over 18 budgets; greedy counterexample             | `b2-items-reused`              | each block may be taken twice (unbounded knapsack); path length and optimum both break                                   | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B2   | exhaustive 16-subset search over 18 budgets; greedy counterexample             | `b2-budget-off-by-one`         | plan exceeds the budget by one unit; the exhaustive oracle checks used <= budget                                         | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B3   | storage bytes of the two non-parameter storages; grad_fn is None               | `b3-not-detached`              | packed object keeps the autograd graph (reference cycle rule); grad_fn check catches it                                  | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B3   | storage bytes of the two non-parameter storages; grad_fn is None               | `b3-view-bytes`                | a saved view keeps its whole storage alive; last-view-wins byte count is wrong                                           | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B3   | storage bytes of the two non-parameter storages; grad_fn is None               | `b3-parameters-counted`        | parameter storages are not activations; counting them inflates the inventory                                             | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B4   | exhaustive enumeration of all mode combinations for budgets 0..15              | `b4-budget-overflow`           | plan uses more units than the budget; exhaustive oracle checks feasibility                                               | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B4   | exhaustive enumeration of all mode combinations for budgets 0..15              | `b4-min-units-not-cost`        | returns the smallest plan instead of the cheapest; cost differs from the exhaustive minimum                              | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B4   | exhaustive enumeration of all mode combinations for budgets 0..15              | `b4-infeasible-silent`         | an infeasible budget returns an empty plan instead of raising                                                            | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B5   | independent hook count on the true per-block input; widening first block       | `b5-input-not-advanced`        | every block calibrated on the raw sample; the 8->16 widening block makes the second block fail                           | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B5   | independent hook count on the true per-block input; widening first block       | `b5-grads-left`                | calibration leaves gradients behind, polluting the next training step                                                    | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| B5   | independent hook count on the true per-block input; widening first block       | `b5-mode-ignored`              | checkpoint row measured as retain; saved_bytes differs from the independent hook count                                   | B                | CAUGHT (cpu) / CAUGHT (cuda) |
| C1   | pack-time snapshot after in-place write; storage pointer; CUDA allocated delta | `c1-host-aliases-input`        | .to('cpu') of a CPU tensor is the same storage; an in-place write after pack changes the saved value                     | C                | CAUGHT (cpu) / n/a (cuda)    |
| C1   | pack-time snapshot after in-place write; storage pointer; CUDA allocated delta | `c1-parameters-offloaded`      | parameter (views) copied to host every forward although they stay resident anyway                                        | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C1   | pack-time snapshot after in-place write; storage pointer; CUDA allocated delta | `c1-min-bytes-ignored`         | tiny tensors copied one by one; min_bytes threshold not honored                                                          | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C1   | pack-time snapshot after in-place write; storage pointer; CUDA allocated delta | `c1-keeps-device-reference`    | pack copies to host but keeps the CUDA tensor alive: no memory saved; the CUDA after-forward allocator check catches it  | C                | CAUGHT                       |
| C2   | repeated unpack after modifying the first result; dtype round trips            | `c2-restore-without-copy`      | on CPU the restored tensor is the payload itself; modifying one restore corrupts the next                                | C                | CAUGHT (cpu) / n/a (cuda)    |
| C2   | repeated unpack after modifying the first result; dtype round trips            | `c2-dtype-not-restored`        | restored float64/bool/int64 tensors come back as float32                                                                 | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C2   | repeated unpack after modifying the first result; dtype round trips            | `c2-h2d-not-counted`           | restore transfers not counted; counters disagree with d2h                                                                | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C2   | repeated unpack after modifying the first result; dtype round trips            | `c2-device-not-restored`       | restored tensor stays on the host; CUDA device check and backward fail                                                   | C                | CAUGHT                       |
| C3   | forward hook counts; owner counters; RNG state equality on rejection           | `c3-no-owner-check`            | missing owner discovered mid-forward (AttributeError) after earlier blocks ran and consumed RNG                          | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C3   | forward hook counts; owner counters; RNG state equality on rejection           | `c3-offload-without-hooks`     | offload mode silently retains; gradients match, only the copy counters reveal it                                         | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C3   | forward hook counts; owner counters; RNG state equality on rejection           | `c3-checkpoint-as-retain`      | checkpoint blocks retained; recompute counts reveal it                                                                   | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C4   | test-side full-batch MSE, manual clip min(1, c/norm), manual SGD; norm >> c    | `c4-mean-of-means`             | unequal microbatches weighted equally; 2+3 split differs from the full-batch oracle                                      | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C4   | test-side full-batch MSE, manual clip min(1, c/norm), manual SGD; norm >> c    | `c4-clip-per-microbatch`       | clipping before accumulation finishes; the test forces norm >> max_grad_norm so it is observable                         | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C4   | test-side full-batch MSE, manual clip min(1, c/norm), manual SGD; norm >> c    | `c4-no-clip`                   | max_grad_norm ignored; distinguishable only because the forced norm exceeds it                                           | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C4   | test-side full-batch MSE, manual clip min(1, c/norm), manual SGD; norm >> c    | `c4-zero-grad-each-microbatch` | only the last microbatch's gradient reaches the step                                                                     | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C4   | test-side full-batch MSE, manual clip min(1, c/norm), manual SGD; norm >> c    | `c4-late-owner-check`          | owner rejected only inside run_policy, after zero_grad already wiped the caller's gradients                              | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C5   | lr=0 invariance; independent hook bytes of the offloaded block; CUDA peaks     | `c5-counters-not-reset`        | warmup transfers included in per-step bytes; oracle counts one step                                                      | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C5   | lr=0 invariance; independent hook bytes of the offloaded block; CUDA peaks     | `c5-nonzero-lr`                | parameters drift between policies; lr=0 invariant (identical losses and weights) catches it                              | C                | CAUGHT (cpu) / CAUGHT (cuda) |
| C5   | lr=0 invariance; independent hook bytes of the offloaded block; CUDA peaks     | `c5-peak-not-reset`            | peak carries over from the previous policy; checkpoint no longer shows a lower peak than retain                          | C                | CAUGHT                       |
| R1   | float64 eager reference; gradcheck; N r 8 saved bytes                          | `r1-grad-b-transposed`         | dB computed as transpose(Z) @ dY, shape [r,out], instead of transpose(dY) @ Z, shape [out,r]; autograd rejects the shape | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R1   | float64 eager reference; gradcheck; N r 8 saved bytes                          | `r1-grad-b-unscaled`           | missing alpha/r in dB; gradcheck and the float64 eager oracle catch it                                                   | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R1   | float64 eager reference; gradcheck; N r 8 saved bytes                          | `r1-dx-base-only`              | input gradient ignores the LoRA branch (B != 0 in the test)                                                              | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R1   | float64 eager reference; gradcheck; N r 8 saved bytes                          | `r1-no-input-grad`             | x gets no gradient although it requires one: upstream layers stop learning                                               | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R1   | float64 eager reference; gradcheck; N r 8 saved bytes                          | `r1-also-saves-x`              | x saved 'just in case': numerically perfect, only the N r 8 saved-bytes check catches it                                 | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R2   | float64 eager reference; gradcheck; jacobian of erf GELU; N (h+f+2r) 8 bytes   | `r2-saves-gelu-output`         | GELU output saved anyway: gradients exact, the whole point (N f bytes) lost; byte checks catch it                        | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R2   | float64 eager reference; gradcheck; jacobian of erf GELU; N (h+f+2r) 8 bytes   | `r2-gelu-grad-cdf-only`        | GELU'(u) taken as Phi(u), dropping u phi(u); gradcheck catches it                                                        | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R2   | float64 eager reference; gradcheck; jacobian of erf GELU; N (h+f+2r) 8 bytes   | `r2-tanh-recompute`            | recompute uses the tanh approximation while forward used erf: dA_down off by ~1e-4                                       | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R2   | float64 eager reference; gradcheck; jacobian of erf GELU; N (h+f+2r) 8 bytes   | `r2-dx-missing-lora`           | input gradient drops the up-projection LoRA path                                                                         | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R2   | float64 eager reference; gradcheck; jacobian of erf GELU; N (h+f+2r) 8 bytes   | `r2-grad-a2-uses-u`            | dA_down built from the pre-activation u instead of GELU(u)                                                               | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R3   | float64 dense reference with the same mask; gradcheck; N (d+r) 8 bytes         | `r3-saves-mask`                | mask saved in addition to the seed; only the N (d + r) 8 saved-bytes check catches it                                    | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R3   | float64 dense reference with the same mask; gradcheck; N (d+r) 8 bytes         | `r3-backward-other-seed`       | backward regenerates a different mask than forward used                                                                  | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R3   | float64 dense reference with the same mask; gradcheck; N (d+r) 8 bytes         | `r3-mask-from-global-rng`      | forward mask drawn from the global RNG, so the seed cannot reproduce it in backward                                      | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R3   | float64 dense reference with the same mask; gradcheck; N (d+r) 8 bytes         | `r3-dx-no-inverted-scale`      | dX uses the 0/1 keep mask without 1/(1-p)                                                                                | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R3   | float64 dense reference with the same mask; gradcheck; N (d+r) 8 bytes         | `r3-grad-a-undropped`          | dA uses x instead of the dropped input x * m                                                                             | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R4   | toy with known 4 MiB saved / 32 MiB transient / 64 MiB held / 128 MiB freed    | `r4-grads-accumulate`          | grads accumulate over the timed passes; contract says .grad holds exactly one pass                                       | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R4   | toy with known 4 MiB saved / 32 MiB transient / 64 MiB held / 128 MiB freed    | `r4-parameters-counted`        | saved parameter storage counted as activation memory                                                                     | R                | CAUGHT (cpu) / CAUGHT (cuda) |
| R4   | toy with known 4 MiB saved / 32 MiB transient / 64 MiB held / 128 MiB freed    | `r4-peak-not-reset`            | peak includes a 128 MiB spike freed before the measurement                                                               | R                | CAUGHT                       |
| R4   | toy with known 4 MiB saved / 32 MiB transient / 64 MiB held / 128 MiB freed    | `r4-absolute-peak`             | absolute peak includes a 64 MiB tensor that is not part of the measured pass                                             | R                | CAUGHT                       |
