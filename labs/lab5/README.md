# Lab 5：前缀缓存，从缓存身份、所有权到多副本缓存感知路由

[讲义 D5](../../course/modules/05-prefix-cache.md) · [评分](grade.md) · [公开来源与借鉴边界](resources.md) · [参考答案导读](../../solutions/lab5/README.md)

> [!important] 你要交付什么
> 用 Rust 实现前缀缓存的身份（vLLM 式链式块哈希、带 namespace 的完整祖先键）、所有权（lease、事务插入、失效、缩容）和单副本的四条不变量（A/B）；再实现准入、全命中规则、多副本缓存感知路由和带事件的回放（C）。R 阶段在 RTX 4060 上测 Qwen3-0.6B 的 prefill 延迟 $T(p,n)$，拟合代价模型，再把 Mooncake 格式的 trace 放进 C 的回放里比较路由策略。A/B/C 的时间全部是 `modeled`；只有 R 阶段本机测得的系数和延迟标为 `measured`。

## 0. 环境

本 lab 的依赖和其他 lab 互不累加。在 `labs/` 下先安装、再检查：

```bash
bash tools/setup_env.sh --lab 5 && bash tools/check_env.sh --lab 5
```

Windows 原生（PowerShell）只做 CPU 部分：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 5
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 5
```

| 阶段 | 需要                                                                    | WSL2 / Linux | Windows 原生 | macOS       |
| ---- | ----------------------------------------------------------------------- | ------------ | ------------ | ----------- |
| A    | Rust                                                                    | 可做         | 可做         | 可做        |
| B    | Rust                                                                    | 可做         | 可做         | 可做        |
| C    | Rust                                                                    | 可做         | 可做         | 可做        |
| R    | Rust；GPU 部分另需 NVIDIA GPU + PyTorch + transformers；模型 Qwen3-0.6B | 可做         | 仅 CPU 部分  | 仅 CPU 部分 |

A/B/C 只需要 Rust。R 阶段分两半：CPU 一半用仓库里提交的 4060 实测数据做 R1 拟合（Python 标准库即可），再用 Rust 回放提交的 Mooncake 格式 trace；GPU 一半加载 Qwen3-0.6B（bf16 约 1.2 GB 显存），在本机重新测 $T(p,n)$ 并重新拟合。模型默认放在 `~/models/Qwen3-0.6B`，可以用 `MLSYS_MODELS` 改目录；不想下载模型就加 `--no-models`（PowerShell 是 `-NoModels`），这时只能做 CPU 一半。缺少依赖时 runner 只打印一行 `MISSING ...` 和对应的 `fix:` 命令，退出码为 3；在 Windows 上请求 `--device cuda`，`fix:` 给出的是一条能直接复制的 WSL 命令。各平台的细节见[环境配置](../setup.md)。

## 文件和实施顺序

| 文件                                        | 学生负责     | 已提供                                                                                               |
| ------------------------------------------- | ------------ | ---------------------------------------------------------------------------------------------------- |
| `src/cache_student.rs`                      | A1–A6、B1–B5 | `Namespace`、`Key`、`Entry`、`Cache::new`、`Lease::len`、`InvariantError`                            |
| `src/cluster_student.rs`                    | C1–C6        | `AdmissionPolicy`、`RoutePolicy`、`ReplicaView`、`Request`、`ClusterConfig`、`ResultRow`、`Metrics`  |
| `src/trace_student.rs`                      | R2、R3       | `TraceConfig`、`TraceError`、`load_mooncake`、`load_cost_model`                                      |
| `real/fit_student.py`                       | R1           | `features`、`predict`                                                                                |
| `src/hash.rs`                               | —            | A1 用的块哈希原语 `hash_block`（64 位 FNV-1a，确定性、非密码学）                                     |
| `src/cost.rs`                               | —            | `CostModel::prefill_ms` 与 `FullHitRule`                                                             |
| `src/json.rs`                               | —            | 最小 JSON 读取器（无外部 crate，离线可编译）                                                         |
| `real/measure.py`                           | —            | GPU 上测 $T(p,n)$：快照、crop 回滚、300 ms 预热、`stable` 列、正确性闸门                             |
| `real/fit.py`                               | —            | 调用 R1，写出 `bench R` 读取的拟合文件                                                               |
| `real/make_trace.py`                        | —            | 生成 Mooncake 格式的合成 trace（标准库）                                                             |
| `real/data/prefill-qwen3-0.6b-rtx4060.json` | —            | 参考测量：RTX 4060 Laptop，驱动 566.24，torch 2.14.1+cu126，transformers 5.18.0，2026-10-18，49 个点 |
| `real/data/cost-qwen3-0.6b-rtx4060.json`    | —            | 参考拟合（同一设备、驱动、日期），CPU 学习者直接用它                                                 |
| `real/data/synthetic-mooncake.jsonl`        | —            | 1200 个请求的合成 trace（124 KB），`make_trace.py --seed 2025` 生成                                  |
| `tests/exercises.rs`、`real/test_real.py`   | —            | 21 个 Rust 测试、R1 的 Python 检查；每项打印 `PASS`/`FAIL`                                           |
| `src/main.rs`、`exercise.py`                | —            | benchmark 与阶段入口                                                                                 |

以下命令都在 **`mlsys-intermediate/labs/`** 运行。按 A、B、C、R 的顺序完成；每个阶段的测试只调用本阶段和更早阶段的函数。

```bash
python tools/run.py --lab 5 --impl student --mode test --part A
python tools/run.py --lab 5 --impl student --mode test --part B
python tools/run.py --lab 5 --impl student --mode test --part C
python tools/run.py --lab 5 --impl student --mode test --part R                 # CPU：R1 拟合检查 + R2/R3 + 提交 trace 的回放
python tools/run.py --lab 5 --impl student --mode test --part R --device cuda   # WSL：再加一次短的真实测量
python tools/run.py --lab 5 --impl student --mode bench --part all
python tools/run.py --lab 5 --impl student --mode bench --part R --device cuda  # 测 T(p,n)、用你的 R1 拟合、再回放
```

直接调试一个测试：`cargo test --offline --release --manifest-path lab5/Cargo.toml --test exercises c_replay_hand_timeline -- --nocapture`。每个测试把每项检查打印成一行 `PASS 测试名/检查名` 或 `FAIL 测试名/检查名: 期望 vs 实际`，跑完所有检查后才判失败，一次就能看到全部问题；未实现的 TODO 会显示为 `panicked: not yet implemented: TODO A1` 这样的 FAIL 行。请保留所有 BEGIN/END TODO 标记；标记前的注释就是契约（输入、输出、不变量、边界、对应测试）。可以加 helper、测试和 benchmark，不要改现有测试的期望值来迁就实现。

## A 基础：缓存身份与单副本结构（5.5h）

A 阶段的测试直接构造缓存 fixture，不调用 B/C 的函数。

### A1 `block_hashes`：vLLM 式链式块哈希（1.5h）

输入 `HashRequest{tokens, lora, mm, cache_salt}` 和块大小 `block`，输出每个**满块**的哈希，不满的尾块不出哈希。第 $i$ 块覆盖 $[s,e)=[iB,(i+1)B)$，哈希为 `hash_block(parent, tokens[s..e], extra_i)`，第 0 块的 parent 是 `NONE_HASH`。`extra_i` 依次放：

1. `Lora{name, path}`，**每一块都放**。vLLM v0.32.0 用 (name, path) 而不是 adapter ID，是为了“把同一个名字指向另一份权重”时不会误复用；
2. 每个与本块重叠的多模态条目一个 `Mm{identifier, offset}`，重叠判定是 `item.start < e && item.start + item.len > s`，`offset = item.start - s`（可以为负）；
3. `Salt(salt)`，**只放在第 0 块**，后面的块靠 parent 链把它带下去。

采样参数（temperature、top-p、seed）故意不在键里：它们只作用于 logits，不改变 prefill 的 KV。测试 `a_hash_chain_extra_keys` 用提供的 `hash_block` 逐块重算期望值，另外检查链式性质：改第 1 块不影响第 0 块的哈希，改第 0 块会改变之后所有块；同名不同路径的 LoRA 每块都不同；只在边界相接、并不重叠的多模态条目不进入下一块的键。

### A2 `block_keys`：带 namespace 的完整祖先键（0.5h）

`block=2, tokens=[1,2,3,4,5]` 生成前缀 `[1,2]` 与 `[1,2,3,4]` 两个键，尾部 `[5]` 不登记。`Namespace` 的五个字段（model、adapter、tokenizer、tenant、position_offset）任意一个不同，键就不同。`tenant` 相当于 vLLM 的 `cache_salt`；`position_offset` 代表“同一段内容放在别的位置”，只有 EPIC、CacheBlend 这类位置无关缓存才可能复用它，精确前缀引擎的前缀总从位置 0 开始。

完整祖先键不会碰撞，但占用的空间是 $O(T^2/B)$；A benchmark 的 12 行同时打印 Vec 键的 token 载荷和链式哈希的 8 字节键，画出两者随长度的增长。这里只计结构载荷，不是运行时内存峰值。

### A3 `Cache::matched_tokens`：遇到第一个缺口就停（0.5h）

返回连续命中的满块所覆盖的 token 数。父块缺失而子块还在时返回缺口之前的长度：缺口后面的子块已经不可达，与 vLLM 链式查找“一个块缺失意味着后面全缺失”一致。函数只读，不改 refs、last、hits 和 clock，测试会比较调用前后的快照。

### A4 `Cache::victim`：只在合法候选里选（1h）

合法候选是 `refs == 0`、不在 `protected` 里、并且是叶子（同 namespace 中没有别的常驻键以它为前缀）的节点。`Lru` 取最小的 `(last, key)`，`Frequency` 取最小的 `(hits, last, key)`，后者就是 SGLang `evict_policy.py` 里 `LFUStrategy` 的排序。只淘汰叶子是 SGLang 的语义；vLLM 的单一 LRU 空闲队列可以先淘汰父块，留下的子块不可达、白占空间（见讲义 §4）。测试含 300 个随机缓存，与按定义筛选再排序的 oracle 比较；还专门检查“另一个租户的更长键不算后代”。

### A5 `Cache::validate`：返回错误枚举，不返回字符串（1h）

先检查 `nodes.len() <= capacity`，再按键序逐个节点依次检查：长度大于 0 且是 `block` 的倍数、父键（去掉最后一块）存在于**同一 namespace**、$\mathrm{refs(child)}\le\mathrm{refs(parent)}$、`last <= clock`。返回第一个违反项，对应 `InvariantError` 的五个变体：`OverCapacity{nodes, capacity}`、`Misaligned(key)`、`Orphan(key)`、`PinAboveParent(key)`、`FutureClock(key)`。测试按变体和携带的键比较，不依赖任何描述字符串；另有一例同时含两个错误，用来检查“按键序取第一个”。

### A6 `Cache::stats`：块数与引用数分开记（0.5h）

`pinned_blocks` 按节点计一次，`references` 累加所有 lease 的引用；同一块被 3 个 lease 引用，pinned 块数是 1，引用数是 3。`key_token_bytes` 每个存储的 token 记 4 字节，不含容器开销和 KV 张量；`max_depth` 以块为单位。

### A 验收（0.5h）

六个 A 测试全部通过。在报告里手算三种状态：namespace 改变、父块缺失、同一块被两次 pin，并说明为什么生产系统还需要把模型 revision、dtype、chat template 版本放进身份，而不是只看模型名。

## B 核心：事务与所有权（7h）

### B1 `Cache::insert`：全有或全无（2h）

在副本上规划：数出缺失的键，把整条路径放进 `protected`，反复淘汰合法候选直到 `len + missing <= capacity`，然后 clock 加 1、补齐缺失键（refs 0、last = 新 clock、hits 0），最后一次性赋值发布。已存在的键不动，插入不算读。路径本身超过容量或被 pin 卡住时返回 false，节点、元数据和 clock 都不变；空路径直接成功且不改 clock。测试里有三个反例：能先淘汰一个叶子、之后被 pin 卡住；路径自己的祖先恰好是 LRU 叶子（必须受保护）；以及 acquire 刷新 LRU 次序后的淘汰顺序。

整表 clone 是为了让回滚一眼可证；改用 undo log 也可以，但要通过同样的测试，并单独记录管理开销。

### B2–B3 `acquire` / `release`：lease 的生命周期（1.5h）

`acquire` 先把 clock 加 1，再沿连续命中链逐块 `refs += 1`、`last = clock`、`hits += 1`，遇到第一个缺失就停；完全未命中也推进 clock，返回空 lease。`release` 消费 lease，从深到浅减引用；引用已为 0 或键不存在时 panic（这是程序错误，不是可恢复状态）；引用归零的块继续留在缓存里供以后复用。Rust 的所有权保证同一个 lease 不会被释放两次，但不保证它被还给创建它的那个 cache，这一点仍是调用方契约（挑战题可以加 cache ID 或 RAII guard）。

### B4 `Cache::invalidate`：按 namespace 失效（1h）

adapter 更新或租户清理时整体删除一个 namespace。任何目标节点被 pin 就返回 `None` 且什么都不改（clock 也不动）；成功时删除、clock 加 1、返回删除数。测试里另一个租户有完全相同的 token，它的节点必须全部保留。

### B5 `Cache::resize`：运行时改容量（1h）

扩容只改 capacity，节点和 clock 都不动。缩容在副本上淘汰合法候选直到 `len <= target`，成功后 clock 加 1；被 pin 卡住时容量、节点、clock 全部回滚。

### B 测试与 benchmark（1.5h）

运行 5 个 B 测试和 B benchmark（4 种容量 $\times$ LRU/Frequency 共 8 行）。结合“稳定前缀 + 变化尾部”的负载解释两种策略为什么在小容量下完全相同，保留这个负结果。

## C 前沿：准入、全命中规则与多副本缓存感知路由（10h）

C 阶段把 A/B 的单副本缓存放进 $K$ 个副本的集群：请求到达时由路由器选副本，副本内 FIFO 排队、`slots` 个抽象服务位并行，prefill 时间由 `CostModel` 给出，完成时才发布 KV。这是 Preble、SGLang 路由器一类机制的可执行简化版：路由器看的是副本缓存的精确状态（相当于 llm-d 订阅 KV 事件），没有批处理干扰、没有网络、没有 decode 与 prefill 的争抢。

### C1 `Admission::observe`：有界 LRU 历史上的两次命中准入（1.5h）

Always 准入所有满块；TwoHit（cache-on-second-hit，来自 Akamai CDN，Maggs 与 Sitaraman，SIGCOMM CCR 2015）只准入被观察到至少两次的块。步骤：clock 加 1，本请求每个满块键计数加 1、`last = clock`；历史超过 `capacity` 时忘掉 `(last, key)` 最小的条目，按**最近观察**而不是首次插入；TwoHit 返回从头开始、计数不小于 2 的最长连续链的 token 数。历史容量是契约的一部分：`ClusterConfig::history_capacity` 显式传入，参考实现不藏任何默认倍数。测试 `c_admission_lru_history` 用容量 2 的手算序列区分 LRU 与 FIFO，并检查租户隔离和 capacity=0。

### C2 `reusable_tokens`：全命中也要重算（0.5h）

前向至少要输入一个 token 才能得到下一个 token 的 logits，所以完整命中也不能跳过全部计算：

- `Sglang`：$\min(m, t-1)$，按 token 粒度（page_size 为 1；page_size 大于 1 时 SGLang 也按页向下取整，与 vLLM 规则一致）；
- `Vllm`：$\lfloor \min(m, t-1)/B\rfloor\cdot B$，只复用整块。

其中 $m$ 是已缓存 token 数，$t$ 是 prompt 长度，$B$ 是以 token 计的块大小。32 个 token、$B=16$、全部命中时，SGLang 复用 31 个，vLLM 只复用 16 个、重算整整一块。测试对 $t\le40$、$B\le6$ 的所有组合与暴力枚举比较。

### C3 `Router::route`：四种路由策略（2h）

| 策略                      | 规则                                                                                                                                                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `RoundRobin`              | `next % K`，然后 `next += 1`（唯一会改自身状态的策略）                                                                                                                                                                   |
| `LeastLoad`               | 最小的 `(load, index)`                                                                                                                                                                                                   |
| `PrefixAffinity`          | 最大的 `matched`，并列时取最小 `(load, index)`；完全不看负载                                                                                                                                                             |
| `Hybrid{abs, rel, cache}` | 设最高、最低负载为 $h$、$l$。若 $h-l>\mathrm{abs}$ 且 $h>\mathrm{rel}\cdot l$，集群失衡，走 LeastLoad；否则取 PrefixAffinity 的选择 $c$，若 $\mathrm{matched}_c/\mathrm{prompt}>\mathrm{cache}$ 就去 $c$，否则 LeastLoad |

Hybrid 是 SGLang 路由器的规则（两个失衡条件同时成立才算失衡）。区别在于 SGLang 路由器在路由时就把请求文本插进自己按字符维护的近似树，而本 lab 看的是副本缓存的真实内容。测试有 9 个手算例子和 2000 个随机视图，后者与测试文件里另写的一份排序式 oracle 比较。

### C4 `replay`：多副本事件回放（3.5h）

每个事件时刻按顺序做四件事：

1. 处理 `finish <= now` 的完成，按 `(finish, replica, id)` 排序：先 release lease，再把冻结的准入长度插入该副本缓存，输出一行；
2. 按 `(arrival, id)` 处理已到达的请求：为每个副本构造 `ReplicaView`（`load` = 排队 + 运行，`matched` = 只读查询的命中 token 数，上限是 prompt 长度），用**同一个** Router 选副本并入队；同一时刻后到的请求看得到前面请求造成的负载变化；
3. 按副本序号，在有空闲服务位时启动队首：`reused = reusable_tokens(matched, prompt, block * unit_tokens, rule)`，acquire，冻结准入长度，`first_token = now + prefill_ms(reused, prompt - reused)`，`finish = first_token + decode_ms`；
4. 跳到下一个到达或完成时刻。

关键反例是两个同时到达、前缀相同的请求都没命中而并发执行，任何一个完成之前，另一个都不能“免费复用”尚未算完的 KV。`c_replay_hand_timeline` 的手算：$B=2$、两个服务位、每 token 1 ms、decode 1 ms，前两个请求在 t=0 都未命中、t=3 完成；第三个在 t=4 命中 2 个 token，SGLang 规则复用 1 个、t=6 完成，vLLM 规则复用 0 个、t=7 完成。`c_replay_routing_oracle` 手算两组路由时间线（亲和性让会话回到持有其历史的副本；热点前缀下 Hybrid 在负载差超过阈值时分流）。`c_replay_conservation_and_isolation` 在 16 种配置下检查请求守恒、lease 全部释放、缓存合法、每副本 FIFO 与服务位上限、prefill 时间与代价模型一致，并检查**跨租户隔离**：一行的命中长度不能超过“同一租户、同一副本、在它开始前已完成的请求”能提供的最长公共前缀，否则就是跨租户复用或复用了未完成的 KV。

`unit_tokens` 让同一套代码处理两种 trace：合成 trace 一个单元就是一个 token；Mooncake trace 一个单元是一个 512-token 块的哈希 ID。

### C5–C6 `summarize`、`duplicated_blocks`：指标（1.5h）

命中率按 token 加权（$\sum \mathrm{reused}/\sum \mathrm{total}$，2/2 与 0/8 合起来是 0.2，不是 0.5），TTFT = `first_token - arrival`（均值和最近秩 p95），排队 = `start - arrival`，插入失败只数容量或 pin 导致的失败（TwoHit 少缓存不算失败）。失衡度 = 各副本忙碌时间的最大值除以**全部** $K$ 个副本的均值，闲置副本也算在内。`duplicated_blocks` 统计同一个键在多个副本上的额外副本数，它衡量“为命中率付出的容量代价”。

### C 实验（1h）

`bench C` 输出 4 种路由 $\times$ 2 种准入 $\times$ 2 种到达率共 16 行（4 副本、每副本 256 个 32-token 块、SGLang 规则、代价系数取自 4060 拟合并固定）。画命中率、p95 TTFT、失衡度和重复块数，回答：纯亲和性的命中率最高，它的代价落在哪个指标上？Hybrid 在什么到达率下与纯亲和性相同、什么时候开始分流？TwoHit 在这个没有扫描的负载上为什么全面变差？保留负结果。默认负载是确定性的，重复运行同一输入不能当作独立样本；改变到达率和会话长度做敏感性实验。

## R 真实系统：在 4060 上测代价，回放 Mooncake 格式 trace（5.5h）

### R1 `fit_cost_model`：拟合 $T(p,n)$（2h，含一次约 1 分钟的 GPU 测量）

先读 `real/measure.py`，弄清计时区间包含什么：bf16、sdpa、batch 1；一个 `DynamicCache` 预填到最大长度，每个 $p$ 用 `copy.deepcopy` 加 `crop(-k)` 得到快照（transformers 5.18 中负参数表示移除的 token 数）；被计时的只有一次 `model(ids[p:p+n], past_key_values=snap, logits_to_keep=1)`，前后各一个 CUDA event；每次调用后 `crop(-n)` 回滚（不计时）。每组配置先用同一个调用让 GPU 连续忙 300 ms，再做 3 次预热和 10 次计时，输出中位数、p25、p75 和 `stable`（$p_{75}\le1.5\,p_{25}$），不稳定的行最多重测两次。正确性闸门比较缓存路径和完整前向的最后一个位置的 logits。

模型是

$$
T(p,n)=\max\bigl(T_0,\ a+bn+cn\,(p+n/2)+dp\bigr),
$$

$T_0$ 是 eager 前向的 launch 下限，$bn$ 是线性层，$cn(p+n/2)$ 是新 token 对缓存前缀和彼此（因果的一半）的 attention，$dp$ 是每次调用对缓存 KV 的访存（HF `DynamicCache` 用 `torch.cat` 追加，还会整份拷贝一次；`measure.py` 单独测了这部分）。契约：$T_0$ 取最小测量值；只用 $t\ge1.3\,T_0$ 的点做普通最小二乘（少于 4 个点抛 `ValueError`）；先按列缩放再解；返回系数、使用点数、**全部点**上的最大相对误差和 $p^*=b/c$。测试用已知系数的合成数据检查精确恢复、下限点不进回归、含噪数据上残差与每一列正交，以及在你自己的目标函数上做有限差分。

参考拟合（RTX 4060 Laptop，2026-10-18）：$T_0=23.2$ ms，$b=66.7\ \mu\text{s/token}$，$c=16.2\ \text{ns/token}^2$，$d=5.7\ \mu\text{s}$ 每个缓存 token，$p^*\approx4.1$k，与讲义用 FLOPs 估的 3.8k 接近；49 个点上的最大相对误差 24.5%，最差的点在下限过渡区。HF eager 前向在小输入上被 CPU launch 卡住，引擎用 CUDA graph 可以把 $T_0$ 压低很多，这一点要写进报告的口径说明。

### R2 `parse_mooncake_line`：trace 适配器（1h）

Mooncake FAST'25 trace 每行是 `{"timestamp", "input_length", "output_length", "hash_ids"}`，`hash_ids` 是重新编号的**链式** 512-token 块哈希，同一位置 ID 相同就代表前缀相同，正好可以直接当作缓存键的单元。契约：解析失败或不是对象返回 `Json`；字段缺失返回 `Missing(name)`；不是有限、非负、整数返回 `NotInteger(name)`；`input_length == 0` 或 `hash_ids` 为空返回 `Empty`；ID 超出 u32 返回 `IdOverflow`。`arrival = timestamp * time_scale`，`prompt_tokens = input_length`（最后一块可以不满，真实 trace 里两者也确实不总是一致），`decode_ms = output_length * decode_ms_per_token`。同一请求里重复的 ID（如 `[7, 7]`）是合法的，完整祖先键 `[7]` 与 `[7, 7]` 不同。

真实 trace 在 [Mooncake 仓库的 FAST25-release 目录](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release)：conversation（12,031 个请求，平均输入 12,035 token）、toolagent（23,608 个，平均 8,596）和 synthetic。仓库许可证是 Apache-2.0，README 没有给 trace 文件单独声明许可证。本 lab 不下载它们；提交的 `synthetic-mooncake.jsonl` 由 `make_trace.py` 生成，模式相同（共享系统提示、多轮会话、工具调用、少量原样重发），长度控制在 8k 以内，正好落在 R1 的测量范围里。下载了真实 trace 的话，用 `MOONCAKE_TRACE=/path/to/toolagent_trace.jsonl python tools/run.py --lab 5 --mode bench --part R` 回放；它的平均长度超出 4060 上测得的范围，结果属于外推，要在报告里写明。

### R3 `compute_saved`：回放指标（1h）

token 命中率和“省下的 prefill 时间”不是一回事。对每行算有缓存的 $T(r, t-r)$ 和无缓存的 $T(0,t)$，返回 token 命中率、$1-\sum T(r,t-r)/\sum T(0,t)$ 和总 prefill 时间。有 attention 项时，复用的前缀仍要被新 token 读一遍；有下限时，小的节省被 $T_0$ 吞掉；有 $dp$ 项时，复用本身也有代价。测试用三个手算例子分别验证这三点。

### R 实验与必交结果（1.5h）

1. CPU：`bench R` 在提交的 trace 和参考拟合上输出 4 种路由 $\times$ 2 种全命中规则 $\times$ 2/4 副本共 16 行，标签是 `modeled_with_measured_cost`。
2. GPU（WSL）：`bench R --device cuda` 重新测量、用你的 R1 拟合、再回放。提交 `results/lab5-real/prefill-*.json`、`cost-*.json`，以及与参考拟合的逐点对比（`fit.py` 写入了 `residuals`）。
3. 回答：哪种路由在 token 命中率和省下的时间上排名不同？两种全命中规则的差别有多大，为什么在这份 trace 上很小（提示：只有原样重发的请求才是完整命中）？把 `time_scale` 调小 2 倍、4 倍后，Hybrid 相对纯亲和性的优势怎样变化？

## 报告与验收（3h）

提交 4 个学生文件、`test --part all` 的完整日志、A12/B8/C16/R16 行 CSV、R 的测量与拟合 JSON、四幅图（A 键载荷增长、C 的命中率与 p95 TTFT、R 的拟合残差、R 的路由对比），以及一份不超过 5 页的报告。报告必须包含：in-flight miss 反例、全命中规则的手算、一个跨租户隔离的反例、纯亲和性造成热点的证据、至少一个负结果、modeled 与 measured 的区分，以及实际用时。按 [grade.md](grade.md) 验收；不要求某个策略一定胜出。

## 可选挑战

以下挑战另计时间，选一项即可：

- 用 radix 压缩或链式哈希替换完整前缀键，保留当前所有测试；报告 CPU 管理开销。
- 给路由器加“近似视图”模式：路由时就把请求前缀插进路由器自己的树（SGLang 路由器的做法），比较与精确视图在突发同前缀请求下的重复块数。
- 实现 vLLM 式的淘汰（单一 LRU 空闲队列、尾块先走、允许父块先于子块被淘汰），把 `Orphan` 改成“不可达的浪费空间”计数，量化它在 C 负载上浪费了多少容量。
- 实现 Preble 的 E2 规则（缺失长度小于命中长度时利用，否则按负载代价探索），与 Hybrid 在同一 trace 上比较。
- 在 R1 之外测一次 KV 的 CPU 到 GPU 拷贝带宽（pinned memory），给 Qwen3-0.6B 与 1.7B 分别算“载入还是重算”的分界点。

## 资料汇总

| 资料                                                                                                        | 类型        | 指定观察                                         | 时间       |
| ----------------------------------------------------------------------------------------------------------- | ----------- | ------------------------------------------------ | ---------- |
| [讲义 D5](../../course/modules/05-prefix-cache.md)                                                          | 讲义        | §2 身份表、§4 两种淘汰模型、§5 代价模型、§7 路由 | 5h（另计） |
| [vLLM 前缀缓存设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)                              | 文档 / 图解 | Example Time 1–6 与空闲队列图，对照 A1、A4       | 20min      |
| [SGLang 博客（2024-01-17）](https://www.lmsys.org/blog/2024-01-17-sglang/)                                  | 图解        | Figure 4 的九步 radix 树变化，对照 A4、B2/B3     | 15min      |
| [Mooncake FAST25-release README](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md) | 数据说明    | `hash_ids` 的定义与三份 trace 的规模，对照 R2    | 10min      |
| [公开来源与借鉴边界](resources.md)                                                                          | 来源表      | 每个 TODO 对应的上游文件与简化边界               | 20min      |
| [评分标准](grade.md)、[参考答案导读](../../solutions/lab5/README.md)                                        | 验收        | 必交证据与 oracle/fault 对照                     | 10min      |
