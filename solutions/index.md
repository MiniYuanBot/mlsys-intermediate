# 参考实现

[课程索引](../course/index.md) · [实验索引](../labs/index.md) · [验证状态](../labs/docs/validation.md) · [编写规范](../labs/docs/authoring.md)

目录布局与 `labs/labN/` 对应，只包含 TODO 所在的文件（`src/`，lab4/lab5 另有 R 阶段的 `real/`），以及每个 lab 的导读 `README.md` 和错误注入表 `faults.json`。测试、benchmark 和 driver 都留在 `labs/`。参考实现用来解释机制，不是论文原系统，也不追求最优性能。

A/B/C/R 四个阶段的全部必修 TODO 都有答案和测试；学生个人反例、报告和开放挑战仍需独立完成。对照答案前，先写下自己的最小失败输入和预期输出。

## 九个 lab

| Lab | 导读                             | 错误注入表                      | TODO 数 | 错误数（其中 CUDA） | 主要实现范围                                                                                |
| --- | -------------------------------- | ------------------------------- | ------: | ------------------: | ------------------------------------------------------------------------------------------- |
| 1   | [lab1/README.md](lab1/README.md) | [faults.json](lab1/faults.json) |      16 |            49（11） | int4 格式、流式残余缓存、PQ 训练/ADC/取回、int4 KV 融合 decode kernel                       |
| 2   | [lab2/README.md](lab2/README.md) | [faults.json](lab2/faults.json) |      22 |            63（31） | int4 格式与 clipping、四条 W4A32 kernel、AWQ 式通道缩放、FP8 编码与 GEMV                    |
| 3   | [lab3/README.md](lab3/README.md) | [faults.json](lab3/faults.json) |      21 |            61（12） | trace 与计时、分桶 DP 与运行时状态、真实 `torch.compile`、融合 RMSNorm、CUDA graph 捕获尺寸 |
| 4   | [lab4/README.md](lab4/README.md) | [faults.json](lab4/faults.json) |      19 |             43（1） | 接受量与成本、删失估计与预算、推测服务模拟、真实接受 trace 的采集与回放                     |
| 5   | [lab5/README.md](lab5/README.md) | [faults.json](lab5/faults.json) |      20 |             64（2） | 缓存身份、事务与所有权、准入与缓存感知路由、prefill 代价拟合与 trace 回放                   |
| 6   | [lab6/README.md](lab6/README.md) | [faults.json](lab6/faults.json) |      21 |             73（5） | LoRA 生命周期、checkpoint 与预算规划、saved-tensor offload、自定义 `autograd.Function`      |
| 7   | [lab7/README.md](lab7/README.md) | [faults.json](lab7/faults.json) |      25 |            62（24） | 三种路由约定、字节缓存与替换策略、INT8 专家执行、pinned 重叠流水线                          |
| 8   | [lab8/README.md](lab8/README.md) | [faults.json](lab8/faults.json) |      20 |             53（0） | 长度历史与组优势、分段账本、裁剪梯度与版本界、真实长度 trace 回放                           |
| 9   | [lab9/README.md](lab9/README.md) | [faults.json](lab9/faults.json) |      22 |             58（8） | roofline 代价模型、max-flow 放置、多模型共置、实测 $\eta$ 与重新规划                        |

合计 186 个 TODO、526 个注入错误；2026-10-18 在 WSL 上全部被测试抓到（见[验证状态](../labs/docs/validation.md)）。

## 运行参考实现

以下命令从 **`labs/` 目录**执行：

```bash
python tools/run.py --lab 4 --impl reference --mode test
python tools/run.py --lab 2 --impl reference --mode bench --device cuda
# 需要手动调试时，创建新副本；目标已存在会拒绝。
python tools/materialize.py --lab 2 --impl reference --to results/lab2-reference-debug
```

`run.py` 每次把整个 lab 复制到临时目录，再用参考 TODO 覆盖，结束后清理副本，运行记录始终保留。它不修改学生实现，不需要先 git commit 或建 worktree。CUDA 运行自动获取 `/tmp/mlsys-gpu.lock`。

## 运行错误注入

```bash
python tools/fault_inject.py --lab N --list                 # 列出错误和每个 TODO 的覆盖数
python tools/fault_inject.py --lab N --device cpu           # 只注入 CPU 错误
python tools/fault_inject.py --lab N --device cuda          # CUDA 错误，需在 WSL/Linux 下跑
python tools/fault_inject.py --lab N --only b2-kv-axis-swap # 只跑指定的错误
python tools/fault_inject.py --lab N --require-coverage     # 有 TODO 没有错误时失败
```

`fault_inject.py` 先按 `run.py --impl reference` 的方式建临时副本，跑一次未修改的参考实现作为基线；基线不通过就中止（退出码 2）。之后每个错误用一份新副本，按 `find`/`replace` 改动参考代码，再运行该阶段的测试。结果分为 CAUGHT（测试失败，通过）、MISSED（测试仍通过，要补测试）、ENV（缺依赖，需在有依赖的机器上重跑）、STALE（`find` 文本对不上，参考实现改过了）和 TIMEOUT。有 MISSED 或 STALE 时退出码为 1。

## oracle 与错误表的约定

- **每个 TODO 都有独立 oracle**：暴力解或小规模穷举、FP64 dense 参考、带推导的手算小例子，或对学生自己的目标函数做有限差分。oracle 不调用被测代码，也不只和写死的输出比对。
- **每个 TODO 至少一个注入错误，核心 TODO 至少两个**，写进 `faults.json`。每条记录有 `id`、`todo`、`file`、`find`、`replace`、`expect_fail_part`、可选的 `device` 与 `count`，以及说明“模拟哪种真实失误、哪个 oracle 应该抓住它”的 `why`。字段定义见 [authoring.md](../labs/docs/authoring.md) §4，示例见 [faults.example.json](../labs/templates/faults.example.json)。
- **每份导读末尾有“oracle 与 fault 对照表”**，逐个 TODO 列出用哪种 oracle、注入了哪些错误、在哪个阶段失败。读答案时先看这张表，弄清楚测试为什么能区分正确实现和“看起来差不多”的实现。

## 怎么用答案

建议每个 TODO 先独立尝试至少 45 分钟，记下最小失败例子再对照。读完答案后关掉文件重新实现一遍，再加一个自己的反例测试。“参考测试通过”不能证明个人实验已经完成。
