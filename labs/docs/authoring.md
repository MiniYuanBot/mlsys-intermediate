# Lab 与讲义编写规范

[实验索引](../index.md) · [公共工具](../tools/README.md) · [格式规范](formatting.md) · [工作量统计](workload.md) · [验证记录](validation.md) · [讲义模板](../../course/templates/lecture.md)

这份规范给重写讲义、lab 和参考答案的人用。讲义骨架见[讲义模板](../../course/templates/lecture.md)，本页讲 lab 的结构、TODO 标记、测试强度、测量方法和交稿清单。所有命令都从 `labs/` 执行。

## 1. 写作规则

讲义、lab README、`grade.md`、`resources.md` 和 solutions 文档都按下面的规则写。

- 数学一律用标准 LaTeX：行内写 `$...$`，行间写 `$$...$$`。符号用 `\times`、`\le`、`\approx`、`\to`、`\alpha`、`\sqrt{n}`、`\sum`、`\ell`、`^\top` 这类命令。
- 不用 Unicode 数学字符，比如乘号、不等号、约等号、希腊字母、上下标数字、Unicode 减号；也不用 `\(...\)` 和 `\[...\]` 作定界符。
- 变量、复杂度、下标都放进公式：写 $T_d$、$O(n^2)$、$x_i$。代码 span 里只放代码，复杂度和变量名不加反引号。
- 表格中的公式不用裸竖线，写 `\lvert x\rvert` 或 `\Vert x\Vert`，否则会把表格切断。
- 散文里的流程箭头（读入 → 量化 → 写回）可以保留；公式里一律写 `\to`。
- ASCII 图和 Mermaid 图只用英文标签，中文解释写在图下方。
- 中文初稿完成后用 humanizer-zh 润色，不改事实、公式、代码和链接。
- 交稿前运行检查：

```bash
python tools/math_check.py --paths ../course/modules/0N-xxx.md labN ../solutions/labN
python tools/math_check.py --render --paths labN   # 需要 node；WSL 或 Linux 下运行
python tools/check_links.py
```

[math_check.py](../tools/math_check.py) 的错误码如下：

| 代码 | 含义                                                           | 怎么改                                           |
| ---- | -------------------------------------------------------------- | ------------------------------------------------ |
| M001 | 正文或公式里有 Unicode 数学字符                                | 换成 LaTeX 命令，放进 `$...$`                    |
| M002 | 用了 `\(` `\)` `\[` `\]` 定界符                                | 改成 `$...$` 或 `$$...$$`                        |
| M003 | 段落内 `$` 或 `$$` 不配对                                      | 补齐；字面美元符号写 `\$`                        |
| M004 | 行内公式两端有空格                                             | `$x$`，不是 `$ x $`                              |
| M005 | 公式里出现箭头字符                                             | `\to`、`\gets`、`\Rightarrow`、`\leftrightarrow` |
| M006 | 表格行里的公式含裸 `\|`                                        | `\lvert x\rvert`、`\vert`、`\Vert`               |
| M010 | ASCII 图或 Mermaid 图里有中文                                  | 标签改成英文，解释移到图下                       |
| M020 | KaTeX 解析失败（只在 `--render` 时检查）                       | 按报错修公式                                     |
| W001 | 公式外出现疑似公式的裸文本，如没进公式的复杂度、下标（只警告） | 放进 `$...$`                                     |
| W002 | 公式写成了代码 span（只警告）                                  | 去掉反引号，放进 `$...$`                         |
| W020 | KaTeX strict 模式警告，如公式里直接写中文（只警告）            | 用 `\text{...}` 包起来                           |

专有名词里必须出现希腊字母时，在该行末尾加 `<!-- math-ok -->`。KaTeX 版本锁在 [formatter-versions.env](../tools/formatter-versions.env) 的 `KATEX=` 一行；首次 `--render` 会把它装进 `~/.cache/mlsys/katex-<版本>`，不做全局安装。没有 node 的机器会打印 `SKIP render` 并只给扫描结果。

## 2. lab 目录与 README 顺序

一个 lab 的目录：

```text
labs/labN/
  README.md        task statement, stage order, commands
  grade.md         scoring table and required evidence
  resources.md     upstream sources and what was simplified
  exercise.py      the only driver run.py calls (--mode --device --part)
  src/             student stubs between BEGIN/END TODO markers
  test.* bench.*   tests with independent oracles; benchmarks
solutions/labN/
  README.md        reference walkthrough + oracle/fault table
  src/             same file names as labs/labN/src, overlaid by run.py
  faults.json      fault list for tools/fault_inject.py
```

README 的二级标题按下面的顺序排，没有的阶段写一句“本 lab 没有 R 阶段，原因是……”，不要直接删掉标题：

1. `## 0. 环境`：给出 `bash tools/setup_env.sh --lab N && bash tools/check_env.sh --lab N`，再附一张阶段与平台对照表（WSL/Linux、原生 Windows、macOS 各能跑哪些阶段，CUDA 阶段只在 WSL/Linux）。
2. `## 文件和实施顺序`：文件表 `| 文件 | 学生负责 | 已提供 |`，再列各阶段的 `run.py` 命令。
3. `## A 基础：……（Nh）`：每个 TODO 一个 `### A1 函数名（Nh）`，写输入、输出、不变量、边界、对应测试。
4. `## B 核心：……（Nh）`
5. `## C 前沿：……（Nh）`：论文机制的可执行简化版，写清楚简化边界。
6. `## R 真实系统：……（Nh）`：真实模型、真实 GPU 计时或真实 trace。缺依赖时退出码为 3，并打印 `fix:`。
7. `## 报告与验收（Nh）`：必交图表、表格和原始记录。
8. `## 可选挑战`
9. `## 资料汇总`：一张表 `| 资料 | 类型 | 指定观察 | 时间 |`。

### grade.md

满分 100，R 阶段必须有分值。表头固定：

| 项         | 分值 | 自动或人工证据                                                       |
| ---------- | ---: | -------------------------------------------------------------------- |
| A1/A2 ……   |   12 | 测试名 + 检查的性质                                                  |
| B ……       |   25 | ……                                                                   |
| C ……       |   20 | ……                                                                   |
| R 真实系统 |   15 | 原始记录 JSON、测量方法（见第 5 节）                                 |
| 实验与报告 |   20 | 图、负结果、与 baseline 的同口径对比                                 |
| 环境证据   |    — | 附 `bash tools/check_env.sh --lab N` 的完整输出；缺失则 R 阶段不给分 |

表后写“必交命令”和“必查错误”两段，格式沿用现有各 lab。

### resources.md

| 资源（论文 / 代码链接） | 具体阅读位置（文件、函数、图号） | 转化为本 lab 的任务 | 简化边界 |
| ----------------------- | -------------------------------- | ------------------- | -------- |

开头写检索与核验日期。上游代码只链接到具体文件或函数，不大段复制。

### solutions/labN/README.md

1. 导航行：任务、评分、来源。
2. 每个阶段一段：实现要点、为什么这样做、常见错误。
3. 运行命令（参考实现的 test 与 bench）。
4. **oracle 与 fault 对照表**，可以手写，也可以从 `python tools/fault_inject.py --lab N --json …` 的输出整理：

| TODO | 独立 oracle                           | fault id                  | 注入的错误        | 在哪个 part 失败 | 状态   |
| ---- | ------------------------------------- | ------------------------- | ----------------- | ---------------- | ------ |
| A1   | 逐元素重算 min/max/scale 的 FP64 版本 | `a1-scale-denominator-16` | scale 分母写成 16 | A                | CAUGHT |

## 3. TODO 标记与学生骨架

- 每个待实现区域用一对注释包起来：`BEGIN TODO A1` 和 `END TODO A1`，注释符号按语言来（`//` 或 `#`）。编号格式是阶段字母加数字，阶段字母只有 A、B、C、R，同一 lab 内编号唯一。[workload_audit.py](../tools/workload_audit.py) 会检查配对和编号。
- 标记前的注释写清输入、输出、不变量和边界，不写实现步骤。
- 学生骨架里只放一行占位，参考答案在 `solutions/labN/` 的同名文件中：

| 语言          | 学生骨架里的占位                                                                  |
| ------------- | --------------------------------------------------------------------------------- |
| Rust          | `todo!("TODO A1")`                                                                |
| C++           | `throw std::logic_error("Complete A1");`                                          |
| CUDA 设备函数 | `asm("trap;");`，有返回值的再补一行 `return 0;`；host 端的 launch 包装用 C++ 写法 |
| Python        | `raise NotImplementedError("A1")`                                                 |

- 学生骨架必须能编译。跑 A 阶段时应该在第一个未完成的 TODO 处给出可读的失败信息，而不是编译错误或段错误。

## 4. 测试强度

测试要能区分正确实现和“看起来差不多”的实现。每个 lab 交付前必须满足下面几条。

1. **每个 TODO 都有独立 oracle**。可选的做法：
   - 暴力解或小规模穷举（例如对 $\le 5$ 个 GPU、$\le 8$ 层枚举所有放置）；
   - FP64 dense 参考（例如先完整反量化，再用 double 算 attention）；
   - 闭式小例子，手算结果写在注释里并说明推导；
   - 对学生自己的目标函数做有限差分，而不是对另一份手写的闭式梯度做差分：

$$
\frac{\partial L}{\partial \theta_i}\approx\frac{L(\theta+h e_i)-L(\theta-h e_i)}{2h}
$$

2. **不复用被测代码**。oracle 不能调用学生实现的函数，也不能只和写死的输出比对。写死的数值只能作为补充，并且要注明来源。
3. **边界和并列情况**：空输入、长度为 1、尾组不满、全部相同的值、得分并列时的确定性规则、预算刚好等于需求、非法输入不改变状态。
4. **能检查到语义错误**：测试数据要让轴交换、符号取反、漏更新、off-by-one 等错误产生可观测的差别。比如对称数据测不出轴交换，要专门构造非对称数据。
5. **逐项报告，不在第一个失败处停下**：每个检查打印一行 `PASS <名称>` 或 `FAIL <名称>: <期望> vs <实际>`，一个阶段跑完后汇总，有失败就以非零码退出。这样学生一次就能看到所有没过的检查。
6. **退出码约定**：0 表示通过；3 只留给缺依赖（`platform_tools.MissingDependency`，打印一行原因加 `fix:`）；其他非零码表示测试失败。
7. **错误注入**：每个 TODO 至少一个 fault，核心 TODO 至少两个，写进 `solutions/labN/faults.json`。`fault_inject` 必须报告 100% CAUGHT。

### faults.json 格式

文件是一个 JSON 数组，示例见 [faults.example.json](../templates/faults.example.json)。

| 字段               | 必填 | 含义                                                                      |
| ------------------ | ---- | ------------------------------------------------------------------------- |
| `id`               | 是   | 唯一名称，建议写成 `<todo>-<错误>`，如 `b2-kv-axis-swap`                  |
| `todo`             | 是   | 对应的 TODO 编号，如 `B2`                                                 |
| `file`             | 是   | 相对 lab 目录的路径，与 solution 覆盖后的路径一致，如 `src/x_student.hpp` |
| `find`             | 是   | 参考实现中要替换的原文，可以跨行（JSON 中写 `\n`）                        |
| `replace`          | 是   | 注入错误后的文本                                                          |
| `count`            | 否   | `find` 必须恰好出现的次数，默认 1；不符时判为 STALE                       |
| `expect_fail_part` | 是   | 预期失败的阶段：`A`、`B`、`C`、`R` 或 `all`                               |
| `device`           | 否   | `cpu`（默认）或 `cuda`                                                    |
| `why`              | 是   | 这个错误模拟的是哪种真实失误，以及哪个 oracle 应该抓住它                  |

### fault_inject 用法

```bash
python tools/fault_inject.py --lab N --list                 # 列出 fault 和每个 TODO 的覆盖数
python tools/fault_inject.py --lab N                        # 全部注入；CUDA fault 需在 WSL/Linux 下跑
python tools/fault_inject.py --lab N --device cpu --only b2-kv-axis-swap
python tools/fault_inject.py --lab N --faults /tmp/my.json --json results/faults-labN.json
python tools/fault_inject.py --lab N --require-coverage     # 有 TODO 没有 fault 时失败
```

[fault_inject.py](../tools/fault_inject.py) 先按 `run.py --impl reference` 的方式（复制 lab、覆盖 solution、放入 `_tools`）建一份临时副本，对每个用到的 (part, device) 跑一次未修改的参考实现。参考实现跑不过就直接中止（退出码 2）。之后每个 fault 用一份新副本，替换文本，运行 `exercise.py --mode test --part P --device D`，结果分为：

| 状态    | 条件                | 处理                                 |
| ------- | ------------------- | ------------------------------------ |
| CAUGHT  | 退出码非 0 且不是 3 | 通过                                 |
| MISSED  | 退出码 0            | 补测试，直到抓住为止                 |
| ENV     | 退出码 3，缺依赖    | 不计为通过，在有依赖的机器上重跑     |
| STALE   | `find` 匹配次数不对 | 参考实现改过了，同步更新 faults.json |
| TIMEOUT | 超过 `--timeout`    | 视为测试过弱或死循环，查明原因       |

有 MISSED 或 STALE 时退出码为 1。CUDA 运行在有 `flock` 的系统上会加 `/tmp/mlsys-gpu.lock`，多个代理可以同时跑而不争 GPU。

## 5. 测量规则

- 先预热（至少 3 次，JIT 或编译类实验另外报告首次编译时间），再取 $\ge 5$ 次重复，报告中位数和 IQR（第 25 到第 75 百分位）。
- GPU 计时用 `cudaEventRecord` / `cudaEventElapsedTime` 包住被测区间，计时前后同步；CPU 用单调时钟。说明计时区间包含什么，不包含什么。
- 每份记录带上环境：GPU 型号、驱动、CUDA、编译器、torch 版本、锁定版本的哈希（`run.py` 会自动记录）。
- 区分 **modeled** 和 **measured**：由公式、规格表或仿真得到的数字在表头或图注里标成 `modeled, not measured`；只有在本机实测的才标 `measured`。两者不放在同一列里比较。
- 报告带宽和吞吐时写出计算口径，例如有效带宽 $B=\text{bytes}/t$ 中 bytes 指哪些读写。
- 笔记本 GPU 有频率和温度波动，比较 overlap 或加速比时给出容差，不用单次结果下结论。
- **GPU 计时前预热到稳定频率**。几次预热不够：主机在两组配置之间准备数据时 GPU 会空闲，驱动随即降到低 P-state。RTX 4060 Laptop 上实测显存频率会从 8000 MHz 掉到 810 MHz，lab1 的访存受限 kernel 因此偶尔慢 8 倍，split 扫描呈双峰。做法：每组计时前先用同一个 launch 让 GPU 连续忙至少 300 ms（参考 `lab1/decode_bench.cu` 的 `time_it`），每行输出 `stable` 列（$p_{75} \le 1.5\,p_{25}$），不稳定的行重跑后再使用。排查时用 `nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv -lms 50` 记录时钟。
- **一张卡同时只跑一个计时实验**。`run.py --device cuda` 和 `fault_inject.py` 都会拿 `/tmp/mlsys-gpu.lock`，记录里的 `gpu_lock` 字段说明是 `acquired` 还是 `inherited`。手动运行 benchmark 时用 `flock /tmp/mlsys-gpu.lock <命令>`。别的进程占着卡时测到的数，即使很稳定也不能用：后台有负载时，lab1 的结果整体慢了 2 倍。

## 6. 交给整合的注册片段

各方向不直接改共享文件，而是在交稿报告末尾附一份注册片段，由整合阶段统一合并。每一项都写成可以直接粘贴的 Markdown 表格行或段落：

| 目标文件                            | 需要给出的内容                                                 |
| ----------------------------------- | -------------------------------------------------------------- |
| `course/index.md`                   | 本讲一行：编号、标题链接、配套 lab、讲义时长、lab 时长         |
| `course/guides/directions.md`       | 方向一行：硬件要求、研究切口、组内对应工作（没有就写“无”）     |
| `course/readings/resources.md`      | 约 10 篇种子论文，每篇带会议年份、论文链接、代码链接、核验日期 |
| `course/readings/code-map.md`       | 上游仓库、固定版本（commit 或 tag）、文件、函数、对应 TODO     |
| `course/readings/visualizations.md` | 图解链接、类型（教学图解、公式演示、实测工具）、观察任务       |
| `labs/index.md`                     | lab 一行：语言、阶段、平台、时长                               |
| `labs/docs/workload.md`             | `workload_audit.py` 的本 lab 统计和各阶段估时                  |
| `labs/docs/validation.md`           | 本 lab 在 WSL 下的实跑记录：命令、通过项、fault 结果、日期     |

另外附上验证结果：reference 各阶段 test 与 bench 的 RECORD 路径，`fault_inject --json` 的汇总，`math_check` 与 `check_links` 的输出。

## 7. 维护检查

**全量验证。** 在 WSL2/Linux 加 NVIDIA GPU 的机器上运行 `bash tools/verify_all.sh`。它依次做：`check_env --lab all`；每个 lab 的参考实现 test 与 bench（CPU，以及有 GPU 部分的 lab 再跑 CUDA）；学生骨架 A 阶段（必须编译成功并在 TODO 处以退出码 1 失败）；`fault_inject` 的 CPU 与 CUDA 两轮；最后是 `workload_audit`、`check_links`、`math_check`、`format_check`、`test_platform` 和对所有 `.sh` 的 `bash -n`。`--quick` 只跑 test，不跑 benchmark。日志写入 `results/verify-<时间戳>/`，有任何一项失败时退出码为 1。

**错误注入。** `fault_inject.py` 读取 `solutions/labN/faults.json`，每个错误用一份新的参考副本注入并运行该阶段的测试，结果分为 CAUGHT、MISSED、ENV、STALE、TIMEOUT。常用参数：`--list` 只列出错误和 TODO 覆盖；`--device cpu|cuda` 只跑某类错误；`--only ID` 只跑指定错误；`--require-coverage` 在有 TODO 没有错误时失败；`--json PATH` 保存结果。格式与规则见 [authoring.md](#4-测试强度)。

**公式检查。** `math_check.py` 默认扫描所有受 Git 跟踪的文档，跳过代码块、行内代码、front matter 和链接目标。它报告 Unicode 数学字符（M001）、`\(` 或 `\[` 定界符（M002）、`$` 不配对（M003）、表格公式里未转义的竖线（M006）、ASCII 或 Mermaid 图里的中文（M010）等。`--paths` 只查指定文件或目录；`--render` 再用锁定版本的 KaTeX 逐个解析公式；`--summary` 只输出每个文件的统计。专有名词确实需要 Unicode 字符时，在该行加 `<!-- math-ok -->`。

`python tools/workload_audit.py --output results/workload.json` 检查每个 lab 的学生代码与答案：BEGIN/END TODO 标记是否配对，编号是否唯一，三个阶段是否都有任务，参考实现中是否还有未实现的占位代码。统计只计参考 TODO 区域的有效物理行，排除注释、空行和纯括号行，不要求学生写出相同的行数。统计口径与时间安排见[工作量说明](workload.md)。

格式工具安装使用 `python tools/setup_format.py`，适用于 Windows、Linux、macOS 和 WSL；编译器选择、CUDA 架构和终端配置见[环境说明](../setup.md)。导出副本自带 `_tools/platform_tools.py`，可在仓库外独立运行。

阅读资料的链接维护工具见 [course/readings/](../../course/readings/README.md)。
