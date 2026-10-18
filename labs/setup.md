# 环境配置

[实验索引](index.md) · [格式工具](docs/formatting.md) · [验证范围](docs/validation.md)

本课程和 mlsys-foundations 共用一个 Python 虚拟环境 `~/venvs/mlsys`。版本锁定在 [tools/env-versions.env](tools/env-versions.env)，这个文件与 foundations 的同名文件逐字节一致；Rust 版本写在 [tools/formatter-versions.env](tools/formatter-versions.env) 的 `RUST`。和 foundations 不同，这里的 lab **互不累加**：`--lab N` 只安装基础依赖和 lab N 自己用到的东西，`--lab all` 安装全部 lab 的并集。本页命令都在 `labs/` 目录执行。

| 平台                                | 能做什么                            | 安装与检查（在 `labs/` 目录执行）                                                                                                              |
| ----------------------------------- | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **WSL2 Ubuntu 24.04**（课程主环境） | 全部 lab 的全部阶段                 | `bash tools/setup_env.sh --lab N`<br>`bash tools/check_env.sh --lab N`                                                                         |
| Linux（Ubuntu 24.04 + NVIDIA 显卡） | 全部                                | 同上                                                                                                                                           |
| macOS（Apple Silicon）              | 各 lab 的 CPU 部分                  | 同上                                                                                                                                           |
| Windows 原生（PowerShell）          | 各 lab 的 CPU 部分；CUDA 阶段转 WSL | `powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab N`<br>`powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab N` |

`N` 取 1–9 或 `all`。`check_env` 只检查 lab N 需要的东西，每行以 `[ OK ]`、`[WARN]`、`[FAIL]` 或 `[ -- ]` 开头：

- `[FAIL]`：必需项不可用，下一行给出 `fix:` 命令；
- `[WARN]`：能用，但版本和锁定的不同，或者只有 R 阶段用到这一项；
- `[ -- ]`：可选项，或者当前平台不做的部分。

退出码 0 表示没有 `[FAIL]`，1 表示至少一项失败，2 表示参数错误或用错了 shell。检查不只看版本号，还会实际运行：编译并运行一个 C++17 程序；在 WSL/Linux 上用 nvcc 编译并运行一个小 kernel；需要 CUDA 时在 GPU 上做一次 torch 矩阵乘；lab3 跑一次 Inductor 编译探测；需要模型时确认 `config.json`、`tokenizer.json` 和 `*.safetensors` 都在。

在 Git Bash 里运行 `.sh` 脚本会被拒绝，并提示改用 PowerShell 或 WSL。其他 Linux 发行版也能用 `setup_env.sh`，但脚本不会替你装系统包，只列出缺少的命令。

**缺依赖时会发生什么。** `run.py` 不会抛出 traceback，只打印一行原因和一行修复命令，以退出码 3 结束，运行记录里的 `status` 为 `env-missing`：

```text
MISSING cargo for lab 4 stage A/B/C: build and test the Rust simulator
  fix: bash tools/setup_env.sh --lab 4
```

在原生 Windows 上请求 CUDA（例如 `--device cuda`）时，`fix:` 后面是一条能直接粘贴进 PowerShell 的 `wsl -d Ubuntu-24.04 -- bash -lc "cd ... && python tools/run.py ..."`，会在 WSL 里重跑同一个请求。

## 你的机器分别做什么

| 机器                           | 适合                                                                    | 不适合                                    |
| ------------------------------ | ----------------------------------------------------------------------- | ----------------------------------------- |
| Windows + RTX 4060 Laptop 8 GB | **全部 lab，都在 WSL2 里做**；不涉及 CUDA 的部分也可以在 Windows 原生做 | 多 GPU 通信、Hopper 专属特性（TMA/WGMMA） |
| Mac（Apple Silicon）           | lab1、lab4、lab5、lab8、lab9 的 A/B/C，其他 lab 的 CPU 部分             | 任何 CUDA 阶段                            |
| 北大 HPC / 组里服务器          | 多卡扩展、论文原始 artifact 的复现                                      | 日常开发与调试（排队太慢）                |

建议整门课都在 WSL 里做。Windows 原生的 nvcc 只认特定版本的 MSVC，CUDA 12.6 不支持 VS 2026 的 `cl.exe`，所以本课程在原生 Windows 上只做 CPU 部分。主实验不需要多卡。申请 [北大 HPC](https://hpc.pku.edu.cn/) 前，先确认 GPU 型号、主机内存、驱动或容器、分区和配额；单卡课程实验通过，不等于完整复现了论文的多卡 artifact。

## 各 lab 需要的内容

每个 lab 都会装的基础依赖：C++17 编译器、CMake、git、direnv，以及虚拟环境里的 numpy、matplotlib、pytest。下表是 `--lab N` 在基础之上加装的内容，与 [tools/lab_meta.py](tools/lab_meta.py) 一致（`tools/test_platform.py` 会核对脚本里的表）。R 阶段的依赖同样会装上，check_env 对它们只报 `[WARN]`，不会挡住 A/B/C。九个 lab 都已实现 R 阶段（lab8 中用 TRL 跑 GRPO 的部分只给指南）。

| Lab | A/B/C 另需                                                | R 阶段另需                                                                         |
| --- | --------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| 1   | 无                                                        | CUDA Toolkit 12.6、GPU；可选模型 Qwen3-0.6B（需自行下载）                          |
| 2   | CUDA Toolkit 12.6、GPU（B，以及 C 的 GPU 部分）           | CUDA Toolkit、GPU                                                                  |
| 3   | PyTorch 2.14.1（C1–C4）；CUDA Toolkit、GPU（C5–C7）       | PyTorch；GPU（graph 捕获，规划部分 CPU 可做）                                      |
| 4   | Rust 1.99.0、`libssl-dev`、`pkg-config`                   | PyTorch、transformers、safetensors、`hf`；Qwen3-0.6B 与 Qwen3-1.7B                 |
| 5   | Rust                                                      | PyTorch、transformers、GPU；Qwen3-0.6B（CPU 回放不需要）                           |
| 6   | PyTorch（CPU 可做全部正确性检查；GPU 用于搬运与显存实测） | GPU（只用于显存实测）                                                              |
| 7   | CUDA Toolkit、GPU（C3/C4/C6）                             | CUDA Toolkit、GPU                                                                  |
| 8   | Rust                                                      | 回放无需额外依赖；可选采样用 Qwen3-0.6B；TRL 指南用 `~/venvs/mlsys-rl`，默认不安装 |
| 9   | 无                                                        | CUDA Toolkit、GPU（R1–R4；R5/R6 CPU 可做）                                         |

安装位置：CUDA Toolkit 在 `/usr/local/cuda-12.6`（约 3 GB，只装 Toolkit，不装显卡驱动），PyTorch 等 Python 包在 `~/venvs/mlsys`，Rust 在 `~/.rustup` 和 `~/.cargo`，模型在 `~/models`（可用 `MLSYS_MODELS` 改）。Qwen3-0.6B 约 1.5 GB，Qwen3-1.7B 约 4 GB，不想现在下载就加 `--no-models`。

系统包的来源：WSL/Ubuntu 用 apt（需要 sudo），macOS 用 Homebrew，Windows 用 winget 安装 MSYS2（UCRT64 的 gcc、cmake、ninja）和 Python 3.12。PATH 里已有可用的编译器或 Python 时，脚本直接用现成的。脚本还会改一个 shell 配置文件：WSL/Linux 是 `~/.bashrc`，macOS 是 `~/.zshrc`，写入 direnv 钩子、`mlsys` 命令，以及 CUDA、cargo 和 `MLSYS_MODELS` 的环境变量；Windows 是 PowerShell 的 profile，写入 `mlsys` 函数。两个脚本都可以反复运行，已经装好的步骤会跳过。

## 虚拟环境放在哪里，怎么生效

虚拟环境放在**仓库外面**：`~/venvs/mlsys`，Windows 上是 `%USERPROFILE%\venvs\mlsys`。这样做有三个原因：

- **同一个文件夹会被两个系统使用。** 仓库通常放在 Windows 盘上，Windows 原生直接访问，WSL 通过 `/mnt/d/...` 访问。虚拟环境和操作系统绑定：Linux 的是 `bin/python`，Windows 的是 `Scripts\python.exe`。放进仓库的同一个 `.venv` 会被两边互相覆盖。
- **WSL 读写 Windows 盘很慢。** PyTorch 有好几 GB，装在 `/mnt/d` 上安装慢，`import torch` 也慢。放在 WSL 自己的文件系统里要快得多。
- **foundations 与本课程、多个 clone 共用一份环境。** 不用重复下载 PyTorch，执行 `git clean -fdx` 也不会把环境删掉。

**锁定版本的哈希。** 因为环境是共用的，`setup_env` 每次结束时都会写入 `$VENV/.mlsys-pins`：第一行是 `env-versions.env` 中全部 `KEY=VALUE` 行的 sha256，后面是这些行本身。下次运行时，如果哈希和当前仓库的不同（比如 foundations 那边升级了版本），脚本会停下，除非你加 `--repin`（PowerShell 是 `-Repin`）。`check_env` 发现哈希不一致，或者相邻的 `../mlsys-foundations` 的版本文件和这里不同，会报 `[WARN]`。两边确实需要不同版本时，给其中一个仓库单独设置 `MLSYS_VENV`。

如果确实想把环境放进项目里，先设置 `MLSYS_VENV`（例如 `export MLSYS_VENV=$PWD/.venv`，并写进 shell 配置文件），再运行安装脚本。`.venv*` 已被 git 忽略。

**环境怎么生效。** 遇到 `ModuleNotFoundError: No module named 'numpy'`，几乎都是因为当前的 `python` 不是课程环境里的那个，而是系统自带的（比如 `/usr/bin/python3`）。Ubuntu 24.04 不允许直接往系统 Python 里 `pip install`（会报 `externally-managed-environment`），所以**不要**用 `pip install --user` 或 `--break-system-packages` 去“修”。文档里的命令一律写 `python`：课程环境里它总是对的，而 Windows 上的 `python3` 默认是 Microsoft Store 的占位程序，运行后直接以退出码 49 结束。

- **WSL / Linux / macOS**：仓库根目录的 `.envrc` 由 [direnv](https://direnv.net/) 执行。`cd` 进仓库时，它把 `${MLSYS_VENV:-~/venvs/mlsys}` 放到 PATH 最前面，离开仓库时自动恢复，所以在仓库里 `python`、`pytest` 都来自课程环境，其他项目不受影响。第一次安装后要**新开一个终端**，钩子才会生效。在另一个 clone 里第一次 `cd` 进去时，direnv 会提示 `.envrc is blocked`，在仓库根目录执行一次 `direnv allow` 即可。
- **Windows PowerShell**：没有自动激活。每个新终端先执行一次 `mlsys`（等价于运行 `~\venvs\mlsys\Scripts\Activate.ps1`）。安装脚本在虚拟环境里额外放了一个 `python3.exe`，激活之后，旧笔记里的 `python3` 命令也会进入虚拟环境，不会跑到占位程序上。为了允许激活脚本运行，安装脚本会把当前用户的执行策略设为 `RemoteSigned`（[Python 官方文档](https://docs.python.org/3/library/venv.html)推荐的做法）。
- **任何平台、仓库外面**：执行 `mlsys` 手动激活。Git Bash 里用 `source ~/venvs/mlsys/Scripts/activate`。

## WSL2 与 GPU 驱动（只做一次）

1. 在 Windows PowerShell 执行 `wsl --install -d Ubuntu-24.04`，按提示重启。
2. Windows 上装好较新的 NVIDIA 驱动即可。**不要在 WSL 里安装 Linux 显卡驱动**：Windows 驱动会自动透传进 WSL，所以 WSL 里的 `nvidia-smi` 能用。但这**不代表**已经有 CUDA Toolkit（`nvcc`）。Toolkit 由 `setup_env.sh` 从 NVIDIA 的 `wsl-ubuntu` 仓库安装，这个仓库不带显卡驱动。参见 [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)。
3. 仓库可以放在 Windows 盘上，WSL 里通过 `/mnt/d/...` 访问。手动构建时，把构建目录放到 Linux 文件系统里会快很多；`run.py` 本来就在临时目录里编译。
4. 从 Windows 一侧调用 WSL 时，`bash -lc` 不会读取 `~/.bashrc`。`run.py`、`check_env.sh` 会自己到 `/usr/local/cuda/bin` 和 `~/.cargo/bin` 里找 nvcc 和 cargo；runner 给出的 WSL 命令还会先激活 `~/venvs/mlsys`。

## Linux（原生）

显卡驱动要自己装，版本要支持 CUDA 12.6（560 或更新），例如在 Ubuntu 上执行 `sudo ubuntu-drivers install` 后重启。然后运行 `bash tools/setup_env.sh --lab N`。在 Ubuntu 24.04 x86_64 上，脚本会从 NVIDIA 的 `ubuntu2404` 仓库安装 `cuda-toolkit-12-6`：这个包只有 Toolkit，不会替换你的驱动。其他发行版请按 [CUDA 12.6 下载页](https://developer.nvidia.com/cuda-12-6-0-download-archive)自行把 Toolkit 装到 `/usr/local/cuda-12.6`，再重跑脚本。

## macOS

先执行 `xcode-select --install`，再安装 [Homebrew](https://brew.sh)，然后运行 `bash tools/setup_env.sh --lab N`。脚本用 Homebrew 的 Python 3.12 创建 `~/venvs/mlsys`，PyTorch 用 PyPI 上的 macOS 版本。macOS 只做 CPU 部分：遇到含 CUDA 的 lab，脚本会说明哪些阶段跳过，仍然安装基础依赖和 CPU 部分要用的东西；runner 收到 `--device cuda` 时以退出码 3 结束。MPS 可以作为研究扩展，各 lab 的设备参数只有 `cpu` 和 `cuda`。

## Windows 原生（只做 CPU 部分）

在 PowerShell 里、`labs\` 目录下执行：

```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_env.ps1 -Lab 1
# open a NEW PowerShell window, cd back into labs\
mlsys
powershell -ExecutionPolicy Bypass -File tools\check_env.ps1 -Lab 1
python tools\run.py --lab 1 --impl student --mode test --part A
```

- **C++ 编译器。** 工具优先用 MSVC 的 `cl`：在 x64 Developer PowerShell 里运行时会选中它，使用 `/std:c++17`、异常和 UTF-8 选项。只装 VS Code 不会带来 `cl.exe`。没有 `cl` 时退回 `clang++` 或 `g++`。PATH 里既没有 `cl`，也没有 `g++` 加 `cmake` 时，安装脚本用 winget 装 MSYS2，位置是 `$env:MSYS2_ROOT`，未设置时为系统盘下的 `msys64`，并把其中的 `ucrt64\bin` 加进用户 PATH。`python tools/doctor.py` 和 `check_env.ps1` 都会打印实际选中的编译器及其类别（msvc、mingw-gcc 等）。MinGW 的链接器在部分非 ASCII 路径下会失败（`cannot open output file`），仓库或临时目录含中文时 `check_env` 会报 `[WARN]`；这时请换成纯 ASCII 路径、改用 MSVC，或者在 WSL 里做。也可以用 `CXX` 指定编译器，值写可执行文件名或完整路径，不要附带参数。
- **Rust。** rustup 装到 `~\.cargo`。检测到 MSVC（`cl` 或带 C++ 工具的 Visual Studio）时用 `x86_64-pc-windows-msvc`，否则用 `x86_64-pc-windows-gnu`。
- **PyTorch。** 安装锁定版本的 CPU wheel，来源是 `https://download.pytorch.org/whl/cpu`。lab3 C 的 Inductor 在 Windows 上通常还需要 MSVC；探测失败时，请在 WSL 里做 lab3 C。
- **CUDA 阶段一律转到 WSL**，因为 CUDA 12.6 的 nvcc 不支持 VS 2026。安装脚本遇到含 CUDA 的 lab，只会打印对应的 WSL 命令；runner 收到 `--device cuda` 时退出码为 3，并给出可以直接复制的 `wsl -d Ubuntu-24.04 -- bash -lc "..."`。
- AddressSanitizer 在 MinGW 上不可用，调试内存错误请去 WSL。

## 版本说明

- **为什么是 CUDA 12.6 版的 PyTorch**：驱动 566.x 最高支持 CUDA 12.7，PyTorch 的 cu128 及更新版本需要更新的驱动。以后升级了驱动，在 foundations 和本仓库的 `tools/env-versions.env` 里同时改 `CUDA`、`TORCH`、`TORCH_INDEX`，再分别重跑安装脚本（需要加 `--repin`）。
- **升级任何版本**都只改 `tools/env-versions.env`（Rust 改 `tools/formatter-versions.env`），并在同一个提交里更新本页。`check_env` 遇到与锁定版本不同的包会报 `[WARN]`：通常还能用，但要在报告里写明。带 `+cu126` 或 `+cpu` 后缀的 torch 版本视为与锁定一致。
- **CUDA 架构**：默认 `-arch=native`，由 nvcc 检测当前可见的 GPU，参见 [nvcc 架构选项](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch)。在登录节点编译，或者旧 Toolkit 不认识 `native` 时，显式指定目标，例如 RTX 4060 的 `python tools/run.py --lab 2 --impl reference --mode test --device cuda --cuda-arch sm_89`。也可以设置 `MLSYS_CUDA_ARCH`；`NVCC` 可以指向 nvcc 的完整路径。
- **不装 vLLM**：各 lab 自己实现被研究的机制。想用 vLLM 当外部强基线，它的当前版本要求比 566.x 更新的驱动，需要先升级驱动，再装进**另一个**虚拟环境，不要装进 `~/venvs/mlsys`。lab8 R 阶段的 verl/TRL 也同样放进独立的 `~/venvs/mlsys-rl`。
- Hugging Face 模型下载太慢时，先执行 `export HF_ENDPOINT=https://hf-mirror.com`，再运行脚本。

## 真实模型采集

[capture_model.py](tools/capture_model.py) 读取显式指定的本地模型目录，默认 `trust_remote_code=False`，不会自动下载。可以先用 lab1 的 [prompts.txt](lab1/prompts.txt)，也可以用 `--prompts` 指定自己的 UTF-8 文本，每行一个 prompt，空行忽略。输出目录必须尚不存在：

```text
python tools/capture_model.py --model ~/models/Qwen3-0.6B --prompts lab1/prompts.txt --output results/model-probe-01 --max-input 128 --max-new 32 --capture-kv --device cpu
python tools/analyze_kv.py results/model-probe-01/prefill-kv-0.pt --out results/kv-analysis-01
```

有 GPU 时可以改成 `--device cuda`。工具逐步同步，测到的是诊断延迟，不含排队和网络。K/V dump 发生在计时之外，但仍可能影响后续状态；正式的延迟对照应关闭 dump，并单独预热。`analyze_kv.py` 导出热图和两轴 fake-quantization 误差，不评估任务质量。记录提示词的来源和长度分布；自己编写的文本只用于检查机制。

## 记录环境

每个 lab 开始时保存一次环境信息，写进报告的“环境”一节（Windows 原生把第一行换成 `check_env.ps1`）：

```bash
bash tools/check_env.sh --lab N > results/environment/labN-check.txt
python tools/doctor.py > results/environment/labN-doctor.txt
```

`run.py` 的每条记录也会保存解释器、编译器及其类别、torch/triton/transformers/safetensors/huggingface_hub 等包的版本，以及 `pins_sha256`（当前仓库）和 `venv_pins_sha256`（虚拟环境里的 `.mlsys-pins`）。

## 卡住了怎么办

1. 先跑 `check_env`，照 `[FAIL]` 下面的 `fix:` 处理。runner 打印的 `MISSING` 行也一样，照它的 `fix:` 做即可。
2. 环境问题超过 30 分钟还没解决：保存能复现问题的最短命令和完整报错，写进[周报](../course/templates/weekly-report.md)。超过 90 分钟就停止盲试，去问人或换一台机器，同时继续推进 CPU 部分和论文阅读。
