# 环境与硬件

[实验索引](index.md) · [格式工具](docs/formatting.md) · [验证范围](docs/validation.md)

## 选择平台

| 环境                    | CPU 工具链                                          | CUDA 实验                                                  | 终端                                                    |
| ----------------------- | --------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| Windows 原生            | Visual Studio C++ Build Tools / MSVC，Cargo，Python | NVIDIA GPU + CUDA Toolkit + 配套 MSVC                      | x64 Developer PowerShell 或 Native Tools Command Prompt |
| Linux                   | GCC 或 Clang，Cargo，Python                         | NVIDIA GPU + 驱动 + CUDA Toolkit                           | bash / zsh                                              |
| macOS，含 Apple Silicon | Xcode Command Line Tools / Clang，Cargo，Python     | 使用 CPU 路线                                              | zsh / bash                                              |
| WSL2                    | 在发行版内安装 Linux 工具链                         | Windows NVIDIA 驱动提供 GPU，WSL 内安装适配的 CUDA Toolkit | WSL 终端或 VS Code Remote WSL                           |

CPU 路线可完成 lab1、lab4、lab5、lab6、lab8，以及 lab2 的 A/C 主机算法和 lab7 的 CPU 部分。lab3 的 A/B 使用 Python 标准库，C1–C4 需要当前平台可用的 PyTorch Inductor；开始前执行下文的编译探测。lab2 B、lab3 C5、lab6 GPU 搬运/峰值和 lab7 CUDA 流水线需要 NVIDIA GPU。各阶段范围见[实验索引](index.md)。

这些是脚本提供的运行路径。本机实测环境是 WSL2；Windows、Linux 和 macOS 的 CPU 回归已配置到 [GitHub Actions](../.github/workflows/portability.yml)，实际通过状态以该平台运行结果为准。macOS 的 MPS 可另作研究扩展，当前 lab 的设备参数为 `cpu` / `cuda`。

## Python 与系统编译器

Python 使用 3.10 或更新版本，建议 3.12。优先复用 foundations 的环境，也可在 `labs/` 中新建虚拟环境。实验命令默认从 `labs/` 执行，阅读资料维护工具按自己的说明运行。

Linux、macOS、WSL：

```bash
cd labs
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python tools/doctor.py
```

Windows PowerShell，可直接使用虚拟环境解释器：

```powershell
cd labs
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe tools/doctor.py
```

后文的 `python` 都表示你选定的解释器；Windows 未激活环境时，替换为 `.\.venv\Scripts\python.exe`。命令参数和相对路径相同，多行 bash 命令中的 `\` 在 PowerShell 中应去掉并合为一行。

- Windows 安装 [C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 的桌面 C++ 工具和 Windows SDK，从其 x64 开发者终端运行。脚本自动选择 MSVC 并使用 `/std:c++17`、异常和 UTF-8 选项；仅安装编辑器不提供 `cl.exe`。
- macOS 安装 Xcode Command Line Tools，可执行 `xcode-select --install`；脚本使用 Clang。
- Linux/WSL 安装发行版的 GCC 或 Clang、Python venv 支持。例如 Debian/Ubuntu 可安装 `build-essential python3-venv`。
- Rust 使用 [rustup](https://rustup.rs/) 配置，确认 `cargo`、`rustc` 可用。实验 crate 不依赖第三方包，可离线运行；共享格式工具的固定版本另见[格式说明](docs/formatting.md)。

如需指定编译器，设置 `CXX` 为可执行文件名或完整路径，路径中可含空格，不在其中追加编译参数：

```bash
export CXX=clang++
```

```powershell
$env:CXX = "cl.exe"
```

脚本自动处理 `.exe` 后缀、绝对程序路径和平台编译参数。运行产物保存在 `results/`，个人报告写入 `reports/`。学生代码不会被参考实现覆盖；[导出的调试副本](../solutions/index.md)也包含公共编译辅助代码。

## CUDA 与 PyTorch

按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装当前系统对应的 CPU 或 CUDA 构建。CUDA Toolkit、显卡驱动和 PyTorch 内置 runtime 需要分别核对。WSL 的解释器、Cargo、编译器与 Toolkit 都应来自 WSL 环境；不要混用 Windows 的虚拟环境。

原生 CUDA 编译需要 `nvcc` 和兼容的 host compiler；Windows 使用 MSVC，Linux/WSL 使用受该 Toolkit 支持的 GCC/Clang。安装依据 [NVIDIA Windows 指南](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)和 [Linux 指南](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)。

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python tools/doctor.py --compile-probe --device cpu
python tools/run.py --lab 6 --impl reference --mode test
```

lab3 的探测实际调用 Inductor。若失败，先处理输出中的编译器或 PyTorch 依赖问题；Windows 可选择 WSL2。探测失败不会静默切成 eager 后记为通过。

有 NVIDIA GPU 时：

```bash
python tools/doctor.py --compile-probe --device cuda
python tools/run.py --lab 2 --impl reference --mode test --device cuda
```

CUDA 默认使用 `-arch=native`，由 nvcc 检测当前可见 GPU，参见 [nvcc 架构选项](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch)。若在登录节点编译、需要指定目标或旧 Toolkit 不识别 `native`，显式传入目标：

```bash
python tools/run.py --lab 2 --impl reference --mode test --device cuda --cuda-arch sm_89
```

`sm_89` 是 RTX 4060 的示例，其他设备应按实际架构修改。也可以设置 `MLSYS_CUDA_ARCH`，直接运行各 lab 的 `exercise.py` 时同样生效；`NVCC` 可指定 nvcc 可执行文件的完整路径。参考运行的记录会保留平台、架构参数、编译器和公共辅助代码哈希。

## 真实模型采集

另需 transformers、accelerate、safetensors；画图需要 matplotlib。不同论文 artifact 的依赖分别放入虚拟环境或容器。

采集工具读取显式指定的已有本地模型目录，默认 `trust_remote_code=False`，不自动下载。可先使用 lab1 的 [prompts.txt](lab1/prompts.txt)，也可用 `--prompts` 指定自己的 UTF-8 文本，每行一个 prompt，空行忽略。以下是一条适用于各平台的单行命令，替换模型路径，输出目录须尚不存在：

```text
python tools/capture_model.py --model "path/to/Qwen3-0.6B" --prompts lab1/prompts.txt --output results/model-probe-01 --max-input 128 --max-new 32 --capture-kv --device cpu
```

Windows 模型路径可写为 `"D:/models/Qwen3-0.6B"`；CUDA 设备上可改为 `--device cuda`。工具逐步同步，产生诊断延迟，不包含排队和网络。K/V dump 在计时外，仍可能影响后续状态；正式延迟对照应关闭 dump，并独立预热。记录提示词来源和长度分布，自拟文本用于机制检查。

K/V 分布诊断：

```text
python tools/analyze_kv.py results/model-probe-01/prefill-kv-0.pt --out results/kv-analysis-01
```

该工具导出热图与两轴 fake-quantization 误差，不评估任务质量。

## 更大设备与复现

主实验无需多卡或超算。申请 [北大 HPC](https://hpc.pku.edu.cn/) 时，先确认 GPU、主机 RAM、驱动/容器、分区和配额，再按目标节点设置 CUDA 架构。原论文的多卡 artifact 是扩展任务，单卡课程实验的通过不能代表完整复现。
