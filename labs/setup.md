# 环境与硬件

[实验索引](index.md) · [验证记录](docs/validation.md)

## 默认约定

优先复用已经跑通 foundations 的 WSL2/Linux 环境；本项目不会修改全局软件。全部命令在 `labs/` 目录运行。`python` 指向你选择的虚拟环境，基础工具仅用 Python 标准库，lab6/torch 编译/模型采集另需 PyTorch。

从项目根目录先执行一次 `cd labs`。结果和报告分别保存在这个目录下的 `results/`、`reports/`；切换目录后不要再在命令里添加 `labs/` 前缀。格式工具单独安装，见 [格式规范](docs/formatting.md)。

```bash
python3 tools/doctor.py
# 若已有 foundations 环境，激活该环境后运行同一个 doctor。
# 新环境示例：
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

C++17 编译器与 Cargo 支持 lab1/4/5/7/8；Rust crate 没有第三方依赖，`cargo --offline` 可工作。runner 使用 GCC/clang 风格参数；Windows 原生 MSVC 路径未配置，Windows 用户默认使用 WSL2。

## CUDA / PyTorch

- CUDA Toolkit 要包含 nvcc、cuBLAS；WSL GPU 使用 Windows NVIDIA 驱动提供的设备支持，避免重复安装不适用的 Linux 显示驱动。
- 当前 CUDA driver 默认编译 `sm_89`（对应本机 4060）；换 GPU 时先核实 compute capability，再改 runner 的 `-arch`，不要只改标签冒充跨平台验证。
- PyTorch 安装按 [官方选择器](https://pytorch.org/get-started/locally/) 与 [官方 CUDA 说明](https://docs.pytorch.org/docs/stable/notes/cuda.html) 选择匹配的构建。CUDA Toolkit 版本、驱动支持上限和 torch 内置 runtime 不是同一件事。
- 本次实际运行使用 PyTorch 2.14.1+cu126、CUDA Toolkit 12.6、Python 3.12；这记录的是已存在环境，不保证所有平台可用同名 wheel。课程使用公开 API，升级后重跑测试。

```bash
# 在已配置好的 PyTorch 环境中：
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python tools/run.py --lab 6 --impl reference --mode test --device cuda
# 仅画图：
python -m pip install matplotlib
```

模型采集另需 transformers/accelerate/safetensors。优先使用已有模型环境，不把 KIVI、FlexLLM、verl 等不同 artifact 的依赖混装；各自新建虚拟环境/容器，固定上游 commit 与依赖清单。

## Mac mini 的安排

C++、Rust、纯 Python 部分可作为 CPU 学习路线，Apple Silicon 的 MPS 也可另作研究平台；本仓库 CUDA `.cu` 不在 macOS 上运行。lab6 的 CPU 测试与 CUDA 测试分别执行；本次未在 macOS 实机验证。不把 CPU 正确性通过记成 CUDA 性能验收。

## 真实模型采集

工具只读取显式指定的**已有本地目录**，默认 `trust_remote_code=False`，不自动下载。以下路径替换为你的实际模型目录；本次用户设备存在 Qwen3-0.6B/1.7B，推荐先用 0.6B。

```bash
python tools/capture_model.py --model /path/to/Qwen3-0.6B \
  --output results/model-probe-01 --max-input 128 --max-new 32 --capture-kv
```

它对每一步同步，产生 diagnostic latency，不包含排队/网络，不是在线服务 TTFT。K/V dump 发生在计时外，可能影响后续缓存/内存状态；正式 latency 对照应关闭 dump，并独立预热。prompt 是教学合成文本，不代表质量基准。

## 升级与复现

默认主实验没有多卡和超算依赖。申请 [北大 HPC](https://hpc.pku.edu.cn/) 时先确认账户、GPU、主机 RAM、驱动/容器、分区和配额。原论文要求多卡 A100 的 artifact 只作为资源明确后的挑战，不能直接当作 4060 安装步骤。

K/V 分布诊断（输出目录须是新目录）：

```bash
python tools/analyze_kv.py results/model-probe-01/prefill-kv-0.pt --out results/kv-analysis-01
```

生成热图与两轴 fake-quantization 误差，未实现真实模型的低比特 cache 后端，也不评估任务质量。
