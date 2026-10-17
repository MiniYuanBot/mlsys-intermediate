# 代码与文档格式

[实验索引](../index.md) · [公共工具](../tools/README.md)

本项目沿用本地 `mlsys-foundations` 的仓库结构与格式规则。共享配置放在**仓库根目录**，使编辑器、单独调用的formatter和统一检查脚本都能读取同一份规则；工具版本放在 [labs/tools/formatter-versions.env](../tools/formatter-versions.env)。

根目录包含 [.editorconfig](../../.editorconfig)、[.clang-format](../../.clang-format)、[.gersemirc](../../.gersemirc)、[.prettierrc.json](../../.prettierrc.json)、[pyproject.toml](../../pyproject.toml) 和两份formatter忽略规则。`.gitattributes`统一文本换行，`.gitignore`排除环境、缓存、构建产物及本地测量/报告。

| 文件                            | Formatter              | 规则                                                                                          |
| ------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| C/C++/CUDA，含头文件和答案      | clang-format 20.1.8    | foundations `.clang-format` 原样复制；4 空格、100 列、同行左括号、`int *p`                    |
| Python                          | Ruff Formatter 0.17.0  | foundations `pyproject.toml` 的格式规则原样复制；4 空格、双引号、100 列                       |
| CMake                           | gersemi 0.29.2         | 2 空格、100 列；当前未使用 CMake 的实验不强行新增构建文件                                     |
| Shell                           | shfmt 3.12.0           | 与 foundations `.editorconfig` 对应的 4 空格、`-bn -ci -sr`                                   |
| Markdown/JSON/YAML、HTML/JS/CSS | Prettier 3.9.10        | foundations `.prettierrc.json` 原样复制；2 空格、100 列；保留段落换行，不重写 Markdown 代码块 |
| Rust                            | Rust 1.99.0 的 rustfmt | edition 2021；默认 4 空格、100 列                                                             |

UTF-8、LF、末尾换行。格式化与 lint 是不同操作；这里不附带 Ruff lint、自动修复或 import 整理。

## 安装与使用

以下命令均在 `labs/` 目录执行。可沿用 foundations 已安装的工具；检查器优先找 `MLSYS_FORMAT_BIN`、`labs/.venv-format/`，随后找 PATH，并核对版本。

```bash
# 首次安装；Windows/Linux/macOS/WSL，需 Python、Node/npm、Rust/rustup。
python tools/setup_format.py
# 检查整个项目，不修改源文件。
python3 tools/format_check.py
# 按同样规则统一格式，然后再次检查。
python3 tools/format_check.py --write
python3 tools/format_check.py --report results/format-check.json
```

安装只在 `labs/.venv-format/` 中放 Python/Node 工具，rustfmt 使用现有 Rust 工具链。Windows 可在 PowerShell 运行同一个 Python 入口；脚本使用 `Scripts/` 和 Windows shfmt，Linux/macOS/WSL 使用 `bin/` 与对应架构的 shfmt。`bash tools/setup-format.sh` 保留为 POSIX 包装入口。用 `python tools/setup_format.py --dry-run` 可先查看安装计划。

Prettier 和 npm 通过 Node 的 JavaScript 入口调用，避免 Windows `.cmd` 包装器对带空格路径的处理差异。Python 工具显式使用 UTF-8 读写课程文档、CSV 和运行记录。

检查器扫描根目录文件、三个内容目录与 `.vscode/`、`.obsidian/`、`.github/` 共享设置中的实际文件，包含尚未加入 Git 的文件；遇到缺工具、版本不一致或格式差异会失败。`build/`、`target/`、虚拟环境、第三方依赖、`results/`、生成文件与锁文件排除。

## 编辑器设置

用 VS Code 打开仓库根目录，[共享设置](../../.vscode/settings.json)为各语言选择 formatter，新终端默认进入 `labs/`。[扩展推荐](../../.vscode/extensions.json)沿用 foundations 并补充 Rust 支持。Prettier 使用仓库内 Node 模块，Ruff 同时配置 `bin/` 和 `Scripts/` 路径；clang-format、shfmt、gersemi 从启动 VS Code 时的 PATH 查找，Rust 使用现有 rustfmt。

安装后，从 `labs/` 目录把格式工具加入当前终端的 PATH，再启动编辑器：

```bash
PATH="$PWD/.venv-format/bin:$PATH" code ..
```

```powershell
$env:Path = (Join-Path $PWD ".venv-format\Scripts") + ";" + $env:Path
code ..
```

也可以先激活格式工具环境再启动。若已有 VS Code 进程，重启相应窗口以使用更新的环境。使用 WSL 时，在 WSL 内安装工具并通过 Remote WSL 打开仓库。

`.editorconfig`和各formatter从文件目录向上查找根目录的共享配置。统一脚本也支持从根目录运行`python3 labs/tools/format_check.py`。Obsidian基础设置保存在根目录`.obsidian/`，个人workspace、cache、plugins不进入版本管理。
