# 代码与文档格式

[实验索引](../index.md) · [公共工具](../tools/README.md)

本项目沿用本地 `mlsys-foundations` 的仓库结构与格式规则。共享配置放在**仓库根目录**，使编辑器、单独调用的formatter和统一检查脚本都能读取同一份规则；工具版本放在 [labs/tools/formatter-versions.env](../tools/formatter-versions.env)。

根目录包含 [.editorconfig](../../.editorconfig)、[.clang-format](../../.clang-format)、[.gersemirc](../../.gersemirc)、[.prettierrc.json](../../.prettierrc.json)、[pyproject.toml](../../pyproject.toml) 和两份formatter忽略规则。`.gitattributes`统一文本换行，`.gitignore`排除环境、缓存、构建产物及本地测量/报告；精选证据JSON与入口README可正常提交。

| 文件                            | Formatter              | 规则                                                                                          |
| ------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| C/C++/CUDA，含头文件和答案      | clang-format 20.1.8    | foundations `.clang-format` 原样复制；4 空格、100 列、同行左括号、`int *p`                    |
| Python                          | Ruff Formatter 0.17.0  | foundations `pyproject.toml` 的格式规则原样复制；4 空格、双引号、100 列                       |
| CMake                           | gersemi 0.29.2         | 2 空格、100 列；当前未使用 CMake 的实验不强行新增构建文件                                     |
| Shell                           | shfmt 3.12.0           | 与 foundations `.editorconfig` 对应的 4 空格、`-bn -ci -sr`                                   |
| Markdown/JSON/YAML、HTML/JS/CSS | Prettier 3.9.10        | foundations `.prettierrc.json` 原样复制；2 空格、100 列；保留段落换行，不重写 Markdown 代码块 |
| Rust                            | Rust 1.99.0 的 rustfmt | edition 2021；默认 4 空格、100 列                                                             |

UTF-8、LF、末尾换行。格式化与 lint 是不同操作；这里不附带 Ruff lint、自动修复或 import 整理。Prettier 额外覆盖本项目的本地 HTML 可视化，沿用相同规则。

## 安装与使用

以下命令均在 `labs/` 目录执行。可沿用 foundations 已安装的工具；检查器优先找 `MLSYS_FORMAT_BIN`、`labs/.venv-format/`，随后找 PATH，并核对版本。

```bash
# 首次安装；WSL2/Linux/macOS，需 Python、Node/npm、curl、Rust/rustup。
bash tools/setup-format.sh
# 检查整个项目，不修改源文件。
python3 tools/format_check.py
# 按同样规则统一格式，然后再次检查。
python3 tools/format_check.py --write
python3 tools/format_check.py --report results/format-check.json
```

安装只在 `labs/.venv-format/` 中放 Python/Node 工具，rustfmt 使用现有 Rust 工具链。未配置原生 Windows 安装脚本，Windows 使用已完成 foundations 的 WSL2 环境。

检查器扫描根目录文件、三个内容目录与`.vscode/`、`.obsidian/`共享设置中的实际文件，包含尚未加入 Git 的文件；遇到缺工具、版本不一致、格式差异会失败，不能出现“无 Git 文件所以全通过”。`build/`、`target/`、虚拟环境、第三方依赖、`results/`、生成文件与锁文件排除；`docs/evidence/` 中原始 JSON/CSV/图片不重写，保留实验来源，README 仍检查。

## 编辑器设置

用VS Code打开仓库根目录，[共享设置](../../.vscode/settings.json)为C/C++/CUDA、Python、Rust、Shell、Markdown和JSON/YAML选择formatter，并将新终端的工作目录设为`labs/`。[扩展推荐](../../.vscode/extensions.json)沿用foundations并补充Rust支持。先运行安装脚本，编辑器再使用`labs/.venv-format/`中的固定版本工具；Rust使用现有rustfmt。

配置已回到根目录，`.editorconfig`和各formatter的向上查找能够正常生效，无需手动指定旧的`labs/format/`路径。统一脚本也支持从根目录运行`python3 labs/tools/format_check.py`。Obsidian基础设置保存在根目录`.obsidian/`，个人workspace、cache、plugins不进入版本管理。

本次仅迁移共享配置并调整发现路径；实验实现、课程任务和已有原始验证记录保持原样。历史日志中出现的旧配置路径表示当时的执行环境。
