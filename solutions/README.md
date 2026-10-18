# 参考实现

[参考实现索引](index.md) · [仓库首页](../README.md)

## 使用方法

先独立完成实验，再对照。命令在 `labs/` 目录运行；参考实现在临时副本中运行，不覆盖学生代码。

```bash
python tools/run.py --lab N --impl reference --mode test       # 运行参考实现
python tools/fault_inject.py --lab N [--device cuda]           # 错误注入：检验测试能否抓住典型错误
```

各 lab 的解题导读在 `labN/README.md`，错误注入表在 `labN/faults.json`；约定见[参考实现索引](index.md)。
