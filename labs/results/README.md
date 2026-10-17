# 原始实验数据

`run.py` 会在此保存带唯一时间戳的 JSON，记录命令、源码 SHA256、环境、stdout/stderr 和退出状态。大体积结果默认被忽略；正式研究时，选择需要共享的结果归档到 artifact，并保留生成命令。

```bash
python tools/extract.py results/某次运行.json
python tools/extract.py results/某次运行.json --table 0 --out results/table.csv
python tools/plot.py results/table.csv --x n --y microseconds --group method --filter group=128 --out results/latency.png
```

上述命令从 `labs/` 目录执行。上例列名适用于 lab2。lab7 有 cache 和 H2D 两张表，先不加 `--table` 查看编号。plot.py 的误差条是样本标准差，不是置信区间。已随项目归档的交付验证记录位于 [docs/evidence/](../docs/evidence/README.md)。
