# 原始实验数据

`run.py` 在此保存带唯一时间戳的 JSON，记录命令、源码 SHA256、环境、stdout/stderr 和退出状态。生成的数据默认不进入 Git；需要共享时，另行归档所选结果并保留生成命令。

从 `labs/` 目录执行，先列出记录中的表，再按编号导出：

```bash
python tools/extract.py results/某次运行.json
python tools/extract.py results/某次运行.json --table 0 --out results/table.csv
```

文件名与表编号替换为实际值。例如，lab2 的 CUDA 表包含 `case,rows,cols,group,batch,mode,device_us`，导出该表后可画出 batch=1、group=128 时四条路径随矩阵大小的变化：

```bash
python tools/plot.py results/table.csv --x cols --y device_us --group mode \
  --filter case=kernels --filter group=128 --filter batch=1 --out results/latency.png
```

其他实验的表头不同，先核对列名再选择参数。`plot.py` 的误差条是样本标准差；单条数据没有可估计的波动，需要多次独立运行后汇总相同配置。报告中保留原始数值和配置，不能只提交图片。
