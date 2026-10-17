# Lab 2 参考答案导读

覆盖 [任务书](../../labs/lab2/README.md) 全部 A1–A6、B1–B6、C1–C5。答案与学生文件使用相同成对标记，位于 [quant_student.hpp](src/quant_student.hpp) 与 [gemv_student.cuh](src/gemv_student.cuh)。有效算法代码共 252 行，无需将模板手工复制到学生目录。

```bash
# cwd: mlsys-intermediate/labs
python tools/run.py --lab 2 --impl reference --mode test --device cpu --part A
python tools/run.py --lab 2 --impl reference --mode test --device cuda --part all
python tools/run.py --lab 2 --impl reference --mode bench --device cuda --part all
```

A 使用每行独立 byte stride，并在奇数尾列保留高 nibble=0。量化误差证书针对未 clipping 的 absmax 格式；A6 明确以总权重 MSE 搜索裁剪范围，因此单个 outlier 误差可以超过原半步长。通过每组的穷举 oracle，检查实现是否完成了搜索，而非仅返回 baseline。

B2 是逐元素 lane 映射，B3 是逐 byte 映射，B4 将每个解码值复用于四个 token，B5/B6 增加 K 维并行度再做确定顺序归约。四者都只实现 SIMT W4A32，不使用 Tensor Core。测量时必须计入 partial buffer 和额外 launch 的代价，不预设哪种方案始终最快。

C 的候选来自 mean-abs activation 的幂次并按 min/max 几何均值归一。搜索始终以原始 float linear 的输出为基准，每次候选从原 W/X 构造，避免尺度累计。包含 alpha=0 保证**校准集**目标不劣于相同 group 的 baseline；不保证 held-out 改善。C5 在搜索前计算 artifact 的完整字节数，将 channel 元数据一并计入预算。

测试通过独立计算和边界输入检查答案：A2 使用全部已知编码，A4 对照独立的 dense double 计算，A6/C3/C5 分别穷举候选，B 使用不规则形状和奇数组边界，C 的 GPU 路径使用冻结的 artifact，不重新校准。CPU 基础与校准测试已用于开发验证；GPU 阶段以项目最终记录为准，编译成功不能证明运行通过。

当前 runner 的 benchmark 输出数据已驻留时的局部 CUDA event 时间。当前实现不支持对完整模型准确率、AWQ 层间 folding、Marlin 线程布局/FP16/Tensor Core、H2D 与校准成本、模型吞吐作出结论。参考源码用于机制理解，不能作为完整论文复现的证明。
