# Lab 6 参考实现导读

[17 个 TODO 的任务说明](../../labs/lab6/README.md) · [实现](src/train_student.py) · [评分](../../labs/lab6/grade.md) · [资源映射](../../labs/lab6/resources.md)

A 的重点是参数生命周期。冻结 W 仍保留输入导数；merge 保存原矩阵，unmerge 精确恢复；注入先验证后替换；adapter state 不包含基座，load 在全部验证后才写。它不是完整 PEFT 文件协议，多 adapter 与量化基座不在范围内。

B 的两个规划器分别最大化保留收益和最小化多选候选成本。后者使用 Pareto 剪枝，测试通过独立枚举验证结果，避免用相同的 DP 重复计算作为检查。SavedInventory 按 storage 去重并排除参数 view，校准使用真实 forward/backward，仍然不能把保存清单当作 allocator peak。

C 的 pack/unpack 实现实际 CUDA→CPU→CUDA 同步搬运；CPU fallback 使用 identity，不生成虚构的传输数据。混合执行使用每层 context，backward 可在 context 退出后恢复。微批 loss 用总元素数归一，clip 在累积结束后，最后只 step 一次。

measure_policy 先初始化 AdamW 状态，lr=0 保持权重用于公平策略比较，optimizer 运算仍在计时内。默认 benchmark 的规划输入来自 retain/checkpoint 校准；offload 单独测量。全部 24 个配置均执行 PyTorch 训练，延迟来自实测，而非公式估算。

```bash
# cwd: labs/
python tools/run.py --lab 6 --impl reference --mode test --device cpu
python tools/run.py --lab 6 --impl reference --mode test --device cuda
python tools/run.py --lab 6 --impl reference --mode bench --device cuda
```

参考测试有 14 个方法，覆盖每个必修 TODO；CPU 与 CUDA 的实际结果及 hash 见[验证](../../labs/docs/validation.md)。合并误差、输入与参数梯度、微批更新按浮点容差比较；base 恢复和字节数则要求精确一致。

推荐错误注入：在 base linear 使用 no_grad；加载时逐项检查逐项写入；checkpoint 关闭 RNG；对微批 mean 做简单平均；把参数 view offload。先看目标测试是否失败，再恢复。扩展到异步 offload 时，仅开启 non_blocking 还不够，必须额外验证 host buffer 的生命周期和 stream event。
