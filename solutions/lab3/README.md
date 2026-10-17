# Lab 3 参考实现导读

参考答案覆盖全部 15 个 TODO：[`pipeline_student.py`](src/pipeline_student.py)含A1–A5/B1–B5，[`execution_student.py`](src/execution_student.py)含C1–C4，[`fusion_student.cuh`](src/fusion_student.cuh)含C5。共 264 行有效代码，包含 DP 实现及其接入执行流程的代码。

```bash
# cwd: mlsys-intermediate/labs
python tools/run.py --lab 3 --impl reference --mode test --device cpu --part all
python tools/run.py --lab 3 --impl reference --mode test --device cuda --part C
python tools/run.py --lab 3 --impl reference --mode bench --device cuda --part all
```

A 先按时间切分数据，再统计训练频次，分别保存 median、MAD 与 first premium。成本表按实测 bound 查找，不强行用线性 cost 拟合所有 shape。padding 按请求频次加权；总变差在联合 support 上计算，因此能检测出平均长度不变、shape 却完全变化的 trace。

B 的 DP 状态由已覆盖的 shape 数和已用的 bucket 数构成，先枚举再回溯，不采用贪心的最近 bucket 选择。穷举测试验证其在成本模型内的最优性。cache 插入只有在找到合法 victim 后才修改状态；lease 计数保护仍在使用的 artifact。admission 使用预计调用数与 MAD 裕量，恰好回本时拒绝；应用 cache 的容量与 Dynamo 内部缓存需要区分。

C2 实际调用 Inductor，保留每个候选函数，并单独报告整个校准过程的 wall time。C3按设备/dtype/width/bound命名应用entry，guard通过且预测收益为正才使用；双输入同时补行、输出切回、finally释放lease。C4使用训练信息产生所有决策，再逐请求对eager核对hold-out。分支测试固定profile只用来保证覆盖，实际benchmark必须使用C2测量。

C5 使用一 block/row 和 shared 二叉归约；尾部线程贡献零，但仍参与 barrier，分母使用真实 width，并保持 gamma 广播正确。当前 C5 只用于局部融合对照，没有注册到 C2 图中，因此不能声称完整编译器选择了自定义 kernel。

A/B 包含 10 项测试和 2 项线性成本接口检查；C1–C4 实际调用 Inductor。CPU runner 会打印 C5 NOT RUN，不能据此认定 GPU 验收通过。首次调用可能复用缓存，因此报告中不能将该数值称为完全冷启动的编译耗时。本实验完成的教学流程也不等于 Mirage/MPK 完整系统复现。
