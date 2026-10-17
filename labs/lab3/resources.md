# Lab 3 主源阅读与任务映射

检查日期：2026-10-17。下面资料用于具体设计，不只是扩展书单。

| 资料                                                                                                           | 已阅读的内容                                                 | 本课吸收                                                 | 未移植                                                          |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- | --------------------------------------------------------------- |
| [Stanford CS336 systems](https://github.com/stanford-cs336/assignment2-systems)                                | 官方README、作业handout的benchmark/profiling/compile实验组织 | 先正确性→稳定计时→系统比较；C将编译冷启动与稳态分离      | 多GPU/DDP和完整FlashAttention重复foundations，不在此必修        |
| [PyTorch compile教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)             | `torch.compile`调用、实际编译与时序对比、warmup行为          | C1/C2真实Inductor校准与eager oracle，first-call单独记账  | 不把教程特定GPU的速度数值迁移到4060                             |
| [PyTorch recompilation文档](https://docs.pytorch.org/docs/stable/compile/programming_model.recompilation.html) | guard失效、重新编译、缓存限制与fallback                      | B2 shape guard、C3显式fallback、C2标记缓存边界           | 应用artifact LRU不控制Dynamo内部缓存；不伪称实现编译器cache淘汰 |
| [Mirage源码](https://github.com/mirage-project/mirage)                                                         | README中程序图、搜索、编译、执行/性能查询入口                | 搜索必须有正确性验证，C4真实执行核对；有限预算下选择候选 | 没有移植multi-level graph搜索、persistent kernel、完整MPK调度   |
| [Mirage论文](https://arxiv.org/abs/2405.05751)                                                                 | 多层次tensor程序搜索与等价验证问题                           | 报告区分shape分桶搜索与算子图搜索                        | DP只在指定分割成本模型内最优，不宣称达成Mirage全搜索            |
| [CUDA编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)                               | block同步、shared memory与异步kernel计时                     | C5共享归约、不规则width、CUDA event benchmark            | 无Hopper/Blackwell专用依赖                                      |

建议用90分钟定向阅读：30min查CS336测量要求；30min读PyTorch guard/recompile；30min画Mirage“候选→验证→计时/选择”的流程。然后回答：若校准已经编译了所有候选，为什么部署replay加速不代表整个实验生命周期更快？答案应引用自己的calibration_ms与replay_ms。
