# 实验索引

[课程索引](../course/index.md) · [环境](setup.md) · [参考答案](../solutions/index.md)

所有命令从 `labs/` 目录执行。A 是小问题与基本接口，B 是核心实现，C 是论文启发的缩小机制与测量，D 是开放研究挑战。完成全部 8 个 lab 不是开始科研的前置条件。

每个 lab 分为基础A、核心B、前沿C，三个阶段都有注释契约、待填代码、完整参考实现和测试；报告与开放挑战另列。必修时长依次为 **24、26、26、24、25、26、26、23h**，合计 **200h**；讲义共40h。估算包含实现、独立反例、调试、测量和报告。见[工作量审计](docs/workload.md)及[36h/周路线](../course/guides/roadmap.md)。

| Lab                 | 三阶段重点                           | 语言         | CPU范围                     | CUDA范围                     |
| ------------------- | ------------------------------------ | ------------ | --------------------------- | ---------------------------- |
| [1](lab1/README.md) | INT4→流式缓存→PQ检索取回             | C++17        | 完整A/B/C                   | 真实模型扩展                 |
| [2](lab2/README.md) | 校准→多种解包kernel→激活感知量化     | C++/CUDA/PTX | A与C主机算法；单独选阶段    | B及C kernel集成              |
| [3](lab3/README.md) | trace成本→分桶缓存→真实Inductor      | Python/CUDA  | A/B及真实CPU Inductor C1–C4 | 真实GPU Inductor与C5融合     |
| [4](lab4/README.md) | 接受统计→预算控制→SLO事件队列        | Rust         | A/B/C离散事件系统           | 真实服务扩展                 |
| [5](lab5/README.md) | 前缀拓扑→事务lease→并发复用调度      | Rust         | A/B/C完整trace实现          | 真实KV/引擎扩展              |
| [6](lab6/README.md) | LoRA生命周期→激活规划→offload训练    | PyTorch      | 全部算法/真实CPU训练        | 实际搬运、GPU梯度、显存峰值  |
| [7](lab7/README.md) | 路由INT8→容量缓存→专家执行流水线     | C++/CUDA     | 路由/缓存/CPU专家数值       | 真实量化专家卸载与单双槽执行 |
| [8](lab8/README.md) | 预测优势→分段调度→有界版本队列与训练 | Rust         | A/B/C含真实小型策略梯度     | 完整LLM rollout扩展          |

默认测试所有阶段；分阶段例如 `python tools/run.py --lab 1 --part A --mode test`。`--impl reference`在临时副本叠加完整答案，不改学生文件。各阶段CPU/GPU边界必须按表识别，命令退出0不表示未运行的硬件部分通过。

## 共用工具

- [doctor.py](tools/doctor.py)：只读环境检查。
- [run.py](tools/run.py)：学生/参考、测试/benchmark；临时副本；保存完整日志与哈希。
- [extract.py](tools/extract.py)：列出运行记录中的表、导出 CSV。
- [plot.py](tools/plot.py)：标准 matplotlib 图与样本标准差；需单独安装 matplotlib。
- [capture_model.py](tools/capture_model.py)：从**已有本地模型**采集串行 decode 与可选 K/V；不自动下载模型。
- [record.py](tools/record.py)：为额外的编译/真实模型实验保存命令与日志。
- [analyze_kv.py](tools/analyze_kv.py)：真实 K/V 的离线分布热图与两轴误差诊断。
- [materialize.py](tools/materialize.py)：创建独立 lab 副本用于调试/profiler，不覆盖现有目录。
- [check_links.py](tools/check_links.py)：仓库内 Markdown 文件链接检查。
- [workload_audit.py](tools/workload_audit.py)：校验学生/答案 TODO 配对，统计各阶段实际填空区域代码。
- [format_check.py](tools/format_check.py)：使用 foundations 的格式参数与固定版本，统一检查学生/参考源码及课程文档。

共用命令与输出说明见 [tools/README.md](tools/README.md)。工具运行成功只验证其覆盖的代码；人工报告、学生新增测试/测量与挑战分别验收。

报告写在 `reports/`，原始数据在 `results/`；不要用图片替代原始数值。验收见每 lab 的 grade.md；整体验证状态见 [validation](docs/validation.md)。
