# Lab 4 参考答案与运行证据

[任务书](../../labs/lab4/README.md) · [评分](../../labs/lab4/grade.md) · [试运行原始记录](validation.json)

参考答案在 [policy_student.rs](src/policy_student.rs) 与 [engine_student.rs](src/engine_student.rs)。两份文件的 BEGIN/END 标记与学生文件完全对应，覆盖 A/B/C 全部 14 项必修任务；tests 和 benchmark 保留在学生侧，替换答案时不会替换测试。

## A 为什么这样算

A1按首次失败位置放概率质量：前缀存活概率乘当前失败概率；最后桶是全接受质量。A2由分布求一、二阶矩。A3固定nearest-rank，确保小样本p95定义可重现。A4 按 context 夹取或插值，不把区间外的值当作实测结果。A5按context/k分组取中位数，缺列报错。

测试 oracle 枚举二进制接受向量，独立于按首次拒绝位置计算的实现，能发现误计后缀拒绝的问题。`a_calibration_grouping_and_missing` 检查缺测项是否被错误地当作零成本 kernel。

## B 两种预算与不完全观测

success/failure不含先验，折扣每轮作用到所有位置。Beta平滑在读均值时添加；Wilson形式直接用折扣计数，仅作为保守分数。B1不把未知后缀记失败，B4检测负向变化后允许重置。

B3 通过枚举求单请求在 round bound 内的最优解；B5 对多请求预算采用边际贪心，并保留了不最优的反例。两者的目标、可行域和最优性不能混用。B5 独立于 C 的单服务台，用最小模型帮助理解 AdaServe 式共享资源问题，并非上游系统的复现。

## C 时间和守恒

引擎只在验证完成时提交。到达发生在一轮中间时，先入ready再把运行请求回队，因此手算r0=[1,3]、r1=[2]。空ready时跳到下一个arrival；超过deadline也继续排空，显式计入infeasible而非删掉请求。

每个潜在接受事件由seed/request/emitted/depth决定。控制器只读历史；真实alpha仅属于环境。summary逐请求核对token数，然后按首到达至末完成算goodput，包含排空，单位req/simulated-second。

## 已执行的检查

2026-10-17在CPU实际运行 `exercise.py --mode test --part A/B/C/all`，分别4/5/5/14个测试通过；all benchmark输出A20、B120、C60行。学生A stub返回101且明确 `not yet implemented`。命令和完整stdout/stderr见validation.json。该记录包含源码SHA；之后格式化可能改变字节SHA，最终统一runner记录以项目验收为准。

`conservative_no_probe`明确是负基线：在当前递增成本表下，冷启动的零下界使它永远停在k0，新增回归验证与普通decode的提交时间线相同。B的固定观测回放不代表在线恢复能力。验收不要求该负基线胜出：在低负载、已过deadline或缺乏探索时都可能表现不同。分析时应使用最小反例解释错误，不能只看平均速度。参考答案未实现完整 AdaServe、多 GPU 真实服务或真实采样正确性，也不覆盖开放挑战。
