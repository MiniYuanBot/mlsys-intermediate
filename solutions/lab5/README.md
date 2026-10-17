# Lab 5 参考答案与运行证据

[任务书](../../labs/lab5/README.md) · [评分](../../labs/lab5/grade.md) · [试运行原始记录](validation.json)

答案在 [cache_student.rs](src/cache_student.rs) 与 [replay_student.rs](src/replay_student.rs)。A/B/C 全部 14 项 TODO 均已实现；独立测试、benchmark 和 CLI 保留在 labs 侧。

## A 前缀身份与拓扑

完整token祖先链加namespace构成key，只有完整块入表。匹配首miss即停止。victim排除pin、protected以及任何存在后代的节点；顺序稳定，便于定位不同策略之间的第一次分歧。validate检查直接父亲、块对齐和refs祖先关系。

stats区分引用总数与被pin节点数。保存完整 prefix 导致的 key payload 增长，是本教学实现的成本；它不代表 GPU KV 大小，也不包含容器开销。测试使用直接构造的fixture，A不会调用B/C。

## B 为什么先clone再发布

插入操作先在副本上计算 missing 并执行淘汰，全部完成后才赋值给 self。因此，即使选出一个 victim 后才发现另一条路径被 pin，操作失败也不会改变任何节点或 clock。resize 使用同样的策略；namespace 失效则先检查所有 pin，再执行删除。

acquire对连续祖先加refs；release消费Lease，从叶到根减引用但保留缓存。Rust所有权避免相同Lease二次释放，API仍假定将Lease交回原Cache；生产实现可使用cache ID或RAII guard强化归属。测试验证并发共享、失败快照、跨namespace隔离与0容量。

## C 完成事件发布缓存

入队只表示请求到达，不会产生 KV。派发时冻结 hit 与 admission 长度，运行期间持有命中路径的 lease。请求完成后先 release，再 insert 已计算的 prefix，随后新请求才能命中。两个同时 miss 的请求都要承担 prefill 成本，这一点由手算时间线单独验证。

TwoHit 记录完整块祖先的近期访问，历史表也有容量限制。它是用于教学的准入基线，不是 TinyLFU，也不是 SGLang 的完整策略。LPM 优先考虑命中，但等待超过 age 阈值后改为先到者优先；在过载情况下，这一规则不保证固定的响应时间上界。

指标按总token加权，而不是平均每请求命中率。capacity failure只统计事务插入因空间/pin失败；TwoHit决定少缓存不属于这个失败计数。C的worker是抽象独立服务台，不等价于单GPU多batch。

## 已执行的检查

2026-10-17在CPU运行 `exercise.py --mode test --part A/B/C/all`，分别5/5/5/15个测试通过；all benchmark产生A12、B8、C36行。学生A stub返回101且明确未实现。原始命令输出与源码SHA在validation.json，最终统一格式化后的哈希以项目runner记录为准。

运行合成负载时已观察到反例：capacity=4 时，two-hit+LPM 的整体 hit 约为 0.236，低于 two-hit+FIFO 的约 0.239，因此不能预设组合策略一定更好。图中的 p95 与 wait 来自显式成本模型，未测量真实 GPU、网络或内核吞吐。
