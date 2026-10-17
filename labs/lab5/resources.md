# Lab 5 公开资料与整合说明

核验日期：2026-10-17。实验实现独立编写；借鉴语义和实验组织，不拷贝上游源码。以下入口均来自作者或项目官方。

| 来源                                                                                                                                                                          | 要读的位置                                              | 实验对应                                      | 简化边界                                                  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------- |
| [SGLang论文](https://arxiv.org/abs/2312.07104)                                                                                                                                | RadixAttention与缓存感知调度                            | A完整前缀、B生命周期、C2 LPM调度              | 不实现GPU attention和连续batch                            |
| [SGLang固定radix_cache.py](https://github.com/sgl-project/sglang/blob/456231e2517dd29e78c77d20126c893b3f1a6675/python/sglang/srt/mem_cache/radix_cache.py)                    | `match_prefix`、`evict`、`inc_lock_ref`、`dec_lock_ref` | A3叶淘汰、B2/B3祖先pin/release                | 使用BTreeMap+完整prefix，未做压缩路径/真实KV分配          |
| [vLLM v0.21.0前缀缓存设计](https://docs.vllm.ai/en/v0.21.0/design/prefix_caching/)                                                                                            | key、refcount、分配与回收                               | A1 namespace/祖先身份、B所有权                | vLLM哈希块管理不要求照搬本lab叶拓扑约束；不要混称实现相同 |
| [CacheBench workload配置](https://cachelib.org/docs/Cache_Library_User_Guides/Configuring_cachebench_parameters/)                                                             | 合成分布、replay、一次性访问                            | C中的热点/扫描/恢复，容量×策略实验            | KV store key-value与模型KV不是同一负载                    |
| [CacheBench扩展结构](https://cachelib.org/docs/Cache_Library_User_Guides/Developing_for_Cachebench/)                                                                          | generator/stressor接口边界                              | 分开Request生成与cache/replay，独立测状态守恒 | 不引入CacheLib编译依赖，也不声称硬件吞吐                  |
| [CacheBlend作者页面](https://www.microsoft.com/en-us/research/publication/you-only-prefill-once-combining-cached-knowledge-for-large-language-model-serving-with-cacheblend/) | 非前缀知识组合与重算                                    | 开放挑战中辨别exact-prefix边界                | 本lab不做任意文档KV复用与质量评估                         |

## 源码阅读交付

在固定SGLang文件画三条调用链：命中并锁定、完成解锁、淘汰可回收leaf。指出root、运行请求和可淘汰节点的差异；再用本lab的两lease测试说明哪些不变量相同。

对照vLLM设计，解释“祖先参与key身份”与“必须保留父节点”的区别：前者是正确性语义，后者是本lab/SGLang树式管理的结构策略，哈希块表可以采用不同结构。

C1 TwoHit与C2 age保护是独立设计的教育基线，不声称来自某篇前沿论文的原算法。它们让你在已有工业机制周围提出可测量的改进；真正的新颖性需要进一步检索与导师讨论。其它课程入口见[资源总表](../../course/readings/courses-and-labs.md)。
