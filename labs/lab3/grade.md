# Lab 3 评分与验收（100分）

| 任务         | 分值 | 验收依据                                                       |
| ------------ | ---: | -------------------------------------------------------------- |
| A1/A2        |    8 | chronological split不泄漏；median/MAD、非法profile测试         |
| A3/A4/A5     |   12 | 手算interval cost、同均值分布反例、频次加权padding/fallback    |
| A分析        |    5 | 五个手算和计时口径；不能把first premium称为纯编译时间          |
| B1           |   12 | `dp_exhaustive`对全部分割、四种capacity/premium独立枚举        |
| B2           |    5 | 上界guard、过度padding与未知长shape fallback                   |
| B3/B4        |   10 | pin生命周期、失败事务不改变cache、两次lease不提前释放          |
| B5           |    5 | break-even等号拒绝、MAD裕量、历史预测无held-out计数            |
| B分析        |    3 | DP模型最优与真实runtime差异、application cache边界             |
| C1/C2        |    8 | 真正同步调用、真实Inductor artifact、first/warm/eager记录      |
| C3/C4        |   12 | 真实compiled和eager分支、补双输入/切回、完整closed loop oracle |
| C5           |    8 | CUDA融合在15组shape上与double CPU和双kernel基线一致            |
| C实验报告    |    8 | 稳定/突变/长尾真实trace、五次独立记录、校准总成本、负结果      |
| 可复现与工时 |    4 | 环境、缓存制度、seed、命令、原日志、实际工时                   |

A/B的10个单测在 `pipeline_test.py`；B还运行旧 `policy_test.py`的2个兼容检查。C1–C4在 `execution_test.py`实际调用Inductor；为覆盖两条路由的固定admission数字仅是测试fixture，不能当性能测量。C5在 `test.cu`。参考实现试运行证据由[项目验证记录](../docs/validation.md)汇总，不能从编译成功推断CUDA运行通过。

```bash
python tools/run.py --lab 3 --impl student --mode test --device cpu --part A
python tools/run.py --lab 3 --impl student --mode test --device cpu --part B
python tools/run.py --lab 3 --impl student --mode test --device cuda --part C
python tools/run.py --lab 3 --impl student --mode bench --device cuda --part all
```

未完成前置任务时，后续测试可以失败；不允许替换oracle让其通过。CUDA缺失时CPU C1–C4仍可验证，C5明确未完成。性能项依据测量方法和分析评分，不要求DP总是胜过dynamic，也不按报告中最大的speedup评分。编译、校准、routing和执行成本漏计会使对应性能分析项不通过。
