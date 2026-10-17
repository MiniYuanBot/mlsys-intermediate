# Revision 3 证据索引

本目录保存本轮实际运行的命令、环境、哈希和输出。正式采用的 21 个参考调用全部通过，另有 8 个学生 A 阶段的预期失败对照。机器可读清单见[summary.json](summary.json)；每个 bench 表均已导出为 CSV，并核对了列数和行数。

| Lab/阶段 | 设备 | 测试/测量 | 原始记录                                                                       | 表格行数    |
| -------- | ---- | --------- | ------------------------------------------------------------------------------ | ----------- |
| Lab1/all | cpu  | test      | [lab1-reference-test-cpu.json](lab1-reference-test-cpu.json)                   | —           |
| Lab1/all | cpu  | bench     | [lab1-reference-bench-cpu.json](lab1-reference-bench-cpu.json)                 | 18          |
| Lab2/all | cuda | test      | [lab2-reference-test-cuda.json](lab2-reference-test-cuda.json)                 | —           |
| Lab2/all | cuda | bench     | [lab2-reference-bench-cuda.json](lab2-reference-bench-cuda.json)               | 3, 40       |
| Lab3/all | cuda | test      | [lab3-reference-test-cuda.json](lab3-reference-test-cuda.json)                 | —           |
| Lab3/all | cuda | bench     | [lab3-reference-bench-cuda-clean.json](lab3-reference-bench-cuda-clean.json)   | 19, 120     |
| Lab4/all | cpu  | test      | [lab4-reference-all-test-cpu.json](lab4-reference-all-test-cpu.json)           | —           |
| Lab4/all | cpu  | bench     | [lab4-reference-all-bench-cpu.json](lab4-reference-all-bench-cpu.json)         | 20, 120, 60 |
| Lab5/all | cpu  | test      | [lab5-reference-all-test-cpu.json](lab5-reference-all-test-cpu.json)           | —           |
| Lab5/all | cpu  | bench     | [lab5-reference-all-bench-cpu.json](lab5-reference-all-bench-cpu.json)         | 12, 8, 36   |
| Lab6/all | cuda | test      | [lab6-reference-test-cuda.json](lab6-reference-test-cuda.json)                 | —           |
| Lab6/all | cuda | bench     | [lab6-reference-bench-cuda.json](lab6-reference-bench-cuda.json)               | 24          |
| Lab7/all | cuda | test      | [lab7-reference-test-cuda.json](lab7-reference-test-cuda.json)                 | 12          |
| Lab7/all | cuda | bench     | [lab7-reference-bench-cuda-final2.json](lab7-reference-bench-cuda-final2.json) | 9, 6, 6, 12 |
| Lab8/all | cpu  | test      | [lab8-reference-test-cpu-final.json](lab8-reference-test-cpu-final.json)       | —           |
| Lab8/all | cpu  | bench     | [lab8-reference-bench-cpu-final.json](lab8-reference-bench-cpu-final.json)     | 3, 12, 108  |
| Lab6/all | cpu  | test      | [lab6-reference-test-cpu.json](lab6-reference-test-cpu.json)                   | —           |
| Lab6/all | cpu  | bench     | [lab6-reference-bench-cpu.json](lab6-reference-bench-cpu.json)                 | 24          |
| Lab2/A   | cpu  | test      | [lab2-reference-A-test-cpu.json](lab2-reference-A-test-cpu.json)               | —           |
| Lab2/C   | cpu  | test      | [lab2-reference-C-test-cpu.json](lab2-reference-C-test-cpu.json)               | —           |
| Lab3/all | cpu  | test      | [lab3-reference-all-test-cpu.json](lab3-reference-all-test-cpu.json)           | —           |

同一流程中，不同策略的 JSONL 可能包含不同的可选字段。CSV 按字段的有序并集导出，缺失字段留空；完整的嵌套 calibration 数据仍保留在原 JSON 中。表编号和字段说明见 summary。

## 审计与历史

- [填空工作量](workload.json)、[foundations对比](baseline-comparison.json)。
- [格式检查](format-check.json)、[交付检查](quality-check.json)。
- Lab3旧`lab3-reference-bench-cuda.json`存在stdout交错，不能用作JSONL数据；正式使用clean记录。
- Lab7旧bench的表头/计时版本已被final2替代；保留旧日志便于追查，不能混合做性能对照。
- 有效 record 中的 source hash 对应执行时的文件内容。之后的格式统一或文字注释更新不代表重新测量了性能；当前交付的源码清单保存在 quality-check 中。
- [上一版证据](../revision2/README.md)与[项目验证范围](../../validation.md)分别列历史和当前结论。
