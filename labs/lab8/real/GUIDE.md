# R 桥接指南：在 8GB 卡上用 TRL 跑 GRPO，并导出长度 trace

[lab8 任务书](../README.md) · [讲义 D8](../../../course/modules/08-rollout-scheduling.md) · [资源映射](../resources.md)

> [!warning] 未在本机验证
> 本指南写于 2026-10-18，**没有在课程机器上安装或运行过**。版本号、默认值和参数名来自 TRL 1.16.0 的 PyPI 元数据与官方文档、verl 与 vLLM 的官方文档（同日核验）。显存和耗时是粗估，下文逐项标了“未验证”。照做时请把实际的 `pip freeze`、显存峰值和每步耗时记进报告；和本指南不一致的地方，以实测为准。

lab8 的 R1–R3 只需要 Rust，回放的是仓库里那份直接采样得到的 trace（Qwen3-0.6B，**不来自 GRPO 训练**）。本指南讲怎样得到一份**训练中**的 trace：每一步的策略版本都在变，长度分布随训练漂移。这一部分是选做，不计入 R 阶段的自动测试，计入报告分。

## 1. 框架选择

| 框架                      | 8GB 单卡                 | 依据                                                                                                                                                                                                                                                  |
| ------------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TRL `GRPOTrainer`（推荐） | 可行（`use_vllm=False`） | TRL 1.16.0（2026-10-15 发布）要求 Python $\ge$ 3.11、`transformers>=4.56.2`、`accelerate>=1.10.1`，不钉 torch；`use_vllm` 默认 False，生成走 `model.generate()`；`beta=0.0` 默认不加载参考模型；`gradient_checkpointing` 默认开启；支持 `peft_config` |
| verl                      | 不可行                   | [quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html) 要求单卡至少 24GB，并使用 vLLM rollout                                                                                                                                      |
| OpenRLHF                  | 不可行                   | README 没有单卡示例，起步配置 8 张 A100；钉死 `vllm==0.29.0` 等版本                                                                                                                                                                                   |

**不要装 vLLM。** 当前 vLLM 钉 `torch==2.13.0`，和课程环境的 torch 2.14.1 不能共存；[安装文档](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)说默认 wheel 按 CUDA 12.9 编译。按 [CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) 的驱动表，CUDA 12.9 在 Windows 上要求驱动 $\ge$ 576.02，12.8 要求 $\ge$ 570.65，本机驱动 566.24 只满足 12.6。WSL 用的是 Windows 驱动，在 WSL 里装不了更新的驱动。cu129 的 wheel 能否靠 minor version compatibility 在 566.24 上跑通（vLLM 依赖 Triton/FlashInfer 的 JIT）没有验证，不要指望。以后想用 vLLM，先把 Windows 驱动升级到 $\ge$ 580，再另建 venv。

## 2. 独立环境（不要装进 `~/venvs/mlsys`）

在 WSL 里：

```bash
python3.12 -m venv ~/venvs/mlsys-rl
source ~/venvs/mlsys-rl/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install "trl==1.16.0" peft datasets     # 不装 trl[vllm]
pip freeze > ~/mlsys-rl-freeze.txt           # 附进报告
python -c "import torch, trl, transformers; print(torch.__version__, torch.cuda.is_available(), trl.__version__, transformers.__version__)"
```

PyTorch cu126 索引上有 `torch-2.14.1+cu126` 的 cp312 wheel（已核验 URL 存在）。`pip` 会给 TRL 选一个满足 `transformers>=4.56.2` 的版本；它和课程 venv 里的 transformers 5.18.0 是否相同、两者组合是否有兼容问题，**未验证**。

模型和数据：

- 模型 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)，verl quickstart 用的也是它。下载到 `~/models/Qwen2.5-0.5B-Instruct`（约 1GB）。
- 数据 [GSM8K](https://huggingface.co/datasets/openai/gsm8k)（`openai/gsm8k`，MIT 许可；train 7,473 条、test 1,319 条；答案最后一行是 `#### <答案>`）。
- Qwen3-0.6B 默认开思考模式，回答更长、长尾更明显，适合做 trace，但显存更紧，0.5B 级别模型的长度分布能否代表大模型也没有证据。报告里把这一点写成局限。

## 3. 显存估算

参数量 $N\approx0.49\times10^9$。

| 配置                                                 | 权重 + 梯度 + Adam 状态                                            | 结论                       |
| ---------------------------------------------------- | ------------------------------------------------------------------ | -------------------------- |
| fp32 全参（模型以字符串传入、没指定 dtype 时的默认） | $N\times(4+4+8)\,\text{B}\approx7.9\,\text{GB}$                    | 必然 OOM                   |
| bf16 全参                                            | $N\times(2+2+4)\,\text{B}\approx3.9\,\text{GB}$，另加激活和生成 KV | 可能放得下，**未验证**     |
| bf16 + LoRA（$r=16$，注意力投影）                    | 约 1GB 权重 + 很小的 LoRA 状态                                     | 粗估峰值 4–7GB，**未验证** |

所以 `model_init_kwargs={"dtype": "bfloat16"}` 必须设。生成阶段 8 条 $\times$ 512 token 的 KV：Qwen2.5-0.5B 有 24 层、2 个 KV 头、head_dim 64（以模型的 `config.json` 为准），每 token 的 KV 为 $24\times2\times2\times64\times2\,\text{B}=12\,\text{KB}$，8 条 $\times$ 约 600 token 约 60MB，不是瓶颈；瓶颈在反向时的激活，靠 `gradient_checkpointing` 压。另设 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，减轻 online RL 的显存碎片。

## 4. 训练脚本骨架

以下参数名和默认值来自 TRL 文档；没有在本机运行过。

```python
# grpo_gsm8k.py -- run inside ~/venvs/mlsys-rl, in WSL, with the GPU lock:
#   flock /tmp/mlsys-gpu.lock python grpo_gsm8k.py
import json, re, time
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "/home/<you>/models/Qwen2.5-0.5B-Instruct"
SUFFIX = "\nSolve step by step. End with a line '#### <integer>'."
MAXLEN = 512

ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.map(lambda row, i: {"prompt": [{"role": "user", "content": row["question"] + SUFFIX}],
                            "gold": row["answer"].split("####")[-1].strip().replace(",", ""),
                            "prompt_id": i}, with_indices=True)

TRACE = open("grpo-trace.jsonl", "a", encoding="utf-8")

def correctness(prompts, completions, completion_ids, gold, prompt_id, trainer_state, **kw):
    """Reward = exact integer after the last '####'; also writes one trace row per sample.
    Rows are grouped by consecutive identical prompt_id (num_generations per group)."""
    rewards, step = [], trainer_state.global_step
    for i, (text, ids) in enumerate(zip(completions, completion_ids)):
        content = text[0]["content"] if isinstance(text, list) else text
        tail = content.rsplit("####", 1)[-1] if "####" in content else ""
        m = re.search(r"-?\d[\d,]*", tail)
        r = float(m is not None and m.group(0).replace(",", "") == gold[i])
        rewards.append(r)
        TRACE.write(json.dumps({
            "prompt_id": int(prompt_id[i]), "group": int(step) * 100000 + int(prompt_id[i]),
            "round": int(step), "sample_idx": i, "policy_version": int(step),
            "response_tokens": len(ids), "truncated": len(ids) >= MAXLEN,
            "gen_ms": None, "reward": r}) + "\n")
    TRACE.flush()
    return rewards

config = GRPOConfig(
    output_dir="out-grpo", model_init_kwargs={"dtype": "bfloat16"}, bf16=True,
    num_generations=8, per_device_train_batch_size=8, gradient_accumulation_steps=4,
    max_completion_length=MAXLEN, temperature=1.0, beta=0.0, epsilon=0.2,
    loss_type="dapo", scale_rewards="group", mask_truncated_completions=False,
    use_vllm=False, logging_steps=1, max_steps=20, report_to="none",
)
peft = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                  task_type="CAUSAL_LM")
trainer = GRPOTrainer(model=MODEL, reward_funcs=correctness, args=config,
                      train_dataset=ds, peft_config=peft)
trainer.train()
```

要点：

- 奖励函数的关键字参数里有 `prompts`、`completions`、`completion_ids`、`trainer_state`，以及数据集中除 `prompt` 以外的列（这里是 `gold` 和 `prompt_id`），返回 `list[float]`。这样不改 TRL 源码就能写 trace。
- `per_device_train_batch_size=8` 配 `num_generations=8`，每个设备步是 1 个 prompt 的 8 条。`policy_version` 用 `global_step`；梯度累积的 4 个微步共用同一个版本。
- **先跑 `max_steps=20`**，记下每步耗时（TRL 日志的 `step_time`）和 `torch.cuda.max_memory_allocated()`，再决定跑多少步。TRL 文档唯一的参考是 Qwen2.5-0.5B 在 DeepMath-103K 上用 8 卡约 1 天，单卡耗时**未验证**。
- 对照 Dr. GRPO：另跑一组 `scale_rewards="none"`、`loss_type="dr_grpo"`，比较长度曲线。
- 做长度 trace 时**不要**打开 `mask_truncated_completions`，否则截断样本不进 loss，分布变了（讲义 §7.1）。

## 5. 时间字段怎么补

TRL 的 `profiling/Time taken: transformers.generate` 等指标是**每个生成 batch** 的耗时，不是每条样本的。HF `generate` 是静态 batch，单条样本的完成时刻只能近似：

- 简单办法：在奖励函数里把 `gen_ms` 写成 `null`，再用本 lab 实测的线性模型补齐，记作 modeled。lab8 参考 trace 上的拟合是 $\text{gen\_ms}\approx27.3\cdot\text{tokens}+36$（batch 8，RTX 4060 Laptop），换模型或 batch 后要重拟合。
- 精确办法：仿照 [`sample_trace.py`](sample_trace.py) 的 `StepClock`，给 `generate` 传一个 `StoppingCriteria`，在每个 decode 步记时间戳和各序列是否已出 EOS。TRL 内部调用 `generate` 的位置要改源码或 monkeypatch，**未验证**。

Rust 的 `RolloutTrace::parse` 要求 `gen_ms` 是数字。补齐后再回放：

```bash
python - <<'EOF'
import json
rows = [json.loads(l) for l in open("grpo-trace.jsonl")]
with open("grpo-trace-filled.jsonl", "w") as f:
    f.write(json.dumps({"type": "header", "label": "TRL GRPO training trace, gen_ms modeled from tokens"}) + "\n")
    for r in rows:
        r["gen_ms"] = round(27.3 * r["response_tokens"] + 36, 3)
        f.write(json.dumps(r) + "\n")
EOF
mkdir -p results/lab8-real && cp grpo-trace-filled.jsonl results/lab8-real/
python tools/run.py --lab 8 --impl student --mode bench --part R     # also replays results/lab8-real/*.jsonl
```

注意 trace 里 `round` 等于训练步，每轮只有一个组。回放时“下一轮的预测来自上一轮”，在训练 trace 上的含义是“用旧策略的长度预测新策略的长度”，这正是讲义 §8.3 要你测的预测漂移。

## 6. 预计会遇到的问题

| 现象                                                       | 原因与处理                                                                                                                                          |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 一加载就 OOM                                               | 没设 `model_init_kwargs={"dtype": "bfloat16"}`，默认 fp32                                                                                           |
| `pip install trl[vllm]` 后 torch 被降级                    | vLLM 钉 torch 2.13；删掉 venv 重建，不装 vllm extra                                                                                                 |
| vLLM 报 `cudaErrorCallRequiresNewerDriver` 或 PTX 相关错误 | 驱动 566.24 太旧（见第 1 节），不要在本机用 vLLM                                                                                                    |
| 想用 verl                                                  | quickstart 要求单卡 $\ge$ 24GB，本机不行                                                                                                            |
| 每步很慢                                                   | HF `generate` 无连续批处理；可试 `use_transformers_continuous_batching=True`（需 transformers $\ge$ 5.8.0，文档称适合单卡显存受限环境，**未验证**） |
| 奖励一直为 0                                               | 0.5B 模型不按格式输出；检查 `####` 的解析，或在提示里给一个示例                                                                                     |
| 长度一直涨                                                 | 检查 `loss_type` 与 `scale_rewards`（讲义 §2.3）                                                                                                    |

## 7. 交付

报告里附：`pip freeze`、`nvidia-smi` 的驱动与显存、20 步试跑的每步耗时和显存峰值、训练 trace（或其前 1MB）、R 阶段对它的回放表，以及和参考 trace 的长度分布对比。凡是本指南标了“未验证”的项，写明你实测的结果。
