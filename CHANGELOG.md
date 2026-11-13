# Changelog

## 0.1.0 — 2026-11-13

First release.

- `gpu-budget train | finetune | infer <model>`: memory per GPU, each part apart (weights, gradients, optimizer state,
  activations, KV cache, draft model, overhead), whether it fits, and what to try when it does not; with `--tokens`,
  compute, time at an MFU and cost at a price.
- Training: mixed-precision AdamW (or 8-bit, BF16 states, SGD), ZeRO 1/2/3 and FSDP, tensor, pipeline and expert
  parallelism, activations with or without recomputation and FlashAttention. Fine-tuning: full, LoRA, QLoRA.
  Inference: BF16/FP8/INT8/INT4 weights, FP8/INT8 KV cache, batch and context, tensor parallelism, a draft model.
- 13 built-in models (Llama 2/3.x, Mistral, Mixtral, Qwen2.5, Gemma 2, DeepSeek-V3, GPT-2 XL, OPT, GPT-3), any
  Hugging Face `config.json`; 10 GPUs from the makers' datasheets.
- Checked against GPT-3's and Llama 3's training compute, Llama 2's GPU-hours, ZeRO's model states and vLLM's KV cache.
- The page: the same engine, the settings in the link.
