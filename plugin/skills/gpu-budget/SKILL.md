---
name: gpu-budget
description: Estimate how much GPU memory, time and money training, fine-tuning (full, LoRA, QLoRA) or serving an LLM takes with gpu-budget, each part shown with its formula (weights, gradients, optimizer state, activations, KV cache), for ZeRO/FSDP, tensor, pipeline and expert parallelism, quantization and speculative decoding. Use when the user asks whether a model fits on a GPU, how many GPUs they need, how much VRAM a fine-tune or a long context takes, or how long and how much a training run would cost.
license: MIT
compatibility: Needs Node 22+. Runs gpu-budget with npx from GitHub (github:zch6a/gpu-budget) unless installed. No network use beyond fetching the tool; any Hugging Face config.json works as the model.
metadata:
  homepage: https://github.com/zch6a/gpu-budget
---

# GPU memory, time and cost with gpu-budget

```bash
npx github:zch6a/gpu-budget infer llama-3.1-70b --quant int4 --kv fp8 --ctx 32768 --batch 4 --gpu h100-sxm
npx github:zch6a/gpu-budget finetune llama-3.1-8b --method qlora --rank 16 --gpu rtx-4090
npx github:zch6a/gpu-budget train llama-3.1-70b --gpus 64 --zero 3 --seq 8192 --recompute full --tokens 2e12 --price 2.5
npx github:zch6a/gpu-budget models     # built-in models;  gpus: built-in GPUs
```

- The model is a built-in name or a path to the model's Hugging Face `config.json`; ask for it (or download only that
  small file) when the model is not built in. Never guess an architecture.
- Read back the per-GPU parts and the total against the GPU's memory, and the verdict. Exit status 1 means it does not
  fit, and the tool says what to try: run the suggestion and report the result rather than giving up.
- These are estimates of what the tensors need; frameworks add fragmentation and buffers. Tell the user to leave
  headroom (10–20%), and say which assumptions matter (FlashAttention, recomputation, micro-batch, sequence).
- Time needs `--tokens` and assumes an MFU (`--mfu`, 0.4 by default); cost needs the user's `--price` per GPU-hour.
  `--json` gives the numbers for further work.
- The page (https://zch6a.github.io/gpu-budget/) takes the same settings and its link can be shared.
