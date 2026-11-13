# Polish log

### 2026-11-13 — development

**Why it exists**: GPU memory calculators abound, but most give one number with no parts, no formula and no check.
gpu-budget shows every part (weights, gradients, optimizer state, activations, KV cache) with the formula it comes
from, and is checked against published runs.

**One engine**: plain JavaScript with JSDoc types (checked by TypeScript, `checkJs`, strict), run by Node for the
command line and loaded as-is by the page. No build step and no runtime dependency; the page imports the same files.

**What it covers**: training (mixed-precision AdamW or 8-bit, BF16 states, SGD; ZeRO 1/2/3 and FSDP; tensor,
pipeline, expert parallelism; activations generalised from Korthikanti et al. to RMSNorm, gated MLPs, GQA and
FlashAttention, with or without recomputation), fine-tuning (full, LoRA, QLoRA), inference (BF16/FP8/INT8/INT4
weights, FP8/INT8 KV cache, multi-head latent attention, a draft model), compute (6 × active parameters × tokens),
time at an MFU, cost at a price. Any Hugging Face `config.json` is a model.

**Checked**:

- The 13 built-in models: their parameter counts, recomputed from their `config.json` architecture, equal the counts
  Hugging Face publishes for their weights, to the parameter (Llama 2's API totals include F32 rotary buffers; GPT-2
  XL's include 48 causal-mask buffers; both noted). GPT-3's paper gives 174,600 million; the formula gives 174.604B.
- Published runs, each from the paper's own words (src/records.js): GPT-3's 3.14E+23 FLOPs; Llama 3 405B's 3.8×10^25;
  Llama 2 70B's 1,720,320 A100 GPU-hours imply 42.8% MFU (Llama 3 reports 38–43%); ZeRO's 16 bytes per parameter and
  at least 24 GB for a 1.5B GPT-2; vLLM's 800 KB per OPT-13B token (819,200 bytes).
- GPU figures from the makers' pages, checked today: H100 SXM, H100 NVL, H200, HGX B200, A100 80GB, L40S (NVIDIA lists
  sparse; dense is half), RTX 4090 and RTX 3090 Ti (the Ada whitepaper's table), Apple M5 Max and M5 Ultra memory.
  Found on the way: NVIDIA's page now lists the H100 NVL (94 GB), not the 80 GB PCIe card I had assumed; the A100 40GB
  is not on the page (dropped); AMD's MI300X page and datasheet timed out (left out rather than guessed). The papers
  and whitepaper fetched for this were deleted after reading.

**Tried and fixed while building**: the CLI printed nothing when run from this path (the "am I the program" check
compared `import.meta.url`, which percent-encodes the Chinese folder name: real paths now, also through npm's bin
link); fine-tuning defaults to recomputation (LoRA/QLoRA practice; without it QLoRA of an 8B model "did not fit" a
24 GB card); the "does not fit" hint depends on the command; sizes under 1 GiB show in MiB; on the page, `hidden` lost
to `label { display: grid }` and the mode buttons' hidden radios overflowed the page sideways.

Tests: 52 (node:test), type check clean.

Next: polishing round 1 (cloud prices with dates and sources; AMD; an independent review).
