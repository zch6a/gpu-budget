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

### 2026-11-13 — polishing round 1

**Prices**: on-demand list prices read today from RunPod's (Secure Cloud) and Lambda's (8-GPU instances) pricing
pages, each with the offer it is for, the date and the link (`src/data.js`). Lambda's page says its prices take effect
on 2026-11-16, which the entries say. `--price runpod` uses a provider's price for the GPU (and the report names it);
`gpu-budget prices` lists them; the page offers them per GPU. RunPod's community prices came from the page's metadata
rather than its cards and were left out.

**AMD**: MI300X, MI325X and MI355X are in with their memory from ROCm's GPU table; AMD's product pages and datasheet
timed out every time, so their tensor throughput is not in (no time estimate, said so).

**First independent review** (11 defects and 4 smaller items, all fixed; tests in `test/review1.test.js`):

1. Expert parallelism with ZeRO sharded the experts' states twice (by EP, then over all data-parallel GPUs): Mixtral on
   8 GPUs, EP 8, ZeRO-3 showed 10.1 GiB of optimizer state for about 65. The experts' states are sharded over the
   experts' own data-parallel group; EP must divide the data-parallel GPUs and the experts.
2. DeepSeek-V3's latent KV cache was divided by tensor parallelism; serving engines keep it whole on each GPU.
3. A config.json without `tie_word_embeddings` was counted untied (Gemma 2 +918M): tied, Hugging Face's default.
   The test had patched the key in.
4. Fine-tuning ignored ZeRO (and then advised adding it): ZeRO-3 / FSDP shards the frozen weights, the adapters'
   states from ZeRO-1/2.
5. The inference buffer grew with the batch (a 512-token chunk per sequence): one 8,192-token prefill budget shared.
6. KV heads were divided fractionally over tensor-parallel GPUs: split when they divide, one per GPU (replicated) when
   there are more GPUs, refused otherwise.
7. Pipeline stages were taken as equal: the embedding sits on the first, the output layer and logits on the last; the
   fuller end is reported (and named).
8. `constructor` was a model (Object's own properties): own keys only.
9. config.json: Qwen-MoE's shared expert ignored, DeepSeek's latent attention with a wrong head size, GPT-NeoX given
   learned positions: all read now. Multi-head latent attention is modelled in full, so DeepSeek-V3 is computed rather
   than hard-coded: 671.03B and 37.55B active, against the paper's 671B and 37B.
10. (plausible) ZeRO-3's gathered layer was not counted: one layer's weights and a gradient buffer are now.
11. (plausible) LoRA compute charged 6N: 4 × frozen + 6 × adapter parameters.

Smaller: ZeRO-2/3 with pipeline parallelism is refused (DeepSpeed refuses it; Megatron shards only the optimizer);
values out of range (MFU 5, a negative price, tp 0.5) are refused in the engine, so the page shows the error too;
`--price` or `--mfu` without `--tokens` is an error, not ignored; the README's claims are exact now (11 models to the
parameter, GPT-3 and DeepSeek-V3 against their papers' rounded counts, OPT-13B unchecked).

The README's example moved from 33.4 to 36.6 GiB (ZeRO-3's gathered layer); screenshot and social card redone.

Tests: 70 (were 52), type check clean.

Next: round 2.
