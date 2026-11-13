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

### 2026-11-13 — polishing round 2

**Two more published checks**, each from the paper's own words:

- Korthikanti et al.'s activation formula, sbh(34 + 5as/h): their count includes dropout's masks (1 byte per element)
  and the attention dropout's output, which today's LLM training mostly leaves out. With `--dropout` (new) and without
  FlashAttention, the engine's formula, written for today's layers, reduces to theirs exactly for a GPT-3 layer
  (1,400,832 bytes per token at sequence 2048). A record and a test.
- QLoRA: "finetune a 65B parameter model on a single 48GB GPU". LLaMA 65B added (its config from Hugging Face, its
  count 65,285,660,672 to the parameter); QLoRA with rank 64 on all linear layers, sequence 512 and recomputation
  comes to 45.8 GiB of 48: the claim holds, narrowly (16-bit LoRA would need about 122 GiB).

**The page**: a Copy link button (the link holds the settings; checked in a browser: it copies the page's address and
says so). **The package**: `npm pack` holds the code, the page and the docs only; a test checks the list.

Docs: the CHANGELOG brought up to date (16 models, 13 GPUs, prices, the checks); the README's table has the two new
rows.

Tests: 74 (were 70), type check clean.

Next: round 3.

### 2026-11-13 — polishing round 3

Checked first: a fresh `npx -y github:zch6a/gpu-budget` (empty cache) installs and runs in about 22 seconds; odd inputs
(a context of 0, a fractional batch, unknown formats, upper-case names) give one clear line each. Huge values printed
"176117051260456992768.0 days" and "33814473842.01T GPU-hours": beyond trillions a power of ten, beyond two years
years ("4.82e+17 years").

**Second independent review** (15; two of them, the number formatting and the wording for GPUs without a checked
throughput, already fixed above; the other 13 fixed, tests in `test/review2.test.js`):

1. LoRA with expert parallelism and ZeRO-3 sharded the frozen experts twice (round 1's fix had gone into training only):
   Mixtral, 8 GPUs, EP 8: 2.15 GiB of weights for about 10.9. Over the experts' own group now.
2. Activations were always divided by tensor parallelism, assuming sequence parallelism without saying so. It is the
   default now and said; `--no-sp` keeps the norms' and projections' inputs whole on each GPU, which reproduces
   Korthikanti et al.'s sbh(10 + 24/t + 5as/ht) exactly (a test).
3. The page still described the old 512-token buffer.
4. Two records were too loose to catch a regression (Llama 2's MFU anywhere in 0.3–0.6; QLoRA only an upper bound):
   their values are pinned in a test (42.8% ± 0.2; 45.8 ± 0.1 GiB), the records keeping what the papers say.
5. Training accepted tensor-parallel sizes that do not split the heads (Megatron refuses), and latent attention under
   tensor parallelism was not checked: refused with the reason.
6. `--fp8` was undocumented, and on an A100 said "no checked tensor throughput": documented (the time at the FP8 peak,
   memory unchanged), and the message names FP8 and points to the BF16 figure.
7. The README said 12 models match to the parameter: 13 (LLaMA 65B joined in round 2).
8. (plausible) ZeRO-3 gathered only a layer: the larger of a layer and the embedding/output unit now (stated).
9. LoRA on latent attention counted plain q/k/v/o matrices (11% too many for DeepSeek-V3): q_a, q_b, kv_a, kv_b, o.
10. Pipeline sizes that do not divide the layers were accepted (Llama 3.1 405B at 31.5 layers a stage): refused.
11. The CLI ignored options that do not apply (`infer --gpus 8 --tokens ...` dropped them silently) and hinted at
    what was already set, or at ZeRO-3 with pipeline parallelism: options that do not apply are an error (with
    "to split the model over GPUs: --tp" for --gpus); the hints skip what is set and what the engine refuses.
12. (plausible) The draft model's KV cache ignored `--kv`: it follows the target's format.
13. (plausible) Megatron keeps FP32 gradients: `--fp32-grads` (4 bytes per parameter).
14. A total just over the limit read "80.0 GiB of 80.0 GiB: does not fit (100%)": two decimals and one more digit of
    percentage near the limit (a test finds such a context and checks).
15. The page dropped a custom config.json from the link (it opened on an error), defaulted training to recomputation
    unlike the command line, and kept old values when a new link was followed: the config is kept, the recompute
    default follows the mode as on the command line, a new link starts from the defaults (checked in a browser).

Also: `package.json` had gone back to Node ≥20 (npm's rewrite of the file); ≥22, as the README says.

The README's example moved to 41.3 GiB (ZeRO-3's gathered unit is the embedding and output layer for Llama 3.1 70B);
screenshot and social card redone.

Tests: 86 (were 75), type check clean.

Next: round 4.

### 2026-11-13 — polishing round 4

**Security and dependencies**: no runtime dependencies (`npm audit --omit=dev`: nothing); the dev dependencies (the
type checker, Node's types) audited against the public registry: 0 vulnerabilities. No eval, no network, no HTML from
data, in the command line or the page; the page's Content-Security-Policy allows its own files only.

**The lockfile pointed at a mirror**: npm on this machine is set to registry.npmmirror.com, and `package-lock.json`
had recorded its URLs, so CI and anyone installing would have fetched from a mirror they never chose. The URLs point
at registry.npmjs.org now (the integrity hashes are the same; npm still swaps in a configured registry), and a test
keeps it so. The same is true of seven earlier projects' lockfiles: noted for them.

**The release**: a test finds this version's CHANGELOG section the way the release workflow's awk finds it (and that
it is not empty), and checks one version everywhere (package.json, the command line, the plugin) and one Node floor
(package.json, the README badge, the Skill, CI).

**The page**: the stacked bar is an image to a screen reader with its parts in words ("Weights 15.0 GiB, Working buffer
576 MiB, …; 17.5 GiB of 80.0 GiB"); every control has a label (checked in a browser). The CHANGELOG and the Skill name
the options for framework differences (`--no-sp`, `--fp32-grads`, `--dropout`, `--fp8`).

Tests: 89 (were 86), type check clean.

Next: round 5 (a last review of everything, then v0.1.0).
