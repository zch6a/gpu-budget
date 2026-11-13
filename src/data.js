// @ts-check
/**
 * The built-in models and GPUs. Every model's architecture is its Hugging Face config.json (linked); every GPU number
 * is its maker's datasheet (linked), dense (no sparsity) tensor throughput. A test recomputes each model's parameter
 * count from its architecture and compares it with the count published for its weights.
 *
 * @typedef {object} Model
 * @property {string} name
 * @property {number} layers
 * @property {number} hidden
 * @property {number} heads attention heads
 * @property {number} kvHeads key/value heads (fewer than heads: grouped-query attention)
 * @property {number} [headDim] when it is not hidden / heads
 * @property {number} ffn the MLP's inner size (each expert's, for a mixture of experts)
 * @property {number} vocab
 * @property {boolean} tied the output layer shares the embedding's weights
 * @property {"gated"|"plain"} [mlp] gated (SwiGLU/GeGLU: three matrices) or plain (two, GPT-style); gated by default
 * @property {boolean} [qkvBias] biases on the query, key and value projections (Qwen)
 * @property {boolean} [biases] biases everywhere (GPT-style)
 * @property {number} [normsPerLayer] 2 by default (Gemma 2 has 4)
 * @property {number} [positions] learned position embeddings (GPT-style), their count
 * @property {number} [experts] a mixture of experts: experts per MoE layer
 * @property {number} [topK] experts each token uses
 * @property {number} [sharedExperts] experts every token uses
 * @property {number} [denseLayers] leading layers with a dense MLP instead of experts
 * @property {number} [denseFfn] those layers' MLP size
 * @property {number} [sharedFfn] a shared expert of its own size (Qwen-MoE)
 * @property {{qLora?: number, kvLora: number, nope: number, rope: number, v: number}} [mla] multi-head latent attention (DeepSeek-V2/V3)
 * @property {number} [kvPerToken] KV cache values per token per layer, when not 2 × kvHeads × headDim (multi-head latent attention)
 * @property {number} [params] the parameter count, when it cannot be derived from the fields above
 * @property {number} [activeParams] the parameters each token uses, likewise
 * @property {number} [published] the parameter count published for the weights (checked by a test)
 * @property {number} [publishedActive] the active parameter count published (a mixture of experts)
 * @property {string} source
 *
 * @typedef {object} Gpu
 * @property {string} name
 * @property {number} memory GiB
 * @property {number|null} bf16 dense BF16/FP16 tensor TFLOPS (FP32 accumulate), null when not published
 * @property {number|null} fp8 dense FP8 tensor TFLOPS, null when the GPU has none
 * @property {number} [usable] the share of memory the GPU may use (Apple: about 75% of unified memory by default)
 * @property {string} kind
 * @property {string} source
 */

/** @type {Record<string, Model>} */
export const MODELS = {
  "llama-3.1-8b": { name: "Llama 3.1 8B", layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 128256, tied: false,
    published: 8030261248, source: "https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json" },
  "llama-3.1-70b": { name: "Llama 3.1 70B", layers: 80, hidden: 8192, heads: 64, kvHeads: 8, ffn: 28672, vocab: 128256, tied: false,
    published: 70553706496, source: "https://huggingface.co/meta-llama/Llama-3.1-70B/blob/main/config.json" },
  "llama-3.1-405b": { name: "Llama 3.1 405B", layers: 126, hidden: 16384, heads: 128, kvHeads: 8, ffn: 53248, vocab: 128256, tied: false,
    published: 405853388800, source: "https://huggingface.co/meta-llama/Llama-3.1-405B/blob/main/config.json" },
  "llama-3.2-1b": { name: "Llama 3.2 1B", layers: 16, hidden: 2048, heads: 32, kvHeads: 8, headDim: 64, ffn: 8192, vocab: 128256, tied: true,
    published: 1235814400, source: "https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/config.json" },
  "llama-2-7b": { name: "Llama 2 7B", layers: 32, hidden: 4096, heads: 32, kvHeads: 32, ffn: 11008, vocab: 32000, tied: false,
    published: 6738415616, source: "https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json" },
  "llama-2-70b": { name: "Llama 2 70B", layers: 80, hidden: 8192, heads: 64, kvHeads: 8, ffn: 28672, vocab: 32000, tied: false,
    published: 68976648192, source: "https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json" },
  "mistral-7b": { name: "Mistral 7B v0.1", layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 32000, tied: false,
    published: 7241732096, source: "https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json" },
  "mixtral-8x7b": { name: "Mixtral 8x7B", layers: 32, hidden: 4096, heads: 32, kvHeads: 8, ffn: 14336, vocab: 32000, tied: false, experts: 8, topK: 2,
    published: 46702792704, source: "https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json" },
  "qwen2.5-7b": { name: "Qwen2.5 7B", layers: 28, hidden: 3584, heads: 28, kvHeads: 4, ffn: 18944, vocab: 152064, tied: false, qkvBias: true,
    published: 7615616512, source: "https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json" },
  "qwen2.5-72b": { name: "Qwen2.5 72B", layers: 80, hidden: 8192, heads: 64, kvHeads: 8, ffn: 29568, vocab: 152064, tied: false, qkvBias: true,
    published: 72706203648, source: "https://huggingface.co/Qwen/Qwen2.5-72B/blob/main/config.json" },
  "gemma-2-9b": { name: "Gemma 2 9B", layers: 42, hidden: 3584, heads: 16, kvHeads: 8, headDim: 256, ffn: 14336, vocab: 256000, tied: true, normsPerLayer: 4,
    published: 9241705984, source: "https://huggingface.co/google/gemma-2-9b/blob/main/config.json" },
  // the paper: "671B total parameters with 37B activated for each token" (the repository's 684.5B adds the multi-token prediction module)
  "deepseek-v3": { name: "DeepSeek-V3 (671B, 37B active)", layers: 61, hidden: 7168, heads: 128, kvHeads: 128, ffn: 2048, vocab: 129280, tied: false,
    experts: 256, topK: 8, sharedExperts: 1, denseLayers: 3, denseFfn: 18432, mla: { qLora: 1536, kvLora: 512, nope: 128, rope: 64, v: 128 }, kvPerToken: 576,
    published: 671e9, publishedActive: 37e9, source: "https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json" },
  "gpt2-xl": { name: "GPT-2 XL (1.5B)", layers: 48, hidden: 1600, heads: 25, kvHeads: 25, ffn: 6400, vocab: 50257, tied: true, mlp: "plain", biases: true, positions: 1024,
    // the safetensors total, 1,607,942,848, less 48 causal-mask buffers of 1024 × 1024 that are not parameters
    published: 1557611200, source: "https://huggingface.co/openai-community/gpt2-xl/blob/main/config.json" },
  "opt-13b": { name: "OPT 13B", layers: 40, hidden: 5120, heads: 40, kvHeads: 40, ffn: 20480, vocab: 50272, tied: true, mlp: "plain", biases: true, positions: 2050,
    source: "https://huggingface.co/facebook/opt-13b/blob/main/config.json" },
  "gpt-3-175b": { name: "GPT-3 175B", layers: 96, hidden: 12288, heads: 96, kvHeads: 96, ffn: 49152, vocab: 50257, tied: true, mlp: "plain", biases: true, positions: 2048, published: 174600e6,
    source: "https://arxiv.org/abs/2005.14165" },
};

/** @type {Record<string, Gpu>} */
export const GPUS = {
  // NVIDIA's pages give tensor throughput with sparsity; dense is half
  "h100-sxm": { name: "NVIDIA H100 SXM 80GB", memory: 80, bf16: 989.5, fp8: 1979, kind: "data center", source: "https://www.nvidia.com/en-us/data-center/h100/" },
  "h100-nvl": { name: "NVIDIA H100 NVL 94GB", memory: 94, bf16: 835.5, fp8: 1670.5, kind: "data center", source: "https://www.nvidia.com/en-us/data-center/h100/" },
  "h200": { name: "NVIDIA H200 SXM 141GB", memory: 141, bf16: 989.5, fp8: 1979, kind: "data center", source: "https://www.nvidia.com/en-us/data-center/h200/" },
  // HGX B200: 1.4 TB for 8 GPUs, 36 PFLOPS BF16 and 72 PFLOPS FP8 sparse for 8
  "b200": { name: "NVIDIA B200 (HGX) 180GB", memory: 180, bf16: 2250, fp8: 4500, kind: "data center", source: "https://www.nvidia.com/en-us/data-center/hgx/" },
  "a100-80": { name: "NVIDIA A100 80GB", memory: 80, bf16: 312, fp8: null, kind: "data center", source: "https://www.nvidia.com/en-us/data-center/a100/" },
  "l40s": { name: "NVIDIA L40S 48GB", memory: 48, bf16: 362.05, fp8: 733, kind: "data center", source: "https://www.nvidia.com/en-us/data-center/l40s/" },
  // the Ada whitepaper's Table 2: BF16 tensor with FP32 accumulate, dense
  "rtx-4090": { name: "NVIDIA GeForce RTX 4090 24GB", memory: 24, bf16: 165.2, fp8: 330.3, kind: "consumer",
    source: "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf" },
  "rtx-3090-ti": { name: "NVIDIA GeForce RTX 3090 Ti 24GB", memory: 24, bf16: 80, fp8: null, kind: "consumer",
    source: "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf" },
  // AMD: memory from ROCm's GPU table; their tensor throughput could not be checked yet (no time estimate)
  "mi300x": { name: "AMD Instinct MI300X 192GB", memory: 192, bf16: null, fp8: null, kind: "data center", source: "https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html" },
  "mi325x": { name: "AMD Instinct MI325X 256GB", memory: 256, bf16: null, fp8: null, kind: "data center", source: "https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html" },
  "mi355x": { name: "AMD Instinct MI355X 288GB", memory: 288, bf16: null, fp8: null, kind: "data center", source: "https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html" },
  // Apple: the GPU may use part of the unified memory (about three quarters by default: an assumption, adjustable in macOS)
  "m5-max-128": { name: "Apple M5 Max 128GB (unified)", memory: 128, bf16: null, fp8: null, usable: 0.75, kind: "Apple",
    source: "https://www.apple.com/macbook-pro/specs/" },
  "m5-ultra-512": { name: "Apple M5 Ultra 512GB (unified)", memory: 512, bf16: null, fp8: null, usable: 0.75, kind: "Apple",
    source: "https://www.apple.com/mac-studio/specs/" },
};

/**
 * @typedef {object} Price
 * @property {string} provider
 * @property {string} gpu a key of GPUS
 * @property {number} usd per GPU-hour, on demand
 * @property {string} offer what the price is for
 * @property {string} checked the date the price was read from the page
 * @property {string} source
 */

/**
 * On-demand list prices, read from the providers' own pricing pages on the date given. Prices change: check the page
 * before relying on one; `--price` takes your own number too.
 * @type {Price[]}
 */
export const PRICES = [
  { provider: "runpod", gpu: "b200", usd: 7.99, offer: "Secure Cloud pod", checked: "2026-11-13", source: "https://www.runpod.io/pricing" },
  { provider: "runpod", gpu: "h200", usd: 5.29, offer: "Secure Cloud pod", checked: "2026-11-13", source: "https://www.runpod.io/pricing" },
  { provider: "runpod", gpu: "h100-sxm", usd: 4.39, offer: "Secure Cloud pod", checked: "2026-11-13", source: "https://www.runpod.io/pricing" },
  { provider: "runpod", gpu: "h100-nvl", usd: 3.19, offer: "Secure Cloud pod", checked: "2026-11-13", source: "https://www.runpod.io/pricing" },
  { provider: "runpod", gpu: "a100-80", usd: 2.19, offer: "Secure Cloud pod, A100 SXM", checked: "2026-11-13", source: "https://www.runpod.io/pricing" },
  { provider: "runpod", gpu: "l40s", usd: 1.09, offer: "Secure Cloud pod", checked: "2026-11-13", source: "https://www.runpod.io/pricing" },
  { provider: "runpod", gpu: "rtx-4090", usd: 0.89, offer: "Secure Cloud pod", checked: "2026-11-13", source: "https://www.runpod.io/pricing" },
  { provider: "lambda", gpu: "b200", usd: 7.69, offer: "8× HGX B200 instance (prices effective 2026-11-16)", checked: "2026-11-13", source: "https://lambda.com/pricing" },
  { provider: "lambda", gpu: "h100-sxm", usd: 4.59, offer: "8× HGX H100 instance (prices effective 2026-11-16)", checked: "2026-11-13", source: "https://lambda.com/pricing" },
  { provider: "lambda", gpu: "a100-80", usd: 3.19, offer: "8× HGX A100 80GB instance (prices effective 2026-11-16)", checked: "2026-11-13", source: "https://lambda.com/pricing" },
];

/** Bytes per value for the formats the calculator knows. int4 is 4 bits plus an FP16 scale per group of 128 (GPTQ/AWQ style). */
export const BYTES = { fp32: 4, bf16: 2, fp16: 2, fp8: 1, int8: 1, int4: (4 + 16 / 128) / 8, nf4: (4 + 0.127) / 8 };
