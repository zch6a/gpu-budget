// @ts-check
/**
 * The calculator: plain functions of a configuration, the same in the command line and the page. Memory is per GPU,
 * in bytes; every part is reported apart, with the formula it comes from:
 *
 * - Model states (ZeRO paper, Rajbhandari et al. 2020, arXiv:1910.02054): with mixed precision and Adam, 2 bytes of
 *   weights, 2 of gradients and 12 of optimizer state (FP32 master copy, momentum, variance) per parameter; ZeRO stage
 *   1 shards the optimizer state over the data-parallel GPUs, stage 2 the gradients too, stage 3 the weights too.
 *   Tensor and pipeline parallelism divide them; expert parallelism divides the experts.
 * - Activations (Korthikanti et al. 2022, arXiv:2205.05198), counted for today's layers (RMSNorm, gated MLP,
 *   grouped-query attention, FlashAttention): per layer and token, the tensors the backward pass needs, in 16-bit;
 *   without FlashAttention the attention scores too (2·heads·seq bytes per token). Full recomputation keeps only each
 *   layer's input. Sequence parallelism divides by the tensor-parallel size; with pipeline parallelism (1F1B) the first
 *   stage holds as many microbatches as there are stages, which comes to the whole model's layers.
 * - Compute (Kaplan et al. 2020; the convention labs report totals in): 6 × parameters × tokens, the active
 *   parameters for a mixture of experts; optionally PaLM's attention term, 12 × layers × attention width × sequence.
 *
 * @typedef {import("./data.js").Model} Model
 * @typedef {import("./data.js").Gpu} Gpu
 */

import { BYTES, GPUS, MODELS } from "./data.js";

export const GiB = 2 ** 30;

/** @param {Model} m */
function headDim(m) {
  return m.headDim ?? m.hidden / m.heads;
}

/**
 * Parameters: total, active (per token), and the parts of one layer.
 * @param {Model} m
 */
export function params(m) {
  const hd = headDim(m);
  const q = m.heads * hd;
  const kv = m.kvHeads * hd;
  const bias = m.biases ? 1 : 0;
  const attn = m.hidden * q + 2 * m.hidden * kv + q * m.hidden + (m.qkvBias || m.biases ? q + 2 * kv : 0) + bias * m.hidden;
  const mats = m.mlp === "plain" ? 2 : 3;
  /** @param {number} f */
  const mlp = (f) => mats * m.hidden * f + bias * (f + m.hidden);
  const norms = (m.normsPerLayer ?? 2) * m.hidden * (m.biases ? 2 : 1);
  const embed = m.vocab * m.hidden + (m.positions ?? 0) * m.hidden;
  const head = m.tied ? 0 : m.vocab * m.hidden;
  const finalNorm = m.hidden * (m.biases ? 2 : 1);
  let total;
  let active;
  let expertParams = 0;
  if (m.experts) {
    const dense = m.denseLayers ?? 0;
    const moe = m.layers - dense;
    const perExpert = mlp(m.ffn);
    const router = m.hidden * m.experts;
    const denseMlp = mlp(m.denseFfn ?? m.ffn);
    expertParams = moe * m.experts * perExpert;
    const shared = moe * (m.sharedExperts ?? 0) * perExpert;
    total = m.layers * (attn + norms) + dense * denseMlp + moe * router + expertParams + shared + embed + head + finalNorm;
    active = m.layers * (attn + norms) + dense * denseMlp + moe * (router + (m.topK ?? 1) * perExpert) + shared + embed + head + finalNorm;
  } else {
    total = m.layers * (attn + mlp(m.ffn) + norms) + embed + head + finalNorm;
    active = total;
  }
  if (m.params) {
    // a model whose parts the fields cannot describe (multi-head latent attention): its published counts
    expertParams = expertParams * (m.params / total);
    total = m.params;
    active = m.activeParams ?? m.params;
  }
  return { total, active, experts: expertParams, attn, mlp: mlp(m.ffn), embed, head };
}

/** LoRA's trainable parameters: rank × (in + out) for each adapted matrix of every layer (all linear layers, or attention only). */
/** @param {Model} m @param {number} rank @param {"all"|"attention"} targets */
export function loraParams(m, rank, targets = "all") {
  const hd = headDim(m);
  const q = m.heads * hd;
  const kv = m.kvHeads * hd;
  const h = m.hidden;
  const attn = rank * (h + q) + 2 * rank * (h + kv) + rank * (q + h);
  const mlpMats = m.mlp === "plain" ? 2 : 3;
  const mlp = targets === "all" ? mlpMats * rank * (h + m.ffn) * (m.experts ? m.experts : 1) : 0;
  return m.layers * (attn + mlp);
}

/** KV cache bytes per token for the whole model. @param {Model} m @param {number} bytes */
export function kvBytesPerToken(m, bytes) {
  const perLayer = m.kvPerToken ?? 2 * m.kvHeads * headDim(m);
  return m.layers * perLayer * bytes;
}

/**
 * Activation bytes per token per layer that the backward pass keeps (16-bit), for today's layers; see the module.
 * @param {Model} m @param {number} seq @param {boolean} flash
 */
export function activationPerTokenLayer(m, seq, flash = true) {
  const h = m.hidden;
  const q = m.heads * headDim(m);
  const kv = m.kvHeads * headDim(m);
  const f = m.experts ? m.ffn * (m.topK ?? 1) + m.ffn * (m.sharedExperts ?? 0) : m.ffn;
  const mlp = m.mlp === "plain" ? 2 * h + 4 * f : 2 * h + 6 * f; // input, then the inner tensors (gate, up, product: 3; plain: 2)
  const attention = 2 * h + 2 * h + 2 * q + 4 * kv + 2 * q; // norm input, projection input, Q, K and V, the output
  const norms = 2 * h;
  const scores = flash ? 0 : 2 * m.heads * seq; // the softmax output, without FlashAttention
  return attention + mlp + norms + scores;
}

/**
 * @typedef {object} Breakdown
 * @property {number} weights
 * @property {number} grads
 * @property {number} optimizer
 * @property {number} activations
 * @property {number} kvCache
 * @property {number} draft
 * @property {number} overhead
 * @property {number} total
 */

/** @param {Partial<Breakdown>} parts @returns {Breakdown} */
function breakdown(parts) {
  const b = { weights: 0, grads: 0, optimizer: 0, activations: 0, kvCache: 0, draft: 0, overhead: 0, ...parts };
  return { ...b, total: b.weights + b.grads + b.optimizer + b.activations + b.kvCache + b.draft + b.overhead };
}

export const OPTIMIZERS = {
  adamw: { bytes: 12, about: "FP32 master weights, momentum and variance (mixed precision)" },
  "adamw-8bit": { bytes: 6, about: "FP32 master weights, 8-bit momentum and variance (bitsandbytes)" },
  "adamw-bf16": { bytes: 4, about: "BF16 momentum and variance, no master copy" },
  sgd: { bytes: 8, about: "FP32 master weights and momentum" },
};

/**
 * @typedef {object} TrainConfig
 * @property {Model} model
 * @property {number} [gpus] all GPUs (data parallel = gpus / (tp × pp))
 * @property {number} [tp] tensor parallel
 * @property {number} [pp] pipeline parallel
 * @property {number} [ep] expert parallel
 * @property {0|1|2|3} [zero] ZeRO stage (FSDP full shard = 3, shard grad op = 2)
 * @property {number} [seq]
 * @property {number} [microBatch]
 * @property {"none"|"full"} [recompute]
 * @property {boolean} [flash]
 * @property {keyof typeof OPTIMIZERS} [optimizer]
 * @property {number} [overheadGiB] CUDA context, allocator: 1 GiB by default
 */

/** @param {TrainConfig} c */
function parallel(c) {
  const gpus = c.gpus ?? 1;
  const tp = c.tp ?? 1;
  const pp = c.pp ?? 1;
  if (gpus % (tp * pp) !== 0) throw new Error(`gpus (${gpus}) must be a multiple of tp × pp (${tp} × ${pp})`);
  return { gpus, tp, pp, dp: gpus / (tp * pp), ep: c.ep ?? 1 };
}

/** Parameters each GPU holds before ZeRO: the model split by tensor and pipeline parallelism, the experts by expert parallelism too. */
/** @param {Model} m @param {{tp: number, pp: number, ep: number}} p */
function paramsPerGpu(m, p) {
  const n = params(m);
  return (n.total - n.experts) / (p.tp * p.pp) + n.experts / (p.tp * p.pp * p.ep);
}

/** @param {TrainConfig} c */
function activations(c) {
  const m = c.model;
  const seq = c.seq ?? 4096;
  const b = c.microBatch ?? 1;
  const { tp } = parallel(c);
  const perLayer = activationPerTokenLayer(m, seq, c.flash ?? true);
  const layers = c.recompute === "full" ? m.layers * 2 * m.hidden + perLayer : m.layers * perLayer; // full: each layer's input, plus one layer being recomputed
  const logits = 4 * m.vocab; // FP32 logits for the loss, per token (vocabulary split over tensor parallelism)
  return (seq * b * (layers + logits)) / tp;
}

/** Full training (pre-training or full fine-tuning) memory per GPU. @param {TrainConfig} c */
export function trainMemory(c) {
  const p = parallel(c);
  const n = paramsPerGpu(c.model, p);
  const zero = c.zero ?? 0;
  const opt = OPTIMIZERS[c.optimizer ?? "adamw"];
  if (!opt) throw new Error(`unknown optimizer ${c.optimizer} (${Object.keys(OPTIMIZERS).join(", ")})`);
  return breakdown({
    weights: (n * 2) / (zero >= 3 ? p.dp : 1),
    grads: (n * 2) / (zero >= 2 ? p.dp : 1),
    optimizer: (n * opt.bytes) / (zero >= 1 ? p.dp : 1),
    activations: activations(c),
    overhead: (c.overheadGiB ?? 1) * GiB,
  });
}

/**
 * @typedef {TrainConfig & {method: "full"|"lora"|"qlora", rank?: number, targets?: "all"|"attention"}} FinetuneConfig
 */

/** Fine-tuning memory per GPU: full (as training), LoRA (frozen 16-bit weights) or QLoRA (frozen 4-bit NF4 weights). @param {FinetuneConfig} c */
export function finetuneMemory(c) {
  if (c.method === "full") return trainMemory(c);
  const p = parallel(c);
  const m = c.model;
  const n = params(m);
  const trainable = loraParams(m, c.rank ?? 16, c.targets ?? "all") / (p.tp * p.pp);
  const frozen = paramsPerGpu(m, p);
  // QLoRA keeps the embeddings and the output layer in 16-bit; the rest in 4-bit NF4 with double quantization
  const sixteen = (n.embed + n.head) / (p.tp * p.pp);
  const weights = c.method === "qlora" ? (frozen - sixteen) * BYTES.nf4 + sixteen * 2 : frozen * 2;
  const opt = OPTIMIZERS[c.optimizer ?? "adamw"];
  return breakdown({
    weights: weights + trainable * 2,
    grads: trainable * 2,
    optimizer: trainable * opt.bytes,
    activations: activations(c),
    overhead: (c.overheadGiB ?? 1) * GiB,
  });
}

/**
 * @typedef {object} InferConfig
 * @property {Model} model
 * @property {keyof typeof BYTES} [quant] the weights' format
 * @property {keyof typeof BYTES} [kv] the KV cache's format
 * @property {number} [ctx] tokens of context per sequence (prompt and output)
 * @property {number} [batch] sequences at once
 * @property {number} [tp] GPUs the model is split over
 * @property {Model} [draft] a draft model for speculative decoding (16-bit)
 * @property {number} [overheadGiB]
 */

/** Inference memory per GPU: the weights, the KV cache, a draft model, a working buffer. @param {InferConfig} c */
export function inferMemory(c) {
  const m = c.model;
  const tp = c.tp ?? 1;
  const q = BYTES[c.quant ?? "bf16"];
  const kvb = BYTES[c.kv ?? "bf16"];
  if (q === undefined || kvb === undefined) throw new Error(`unknown format (${Object.keys(BYTES).join(", ")})`);
  const n = params(m);
  // the embeddings stay 16-bit when the rest is quantized below 8 bits (as GPTQ/AWQ checkpoints keep them)
  const sixteen = q < 1 ? n.embed + n.head : 0;
  const weights = ((n.total - sixteen) * q + sixteen * 2) / tp;
  const ctx = c.ctx ?? 4096;
  const batch = c.batch ?? 1;
  const kvCache = (kvBytesPerToken(m, kvb) * ctx * batch) / tp;
  const draft = c.draft ? params(c.draft).total * 2 + kvBytesPerToken(c.draft, 2) * ctx * batch : 0;
  // a working buffer for one prefill chunk of 512 tokens per sequence, and the last token's logits in FP32
  const work = (batch * 512 * (2 * m.hidden + 2 * (m.experts ? m.ffn * (m.topK ?? 1) : m.ffn)) * 2) / tp + (batch * m.vocab * 4) / tp;
  return breakdown({ weights, kvCache, draft, activations: work, overhead: (c.overheadGiB ?? 1) * GiB });
}

/**
 * Training compute, time and cost.
 * @param {{model: Model, tokens: number, gpus: number, gpu: Gpu, mfu?: number, price?: number, seq?: number, attention?: boolean, fp8?: boolean}} c
 */
export function trainTime(c) {
  const m = c.model;
  const n = params(m).active;
  const seq = c.seq ?? 4096;
  const perToken = 6 * n + (c.attention ? 12 * m.layers * m.heads * headDim(m) * seq : 0);
  const flops = perToken * c.tokens;
  const peak = (c.fp8 ? c.gpu.fp8 : c.gpu.bf16) ?? null;
  const mfu = c.mfu ?? 0.4;
  if (peak === null) return { flops, seconds: null, gpuHours: null, cost: null, peak: null, mfu };
  const seconds = flops / (c.gpus * peak * 1e12 * mfu);
  const gpuHours = (seconds * c.gpus) / 3600;
  return { flops, seconds, gpuHours, cost: c.price ? gpuHours * c.price : null, peak, mfu };
}

/** The MFU a published run implies: its compute over (GPU-hours × peak). @param {number} flops @param {number} gpuHours @param {Gpu} gpu */
export function impliedMfu(flops, gpuHours, gpu) {
  return flops / (gpuHours * 3600 * (gpu.bf16 ?? Number.NaN) * 1e12);
}

/** Memory a GPU offers (Apple: the share of unified memory the GPU may use). @param {Gpu} gpu */
export function usableBytes(gpu) {
  return gpu.memory * (gpu.usable ?? 1) * GiB;
}

/**
 * A model from a Hugging Face config.json (Llama, Mistral, Mixtral, Qwen2, Gemma and their kin).
 * @param {Record<string, any>} cfg
 * @param {string} [name]
 * @returns {Model}
 */
export function fromHFConfig(cfg, name = "custom") {
  const c = cfg.text_config ?? cfg;
  /** @param {string[]} keys */
  const num = (...keys) => {
    for (const k of keys) if (typeof c[k] === "number" && Number.isFinite(c[k]) && c[k] > 0) return c[k];
    return undefined;
  };
  const layers = num("num_hidden_layers", "n_layer", "num_layers");
  const hidden = num("hidden_size", "n_embd", "d_model");
  const heads = num("num_attention_heads", "n_head");
  const vocab = num("vocab_size");
  const ffn = num("moe_intermediate_size", "intermediate_size", "n_inner", "ffn_dim") ?? (hidden ? 4 * hidden : undefined);
  if (!layers || !hidden || !heads || !vocab || !ffn) {
    throw new Error("config.json needs num_hidden_layers, hidden_size, num_attention_heads, vocab_size and intermediate_size");
  }
  const type = String(c.model_type ?? "");
  const gpt = /^(gpt2|gpt_neo|opt)/.test(type);
  /** @type {Model} */
  const m = {
    name,
    layers,
    hidden,
    heads,
    kvHeads: num("num_key_value_heads") ?? heads,
    ffn,
    vocab,
    tied: c.tie_word_embeddings ?? gpt,
    source: "config.json",
  };
  const hd = num("head_dim");
  if (hd && hd !== hidden / heads) m.headDim = hd;
  if (gpt) {
    m.mlp = "plain";
    m.biases = true;
    m.positions = num("n_positions", "max_position_embeddings");
  }
  if (type.startsWith("qwen2")) m.qkvBias = true;
  if (type.startsWith("gemma2") || type.startsWith("gemma3")) m.normsPerLayer = 4;
  const experts = num("num_local_experts", "num_experts", "n_routed_experts");
  if (experts) {
    m.experts = experts;
    m.topK = num("num_experts_per_tok", "moe_top_k") ?? 2;
    const shared = num("n_shared_experts");
    if (shared) m.sharedExperts = shared;
    const dense = num("first_k_dense_replace");
    if (dense) {
      m.denseLayers = dense;
      m.denseFfn = num("intermediate_size") ?? ffn;
    }
  }
  if (num("kv_lora_rank")) m.kvPerToken = (num("kv_lora_rank") ?? 0) + (num("qk_rope_head_dim") ?? 0);
  return m;
}

/** @param {string} id */
export function model(id) {
  const m = MODELS[id.toLowerCase()];
  if (!m) throw new Error(`unknown model ${id} (models: ${Object.keys(MODELS).join(", ")}; or a config.json)`);
  return m;
}

/** @param {string} id */
export function gpu(id) {
  const g = GPUS[id.toLowerCase()];
  if (!g) throw new Error(`unknown GPU ${id} (GPUs: ${Object.keys(GPUS).join(", ")})`);
  return g;
}

/** Bytes as GiB with one decimal (MiB below 1 GiB, so that a small part does not read 0.0). @param {number} b */
export function gib(b) {
  return b < GiB ? `${Math.round(b / 2 ** 20)} MiB` : `${(b / GiB).toFixed(1)} GiB`;
}
