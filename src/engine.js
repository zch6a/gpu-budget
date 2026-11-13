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
 *   without FlashAttention the attention scores too (2·heads·seq bytes per token); with dropout its masks (1 byte per
 *   element) and the attention dropout's output, which for a GPT layer gives the paper's sbh(34 + 5as/h) exactly. Full recomputation keeps only each
 *   layer's input. Sequence parallelism divides by the tensor-parallel size; with pipeline parallelism (1F1B) the first
 *   stage holds as many microbatches as there are stages, which comes to the whole model's layers.
 * - Compute (Kaplan et al. 2020; the convention labs report totals in): 6 × parameters × tokens, the active
 *   parameters for a mixture of experts; optionally PaLM's attention term, 12 × layers × attention width × sequence.
 *
 * @typedef {import("./data.js").Model} Model
 * @typedef {import("./data.js").Gpu} Gpu
 */

import { BYTES, GPUS, MODELS, PRICES } from "./data.js";

export const GiB = 2 ** 30;

/** @param {Model} m */
function headDim(m) {
  return m.headDim ?? m.hidden / m.heads;
}

/** The attention's widths: query, key, value and output, for grouped-query or multi-head latent attention. @param {Model} m */
function dims(m) {
  if (m.mla) {
    const qk = m.heads * (m.mla.nope + m.mla.rope);
    return { q: qk, k: qk, v: m.heads * m.mla.v, o: m.heads * m.mla.v };
  }
  const hd = headDim(m);
  return { q: m.heads * hd, k: m.kvHeads * hd, v: m.kvHeads * hd, o: m.heads * hd };
}

/**
 * Parameters: total, active (per token), the experts', and the parts that sit on the first and last pipeline stage.
 * @param {Model} m
 */
export function params(m) {
  const d = dims(m);
  const h = m.hidden;
  const bias = m.biases ? 1 : 0;
  let attn;
  if (m.mla) {
    // multi-head latent attention (DeepSeek-V2/V3): queries through a low-rank projection (when qLora), keys and values
    // through a shared latent of kvLora values plus a rotary part, each projection with its norm
    const { qLora, kvLora, nope, rope, v } = m.mla;
    const q = qLora ? h * qLora + qLora + qLora * m.heads * (nope + rope) : h * m.heads * (nope + rope);
    const kv = h * (kvLora + rope) + kvLora + kvLora * m.heads * (nope + v);
    attn = q + kv + m.heads * v * h;
  } else {
    attn = h * d.q + h * d.k + h * d.v + d.o * h + (m.qkvBias || m.biases ? d.q + d.k + d.v : 0) + bias * h;
  }
  const mats = m.mlp === "plain" ? 2 : 3;
  /** @param {number} f */
  const mlp = (f) => mats * h * f + bias * (f + h);
  const norms = (m.normsPerLayer ?? 2) * h * (m.biases ? 2 : 1);
  const embed = m.vocab * h + (m.positions ?? 0) * h;
  const head = m.tied ? 0 : m.vocab * h;
  const finalNorm = h * (m.biases ? 2 : 1);
  let total;
  let active;
  let experts = 0;
  if (m.experts) {
    const dense = m.denseLayers ?? 0;
    const moe = m.layers - dense;
    const perExpert = mlp(m.ffn);
    const router = h * m.experts;
    const shared = moe * (m.sharedFfn ? mlp(m.sharedFfn) : (m.sharedExperts ?? 0) * perExpert);
    experts = moe * m.experts * perExpert;
    const common = m.layers * (attn + norms) + dense * mlp(m.denseFfn ?? m.ffn) + moe * router + shared + embed + head + finalNorm;
    total = common + experts;
    active = common + moe * (m.topK ?? 1) * perExpert;
  } else {
    total = m.layers * (attn + mlp(m.ffn) + norms) + embed + head + finalNorm;
    active = total;
  }
  if (m.params) {
    // a model whose parts the fields cannot describe: its published counts
    experts *= m.params / total;
    total = m.params;
    active = m.activeParams ?? m.params;
  }
  return { total, active, experts, embed, head: head + finalNorm, attn, mlp: mlp(m.ffn) };
}

/** LoRA's trainable parameters: rank × (in + out) for each adapted matrix of every layer (all linear layers, or attention only). */
/** @param {Model} m @param {number} rank @param {"all"|"attention"} targets */
export function loraParams(m, rank, targets = "all") {
  const p = loraParts(m, rank, targets);
  return p.dense + p.experts;
}

/**
 * LoRA's trainable parameters in two parts: on the routed experts (split by expert parallelism; a token uses topK of
 * them) and the rest.
 * @param {Model} m @param {number} rank @param {"all"|"attention"} targets
 */
export function loraParts(m, rank, targets = "all") {
  const d = dims(m);
  const h = m.hidden;
  let attn = rank * (h + d.q) + rank * (h + d.k) + rank * (h + d.v) + rank * (d.o + h);
  if (m.mla) {
    // latent attention's own matrices: q_a and q_b (or q), kv_a (with the rotary part), kv_b, o
    const { qLora, kvLora, nope, rope, v } = m.mla;
    const q = qLora ? rank * (h + qLora) + rank * (qLora + m.heads * (nope + rope)) : rank * (h + m.heads * (nope + rope));
    attn = q + rank * (h + kvLora + rope) + rank * (kvLora + m.heads * (nope + v)) + rank * (m.heads * v + h);
  }
  const mats = m.mlp === "plain" ? 2 : 3;
  const perMlp = (/** @type {number} */ f) => mats * rank * (h + f);
  let mlp = 0;
  let experts = 0;
  if (targets === "all") {
    if (m.experts) {
      const dense = m.denseLayers ?? 0;
      const shared = m.sharedFfn ? perMlp(m.sharedFfn) : (m.sharedExperts ?? 0) * perMlp(m.ffn);
      mlp = dense * perMlp(m.denseFfn ?? m.ffn) + (m.layers - dense) * shared;
      experts = (m.layers - dense) * m.experts * perMlp(m.ffn);
    } else {
      mlp = m.layers * perMlp(m.ffn);
    }
  }
  return { dense: m.layers * attn + mlp, experts };
}

/**
 * The adapters one token uses: all but the routed experts', and topK of those.
 * @param {Model} m @param {number} rank @param {"all"|"attention"} targets
 */
export function activeLora(m, rank, targets = "all") {
  const p = loraParts(m, rank, targets);
  return p.dense + (m.experts ? (p.experts * (m.topK ?? 1)) / m.experts : 0);
}

/**
 * KV cache bytes per token for the whole model, on one GPU when the model is split over `tp`: each GPU keeps its share
 * of the KV heads (one head at least: with more GPUs than KV heads they are replicated); a latent cache (multi-head
 * latent attention) is kept whole on every GPU.
 * @param {Model} m @param {number} bytes @param {number} [tp]
 */
export function kvBytesPerToken(m, bytes, tp = 1) {
  if (m.kvPerToken) {
    if (tp > 1 && m.heads % tp !== 0) throw new Error(`tensor parallel ${tp} does not split the ${m.heads} attention heads evenly`);
    return m.layers * m.kvPerToken * bytes;
  }
  let heads = m.kvHeads;
  if (tp > 1) {
    if (m.kvHeads >= tp && m.kvHeads % tp === 0) heads = m.kvHeads / tp;
    else if (tp > m.kvHeads && tp % m.kvHeads === 0) heads = 1;
    else throw new Error(`tensor parallel ${tp} does not split ${m.kvHeads} KV heads evenly (use a divisor or a multiple of ${m.kvHeads})`);
  }
  return m.layers * 2 * heads * headDim(m) * bytes;
}

/**
 * Activation bytes per token per layer that the backward pass keeps (16-bit), for today's layers; see the module.
 * @param {Model} m @param {number} seq @param {boolean} flash @param {boolean} [dropout]
 */
export function activationPerTokenLayer(m, seq, flash = true, dropout = false) {
  const h = m.hidden;
  const d = dims(m);
  const f = m.experts ? m.ffn * (m.topK ?? 1) + (m.sharedFfn ?? m.ffn * (m.sharedExperts ?? 0)) : m.ffn;
  const mlp = m.mlp === "plain" ? 2 * h + 4 * f : 2 * h + 6 * f; // input, then the inner tensors (gate, up, product: 3; plain: 2)
  const attention = 2 * h + 2 * h + 2 * d.q + 2 * d.k + 2 * d.v + 2 * d.o; // norm input, projection input, Q, K, V, the output
  const norms = 2 * h;
  const scores = flash ? 0 : 2 * m.heads * seq; // the softmax output, without FlashAttention
  // dropout: a 1-byte mask after the attention and after the MLP; without FlashAttention also the attention dropout's mask
  // (1 byte) and output (2 bytes) over the scores (FlashAttention regenerates its dropout instead of storing it)
  const masks = dropout ? 2 * h + (flash ? 0 : 3 * m.heads * seq) : 0;
  return attention + mlp + norms + scores + masks;
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
 * @property {"first"|"last"} [stage] with pipeline parallelism, the stage this is (the fuller of the two ends)
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
 * @property {number} [ep] expert parallel (the experts split over this many of the data-parallel GPUs)
 * @property {0|1|2|3} [zero] ZeRO stage (FSDP full shard = 3, shard grad op = 2)
 * @property {number} [seq]
 * @property {number} [microBatch]
 * @property {"none"|"full"} [recompute]
 * @property {boolean} [flash]
 * @property {boolean} [dropout] dropout in training (most LLM training turns it off)
 * @property {boolean} [sequenceParallel] with tensor parallelism, split the norms' and projections' inputs too (Megatron's
 *   sequence parallelism; true by default). Without it they are kept whole on every tensor-parallel GPU.
 * @property {2|4} [gradBytes] bytes per gradient: 2 (BF16, DeepSpeed), 4 (Megatron's FP32 main gradients)
 * @property {keyof typeof OPTIMIZERS} [optimizer]
 * @property {number} [overheadGiB] CUDA context, allocator: 1 GiB by default
 */

/** @param {unknown} v @param {string} name @param {number} fallback */
function whole(v, name, fallback) {
  if (v === undefined) return fallback;
  if (typeof v !== "number" || !Number.isInteger(v) || v < 1) throw new Error(`${name} must be a whole number ≥ 1, not ${String(v)}`);
  return v;
}

/**
 * The parallel layout, checked: tensor × pipeline must divide the GPUs; expert parallelism splits the experts over part
 * of the data-parallel GPUs (it must divide both); ZeRO-2/3 do not combine with pipeline parallelism (DeepSpeed refuses
 * it; Megatron's distributed optimizer shards only the optimizer state, as ZeRO-1).
 * @param {TrainConfig} c
 */
function parallel(c) {
  const gpus = whole(c.gpus, "gpus", 1);
  const tp = whole(c.tp, "tp", 1);
  const pp = whole(c.pp, "pp", 1);
  const ep = whole(c.ep, "ep", 1);
  const zero = c.zero ?? 0;
  if (![0, 1, 2, 3].includes(zero)) throw new Error(`zero must be 0, 1, 2 or 3, not ${zero}`);
  if (gpus % (tp * pp) !== 0) throw new Error(`gpus (${gpus}) must be a multiple of tp × pp (${tp} × ${pp})`);
  const dp = gpus / (tp * pp);
  if (ep > 1) {
    if (!c.model.experts) throw new Error(`ep ${ep}: ${c.model.name} has no experts`);
    if (dp % ep !== 0) throw new Error(`ep (${ep}) must divide the data-parallel GPUs (${dp} = gpus / (tp × pp))`);
    if (c.model.experts % ep !== 0) throw new Error(`ep (${ep}) must divide the ${c.model.experts} experts`);
  }
  if (pp > 1 && zero >= 2) throw new Error("ZeRO-2/3 with pipeline parallelism: DeepSpeed refuses it, Megatron shards only the optimizer (use --zero 1)");
  if (c.model.layers % pp !== 0) throw new Error(`pp (${pp}) must divide the ${c.model.layers} layers (an even split; frameworks need it unless told otherwise)`);
  headsSplit(c.model, tp);
  return { gpus, tp, pp, dp, ep, zero };
}

/**
 * Tensor parallelism splits the attention heads, and the KV heads (or replicates them when there are more GPUs than KV
 * heads): the heads must divide.
 * @param {Model} m @param {number} tp
 */
function headsSplit(m, tp) {
  if (tp === 1) return;
  if (m.heads % tp !== 0) throw new Error(`tensor parallel ${tp} does not split the ${m.heads} attention heads evenly`);
  if (!m.mla && !(m.kvHeads % tp === 0 || tp % m.kvHeads === 0)) {
    throw new Error(`tensor parallel ${tp} does not split ${m.kvHeads} KV heads evenly (use a divisor or a multiple of ${m.kvHeads})`);
  }
}

/**
 * Parameters on one GPU of a pipeline stage, before ZeRO: the layers split over the stages and tensor parallelism, the
 * embedding on the first stage, the output layer on the last; the experts split by expert parallelism too.
 * @param {Model} m @param {{tp: number, pp: number, ep: number}} p @param {"first"|"last"} stage
 */
function stageParams(m, p, stage) {
  const n = params(m);
  const layers = n.total - n.experts - n.embed - n.head;
  const ends = (stage === "first" || p.pp === 1 ? n.embed : 0) + (stage === "last" || p.pp === 1 ? n.head : 0);
  // what ZeRO-3 / FSDP gathers whole at a time: a layer, or the stage's embedding / output layer when larger
  const layer = (layers + n.experts / p.ep) / m.layers / p.tp;
  return { dense: (layers / p.pp + ends) / p.tp, experts: n.experts / (p.pp * p.tp * p.ep), gathered: Math.max(layer, ends / p.tp), ends: ends / p.tp, layer };
}

/**
 * Activations on one GPU of a stage: with 1F1B the first stage keeps as many microbatches as there are stages, which comes
 * to the whole model's layers; the last stage keeps one, and the FP32 logits.
 * @param {TrainConfig} c @param {{tp: number, pp: number}} p @param {"first"|"last"} stage
 */
function activations(c, p, stage) {
  const m = c.model;
  const seq = c.seq ?? 4096;
  const b = c.microBatch ?? 1;
  const sp = c.sequenceParallel ?? true;
  const all = activationPerTokenLayer(m, seq, c.flash ?? true, c.dropout ?? false);
  // without sequence parallelism the norms' inputs, the attention's and the MLP's input (and dropout's two masks) are
  // whole on each tensor-parallel GPU: Korthikanti et al.'s sbh(10 + 24/t + 5as/ht)
  const whole = sp || p.tp === 1 ? 0 : 8 * m.hidden + (c.dropout ? 2 * m.hidden : 0);
  const perLayer = whole + (all - whole) / p.tp;
  const input = sp ? (2 * m.hidden) / p.tp : 2 * m.hidden;
  const layersHeld = p.pp === 1 || stage === "first" ? m.layers : m.layers / p.pp;
  const kept = c.recompute === "full" ? layersHeld * input + perLayer : layersHeld * perLayer; // full: each layer's input, and one being recomputed
  const logits = p.pp === 1 || stage === "last" ? (4 * m.vocab) / p.tp : 0; // FP32 logits for the loss, the vocabulary split
  return seq * b * (kept + logits);
}

/**
 * The fuller end of the pipeline (or the one stage without it).
 * @param {(stage: "first"|"last") => Breakdown} at @param {number} pp
 */
function fuller(at, pp) {
  if (pp === 1) return at("first");
  const first = { ...at("first"), stage: /** @type {const} */ ("first") };
  const last = { ...at("last"), stage: /** @type {const} */ ("last") };
  return first.total >= last.total ? first : last;
}

/**
 * Full training (pre-training or full fine-tuning) memory per GPU. With ZeRO-3 one layer's weights are gathered whole for
 * its computation, and its gradients reduced from a whole buffer: both are counted.
 * @param {TrainConfig} c
 */
export function trainMemory(c) {
  const p = parallel(c);
  const opt = OPTIMIZERS[c.optimizer ?? "adamw"];
  if (!opt) throw new Error(`unknown optimizer ${c.optimizer} (${Object.keys(OPTIMIZERS).join(", ")})`);
  const z = p.zero;
  const edp = p.dp / p.ep; // the experts' data-parallel group
  return fuller((stage) => {
    const s = stageParams(c.model, p, stage);
    /** @param {number} bytes @param {number} from */
    const state = (bytes, from) => (s.dense * bytes) / (z >= from ? p.dp : 1) + (s.experts * bytes) / (z >= from ? edp : 1);
    const gathered = z >= 3 && p.dp > 1 ? s.gathered * 2 : 0;
    const gb = c.gradBytes ?? 2;
    return breakdown({
      weights: state(2, 3) + gathered,
      grads: state(gb, 2) + (gathered * gb) / 2,
      optimizer: state(opt.bytes, 1),
      activations: activations(c, p, stage),
      overhead: (c.overheadGiB ?? 1) * GiB,
    });
  }, p.pp);
}

/**
 * @typedef {TrainConfig & {method: "full"|"lora"|"qlora", rank?: number, targets?: "all"|"attention"}} FinetuneConfig
 */

/**
 * Fine-tuning memory per GPU: full (as training), LoRA (frozen 16-bit weights) or QLoRA (frozen 4-bit NF4 weights,
 * the embeddings and output layer in 16-bit). ZeRO-3 / FSDP shards the frozen weights too (as FSDP with QLoRA does);
 * the adapters' gradients and optimizer state are sharded from ZeRO-2 and ZeRO-1.
 * @param {FinetuneConfig} c
 */
export function finetuneMemory(c) {
  if (c.method === "full") return trainMemory(c);
  const p = parallel(c);
  const m = c.model;
  const opt = OPTIMIZERS[c.optimizer ?? "adamw"];
  if (!opt) throw new Error(`unknown optimizer ${c.optimizer} (${Object.keys(OPTIMIZERS).join(", ")})`);
  const z = p.zero;
  const lora = loraParts(m, whole(c.rank, "rank", 16), c.targets ?? "all");
  const edp = p.dp / p.ep;
  // the adapters follow their matrices: the experts' over tensor, pipeline and expert parallelism, their states sharded
  // over the experts' own data-parallel group
  const denseA = lora.dense / (p.tp * p.pp);
  const expertA = lora.experts / (p.tp * p.pp * p.ep);
  /** @param {number} bytes @param {number} from */
  const adapters = (bytes, from) => (denseA * bytes) / (p.zero >= from ? p.dp : 1) + (expertA * bytes) / (p.zero >= from ? edp : 1);
  return fuller((stage) => {
    const s = stageParams(m, p, stage);
    const perParam = c.method === "qlora" ? BYTES.nf4 : 2;
    // QLoRA keeps the embedding and the output layer in 16-bit; the frozen dense weights shard over the data-parallel
    // GPUs with ZeRO-3, the frozen experts over the experts' own data-parallel group
    const dense = (s.dense - s.ends) * perParam + s.ends * 2;
    const frozen = dense / (z >= 3 ? p.dp : 1) + (s.experts * perParam) / (z >= 3 ? edp : 1);
    // ZeRO-3 / FSDP gathers one unit whole: a layer (in its format), or the embedding and output layer (16-bit in QLoRA)
    const gathered = z >= 3 && p.dp > 1 ? Math.max(s.layer * perParam, s.ends * 2) : 0;
    return breakdown({
      weights: frozen + gathered + adapters(2, 3),
      grads: adapters(2, 2),
      optimizer: adapters(opt.bytes, 1),
      activations: activations(c, p, stage),
      overhead: (c.overheadGiB ?? 1) * GiB,
    });
  }, p.pp);
}

/**
 * @typedef {object} InferConfig
 * @property {Model} model
 * @property {keyof typeof BYTES} [quant] the weights' format
 * @property {keyof typeof BYTES} [kv] the KV cache's format
 * @property {number} [ctx] tokens of context per sequence (prompt and output)
 * @property {number} [batch] sequences at once
 * @property {number} [tp] GPUs the model is split over
 * @property {Model} [draft] a draft model for speculative decoding (16-bit, on each GPU)
 * @property {number} [overheadGiB]
 */

export const PREFILL_TOKENS = 8192; // a serving engine's token budget for one step of chunked prefill, shared by the batch

/** Inference memory per GPU: the weights, the KV cache, a draft model, a working buffer. @param {InferConfig} c */
export function inferMemory(c) {
  const m = c.model;
  const tp = whole(c.tp, "tp", 1);
  headsSplit(m, tp); // as serving engines do: the attention heads must split evenly
  const ctx = whole(c.ctx, "ctx", 4096);
  const batch = whole(c.batch, "batch", 1);
  const q = Object.hasOwn(BYTES, c.quant ?? "bf16") ? BYTES[c.quant ?? "bf16"] : undefined;
  const kvb = Object.hasOwn(BYTES, c.kv ?? "bf16") ? BYTES[c.kv ?? "bf16"] : undefined;
  if (q === undefined || kvb === undefined) throw new Error(`unknown format (${Object.keys(BYTES).join(", ")})`);
  const n = params(m);
  // the embeddings stay 16-bit when the rest is quantized below 8 bits (as GPTQ/AWQ checkpoints keep them)
  const sixteen = q < 1 ? n.embed + n.head : 0;
  const weights = ((n.total - sixteen) * q + sixteen * 2) / tp;
  const kvCache = kvBytesPerToken(m, kvb, tp) * ctx * batch;
  // a draft model in 16-bit, whole on each GPU, its KV cache in the same format as the target's
  const draft = c.draft ? params(c.draft).total * 2 + kvBytesPerToken(c.draft, kvb) * ctx * batch : 0;
  // one step's working tensors for a prefill budget shared by the batch, and the last tokens' logits in FP32
  const tokens = Math.min(batch * ctx, PREFILL_TOKENS);
  const f = m.experts ? m.ffn * (m.topK ?? 1) + (m.sharedFfn ?? m.ffn * (m.sharedExperts ?? 0)) : m.ffn;
  const work = (tokens * (2 * m.hidden + 2 * f) * 2 + batch * m.vocab * 4) / tp;
  return breakdown({ weights, kvCache, draft, activations: work, overhead: (c.overheadGiB ?? 1) * GiB });
}

/**
 * Training compute, time and cost. A fine-tune with frozen weights (LoRA/QLoRA: give `trainable`, the adapters a token
 * uses, as `activeLora` gives them) costs 4 × parameters per token (no weight gradients but the adapters') rather than 6.
 * @param {{model: Model, tokens: number, gpus: number, gpu: Gpu, mfu?: number, price?: number, seq?: number, attention?: boolean, fp8?: boolean, trainable?: number}} c
 */
export function trainTime(c) {
  const m = c.model;
  const mfu = c.mfu ?? 0.4;
  if (!(mfu > 0 && mfu <= 1)) throw new Error(`mfu must be more than 0 and at most 1, not ${mfu}`);
  if (c.price !== undefined && !(c.price >= 0 && Number.isFinite(c.price))) throw new Error(`price must be 0 or more, not ${c.price}`);
  if (!(c.tokens >= 0 && Number.isFinite(c.tokens))) throw new Error(`tokens must be 0 or more, not ${c.tokens}`);
  const gpus = whole(c.gpus, "gpus", 1);
  const n = params(m).active;
  const seq = c.seq ?? 4096;
  const d = dims(m);
  const perToken = (c.trainable !== undefined ? 4 * n + 6 * c.trainable : 6 * n) + (c.attention ? 12 * m.layers * d.q * seq : 0);
  const flops = perToken * c.tokens;
  const peak = (c.fp8 ? c.gpu.fp8 : c.gpu.bf16) ?? null;
  if (peak === null) return { flops, seconds: null, gpuHours: null, cost: null, peak: null, mfu };
  const seconds = flops / (gpus * peak * 1e12 * mfu);
  const gpuHours = (seconds * gpus) / 3600;
  return { flops, seconds, gpuHours, cost: c.price !== undefined ? gpuHours * c.price : null, peak, mfu };
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
 * A model from a Hugging Face config.json (Llama, Mistral, Mixtral, Qwen2 and Qwen-MoE, Gemma, DeepSeek-V2/V3,
 * GPT-2/GPT-Neo/OPT, GPT-NeoX and their kin). Embeddings are tied unless the config says not (Hugging Face's default).
 * @param {Record<string, any>} cfg
 * @param {string} [name]
 * @returns {Model}
 */
export function fromHFConfig(cfg, name = "custom") {
  if (!cfg || typeof cfg !== "object") throw new Error("config.json is an object of fields");
  const c = cfg.text_config && typeof cfg.text_config === "object" ? cfg.text_config : cfg;
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
  const learnedPositions = ["gpt2", "gpt_neo", "opt"].includes(type);
  /** @type {Model} */
  const m = {
    name,
    layers,
    hidden,
    heads,
    kvHeads: num("num_key_value_heads") ?? heads,
    ffn,
    vocab,
    tied: typeof c.tie_word_embeddings === "boolean" ? c.tie_word_embeddings : true,
    source: "config.json",
  };
  const hd = num("head_dim");
  if (hd && hd !== hidden / heads) m.headDim = hd;
  if (learnedPositions || type === "gpt_neox") {
    m.mlp = "plain";
    m.biases = true;
  }
  if (learnedPositions) m.positions = num("n_positions", "max_position_embeddings");
  if (type.startsWith("qwen2")) m.qkvBias = true;
  if (type.startsWith("gemma2") || type.startsWith("gemma3")) m.normsPerLayer = 4;
  const experts = num("num_local_experts", "num_experts", "n_routed_experts");
  if (experts) {
    m.experts = experts;
    m.topK = num("num_experts_per_tok", "moe_top_k") ?? 2;
    const shared = num("n_shared_experts");
    if (shared) m.sharedExperts = shared;
    const sharedFfn = num("shared_expert_intermediate_size"); // Qwen-MoE: one shared expert of its own size
    if (sharedFfn) m.sharedFfn = sharedFfn;
    const dense = num("first_k_dense_replace");
    if (dense) {
      m.denseLayers = dense;
      m.denseFfn = num("intermediate_size") ?? ffn;
    }
  }
  const kvLora = num("kv_lora_rank");
  if (kvLora) {
    m.mla = { qLora: num("q_lora_rank"), kvLora, nope: num("qk_nope_head_dim") ?? 128, rope: num("qk_rope_head_dim") ?? 64, v: num("v_head_dim") ?? 128 };
    m.kvPerToken = kvLora + m.mla.rope;
  }
  return m;
}

/**
 * A provider's listed price for a GPU, or undefined.
 * @param {string} provider @param {string} gpuId
 */
export function priceFor(provider, gpuId) {
  return PRICES.find((p) => p.provider === provider.toLowerCase() && p.gpu === gpuId.toLowerCase());
}

/** @param {string} id */
export function model(id) {
  const key = id.toLowerCase();
  if (!Object.hasOwn(MODELS, key)) throw new Error(`unknown model ${id} (models: ${Object.keys(MODELS).join(", ")}; or a config.json)`);
  return MODELS[key];
}

/** @param {string} id */
export function gpu(id) {
  const key = id.toLowerCase();
  if (!Object.hasOwn(GPUS, key)) throw new Error(`unknown GPU ${id} (GPUs: ${Object.keys(GPUS).join(", ")})`);
  return GPUS[key];
}

/**
 * The verdict, as the command line and the page both say it: the total against the GPU's memory, with the difference
 * in MiB when the rounded figures would look equal ("80.0 GiB of 80.0 GiB: does not fit" must say by how much).
 * @param {number} total @param {number} usable
 */
export function verdict(total, usable) {
  const fits = total <= usable;
  const share = (total / usable) * 100;
  const pct = share >= 99.5 && share < 100.5 ? share.toFixed(1) : share.toFixed(0);
  const near = gib(total) === gib(usable);
  const diff = Math.abs(total - usable) / 2 ** 20;
  const by = near ? `, ${diff < 1 ? "<1" : Math.round(diff)} MiB ${fits ? "to spare" : "over"}` : "";
  return { fits, text: `${gib(total)} of ${gib(usable)}: ${fits ? "fits" : "does not fit"} (${pct}%${by})` };
}

/** Bytes as GiB with one decimal (MiB below 1 GiB, so that a small part does not read 0.0). @param {number} b */
export function gib(b) {
  return b < GiB ? `${Math.round(b / 2 ** 20)} MiB` : `${(b / GiB).toFixed(1)} GiB`;
}
