// @ts-check
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { BYTES, GPUS, MODELS } from "../src/data.js";
import {
  activationPerTokenLayer,
  finetuneMemory,
  fromHFConfig,
  GiB,
  inferMemory,
  kvBytesPerToken,
  loraParams,
  params,
  trainMemory,
  trainTime,
  usableBytes,
} from "../src/engine.js";
import { RECORDS } from "../src/records.js";

describe("parameters", () => {
  for (const [id, m] of Object.entries(MODELS)) {
    if (!m.published) continue;
    it(`${id}: the architecture gives the published count`, () => {
      const exact = id !== "gpt-3-175b"; // GPT-3's paper rounds to 174,600 million
      const total = params(m).total;
      if (exact) assert.equal(total, m.published);
      else assert.ok(Math.abs(total / /** @type {number} */ (m.published) - 1) < 1e-4, `${total}`);
    });
  }

  it("a mixture of experts: Mixtral uses 12.9B of its 46.7B per token", () => {
    const p = params(MODELS["mixtral-8x7b"]);
    assert.equal(Math.round(p.active / 1e8) / 10, 12.9);
    assert.ok(p.experts > 0.9 * p.total);
  });
});

describe("published records", () => {
  for (const r of RECORDS) {
    it(r.what, () => {
      const c = r.computed();
      assert.ok(r.agrees(c.value), `${c.text} against ${r.published}`);
    });
  }
});

describe("training memory", () => {
  const m = MODELS["llama-3.1-8b"];
  const n = params(m).total;

  it("ZeRO shards the optimizer, then the gradients, then the weights over the data-parallel GPUs", () => {
    const at = (/** @type {0|1|2|3} */ zero) => trainMemory({ model: m, gpus: 8, zero });
    assert.equal(at(0).optimizer, n * 12);
    assert.equal(at(1).optimizer, (n * 12) / 8);
    assert.equal(at(1).grads, n * 2);
    assert.equal(at(2).grads, (n * 2) / 8);
    assert.equal(at(2).weights, n * 2);
    assert.equal(at(3).weights, (n * 2) / 8);
  });

  it("tensor and pipeline parallelism divide the model; the GPU count must fit them", () => {
    const a = trainMemory({ model: m, gpus: 8, tp: 2, pp: 2 });
    assert.equal(a.weights, (n * 2) / 4);
    assert.throws(() => trainMemory({ model: m, gpus: 6, tp: 4 }), /multiple of tp × pp/);
  });

  it("expert parallelism divides only the experts", () => {
    const mx = MODELS["mixtral-8x7b"];
    const p = params(mx);
    const a = trainMemory({ model: mx, ep: 8 });
    assert.equal(a.weights, (p.total - p.experts + p.experts / 8) * 2);
  });

  it("recomputation and FlashAttention shrink the activations; longer sequences without FlashAttention grow them quadratically", () => {
    const base = { model: m, seq: 4096, microBatch: 1 };
    const none = trainMemory(base).activations;
    assert.ok(trainMemory({ ...base, recompute: "full" }).activations < none / 3);
    assert.ok(trainMemory({ ...base, flash: false }).activations > none);
    const noFlash = (/** @type {number} */ s) => activationPerTokenLayer(m, s, false) - activationPerTokenLayer(m, s, true);
    assert.equal(noFlash(8192), 2 * noFlash(4096)); // per token: the scores grow with the sequence
  });

  it("the activation formula equals Korthikanti et al.'s 34·h bytes per token per layer for Llama 3.1 8B", () => {
    // RMSNorm, SwiGLU, GQA and FlashAttention happen to sum to GPT-3's 34·sbh for this shape
    assert.equal(activationPerTokenLayer(m, 4096, true), 34 * m.hidden);
  });

  it("optimizers: 8-bit Adam halves the state, plain SGD keeps a master copy and momentum", () => {
    assert.equal(trainMemory({ model: m, optimizer: "adamw-8bit" }).optimizer, n * 6);
    assert.equal(trainMemory({ model: m, optimizer: "sgd" }).optimizer, n * 8);
    assert.throws(() => trainMemory({ model: m, optimizer: /** @type {any} */ ("lion") }), /unknown optimizer/);
  });
});

describe("fine-tuning memory", () => {
  const m = MODELS["llama-3.1-8b"];
  it("QLoRA < LoRA < full, with LoRA's trainable parameters only in the gradients and optimizer", () => {
    const base = { model: m, recompute: /** @type {const} */ ("full") };
    const full = finetuneMemory({ ...base, method: "full" }).total;
    const lora = finetuneMemory({ ...base, method: "lora", rank: 16 });
    const qlora = finetuneMemory({ ...base, method: "qlora", rank: 16 }).total;
    assert.ok(qlora < lora.total && lora.total < full);
    assert.equal(lora.grads, loraParams(m, 16) * 2);
  });

  it("LoRA's trainable count: rank × (in + out) for each adapted matrix", () => {
    // Llama 3.1 8B, rank 16, q/k/v/o and gate/up/down: per layer 16 × ((4096+4096) + 2(4096+1024) + (4096+4096) + 3(4096+14336)), × 32
    assert.equal(loraParams(m, 16), 41943040);
    assert.ok(loraParams(m, 16, "attention") < loraParams(m, 16));
  });
});

describe("inference memory", () => {
  const m = MODELS["llama-3.1-70b"];
  it("the KV cache: 2 × layers × KV heads × head size × bytes, per token", () => {
    assert.equal(kvBytesPerToken(m, 2), 2 * 80 * 8 * 128 * 2);
    const a = inferMemory({ model: m, ctx: 32768, batch: 4 });
    assert.equal(a.kvCache, kvBytesPerToken(m, 2) * 32768 * 4);
    assert.equal(inferMemory({ model: m, ctx: 32768, batch: 4, kv: "fp8" }).kvCache, a.kvCache / 2);
  });

  it("quantized weights, the embeddings kept in 16-bit below 8 bits; tensor parallelism divides", () => {
    const p = params(m);
    assert.equal(inferMemory({ model: m, quant: "fp8" }).weights, p.total);
    assert.equal(inferMemory({ model: m, quant: "int4" }).weights, (p.total - p.embed - p.head) * BYTES.int4 + (p.embed + p.head) * 2);
    assert.equal(inferMemory({ model: m, tp: 4 }).weights, (p.total * 2) / 4);
  });

  it("multi-head latent attention: DeepSeek-V3 caches 576 values per token per layer", () => {
    assert.equal(kvBytesPerToken(MODELS["deepseek-v3"], 2), 61 * 576 * 2);
  });

  it("a draft model for speculative decoding adds its weights and cache", () => {
    const d = inferMemory({ model: m, draft: MODELS["llama-3.2-1b"] });
    assert.ok(d.draft > params(MODELS["llama-3.2-1b"]).total * 2);
  });
});

describe("time and cost", () => {
  it("GPU-hours and cost follow from compute, peak and MFU", () => {
    const t = trainTime({ model: MODELS["llama-3.1-8b"], tokens: 1e12, gpus: 8, gpu: GPUS["h100-sxm"], mfu: 0.5, price: 2 });
    const expected = t.flops / (989.5e12 * 0.5) / 3600;
    assert.ok(t.gpuHours !== null && Math.abs(t.gpuHours / expected - 1) < 1e-12);
    assert.equal(t.cost, /** @type {number} */ (t.gpuHours) * 2);
  });

  it("a GPU with no published throughput gives no time", () => {
    assert.equal(trainTime({ model: MODELS["llama-3.1-8b"], tokens: 1e9, gpus: 1, gpu: GPUS["m5-max-128"] }).seconds, null);
  });

  it("Apple: the GPU may use three quarters of the unified memory", () => {
    assert.equal(usableBytes(GPUS["m5-max-128"]), 96 * GiB);
  });
});

describe("Hugging Face config.json", () => {
  const configs = {
    "llama-3.1-8b": { model_type: "llama", num_hidden_layers: 32, hidden_size: 4096, num_attention_heads: 32, num_key_value_heads: 8, intermediate_size: 14336,
      vocab_size: 128256, tie_word_embeddings: false }, // prettier-ignore
    "mixtral-8x7b": { model_type: "mixtral", num_hidden_layers: 32, hidden_size: 4096, num_attention_heads: 32, num_key_value_heads: 8, intermediate_size: 14336,
      vocab_size: 32000, num_local_experts: 8, num_experts_per_tok: 2, tie_word_embeddings: false }, // prettier-ignore
    "qwen2.5-7b": { model_type: "qwen2", num_hidden_layers: 28, hidden_size: 3584, num_attention_heads: 28, num_key_value_heads: 4, intermediate_size: 18944,
      vocab_size: 152064, tie_word_embeddings: false }, // prettier-ignore
    "gemma-2-9b": { model_type: "gemma2", num_hidden_layers: 42, hidden_size: 3584, num_attention_heads: 16, num_key_value_heads: 8, head_dim: 256,
      intermediate_size: 14336, vocab_size: 256000 }, // prettier-ignore
    "gpt2-xl": { model_type: "gpt2", n_layer: 48, n_embd: 1600, n_head: 25, vocab_size: 50257, n_positions: 1024 },
  };
  for (const [id, cfg] of Object.entries(configs)) {
    it(`${id}: the same count as the built-in model`, () => {
      const tied = id === "gemma-2-9b" ? { ...cfg, tie_word_embeddings: true } : cfg;
      assert.equal(params(fromHFConfig(tied)).total, params(MODELS[id]).total);
    });
  }

  it("a config without the essentials is refused", () => {
    assert.throws(() => fromHFConfig({ model_type: "llama" }), /needs num_hidden_layers/);
  });

  it("a multimodal config: its text model", () => {
    const m = fromHFConfig({ text_config: configs["llama-3.1-8b"] });
    assert.equal(params(m).total, params(MODELS["llama-3.1-8b"]).total);
  });
});
