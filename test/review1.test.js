// @ts-check
// Regression tests for what the first independent review found (polishing round 1).
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { MODELS } from "../src/data.js";
import { finetuneMemory, fromHFConfig, GiB, gpu, inferMemory, kvBytesPerToken, loraParams, model, params, PREFILL_TOKENS, trainMemory, trainTime } from "../src/engine.js";
import { main } from "../src/cli.js";

/** @param {string[]} argv */
function run(argv) {
  const out = /** @type {string[]} */ ([]);
  const err = /** @type {string[]} */ ([]);
  const code = main(argv, (s) => out.push(s), (s) => err.push(s));
  return { code, out: out.join("\n"), err: err.join("\n") };
}

describe("first review", () => {
  const mixtral = MODELS["mixtral-8x7b"];

  it("experts' states are sharded over the experts' data-parallel group, not all data-parallel GPUs", () => {
    const p = params(mixtral);
    const a = trainMemory({ model: mixtral, gpus: 8, ep: 8, zero: 3 });
    const dense = p.total - p.experts;
    // dense states over 8 GPUs; each GPU's eighth of the experts over 8 / 8 = 1
    assert.ok(Math.abs(a.optimizer - ((dense * 12) / 8 + (p.experts / 8) * 12)) < 1);
    assert.ok(a.optimizer / GiB > 60);
  });

  it("expert parallelism is checked against the GPUs and the experts", () => {
    assert.throws(() => trainMemory({ model: mixtral, gpus: 1, ep: 8 }), /ep \(8\) must divide the data-parallel GPUs/);
    assert.throws(() => trainMemory({ model: mixtral, gpus: 64, ep: 64 }), /must divide the 8 experts/);
    assert.throws(() => trainMemory({ model: MODELS["llama-3.1-8b"], gpus: 8, ep: 2 }), /has no experts/);
  });

  it("a latent KV cache is kept whole on every tensor-parallel GPU", () => {
    const ds = MODELS["deepseek-v3"];
    const one = inferMemory({ model: ds, ctx: 131072, batch: 8 }).kvCache;
    assert.equal(inferMemory({ model: ds, ctx: 131072, batch: 8, tp: 8 }).kvCache, one);
  });

  it("KV heads: split when they divide, one per GPU when there are more GPUs, refused otherwise", () => {
    const m = MODELS["llama-3.1-70b"]; // 8 KV heads
    assert.equal(kvBytesPerToken(m, 2, 8), kvBytesPerToken(m, 2) / 8);
    assert.equal(kvBytesPerToken(m, 2, 16), kvBytesPerToken(m, 2) / 8);
    assert.throws(() => kvBytesPerToken(m, 2, 3), /does not split 8 KV heads/);
  });

  it("a config.json without tie_word_embeddings is tied (Hugging Face's default): Gemma 2 as published", () => {
    const g = fromHFConfig({ model_type: "gemma2", num_hidden_layers: 42, hidden_size: 3584, num_attention_heads: 16, num_key_value_heads: 8, head_dim: 256,
      intermediate_size: 14336, vocab_size: 256000 }); // prettier-ignore
    assert.equal(params(g).total, MODELS["gemma-2-9b"].published);
  });

  it("fine-tuning shards with ZeRO-3 / FSDP, frozen weights too", () => {
    const m = MODELS["llama-3.1-70b"];
    const zero0 = finetuneMemory({ model: m, method: "qlora", gpus: 8 }).weights;
    const zero3 = finetuneMemory({ model: m, method: "qlora", gpus: 8, zero: 3 }).weights;
    assert.ok(zero3 < zero0 / 6, `${zero0 / GiB} ${zero3 / GiB}`); // an eighth, plus the one layer gathered whole
  });

  it("the prefill buffer is one budget for the batch, not one chunk per sequence", () => {
    const m = MODELS["llama-3.1-8b"];
    const big = inferMemory({ model: m, batch: 256, ctx: 4096 }).activations;
    assert.ok(big < 1.5 * GiB);
    assert.equal(PREFILL_TOKENS, 8192);
  });

  it("pipeline stages: the embedding on the first, the output layer and the logits on the last", () => {
    const m = MODELS["llama-3.1-8b"];
    const p4 = trainMemory({ model: m, gpus: 4, pp: 4 });
    const even = trainMemory({ model: m }).weights / 4;
    assert.ok(p4.weights > even); // an end stage holds a quarter of the layers plus the embedding or the output layer
    assert.ok(p4.stage === "first" || p4.stage === "last");
  });

  it("ZeRO-2/3 with pipeline parallelism is refused, ZeRO-1 allowed", () => {
    const m = MODELS["llama-3.1-8b"];
    assert.throws(() => trainMemory({ model: m, gpus: 8, pp: 4, zero: 3 }), /ZeRO-2\/3 with pipeline parallelism/);
    assert.ok(trainMemory({ model: m, gpus: 8, pp: 4, zero: 1 }).total > 0);
  });

  it("names of Object's own properties are not models or GPUs", () => {
    for (const name of ["constructor", "__proto__", "toString", "hasOwnProperty"]) {
      assert.throws(() => model(name), /unknown model/);
      assert.throws(() => gpu(name), /unknown GPU/);
    }
    assert.equal(run(["train", "constructor"]).code, 2);
  });

  it("config.json: Qwen-MoE's shared expert, DeepSeek's latent attention, GPT-NeoX without learned positions", () => {
    const qwenMoe = fromHFConfig({ model_type: "qwen2_moe", num_hidden_layers: 24, hidden_size: 2048, num_attention_heads: 16, num_key_value_heads: 16,
      intermediate_size: 5632, moe_intermediate_size: 1408, shared_expert_intermediate_size: 5632, num_experts: 60, num_experts_per_tok: 4,
      vocab_size: 151936, tie_word_embeddings: false }); // prettier-ignore
    const p = params(qwenMoe);
    assert.ok(Math.abs(p.total / 14.3e9 - 1) < 0.01, `${p.total}`); // Qwen1.5-MoE-A2.7B: 14.3B
    assert.ok(Math.abs(p.active / 2.7e9 - 1) < 0.02, `${p.active}`); // 2.7B activated
    const ds = fromHFConfig({ model_type: "deepseek_v3", num_hidden_layers: 61, hidden_size: 7168, num_attention_heads: 128, num_key_value_heads: 128,
      intermediate_size: 18432, moe_intermediate_size: 2048, n_routed_experts: 256, n_shared_experts: 1, num_experts_per_tok: 8, first_k_dense_replace: 3,
      q_lora_rank: 1536, kv_lora_rank: 512, qk_nope_head_dim: 128, qk_rope_head_dim: 64, v_head_dim: 128, vocab_size: 129280, tie_word_embeddings: false }); // prettier-ignore
    assert.equal(params(ds).total, params(MODELS["deepseek-v3"]).total);
    const neox = fromHFConfig({ model_type: "gpt_neox", num_hidden_layers: 16, hidden_size: 2048, num_attention_heads: 8, intermediate_size: 8192,
      vocab_size: 50304, max_position_embeddings: 2048, tie_word_embeddings: false }); // prettier-ignore
    assert.equal(neox.positions, undefined);
  });

  it("LoRA compute: 4 × frozen parameters + 6 × adapters per token, not 6 × all", () => {
    const m = MODELS["llama-3.1-8b"];
    const g = gpu("h100-sxm");
    const lora = trainTime({ model: m, tokens: 1e9, gpus: 1, gpu: g, trainable: loraParams(m, 16) }).flops;
    const full = trainTime({ model: m, tokens: 1e9, gpus: 1, gpu: g }).flops;
    assert.ok(lora / full < 0.7 && lora / full > 0.66);
  });

  it("values out of range are refused, in the engine (the page and the command line both)", () => {
    const m = MODELS["llama-3.1-8b"];
    const g = gpu("h100-sxm");
    assert.throws(() => trainTime({ model: m, tokens: 1e9, gpus: 1, gpu: g, mfu: 5 }), /mfu must be/);
    assert.throws(() => trainTime({ model: m, tokens: 1e9, gpus: 1, gpu: g, price: -1 }), /price must be/);
    assert.throws(() => trainMemory({ model: m, tp: 0.5 }), /tp must be a whole number/);
  });

  it("--price or --mfu without --tokens is an error, not ignored", () => {
    const r = run(["train", "llama-3.1-8b", "--gpus", "8", "--zero", "3", "--price", "bogus"]);
    assert.equal(r.code, 2);
    assert.match(r.err, /--price and --mfu need --tokens/);
  });
});
