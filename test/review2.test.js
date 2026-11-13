// @ts-check
// Regression tests for what the second independent review found (polishing round 3).
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { main } from "../src/cli.js";
import { MODELS } from "../src/data.js";
import { activationPerTokenLayer, finetuneMemory, GiB, inferMemory, loraParams, params, trainMemory } from "../src/engine.js";
import { RECORDS } from "../src/records.js";

/** @param {string[]} argv */
function run(argv) {
  const out = /** @type {string[]} */ ([]);
  const err = /** @type {string[]} */ ([]);
  const code = main(argv, (s) => out.push(s), (s) => err.push(s));
  return { code, out: out.join("\n"), err: err.join("\n") };
}

describe("second review", () => {
  it("LoRA with expert parallelism and ZeRO-3: the frozen experts over the experts' group, not twice", () => {
    const mx = MODELS["mixtral-8x7b"];
    const ep8 = finetuneMemory({ model: mx, method: "lora", gpus: 8, ep: 8, zero: 3 }).weights;
    const ep1 = finetuneMemory({ model: mx, method: "lora", gpus: 8, ep: 1, zero: 3 }).weights;
    assert.ok(ep8 / GiB > 9 && ep8 / GiB < 13, `${ep8 / GiB}`);
    assert.ok(ep8 < ep1);
  });

  it("without sequence parallelism the replicated tensors are whole on each GPU: Korthikanti's sbh(10 + 24/t + 5as/ht)", () => {
    const m = MODELS["gpt-3-175b"];
    const t = 8;
    const seq = 2048;
    const sp = trainMemory({ model: m, gpus: t, tp: t, seq, flash: false, dropout: true, overheadGiB: 0 }).activations;
    const noSp = trainMemory({ model: m, gpus: t, tp: t, seq, flash: false, dropout: true, sequenceParallel: false, overheadGiB: 0 }).activations;
    const logits = (seq * 4 * m.vocab) / t;
    const h = m.hidden;
    const a = m.heads;
    assert.equal(noSp - logits, seq * m.layers * h * (10 + 24 / t + (5 * a * seq) / (h * t)));
    assert.ok(Math.abs((sp - logits) / (seq * m.layers) - activationPerTokenLayer(m, seq, false, true) / t) < 1e-6);
  });

  it("training refuses a tensor-parallel size that does not split the heads, and pipeline sizes that do not split the layers", () => {
    assert.throws(() => trainMemory({ model: MODELS["llama-3.1-8b"], gpus: 3, tp: 3 }), /does not split the 32 attention heads/);
    assert.throws(() => trainMemory({ model: MODELS["llama-3.1-405b"], gpus: 4, pp: 4 }), /must divide the 126 layers/);
    assert.throws(() => inferMemory({ model: MODELS["deepseek-v3"], tp: 3 }), /does not split the 128 attention heads/);
  });

  it("LoRA on latent attention: its own matrices (DeepSeek-V3, rank 1: 99,392 per layer)", () => {
    const ds = MODELS["deepseek-v3"];
    const attention = loraParams(ds, 1, "attention") / ds.layers;
    assert.equal(attention, 7168 + 1536 + (1536 + 128 * 192) + (7168 + 576) + (512 + 128 * 256) + (128 * 128 + 7168));
    assert.equal(attention, 99392);
  });

  it("ZeRO-3 gathers the larger of a layer and the embedding/output unit", () => {
    const m = MODELS["llama-3.1-8b"];
    const p = params(m);
    const z3 = trainMemory({ model: m, gpus: 8, zero: 3 });
    assert.equal(z3.weights, (p.total * 2) / 8 + (p.embed + p.head) * 2);
  });

  it("Megatron's FP32 gradients: --fp32-grads doubles the gradient memory", () => {
    const m = MODELS["llama-3.1-8b"];
    assert.equal(trainMemory({ model: m, gradBytes: 4 }).grads, 2 * trainMemory({ model: m }).grads);
  });

  it("the draft model's KV cache uses the target's format", () => {
    const m = MODELS["llama-3.1-70b"];
    const d = MODELS["llama-3.2-1b"];
    const bf = inferMemory({ model: m, draft: d, ctx: 32768 }).draft;
    const f8 = inferMemory({ model: m, draft: d, ctx: 32768, kv: "fp8" }).draft;
    assert.ok(f8 < bf);
  });

  it("options that do not apply are refused, and the hint skips what is set or refused", () => {
    const r = run(["infer", "llama-3.1-70b", "--gpus", "8", "--tokens", "1e12"]);
    assert.equal(r.code, 2);
    assert.match(r.err, /--gpus, --tokens do not apply to infer \(to split the model over GPUs: --tp\)/);
    assert.match(run(["train", "llama-3.1-8b", "--ctx", "8192"]).err, /--ctx does not apply to train/);
    const q = run(["infer", "llama-3.1-405b", "--quant", "int4"]);
    assert.doesNotMatch(q.err, /--quant int4/);
    const pp = run(["train", "llama-3.1-70b", "--gpus", "4", "--pp", "4"]);
    assert.doesNotMatch(pp.err, /--zero 3/);
    assert.match(pp.err, /--zero 1/);
  });

  it("a total just over the limit does not read as 100%", () => {
    // Llama 3.1 8B in BF16 with a context sized to land just above 80 GiB
    let ctx = 1;
    for (const step of [65536, 4096, 256, 16, 1]) while (inferMemory({ model: MODELS["llama-3.1-8b"], ctx: ctx + step, batch: 4 }).total <= 80 * GiB) ctx += step;
    const r = run(["infer", "llama-3.1-8b", "--ctx", String(ctx + 1), "--batch", "4"]);
    assert.equal(r.code, 1);
    assert.match(r.out, /does not fit \(100\.0%\)/);
    assert.match(r.out, /80\.0\d GiB {2}of 80\.0 GiB/);
  });

  it("--fp8 is documented, and a GPU without an FP8 figure is named as such", () => {
    assert.match(run(["--help"]).out, /--fp8/);
    const r = run(["train", "llama-3.1-8b", "--gpus", "8", "--zero", "3", "--tokens", "1e9", "--fp8", "--gpu", "a100-80"]);
    assert.match(r.out, /no checked FP8 tensor throughput for NVIDIA A100 80GB, so no time estimate \(without --fp8: its BF16 peak\)/);
  });

  it("the records' values are pinned, so that a change in the engine shows", () => {
    const pinned = {
      "Llama 2 70B": [0.428, 0.002],
      "QLoRA": [45.8 * GiB, 0.1 * GiB],
    };
    for (const [start, [value, tolerance]] of Object.entries(pinned)) {
      const r = RECORDS.find((x) => x.what.startsWith(start));
      assert.ok(r, start);
      const v = r.computed().value;
      assert.ok(Math.abs(v - value) <= tolerance, `${start}: ${v}`);
    }
  });
});
