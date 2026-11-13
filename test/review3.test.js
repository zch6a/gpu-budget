// @ts-check
// Regression tests for what the final review found (polishing round 5).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { main } from "../src/cli.js";
import { GPUS, MODELS } from "../src/data.js";
import { activeLora, finetuneMemory, GiB, inferMemory, loraParams, loraParts, params, trainTime, verdict } from "../src/engine.js";

/** @param {string[]} argv */
function run(argv) {
  const out = /** @type {string[]} */ ([]);
  const err = /** @type {string[]} */ ([]);
  const code = main(argv, (s) => out.push(s), (s) => err.push(s));
  return { code, out: out.join("\n"), err: err.join("\n") };
}

const app = readFileSync(fileURLToPath(new URL("../site/app.js", import.meta.url)), "utf8");

describe("final review", () => {
  it("the page clears its result when the engine refuses the settings", () => {
    const handler = app.slice(app.indexOf("} catch (e) {"), app.indexOf("save();", app.indexOf("} catch (e) {")));
    for (const part of ['$("verdict").textContent = ""', '$("bar").replaceChildren()', '"aria-label", ""', '$("compute").textContent = ""']) {
      assert.ok(handler.includes(part), part);
    }
  });

  it("the page says the verdict as the command line does (the engine's verdict)", () => {
    assert.match(app, /say\(memory\.total, usable\)\.text/);
    assert.equal(verdict(80 * GiB + 3 * 2 ** 20, 80 * GiB).text, "80.0 GiB of 80.0 GiB: does not fit (100.0%, 3 MiB over)");
    assert.equal(verdict(24 * GiB - 2 * 2 ** 20, 24 * GiB).text, "24.0 GiB of 24.0 GiB: fits (100.0%, 2 MiB to spare)");
    assert.equal(verdict(40 * GiB, 80 * GiB).text, "40.0 GiB of 80.0 GiB: fits (50%)");
  });

  it("LoRA adapters on the experts are split by expert parallelism", () => {
    const mx = MODELS["mixtral-8x7b"];
    const parts = loraParts(mx, 16);
    assert.ok(parts.experts > 0.9 * (parts.dense + parts.experts));
    const ep8 = finetuneMemory({ model: mx, method: "lora", gpus: 8, ep: 8 });
    const ep1 = finetuneMemory({ model: mx, method: "lora", gpus: 8, ep: 1 });
    assert.ok(Math.abs(ep8.grads - ((parts.dense + parts.experts / 8) * 2)) < 1);
    assert.ok(ep8.optimizer < ep1.optimizer / 4);
  });

  it("QLoRA with ZeRO-3 gathers the embedding/output unit in 16-bit, not NF4", () => {
    const m = MODELS["llama-3.1-8b"];
    const p = params(m);
    const z3 = finetuneMemory({ model: m, method: "qlora", gpus: 8, zero: 3 }).weights;
    assert.ok(z3 > (p.embed + p.head) * 2, `${z3 / GiB}`);
  });

  it("serving refuses a tensor-parallel size that does not split the attention heads, as training does", () => {
    assert.throws(() => inferMemory({ model: MODELS["qwen2.5-7b"], tp: 8 }), /does not split the 28 attention heads/);
    assert.throws(() => inferMemory({ model: MODELS["llama-3.1-8b"], tp: 64 }), /does not split the 32 attention heads/);
    assert.ok(inferMemory({ model: MODELS["qwen2.5-7b"], tp: 4 }).total > 0);
  });

  it("fine-tuning that turned recomputation off is told to turn it on", () => {
    assert.match(run(["finetune", "llama-3.1-70b", "--recompute", "none"]).err, /--recompute full/);
  });

  it("--fp8 without --tokens, and --rank or --targets with --method full, are errors", () => {
    assert.match(run(["train", "llama-3.1-8b", "--gpus", "8", "--zero", "3", "--fp8"]).err, /--fp8 need --tokens/);
    assert.match(run(["finetune", "llama-3.1-8b", "--method", "full", "--rank", "64"]).err, /--rank and --targets are for --method lora or qlora/);
  });

  it("LoRA compute counts the adapters a token uses: topK of the experts'", () => {
    const mx = MODELS["mixtral-8x7b"];
    const parts = loraParts(mx, 16);
    assert.equal(activeLora(mx, 16), parts.dense + (parts.experts * 2) / 8);
    assert.ok(activeLora(mx, 16) < loraParams(mx, 16) / 2);
    const t = trainTime({ model: mx, tokens: 1e9, gpus: 1, gpu: GPUS["h100-sxm"], trainable: activeLora(mx, 16) });
    assert.equal(t.flops, (4 * params(mx).active + 6 * activeLora(mx, 16)) * 1e9);
  });
});
