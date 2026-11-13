// @ts-check
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import { main } from "../src/cli.js";

/** @param {string[]} argv */
function run(argv) {
  const out = /** @type {string[]} */ ([]);
  const err = /** @type {string[]} */ ([]);
  const code = main(argv, (s) => out.push(s), (s) => err.push(s));
  return { code, out: out.join("\n"), err: err.join("\n") };
}

describe("command line", () => {
  it("train: the parts, the total, whether it fits, and the time and cost", () => {
    const r = run(["train", "llama-3.1-8b", "--gpus", "8", "--zero", "3", "--tokens", "1e12", "--price", "2"]);
    assert.equal(r.code, 0);
    for (const s of ["weights", "gradients", "optimizer state", "activations", "fits", "FLOPs", "GPU-hours", "$"]) assert.ok(r.out.includes(s), s);
  });

  it("exit 1 and a hint when it does not fit", () => {
    const r = run(["infer", "llama-3.1-405b"]);
    assert.equal(r.code, 1);
    assert.match(r.err, /does not fit: try --quant int4/);
  });

  it("finetune recomputes by default; --json gives the numbers", () => {
    const r = run(["finetune", "llama-3.1-8b", "--method", "qlora", "--gpu", "rtx-4090", "--json"]);
    const j = JSON.parse(r.out);
    assert.equal(r.code, 0);
    assert.equal(j.method, "qlora");
    assert.ok(j.fits && j.memory.total < j.usable);
  });

  it("a Hugging Face config.json as the model", () => {
    const dir = mkdtempSync(join(tmpdir(), "gpu-budget-"));
    const cfg = join(dir, "config.json");
    writeFileSync(cfg, JSON.stringify({ model_type: "llama", num_hidden_layers: 16, hidden_size: 2048, num_attention_heads: 32, num_key_value_heads: 8,
      head_dim: 64, intermediate_size: 8192, vocab_size: 128256, tie_word_embeddings: true })); // prettier-ignore
    const r = run(["infer", cfg, "--json"]);
    assert.equal(r.code, 0);
    assert.ok(JSON.parse(r.out).memory.weights > 2.4e9);
  });

  it("errors are one line and exit 2", () => {
    for (const [argv, message] of /** @type {[string[], RegExp][]} */ ([
      [["train", "gpt-9"], /unknown model gpt-9/],
      [["train", "llama-3.1-8b", "--zero", "4"], /--zero 4: expected a whole number ≥ 0 ≤ 3/],
      [["train", "llama-3.1-8b", "--gpus", "6", "--tp", "4"], /multiple of tp × pp/],
      [["infer", "llama-3.1-8b", "--quant", "int3"], /--quant int3: one of/],
      [["infer", "llama-3.1-8b", "--gpu", "tpu"], /unknown GPU tpu/],
      [["fly"], /unknown command fly/],
      [["train"], /which model/],
      [["train", "llama-3.1-8b", "--colour"], /Unknown option/],
      [["infer", "missing.json"], /missing.json: cannot be read/],
    ])) {
      const r = run(argv);
      assert.equal(r.code, 2, argv.join(" "));
      assert.match(r.err, message);
    }
  });

  it("help with no arguments, --help, --version, the lists", () => {
    assert.equal(run([]).code, 2);
    assert.match(run(["--help"]).out, /gpu-budget train/);
    assert.equal(run(["--version"]).out, "gpu-budget 0.1.0");
    assert.match(run(["models"]).out, /mixtral-8x7b.*12\.88B active/);
    assert.match(run(["gpus"]).out, /h100-sxm.*989\.5 TFLOPS/);
  });

  it("runs as a program from a path with non-ASCII characters", () => {
    const cli = fileURLToPath(new URL("../src/cli.js", import.meta.url));
    const out = execFileSync(process.execPath, [cli, "--version"], { encoding: "utf8" });
    assert.equal(out.trim(), "gpu-budget 0.1.0");
  });
});

describe("the README", () => {
  it("shows what the command prints", () => {
    const readme = readFileSync(fileURLToPath(new URL("../README.md", import.meta.url)), "utf8");
    const command = /npx github:zch6a\/gpu-budget (train [^\n]+)/.exec(readme);
    const shown = /```\n(Llama 3\.1 70B on [\s\S]*?)\n```/.exec(readme);
    assert.ok(command && shown);
    assert.equal(run(command[1].split(" ")).out, shown[1]);
  });
});

describe("prices", () => {
  it("a provider's listed price, with its offer, date and source", () => {
    const r = run(["train", "llama-3.1-8b", "--gpus", "8", "--zero", "3", "--tokens", "1e11", "--price", "runpod"]);
    assert.equal(r.code, 0);
    assert.match(r.out, /price: runpod Secure Cloud pod, \$4\.39\/GPU-hour, checked 2026-11-13 \(https:\/\/www\.runpod\.io\/pricing\)/);
    const j = JSON.parse(run(["train", "llama-3.1-8b", "--gpus", "8", "--zero", "3", "--tokens", "1e11", "--price", "runpod", "--json"]).out);
    assert.equal(j.compute.cost, j.compute.gpuHours * 4.39);
  });

  it("no listed price for that GPU: one line naming the providers that list it", () => {
    const r = run(["train", "llama-3.1-8b", "--tokens", "1e11", "--price", "lambda", "--gpu", "l40s"]);
    assert.equal(r.code, 2);
    assert.match(r.err, /no listed price for l40s \(listed: runpod\)/);
  });

  it("gpu-budget prices lists every price with its date and source", () => {
    const r = run(["prices"]);
    assert.match(r.out, /lambda\s+h100-sxm\s+\$ 4\.59\/GPU-hour/);
    assert.match(r.out, /check the provider's page/);
  });
});
