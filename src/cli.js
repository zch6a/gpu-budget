#!/usr/bin/env node
// @ts-check
import { readFileSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { BYTES, GPUS, MODELS } from "./data.js";
import { finetuneMemory, fromHFConfig, gib, gpu, inferMemory, model, OPTIMIZERS, params, trainMemory, trainTime, usableBytes } from "./engine.js";

const VERSION = "0.1.0";

const HELP = `gpu-budget: how much GPU memory, time and money an LLM takes.

  gpu-budget train llama-3.1-8b --gpu h100-sxm --gpus 8 --zero 3 --seq 4096 --tokens 1e12
  gpu-budget finetune llama-3.1-8b --method qlora --rank 16 --gpu rtx-4090
  gpu-budget infer llama-3.1-70b --quant int4 --ctx 32768 --batch 4 --gpu h100-sxm
  gpu-budget models | gpus

A model is a built-in name (gpu-budget models) or a Hugging Face config.json (a path).
Memory is per GPU; each part is shown apart.

training and fine-tuning:
  --gpus N            GPUs in all (1)          --tp N, --pp N, --ep N   tensor, pipeline, expert parallel (1)
  --zero 0|1|2|3      ZeRO stage / FSDP (0)    --optimizer adamw|adamw-8bit|adamw-bf16|sgd (adamw)
  --seq N             sequence length (4096)   --micro-batch N   sequences per GPU per step (1)
  --recompute none|full (none; full for finetune, as LoRA/QLoRA practice)   --no-flash   attention without FlashAttention
  --tokens N          training tokens: adds compute, time (with --mfu, 0.4) and cost (with --price, $/GPU-hour)
  --method full|lora|qlora, --rank N (16), --targets all|attention (fine-tuning)
inference:
  --quant bf16|fp8|int8|int4 (bf16)   --kv bf16|fp8|int8 (bf16)   --ctx N (4096)   --batch N (1)
  --tp N   GPUs the model is split over   --draft MODEL   a draft model for speculative decoding
all:
  --gpu NAME (h100-sxm)   --overhead GiB (1)   --json

exit status: 0 it fits, 1 it does not fit, 2 an error`;

const OPTIONS = /** @type {const} */ ({
  gpu: { type: "string", default: "h100-sxm" },
  gpus: { type: "string" },
  tp: { type: "string" },
  pp: { type: "string" },
  ep: { type: "string" },
  zero: { type: "string" },
  optimizer: { type: "string" },
  seq: { type: "string" },
  "micro-batch": { type: "string" },
  recompute: { type: "string" },
  "no-flash": { type: "boolean" },
  tokens: { type: "string" },
  mfu: { type: "string" },
  price: { type: "string" },
  fp8: { type: "boolean" },
  method: { type: "string" },
  rank: { type: "string" },
  targets: { type: "string" },
  quant: { type: "string" },
  kv: { type: "string" },
  ctx: { type: "string" },
  batch: { type: "string" },
  draft: { type: "string" },
  overhead: { type: "string" },
  json: { type: "boolean" },
  help: { type: "boolean", short: "h" },
  version: { type: "boolean" },
});

/** @param {string | undefined} v @param {string} name @param {{int?: boolean, min?: number, max?: number}} [o] */
function number(v, name, o = {}) {
  if (v === undefined) return undefined;
  const n = Number(v);
  if (!Number.isFinite(n) || (o.int && !Number.isInteger(n)) || n < (o.min ?? 0) || (o.max !== undefined && n > o.max)) {
    throw new Error(`--${name} ${v}: expected ${o.int ? "a whole number" : "a number"}${o.min !== undefined ? ` ≥ ${o.min}` : ""}${o.max !== undefined ? ` ≤ ${o.max}` : ""}`);
  }
  return n;
}

/** @param {string} v @param {string} name @param {readonly string[]} choices */
function choice(v, name, choices) {
  if (!choices.includes(v)) throw new Error(`--${name} ${v}: one of ${choices.join(", ")}`);
  return v;
}

/** @param {string} id */
function loadModel(id) {
  if (/\.json$/i.test(id)) {
    let text;
    try {
      text = readFileSync(id, "utf8");
    } catch (e) {
      throw new Error(`${id}: cannot be read (${/** @type {Error} */ (e).message})`);
    }
    let cfg;
    try {
      cfg = JSON.parse(text);
    } catch (e) {
      throw new Error(`${id}: not JSON (${/** @type {Error} */ (e).message})`);
    }
    return fromHFConfig(cfg, id);
  }
  return model(id);
}

/** @param {number} n */
function short(n) {
  /** @type {[number, string][]} */
  const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];
  for (const [v, u] of units) if (Math.abs(n) >= v) return `${(n / v).toFixed(2)}${u}`;
  return String(n);
}

/** @param {number} s */
function duration(s) {
  const d = s / 86400;
  return d >= 2 ? `${d.toFixed(1)} days` : `${(s / 3600).toFixed(1)} hours`;
}

/**
 * @param {string[]} argv
 * @param {(s: string) => void} out
 * @param {(s: string) => void} err
 */
export function main(argv, out = (s) => process.stdout.write(`${s}\n`), err = (s) => process.stderr.write(`${s}\n`)) {
  try {
    if (argv.length === 0) {
      out(HELP);
      return 2;
    }
    const { values: v, positionals } = parseArgs({ args: argv, options: OPTIONS, allowPositionals: true, strict: true });
    if (v.version) return (out(`gpu-budget ${VERSION}`), 0);
    if (v.help) return (out(HELP), 0);
    const [command, id, ...rest] = positionals;
    if (rest.length) throw new Error(`unexpected ${rest.join(" ")}`);
    if (command === "models") {
      for (const [k, m] of Object.entries(MODELS)) {
        const p = params(m);
        out(`${k.padEnd(16)} ${m.name.padEnd(32)} ${short(p.total).padStart(8)}${p.active !== p.total ? ` (${short(p.active)} active)` : ""}`);
      }
      return 0;
    }
    if (command === "gpus") {
      for (const [k, g] of Object.entries(GPUS)) {
        out(`${k.padEnd(14)} ${g.name.padEnd(32)} ${String(g.memory).padStart(4)} GiB  ${g.bf16 ? `${g.bf16} TFLOPS BF16` : "compute not published"}`);
      }
      return 0;
    }
    if (!["train", "finetune", "infer"].includes(command ?? "")) throw new Error(`unknown command ${command ?? ""} (train, finetune, infer, models, gpus)`);
    if (!id) throw new Error(`${command}: which model? (gpu-budget models, or a config.json)`);
    const m = loadModel(id);
    const g = gpu(v.gpu ?? "h100-sxm");
    const overheadGiB = number(v.overhead, "overhead");
    let memory;
    /** @type {Record<string, unknown>} */
    const report = { command, model: m.name, gpu: g.name };
    if (command === "infer") {
      const quant = /** @type {keyof typeof BYTES} */ (choice(v.quant ?? "bf16", "quant", ["bf16", "fp16", "fp8", "int8", "int4"]));
      const kv = /** @type {keyof typeof BYTES} */ (choice(v.kv ?? "bf16", "kv", ["bf16", "fp16", "fp8", "int8"]));
      memory = inferMemory({ model: m, quant, kv, ctx: number(v.ctx, "ctx", { int: true, min: 1 }), batch: number(v.batch, "batch", { int: true, min: 1 }),
        tp: number(v.tp, "tp", { int: true, min: 1 }), draft: v.draft ? loadModel(v.draft) : undefined, overheadGiB }); // prettier-ignore
    } else {
      const cfg = {
        model: m,
        gpus: number(v.gpus, "gpus", { int: true, min: 1 }),
        tp: number(v.tp, "tp", { int: true, min: 1 }),
        pp: number(v.pp, "pp", { int: true, min: 1 }),
        ep: number(v.ep, "ep", { int: true, min: 1 }),
        zero: /** @type {0|1|2|3|undefined} */ (number(v.zero, "zero", { int: true, min: 0, max: 3 })),
        seq: number(v.seq, "seq", { int: true, min: 1 }),
        microBatch: number(v["micro-batch"], "micro-batch", { int: true, min: 1 }),
        recompute: /** @type {"none"|"full"} */ (choice(v.recompute ?? (command === "finetune" ? "full" : "none"), "recompute", ["none", "full"])),
        flash: !v["no-flash"],
        optimizer: /** @type {keyof typeof OPTIMIZERS} */ (choice(v.optimizer ?? "adamw", "optimizer", Object.keys(OPTIMIZERS))),
        overheadGiB,
      };
      if (command === "finetune") {
        const method = /** @type {"full"|"lora"|"qlora"} */ (choice(v.method ?? "lora", "method", ["full", "lora", "qlora"]));
        const targets = /** @type {"all"|"attention"} */ (choice(v.targets ?? "all", "targets", ["all", "attention"]));
        memory = finetuneMemory({ ...cfg, method, rank: number(v.rank, "rank", { int: true, min: 1 }), targets });
        report.method = method;
      } else {
        memory = trainMemory(cfg);
      }
      if (v.tokens !== undefined) {
        const t = trainTime({ model: m, tokens: /** @type {number} */ (number(v.tokens, "tokens", { min: 1 })), gpus: cfg.gpus ?? 1, gpu: g,
          mfu: number(v.mfu, "mfu", { min: 0.01, max: 1 }), price: number(v.price, "price"), seq: cfg.seq, fp8: v.fp8 }); // prettier-ignore
        report.compute = t;
      }
    }
    const usable = usableBytes(g);
    const fits = memory.total <= usable;
    Object.assign(report, { memory, usable, fits });
    if (v.json) {
      out(JSON.stringify(report, null, 2));
      return fits ? 0 : 1;
    }
    out(`${m.name} on ${g.name}, ${command}${report.method ? ` (${report.method})` : ""}: per GPU`);
    const labels = { weights: "weights", grads: "gradients", optimizer: "optimizer state", activations: command === "infer" ? "working buffer" : "activations",
      kvCache: "KV cache", draft: "draft model", overhead: "CUDA context, allocator" }; // prettier-ignore
    for (const [k, label] of Object.entries(labels)) {
      const b = /** @type {Record<string, number>} */ (/** @type {unknown} */ (memory))[k];
      if (b > 0) out(`  ${label.padEnd(24)} ${gib(b).padStart(11)}`);
    }
    out(`  ${"total".padEnd(24)} ${gib(memory.total).padStart(11)}  of ${gib(usable)}: ${fits ? "fits" : "does not fit"} (${((memory.total / usable) * 100).toFixed(0)}%)`);
    const t = /** @type {ReturnType<typeof trainTime> | undefined} */ (report.compute);
    if (t) {
      out(`  compute ${t.flops.toExponential(2)} FLOPs (6 × ${short(params(m).active)} active params × tokens)`);
      if (t.seconds !== null && t.gpuHours !== null) {
        out(`  time ${duration(t.seconds)} on ${cfg_gpus(v.gpus)} GPU(s) at ${(t.mfu * 100).toFixed(0)}% MFU of ${t.peak} TFLOPS: ${short(t.gpuHours)} GPU-hours` +
          (t.cost !== null ? `, $${short(t.cost)}` : " (add --price for the cost)")); // prettier-ignore
      } else {
        out(`  time: ${g.name} has no published tensor throughput`);
      }
    }
    if (!fits) err(`gpu-budget: it does not fit: try ${command === "infer" ? "--quant int4, --kv fp8, a shorter --ctx or a smaller --batch, or --tp over more GPUs" : command === "finetune" ? "--method qlora, a smaller --micro-batch or --seq, or more --gpus with --zero 3" : "--zero 3, more --gpus (or --tp/--pp), --recompute full, a smaller --micro-batch or --seq"}`);
    return fits ? 0 : 1;
  } catch (e) {
    err(`gpu-budget: ${/** @type {Error} */ (e).message}`);
    return 2;
  }
}

/** @param {string | undefined} gpus */
function cfg_gpus(gpus) {
  return gpus ?? "1";
}

/** Run as a program (directly, or through npm's bin link), not imported by the tests. */
function isMain() {
  try {
    return Boolean(process.argv[1]) && realpathSync(process.argv[1]) === realpathSync(fileURLToPath(import.meta.url));
  } catch {
    return false;
  }
}

if (isMain()) process.exitCode = main(process.argv.slice(2));
