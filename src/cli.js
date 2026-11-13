#!/usr/bin/env node
// @ts-check
import { readFileSync, realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { BYTES, GPUS, MODELS, PRICES } from "./data.js";
import { finetuneMemory, fromHFConfig, gib, gpu, inferMemory, loraParams, model, OPTIMIZERS, params, priceFor, trainMemory, trainTime, usableBytes } from "./engine.js";

const VERSION = "0.1.0";

const HELP = `gpu-budget: how much GPU memory, time and money an LLM takes.

  gpu-budget train llama-3.1-8b --gpu h100-sxm --gpus 8 --zero 3 --seq 4096 --tokens 1e12
  gpu-budget finetune llama-3.1-8b --method qlora --rank 16 --gpu rtx-4090
  gpu-budget infer llama-3.1-70b --quant int4 --ctx 32768 --batch 4 --gpu h100-sxm
  gpu-budget models | gpus | prices

A model is a built-in name (gpu-budget models) or a Hugging Face config.json (a path).
Memory is per GPU; each part is shown apart.

training and fine-tuning:
  --gpus N            GPUs in all (1)          --tp N, --pp N, --ep N   tensor, pipeline, expert parallel (1)
  --zero 0|1|2|3      ZeRO stage / FSDP (0)    --optimizer adamw|adamw-8bit|adamw-bf16|sgd (adamw)
  --seq N             sequence length (4096)   --micro-batch N   sequences per GPU per step (1)
  --recompute none|full (none; full for finetune, as LoRA/QLoRA practice)   --no-flash   attention without FlashAttention
  --dropout           count dropout's masks (most LLM training has dropout off)
  --no-sp             tensor parallelism without sequence parallelism (DeepSpeed AutoTP, FSDP + TP): the norms' and
                      projections' inputs are whole on each GPU
  --fp32-grads        FP32 gradients (Megatron's main grads), 4 bytes per parameter instead of 2
  --fp8               the time at the GPU's FP8 peak (memory is the same: master weights and states stay as they are)
  --tokens N          training tokens: adds compute, time (with --mfu, 0.4) and cost (with --price: $ per GPU-hour, or a
                      provider whose listed price for the GPU is used: runpod, lambda; gpu-budget prices)
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
  dropout: { type: "boolean" },
  "no-sp": { type: "boolean" },
  "fp32-grads": { type: "boolean" },
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

/**
 * --price: a number of dollars per GPU-hour, or a provider whose listed price for the GPU is used (noted in the report).
 * @param {string | undefined} v @param {string} gpuId @param {Record<string, unknown>} report
 */
function price(v, gpuId, report) {
  if (v === undefined) return undefined;
  if (/^[a-z]+$/i.test(v)) {
    const listed = priceFor(v, gpuId);
    if (!listed) {
      const here = PRICES.filter((p) => p.gpu === gpuId.toLowerCase()).map((p) => p.provider);
      throw new Error(`--price ${v}: no listed price for ${gpuId}${here.length ? ` (listed: ${here.join(", ")})` : ""}; give a number of $ per GPU-hour`);
    }
    report.listedPrice = listed;
    return listed.usd;
  }
  return number(v, "price");
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
export function short(n) {
  if (Math.abs(n) >= 1e15) return n.toExponential(2); // past trillions, a power of ten reads better than "33814473842.01T"
  /** @type {[number, string][]} */
  const units = [[1e12, "T"], [1e9, "B"], [1e6, "M"], [1e3, "K"]];
  for (const [v, u] of units) if (Math.abs(n) >= v) return `${(n / v).toFixed(2)}${u}`;
  return String(Math.round(n * 100) / 100);
}

/** @param {number} s */
export function duration(s) {
  const d = s / 86400;
  if (d >= 730) return `${short(d / 365.25)} years`;
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
    if (command === "prices") {
      for (const p of PRICES) out(`${p.provider.padEnd(8)} ${p.gpu.padEnd(12)} $${p.usd.toFixed(2).padStart(5)}/GPU-hour  ${p.offer}, checked ${p.checked}  ${p.source}`);
      out("Listed on-demand prices change: check the provider's page before relying on one.");
      return 0;
    }
    if (command === "gpus") {
      for (const [k, g] of Object.entries(GPUS)) {
        out(`${k.padEnd(14)} ${g.name.padEnd(32)} ${String(g.memory).padStart(4)} GiB  ${g.bf16 ? `${g.bf16} TFLOPS BF16` : "compute not published"}`);
      }
      return 0;
    }
    if (!["train", "finetune", "infer"].includes(command ?? "")) throw new Error(`unknown command ${command ?? ""} (train, finetune, infer, models, gpus, prices)`);
    if (!id) throw new Error(`${command}: which model? (gpu-budget models, or a config.json)`);
    const notFor = command === "infer"
      ? ["gpus", "pp", "ep", "zero", "optimizer", "seq", "micro-batch", "recompute", "no-flash", "dropout", "no-sp", "fp32-grads", "tokens", "mfu", "price", "fp8", "method", "rank", "targets"]
      : ["quant", "kv", "ctx", "batch", "draft", ...(command === "train" ? ["method", "rank", "targets"] : [])]; // prettier-ignore
    const stray = notFor.filter((k) => /** @type {Record<string, unknown>} */ (v)[k] !== undefined);
    if (stray.length) throw new Error(`${stray.map((k) => `--${k}`).join(", ")} ${stray.length === 1 ? "does" : "do"} not apply to ${command}${command === "infer" && stray.includes("gpus") ? " (to split the model over GPUs: --tp)" : ""}`);
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
        dropout: Boolean(v.dropout),
        sequenceParallel: !v["no-sp"],
        gradBytes: /** @type {2|4} */ (v["fp32-grads"] ? 4 : 2),
        optimizer: /** @type {keyof typeof OPTIMIZERS} */ (choice(v.optimizer ?? "adamw", "optimizer", Object.keys(OPTIMIZERS))),
        overheadGiB,
      };
      if (command === "finetune") {
        const method = /** @type {"full"|"lora"|"qlora"} */ (choice(v.method ?? "lora", "method", ["full", "lora", "qlora"]));
        const targets = /** @type {"all"|"attention"} */ (choice(v.targets ?? "all", "targets", ["all", "attention"]));
        const rank = number(v.rank, "rank", { int: true, min: 1 });
        memory = finetuneMemory({ ...cfg, method, rank, targets });
        report.method = method;
        if (method !== "full") report.trainable = loraParams(m, rank ?? 16, targets);
      } else {
        memory = trainMemory(cfg);
      }
      if (v.tokens === undefined && (v.price !== undefined || v.mfu !== undefined)) throw new Error("--price and --mfu need --tokens (the tokens to train on)");
      if (v.tokens !== undefined) {
        const t = trainTime({ model: m, tokens: /** @type {number} */ (number(v.tokens, "tokens", { min: 1 })), gpus: cfg.gpus ?? 1, gpu: g,
          mfu: number(v.mfu, "mfu", { min: 0.01, max: 1 }), price: price(v.price, v.gpu ?? "h100-sxm", report), seq: cfg.seq, fp8: v.fp8,
          trainable: /** @type {number | undefined} */ (report.trainable) }); // prettier-ignore
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
    out(`${m.name} on ${g.name}, ${command}${report.method ? ` (${report.method})` : ""}: per GPU${memory.stage ? ` (the fuller end of the pipeline: its ${memory.stage} stage)` : ""}`);
    const labels = { weights: "weights", grads: "gradients", optimizer: "optimizer state", activations: command === "infer" ? "working buffer" : "activations",
      kvCache: "KV cache", draft: "draft model", overhead: "CUDA context, allocator" }; // prettier-ignore
    for (const [k, label] of Object.entries(labels)) {
      const b = /** @type {Record<string, number>} */ (/** @type {unknown} */ (memory))[k];
      if (b > 0) out(`  ${label.padEnd(24)} ${gib(b).padStart(11)}`);
    }
    const share = (memory.total / usable) * 100;
    const pct = share >= 99.5 && share < 100.5 ? share.toFixed(1) : share.toFixed(0); // 100% must not hide an overage
    const close = Math.abs(memory.total - usable) < 0.05 * 2 ** 30;
    const total = close ? `${(memory.total / 2 ** 30).toFixed(2)} GiB` : gib(memory.total);
    out(`  ${"total".padEnd(24)} ${total.padStart(11)}  of ${gib(usable)}: ${fits ? "fits" : "does not fit"} (${pct}%)`);
    const t = /** @type {ReturnType<typeof trainTime> | undefined} */ (report.compute);
    if (t) {
      out(`  compute ${t.flops.toExponential(2)} FLOPs (${report.trainable !== undefined ? `4 × ${short(params(m).active)} frozen + 6 × ${short(/** @type {number} */ (report.trainable))} adapter` : `6 × ${short(params(m).active)} active`} params × tokens)`);
      if (t.seconds !== null && t.gpuHours !== null) {
        out(`  time ${duration(t.seconds)} on ${cfg_gpus(v.gpus)} GPU(s) at ${(t.mfu * 100).toFixed(0)}% MFU of ${t.peak} TFLOPS: ${short(t.gpuHours)} GPU-hours` +
          (t.cost !== null ? `, $${short(t.cost)}` : " (add --price for the cost)")); // prettier-ignore
        const listed = /** @type {import("./data.js").Price | undefined} */ (report.listedPrice);
        if (listed) out(`  price: ${listed.provider} ${listed.offer}, $${listed.usd}/GPU-hour, checked ${listed.checked} (${listed.source})`);
      } else {
        out(`  time: no checked ${v.fp8 ? "FP8 " : ""}tensor throughput for ${g.name}, so no time estimate${v.fp8 && g.bf16 ? " (without --fp8: its BF16 peak)" : ""}`);
      }
    }
    if (!fits) err(`gpu-budget: it does not fit: try ${hints(command ?? "", v).join(", ")}`);
    return fits ? 0 : 1;
  } catch (e) {
    err(`gpu-budget: ${/** @type {Error} */ (e).message}`);
    return 2;
  }
}

/**
 * What to try when it does not fit: only what is not set already, and nothing the engine refuses (ZeRO-2/3 with pipeline
 * parallelism).
 * @param {string} command @param {Record<string, string | boolean | undefined>} v
 */
function hints(command, v) {
  const out = [];
  if (command === "infer") {
    if (v.quant !== "int4") out.push("--quant int4");
    if (v.kv !== "fp8" && v.kv !== "int8") out.push("--kv fp8");
    out.push("a shorter --ctx or a smaller --batch", "--tp over more GPUs");
  } else {
    if (command === "finetune" && v.method !== "qlora") out.push("--method qlora");
    if (v.zero !== "3" && (v.pp ?? "1") === "1") out.push("--zero 3");
    if (v.zero !== "1" && (v.pp ?? "1") !== "1") out.push("--zero 1");
    if (v.recompute !== "full" && command === "train") out.push("--recompute full");
    out.push("more --gpus (or --tp/--pp)", "a smaller --micro-batch or --seq");
  }
  return out;
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
