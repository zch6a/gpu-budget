// @ts-check
// The page: the form's values in the link (so a configuration can be shared), the same engine as the command line.
import { GPUS, MODELS, PRICES } from "./data.js";
import { finetuneMemory, fromHFConfig, gib, inferMemory, params, priceFor, trainMemory, trainTime, usableBytes } from "./engine.js";
import { RECORDS } from "./records.js";

const form = /** @type {HTMLFormElement} */ (document.getElementById("form"));
const $ = (/** @type {string} */ id) => /** @type {HTMLElement} */ (document.getElementById(id));

const PARTS = [
  ["weights", "Weights", "var(--c1)", "parameters × bytes per value (sharded by ZeRO-3, split by tensor/pipeline parallelism)"],
  ["grads", "Gradients", "var(--c2)", "trainable parameters × 2 bytes (sharded from ZeRO-2)"],
  ["optimizer", "Optimizer state", "var(--c3)", "trainable parameters × the optimizer's bytes (AdamW mixed precision: 12; sharded from ZeRO-1)"],
  ["activations", "Activations", "var(--c4)", "what the backward pass keeps per token and layer, × sequence × micro-batch (Korthikanti et al.)"],
  ["kvCache", "KV cache", "var(--c5)", "2 × layers × KV heads × head size × bytes, per token of context, × batch"],
  ["draft", "Draft model", "var(--c6)", "a draft model's weights and KV cache, for speculative decoding"],
  ["overhead", "CUDA context, allocator", "var(--c7)", "an allowance of 1 GiB"],
];

/** @param {HTMLSelectElement} select @param {[string, string][]} options */
function fill(select, options) {
  for (const [value, text] of options) {
    const o = document.createElement("option");
    o.value = value;
    o.textContent = text;
    select.append(o);
  }
}

fill(/** @type {HTMLSelectElement} */ ($("model")), [...Object.entries(MODELS).map(([k, m]) => /** @type {[string, string]} */ ([k, m.name])), ["custom", "Custom: paste config.json"]]);
fill(/** @type {HTMLSelectElement} */ ($("gpu")), Object.entries(GPUS).map(([k, g]) => [k, g.name]));
fill(/** @type {HTMLSelectElement} */ ($("draft")), Object.entries(MODELS).filter(([, m]) => params(m).total < 10e9).map(([k, m]) => [k, m.name]));

/** The listed prices for the GPU chosen (keeping the provider chosen when it lists this GPU too). */
function clouds() {
  const select = /** @type {HTMLSelectElement} */ ($("cloud"));
  const keep = select.value;
  select.replaceChildren();
  fill(select, [["", "your own"], ...PRICES.filter((p) => p.gpu === val("gpu")).map((p) => /** @type {[string, string]} */ ([p.provider, `${p.provider}: $${p.usd.toFixed(2)} (${p.checked})`]))]);
  select.value = [...select.options].some((o) => o.value === keep) ? keep : "";
}

/** The form's values as a link query. */
function save() {
  const q = new URLSearchParams();
  for (const [k, v] of new FormData(form)) if (typeof v === "string" && v !== "" && k !== "config") q.set(k, v);
  history.replaceState(null, "", `#${q}`);
}

/** The link's values into the form (only fields the form has; values the selects lack are ignored). */
function load() {
  const q = new URLSearchParams(location.hash.slice(1));
  for (const [k, v] of q) {
    const els = form.elements.namedItem(k);
    if (els instanceof RadioNodeList) {
      for (const r of els) if (r instanceof HTMLInputElement && r.value === v) r.checked = true;
    } else if (els instanceof HTMLSelectElement) {
      if ([...els.options].some((o) => o.value === v)) els.value = v;
    } else if (els instanceof HTMLInputElement) {
      els.value = v;
    }
  }
}

/** @param {string} name */
function val(name) {
  const v = new FormData(form).get(name);
  return typeof v === "string" ? v : "";
}

/** @param {string} name @param {number} fallback */
function num(name, fallback) {
  const n = Number(val(name));
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function render() {
  const mode = val("mode");
  for (const g of document.querySelectorAll(".group")) {
    /** @type {HTMLElement} */ (g).hidden = !(/** @type {HTMLElement} */ (g).dataset.mode ?? "").split(" ").includes(mode);
  }
  $("custom-row").hidden = val("model") !== "custom";
  $("error").hidden = true;
  try {
    const m = val("model") === "custom" ? fromHFConfig(JSON.parse(val("config") || "{}"), "custom") : MODELS[val("model")];
    const g = GPUS[val("gpu")];
    const base = { model: m, gpus: num("gpus", 1), tp: num("tp", 1), pp: num("pp", 1), zero: /** @type {0|1|2|3} */ (Number(val("zero")) || 0),
      seq: num("seq", 4096), microBatch: num("mb", 1), recompute: /** @type {"none"|"full"} */ (val("recompute") || "full"),
      optimizer: /** @type {any} */ (val("optimizer") || "adamw") }; // prettier-ignore
    const memory =
      mode === "infer"
        ? inferMemory({ model: m, quant: /** @type {any} */ (val("quant")), kv: /** @type {any} */ (val("kv")), ctx: num("ctx", 8192), batch: num("batch", 1),
            tp: num("tpi", 1), draft: val("draft") ? MODELS[val("draft")] : undefined }) // prettier-ignore
        : mode === "finetune"
          ? finetuneMemory({ ...base, method: /** @type {any} */ (val("method")), rank: num("rank", 16) })
          : trainMemory(base);
    const usable = usableBytes(g);
    const fits = memory.total <= usable;
    const verdict = $("verdict");
    verdict.textContent = `${gib(memory.total)} per GPU of ${gib(usable)}: ${fits ? "fits" : "does not fit"} (${((memory.total / usable) * 100).toFixed(0)}%)`;
    verdict.className = fits ? "fits" : "nofit";
    const bar = $("bar");
    bar.replaceChildren();
    const scale = Math.max(memory.total, usable);
    const rows = /** @type {HTMLTableSectionElement} */ ($("parts").querySelector("tbody"));
    rows.replaceChildren();
    for (const [key, label, color, how] of PARTS) {
      const b = /** @type {Record<string, number>} */ (/** @type {unknown} */ (memory))[key];
      if (!(b > 0)) continue;
      const seg = document.createElement("span");
      seg.style.width = `${(b / scale) * 100}%`;
      seg.style.background = color;
      seg.title = `${label}: ${gib(b)}`;
      bar.append(seg);
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      const sw = document.createElement("span");
      sw.className = "swatch";
      sw.style.background = color;
      const small = document.createElement("span");
      small.className = "how";
      small.textContent = mode === "infer" && key === "activations" ? "a working buffer: one 512-token prefill chunk per sequence, the last logits" : how;
      name.append(sw, mode === "infer" && key === "activations" ? "Working buffer" : label, small);
      const amount = document.createElement("td");
      amount.className = "num";
      amount.textContent = gib(b);
      tr.append(name, amount);
      rows.append(tr);
    }
    const limit = document.createElement("span");
    limit.className = "limit";
    limit.style.left = `${(usable / scale) * 100}%`;
    limit.title = `${g.name}: ${gib(usable)}`;
    bar.append(limit);
    const p = params(m);
    $("note").textContent = `${m.name}: ${(p.total / 1e9).toFixed(2)}B parameters${p.active !== p.total ? `, ${(p.active / 1e9).toFixed(2)}B active per token` : ""}.` +
      (g.usable ? ` ${g.name}: the GPU may use about ${Math.round(g.usable * 100)}% of the unified memory by default.` : "") +
      (memory.stage ? ` Per GPU of the pipeline's ${memory.stage} stage, the fuller end.` : "") +
      (g.bf16 === null && mode === "train" ? ` ${g.name}: no checked tensor throughput, so no time estimate.` : ""); // prettier-ignore
    const compute = $("compute");
    compute.textContent = "";
    if (mode === "train") {
      const listed = val("cloud") ? priceFor(val("cloud"), val("gpu")) : undefined;
      const t = trainTime({ model: m, tokens: Number(val("tokens")) || 0, gpus: num("gpus", 1), gpu: g, mfu: num("mfu", 0.4),
        price: listed ? listed.usd : Number(val("price")) || undefined, seq: num("seq", 4096) }); // prettier-ignore
      compute.textContent =
        t.seconds === null || t.gpuHours === null
          ? `${t.flops.toExponential(2)} FLOPs; ${g.name} has no published tensor throughput.`
          : `${t.flops.toExponential(2)} FLOPs: ${(t.seconds / 86400).toFixed(1)} days on ${num("gpus", 1)} GPU(s) at ${(t.mfu * 100).toFixed(0)}% MFU, ` +
            `${Math.round(t.gpuHours).toLocaleString("en-US")} GPU-hours${t.cost !== null ? `, $${Math.round(t.cost).toLocaleString("en-US")}` : ""}.` +
            (listed ? ` Price: ${listed.provider} ${listed.offer}, $${listed.usd}/GPU-hour, checked ${listed.checked}.` : ""); // prettier-ignore
    }
  } catch (e) {
    $("error").hidden = false;
    $("error").textContent = /** @type {Error} */ (e).message;
  }
  save();
}

function records() {
  const body = /** @type {HTMLTableSectionElement} */ ($("records").querySelector("tbody"));
  for (const r of RECORDS) {
    const tr = document.createElement("tr");
    const what = document.createElement("td");
    const link = document.createElement("a");
    link.href = r.source;
    link.textContent = r.what;
    what.append(link);
    const pub = document.createElement("td");
    pub.textContent = r.published;
    pub.title = r.quote;
    const ours = document.createElement("td");
    const c = r.computed();
    ours.textContent = `${c.text} ${r.agrees(c.value) ? "✓" : "✗"}`;
    tr.append(what, pub, ours);
    body.append(tr);
  }
}

clouds();
load();
clouds();
load(); // the provider options depend on the GPU the link names
form.addEventListener("input", (e) => {
  if (/** @type {HTMLElement} */ (e.target).getAttribute("name") === "gpu") clouds();
  render();
});
window.addEventListener("hashchange", () => (load(), clouds(), load(), render()));
records();
render();
