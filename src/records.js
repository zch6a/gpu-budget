// @ts-check
/**
 * Published numbers the calculator is checked against (each with its source and the source's own words), and what the
 * calculator gives for them. The tests check each; the README and the page show them.
 */

import { BYTES, GPUS, MODELS } from "./data.js";
import { impliedMfu, kvBytesPerToken, params, trainMemory, trainTime } from "./engine.js";

/**
 * @typedef {object} Record
 * @property {string} what
 * @property {string} published the published number, as the source puts it
 * @property {string} quote the source's words
 * @property {string} source
 * @property {() => {value: number, text: string}} computed what the calculator gives
 * @property {(v: number) => boolean} agrees
 */

/** @type {Record[]} */
export const RECORDS = [
  {
    what: "GPT-3 175B, training compute for 300B tokens",
    published: "3.14E+23 FLOPs",
    quote: "Table D.1: GPT-3 175B, total train compute 3.64E+03 PF-days, 3.14E+23 flops; 300 billion tokens",
    source: "https://arxiv.org/abs/2005.14165",
    computed: () => {
      const v = trainTime({ model: MODELS["gpt-3-175b"], tokens: 300e9, gpus: 1, gpu: GPUS["a100-80"] }).flops;
      return { value: v, text: `${v.toExponential(2)} FLOPs` };
    },
    agrees: (v) => Math.abs(v / 3.14e23 - 1) < 0.01,
  },
  {
    what: "Llama 3 405B, training compute for 15.6T tokens",
    published: "3.8 × 10^25 FLOPs",
    quote: "our flagship language model was pre-trained using 3.8×10^25 FLOPs … on 15.6T text tokens",
    source: "https://arxiv.org/abs/2407.21783",
    computed: () => {
      const v = trainTime({ model: MODELS["llama-3.1-405b"], tokens: 15.6e12, gpus: 1, gpu: GPUS["h100-sxm"] }).flops;
      return { value: v, text: `${v.toExponential(2)} FLOPs` };
    },
    agrees: (v) => Math.abs(v / 3.8e25 - 1) < 0.02,
  },
  {
    what: "Llama 2 70B, 1,720,320 A100-80GB GPU-hours for 2T tokens: the MFU that implies",
    published: "1,720,320 GPU hours, 2T tokens",
    quote: "Table 2: 70B, 1720320 GPU hours … A100-80GB … We trained on 2 trillion tokens of data",
    source: "https://arxiv.org/abs/2307.09288",
    computed: () => {
      const flops = trainTime({ model: MODELS["llama-2-70b"], tokens: 2e12, gpus: 1, gpu: GPUS["a100-80"] }).flops;
      const v = impliedMfu(flops, 1720320, GPUS["a100-80"]);
      return { value: v, text: `${(v * 100).toFixed(1)}% MFU implied` };
    },
    agrees: (v) => v > 0.3 && v < 0.6, // the usual range for large dense training (Llama 3 reports 38–43%)
  },
  {
    what: "Model states of a 1.5B-parameter GPT-2 with mixed-precision Adam",
    published: "16 bytes per parameter, at least 24 GB",
    quote: "In total, this results in 2Ψ+2Ψ+KΨ=16Ψ bytes … For a model such as GPT-2 with 1.5 Billion parameters, this leads to a memory requirement of at least 24 GB",
    source: "https://arxiv.org/abs/1910.02054",
    computed: () => {
      const m = trainMemory({ model: MODELS["gpt2-xl"], overheadGiB: 0 });
      const v = m.weights + m.grads + m.optimizer;
      return { value: v, text: `${(v / 1e9).toFixed(1)} GB (${(v / params(MODELS["gpt2-xl"]).total).toFixed(0)} bytes per parameter)` };
    },
    agrees: (v) => v >= 24e9 && Math.abs(v / params(MODELS["gpt2-xl"]).total - 16) < 1e-9,
  },
  {
    what: "KV cache of one token, OPT 13B in FP16",
    published: "800 KB",
    quote: "KV cache of a single token demands 800 KB of space, calculated as 2 (key and value vectors) × 5120 (hidden state size) × 40 (number of layers) × 2 (bytes per FP16)",
    source: "https://arxiv.org/abs/2309.06180",
    computed: () => {
      const v = kvBytesPerToken(MODELS["opt-13b"], BYTES.fp16);
      return { value: v, text: `${v.toLocaleString("en-US")} bytes` };
    },
    agrees: (v) => v === 2 * 5120 * 40 * 2,
  },
];
