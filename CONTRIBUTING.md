# Contributing

```bash
npm ci --ignore-scripts
npm run check        # the type check (TypeScript on the JSDoc) and the tests
npm run site         # the page in _site/ (serve it: modules do not load from file://)
```

## Where things are

| File | What it holds |
| --- | --- |
| `src/data.js` | the models (each its Hugging Face config.json) and GPUs (each its datasheet), with sources |
| `src/engine.js` | the formulas: parameters, training, fine-tuning and inference memory, compute and time |
| `src/records.js` | published numbers the engine is checked against, with the sources' words |
| `src/cli.js` | the command line |
| `site/` | the page (it imports the same `engine.js`, `data.js` and `records.js`) |

## Adding a model or a GPU

- A model: its architecture from its `config.json`, and `published` set to the parameter count Hugging Face gives for
  its weights (`https://huggingface.co/api/models/<repo>?expand[]=safetensors`). The tests recompute the count from
  the architecture; if it does not match, a field is missing (a bias, an extra norm, tied embeddings).
- A GPU: memory and dense BF16/FP8 tensor throughput from the maker's own page or datasheet (NVIDIA lists sparse;
  dense is half), with the link.
- A published record: the number, the source's own words, and what the engine gives, in `src/records.js`.

Changing a formula should come with the paper it follows, and a record that checks it if one exists.
