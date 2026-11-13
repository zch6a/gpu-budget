// @ts-check
// The site as GitHub Pages serves it: site/ with the engine beside it (the page imports the same files the command
// line runs). Writes _site/.
import { copyFileSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "_site");
rmSync(out, { recursive: true, force: true });
mkdirSync(out);
for (const f of readdirSync(join(root, "site"))) copyFileSync(join(root, "site", f), join(out, f));
for (const f of ["engine.js", "data.js", "records.js"]) copyFileSync(join(root, "src", f), join(out, f));
console.log(`site: ${out}`);
