// @ts-check
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

describe("the package", () => {
  it("holds the code, the page and the docs: no tests, scripts or images", () => {
    const root = fileURLToPath(new URL("..", import.meta.url));
    const out = execSync("npm pack --dry-run --json", { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    const files = JSON.parse(out)[0].files.map((/** @type {{path: string}} */ f) => f.path.replace(/\\/g, "/")).sort();
    assert.deepEqual(files, [
      "LICENSE", "README.md", "README.zh-CN.md", "package.json",
      "site/app.js", "site/index.html", "site/style.css",
      "src/cli.js", "src/data.js", "src/engine.js", "src/records.js",
    ]); // prettier-ignore
  });
});
