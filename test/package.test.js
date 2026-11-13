// @ts-check
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

describe("the package", () => {
  it("holds the command line and the docs: no tests, scripts, images, or the page (it is the site)", () => {
    const root = fileURLToPath(new URL("..", import.meta.url));
    const out = execSync("npm pack --dry-run --json", { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
    const files = JSON.parse(out)[0].files.map((/** @type {{path: string}} */ f) => f.path.replace(/\\/g, "/")).sort();
    assert.deepEqual(files, [
      "LICENSE", "README.md", "README.zh-CN.md", "package.json",
      "src/cli.js", "src/data.js", "src/engine.js", "src/records.js",
    ]); // prettier-ignore
  });
});

describe("the release", () => {
  const root = fileURLToPath(new URL("..", import.meta.url));
  const read = (/** @type {string} */ f) => readFileSync(join(root, f), "utf8");
  const pkg = JSON.parse(read("package.json"));

  it("the CHANGELOG has a section for this version, found as the release workflow finds it", () => {
    // release.yml: awk -v v="$version" '$0 ~ "^## " v "( |$)" {on=1; next} /^## / {on=0} on'
    let on = false;
    const notes = [];
    for (const line of read("CHANGELOG.md").split("\n")) {
      if (new RegExp(`^## ${pkg.version}( |$)`).test(line)) on = true;
      else if (/^## /.test(line)) on = false;
      else if (on) notes.push(line);
    }
    assert.ok(notes.join("").trim().length > 100, "the release notes would be empty");
  });

  it("one version everywhere, and one Node floor", () => {
    const plugin = JSON.parse(read("plugin/.claude-plugin/plugin.json"));
    assert.equal(plugin.version, pkg.version);
    assert.match(read("src/cli.js"), new RegExp(`const VERSION = "${pkg.version.replace(/\./g, "\\.")}"`));
    assert.equal(pkg.engines.node, ">=22");
    assert.match(read("README.md"), /node-22%2B/);
    assert.match(read("plugin/skills/gpu-budget/SKILL.md"), /Node 22\+/);
    assert.match(read(".github/workflows/ci.yml"), /node: \["22", "24"\]/);
  });

  it("the lockfile points at the public npm registry, not a mirror", () => {
    const urls = [...read("package-lock.json").matchAll(/"resolved": "([^"]+)"/g)].map((m) => m[1]);
    assert.ok(urls.length > 0 && urls.every((u) => u.startsWith("https://registry.npmjs.org/")), urls.join(" "));
  });
});