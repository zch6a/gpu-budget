// @ts-check
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const html = readFileSync(join(root, "site", "index.html"), "utf8");
const app = readFileSync(join(root, "site", "app.js"), "utf8");

describe("the page", () => {
  it("loads only its own files: a strict Content-Security-Policy, no outside scripts or styles", () => {
    assert.match(html, /http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self'; style-src 'self';/);
    assert.doesNotMatch(html, /<script[^>]+src="https?:/);
    assert.doesNotMatch(html, /<link[^>]+href="https?:/);
    assert.doesNotMatch(html, /<script>(?!<\/script>)/); // no inline script
  });

  it("sets text with textContent, never as HTML", () => {
    assert.doesNotMatch(app, /innerHTML|outerHTML|insertAdjacentHTML|document\.write/);
  });

  it("the built site has every file the page imports, the engine the command line runs", () => {
    execFileSync(process.execPath, [join(root, "scripts", "site.js")], { encoding: "utf8" });
    for (const m of app.matchAll(/from "\.\/([^"]+)"/g)) assert.ok(existsSync(join(root, "_site", m[1])), m[1]);
    for (const f of ["index.html", "style.css", "app.js"]) assert.ok(existsSync(join(root, "_site", f)), f);
    assert.equal(readFileSync(join(root, "_site", "engine.js"), "utf8"), readFileSync(join(root, "src", "engine.js"), "utf8"));
  });
});
