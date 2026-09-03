import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

const html = readFileSync(new URL("../dist-pages/index.html", import.meta.url), "utf8");
const assets = readdirSync(new URL("../dist-pages/assets", import.meta.url));
const script = assets.find((name) => name.endsWith(".js"));
assert.ok(script, "built JavaScript asset is missing");
const bundle = readFileSync(new URL(`../dist-pages/assets/${script}`, import.meta.url), "utf8");

test("build targets the GitHub Pages base path", () => {
  assert.match(html, /\/SAE-Bench\/assets\//);
  assert.match(html, /SAE-Bench · Autonomous SAE interpretability research/);
});

test("research blog contains feature meaning, activation depth, and steering contrast", () => {
  assert.match(bundle, /Benchmark definition and evaluation protocol/);
  assert.match(bundle, /Experimental setting/);
  assert.match(bundle, /A complete portrait of feature 62610/);
  assert.match(bundle, /Five feature-ID case studies/);
  assert.match(bundle, /Where does it activate/);
  assert.match(bundle, /What changes when the direction is written back/);
  assert.match(bundle, /Before steering/);
  assert.match(bundle, /After steering/);
  assert.match(bundle, /Association between natural activation and steering/);
});

test("public bundle presents feature IDs but excludes private infrastructure", () => {
  assert.match(bundle, /selected_feature_id/);
  assert.match(bundle, /expert_feature_id/);
  assert.match(bundle, /62610/);
  for (const forbidden of ["/mnt/bn/", "ep-2025", "tanyuqiao.ws", "Authorization: Bearer"]) {
    assert.equal(bundle.includes(forbidden), false, `public bundle contains ${forbidden}`);
  }
});
