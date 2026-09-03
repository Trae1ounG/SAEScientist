import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

const html = readFileSync(new URL("../dist-pages/index.html", import.meta.url), "utf8");
const assets = readdirSync(new URL("../dist-pages/assets", import.meta.url));
const script = assets.find((name) => name.endsWith(".js"));
assert.ok(script, "built JavaScript asset is missing");
const bundle = readFileSync(new URL(`../dist-pages/assets/${script}`, import.meta.url), "utf8");

test("build targets the GitHub Pages base path", () => {
  assert.match(html, /\/SAEScientist\/assets\//);
  assert.match(html, /SAEScientist-Bench · Autonomous SAE interpretability research/);
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
  assert.match(bundle, /Distribution of all 20 tasks/);
  assert.match(bundle, /Archaeological excavation/);
  assert.match(bundle, /One task, different agent-submitted directions/);
  assert.match(bundle, /How Rank, Activation, and Overall are scored/);
  assert.match(bundle, /Expert feature baseline/);
  assert.match(bundle, /not a ceiling/);
  assert.match(bundle, /Positive, hard negative, and neutral/);
  assert.match(bundle, /How AUROC is computed/);
  assert.match(bundle, /How steering is executed and judged/);
  assert.match(bundle, /One real judgment record/);
  assert.match(bundle, /Complete judge prompt/);
  assert.match(bundle, /Return exactly three ratings with labels A, B, and C/);
  assert.match(bundle, /GPT-4o-2024-11-20/);
  assert.match(bundle, /Expert steering is/);
  assert.doesNotMatch(bundle, /SUMMARY/);
  assert.match(bundle, /Association between natural activation and steering/);
});

test("public bundle presents feature IDs but excludes private infrastructure", () => {
  assert.match(bundle, /selected_feature_id/);
  assert.match(bundle, /expert_feature_id/);
  assert.match(bundle, /62610/);
  const forbidden = [
    ["internal mount path", /\/m[n]t\/b[n]\//],
    ["internal endpoint id", /e[p]-20\d{8,}/],
    ["personal workspace hostname", /[a-z0-9-]+\.w[s]/],
    ["embedded bearer authorization", /Authorizatio[n]:\s*Beare[r]/i],
  ];
  for (const [name, pattern] of forbidden) {
    assert.doesNotMatch(bundle, pattern, `public bundle contains ${name}`);
  }
});
