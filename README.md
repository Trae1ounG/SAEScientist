# SAE-Bench

[中文](README.zh-CN.md) · [Methods and analysis](docs/benchmark_v2_blog.md) · [Machine-readable results](results/leaderboard.json)

**Can an LLM agent discover the right sparse-autoencoder feature from a semantic description and causal feedback?**

SAE-Bench evaluates feature discovery as an agent task. An agent receives an English target description and a restricted probe interface to an official sparse autoencoder (SAE). It must construct its own diagnostic texts, inspect activation measurements, and submit one feature ID. The evaluator then separates three questions that are often conflated:

1. Did the agent recover the expert feature exactly?
2. If not, does its feature reproduce the expert activation pattern?
3. Does steering along that feature causally induce the target behavior without collapsing the model output?

The current release contains 20 expert tasks built on the official Google Gemma Scope SAE for `google/gemma-2-9b-it`, eight agent configurations, and 160 completed discovery runs. Scored agents had no web access and could not read benchmark source, expert IDs, hidden evaluation cases, or public feature labels.

## Results

The primary ordering is macro GT-normalized activation, where the frozen expert feature is `1.0` on each task. Exact match and steering outcomes are reported independently rather than folded into one opaque score.

| Rank | Agent configuration | GT activation | Exact | Causal | Usable | Median time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Kimi K3 High · Cursor | 0.794 | 35% | 45% | 5% | 5.0 min |
| 2 | Grok 4.6 High · Cursor | 0.786 | 35% | 40% | 5% | 9.8 min |
| 3 | Claude Sonnet 5 High · Cursor | 0.783 | 30% | 35% | 10% | 8.2 min |
| 4 | Claude Opus 4.8 High · Cursor | 0.776 | 35% | 45% | 5% | 6.5 min |
| 5 | GPT-5.6 Sol High · Codex | 0.752 | 35% | 40% | 5% | 4.1 min |
| 6 | GPT-5.5 High · Cursor | 0.697 | 30% | 35% | 5% | 7.6 min |
| 7 | GLM-5.2 High · Cursor | 0.692 | 20% | 30% | 10% | 5.7 min |
| 8 | GPT-5.6 Luna High · Codex | 0.645 | 15% | 20% | 5% | 4.0 min |

Across all 160 runs, agents exactly recovered the expert ID in 47 cases (29.4%). Activation similarity predicted causal success at the aggregate level, but was much weaker among non-exact alternatives: a feature can look semantically close under natural activation and still fail to steer in the expert direction.

![Discovery quality versus causal steering](artifacts/leaderboard/discovery_vs_causal.svg)

## What is measured

- **Exact match**: submitted feature ID equals the frozen expert ID.
- **GT-normalized activation**: mean recovery of positive rank, AUROC, positive-control contrast, and activation-pattern correlation relative to the expert.
- **Causal pass**: target induction beats unsteered and matched-random controls under the frozen steering and blinded-judge protocol.
- **Usable pass**: the steered output additionally preserves the requested task and avoids degeneration.
- **Latency**: wall-clock discovery time, reported separately from quality.

There is deliberately no single composite score. A benchmark that hides the difference between semantic retrieval and causal control would make the central failure mode impossible to see.

## Benchmark shape

```text
semantic target
      │
      ▼
offline agent ── writes diagnostic probes ──► restricted SAE probe
      │                                      activations + ranks only
      ▼
one feature ID
      │
      ├── natural-activation comparison against the expert
      └── feature steering against baseline and random direction
                         │
                         ▼
                  blinded PE judgment
```

The expert set uses 20 unique feature/layer pairs: 12 at residual layer 9 and eight at residual layer 20. The tasks span languages, technical registers, domain-specific writing, and ordinary semantic concepts. Every expert direction passes the frozen causal admission gate: 15 remain in the causal-only tier, while five also pass the stricter task-preserving `usable` tier.

## Public release boundary

This repository publishes the methodology, aggregate leaderboard, behavior analysis, and website source. Hidden task payloads, expert IDs, raw agent traces, internal model paths, judge endpoints, and per-prompt PE records remain in a separate private research repository. That split keeps future offline evaluations meaningful and prevents infrastructure details from entering the public Git history.

The full evaluator will be released only after its hidden/public split is finalized. The current JSON files are sufficient to reproduce every aggregate number and visualization on this site without exposing the sealed evaluation inputs.

## Reproduce the public analysis

The `website` branch contains the interactive bilingual research site and its deterministic figure scripts. After checking out that branch:

```bash
npm ci
npm test
```

The generated static site is written to `dist-pages/` and is compatible with GitHub Pages.

## Status

- Benchmark v2: 20 tasks, 160 scored runs, complete.
- Claude Opus 5 High: full 20-task evaluation in progress; it will enter the table only after discovery, activation scoring, steering, and judge validation all complete.
- Paper: in preparation in the private research repository.

## Citation

SAE-Bench is an active research release. A BibTeX entry will be added with the first paper draft; until then, please cite this repository URL and the accessed commit.
