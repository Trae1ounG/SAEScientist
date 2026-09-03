# SAE-Bench

[中文](README.zh-CN.md) · [Interactive research blog](https://trae1oung.github.io/SAE-Bench/) · [Machine-readable results](results/leaderboard.json)

**SAE-Bench evaluates whether an AI agent can independently investigate and identify a sparse-autoencoder feature.**

The agent receives an English semantic target and a restricted activation-probe API. It writes diagnostic examples, compares feature activations, revises its hypothesis, and submits one Feature ID. A hidden evaluator then checks whether the submitted feature matches the expert feature's activation pattern and causal steering behavior.

The current snapshot uses the official Google Gemma Scope SAE for `google/gemma-2-9b-it`: 20 expert feature/layer tasks, eight agent configurations, and 160 trace-audited discovery episodes. Agents have no web access and cannot inspect benchmark code, hidden cases, expert IDs, or public feature labels.

## Architecture

```text
semantic target
      │
      ▼
isolated agent ── writes positive / hard-negative / neutral probes
      │
      ├── query: text + candidate Feature IDs
      └── receive: activations + ranks
      │
      ▼
one submitted Feature ID
      │
      ▼
frozen hidden evaluator
      ├── exact expert-ID recovery
      ├── held-out activation fidelity
      └── baseline / feature / norm-matched-random steering
                                   │
                                   ▼
                         blinded GPT-4o judge
```

The benchmark reports four outcomes separately:

- **Exact:** whether the submitted ID equals the frozen expert ID.
- **Activation:** expert-normalized recovery of held-out rank, AUROC, activation contrast, and per-case activation pattern.
- **Causal:** whether feature steering induces the target more strongly than both control conditions.
- **Usable:** whether the target is induced without destroying the original user task.

There is no composite score that hides these distinctions.

## Current evidence

| Rank | Agent configuration | Activation | Exact | GPT-4o target | Causal | Usable |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Kimi K3 High · Cursor | 0.794 | 35% | 0.281 | 15% | 0% |
| 2 | Grok 4.6 High · Cursor | 0.786 | 35% | 0.303 | 15% | 0% |
| 3 | Claude Sonnet 5 High · Cursor | 0.783 | 30% | 0.277 | 10% | 0% |
| 4 | Claude Opus 4.8 High · Cursor | 0.776 | 35% | 0.226 | 10% | 0% |
| 5 | GPT-5.6 Sol High · Codex | 0.752 | 35% | 0.298 | 15% | 0% |
| 6 | GPT-5.5 High · Cursor | 0.697 | 30% | 0.217 | 10% | 0% |
| 7 | GLM-5.2 High · Cursor | 0.692 | 20% | 0.242 | 15% | 0% |
| 8 | GPT-5.6 Luna High · Codex | 0.645 | 15% | 0.170 | 5% | 0% |

The single-run snapshot supports three observations:

1. Agents often find semantically related features without recovering the exact expert ID.
2. Activation similarity is informative but insufficient for causal equivalence: under the GPT-4o re-judgment, the activation–steering Spearman correlation is 0.690 overall and 0.378 among non-exact candidates.
3. Strong steering can erase the requested task. Causal target induction and usable control therefore need separate metrics.

GPT-4o marks 19/160 runs causal and none usable; only 1/113 non-exact selections passes the causal gate. Eight of the 20 expert directions also fall below the frozen 70% target-success threshold under this judge. The latter is a judge-sensitivity finding and requires expert re-calibration or a restricted causal subset before a final paper claim. These are snapshot results, not final variance estimates.

## Steering evaluation

For submitted feature `k`, the evaluator intervenes at the SAE layer with

```text
h'_t = h_t + alpha × W_dec[:, k]
```

`alpha` is chosen on five calibration prompts. Evaluation uses 20 separate held-out instructions and three conditions: baseline, the submitted feature, and a norm-matched random decoder direction. Their outputs are shuffled and anonymized before GPT-4o scores target relevance, task preservation, and degeneration twice at temperature zero.

The full judge prompt, label definitions, aggregation formulas, per-feature activation distributions, and steering examples are reported in the [research blog](https://trae1oung.github.io/SAE-Bench/).

## Reproduce the public release

```bash
git clone https://github.com/Trae1ounG/SAE-Bench.git
cd SAE-Bench
git checkout website
npm ci
npm test
npm run dev
```

`npm test` builds the static GitHub Pages site and runs its data/content consistency checks. `npm run dev` starts the local interactive blog. The committed result snapshot is available directly:

```bash
jq '.benchmark, .configurations' results/leaderboard.json
```

The public repository reproduces the reported analysis and website. Hidden prompts and sanitized raw agent traces remain in the private research repository; credentials are never stored, and machine-specific paths are removed from both repositories.

## Citation

Please cite this repository and the accessed commit when using the benchmark or reported results.
