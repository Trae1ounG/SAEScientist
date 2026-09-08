# Evaluation and scores

[Back to README](../README.md) · [Run experiments](experiments.md)

## What is measured?

During discovery, the Agent writes probes and observes activations. After submission,
the evaluator uses the task's evaluation set to measure the selected feature.

**Rank.** For each positive text, a feature's activation is the mean of its three
largest non-special-token activations. Rank compares its mean position in the
full dictionary with Expert's mean position. An inactive feature receives rank
131,072.

**Activation.** AUROC measures how often a positive text produces a larger activation
than a control text, with half credit for ties. Controls include hard negatives
and neutral texts.

**Steering.** The model generates an unmodified answer, a feature-steered answer,
and an answer using a norm-matched random direction. The judge scores target
relevance and instruction preservation from 0 to 4 and separately flags degeneration.
Target relevance determines Steering. Preservation and degeneration are reported
separately.

## Current paper formulas

Let `r` and `r_expert` be mean positive-text ranks. Let `T_feature`,
`T_baseline`, and `T_random` be mean target-relevance ratings over evaluation
prompts and judge passes.

```text
Rank       = 200 * r_expert / (r + r_expert)
Activation = 100 * max(0, 2 * AUROC - 1)
Steering   = 100 * max(0, (T_feature - max(T_baseline, T_random)) / 4)
Overall    = mean(Rank, Activation, Steering)
```

Rank is 100 when the mean rank matches Expert and can exceed 100. Activation and
Steering range from 0 to 100. Overall is the mean of the three scores, not a
percentage accuracy. Expert receives the same formulas, so only Expert Rank is
fixed at 100.

Average each task's scores with equal weight to obtain a run-level result.
For three independent runs, report the mean and **sample** standard deviation
of the three run-level values. Average raw target ratings before taking the maximum
of the controls and clipping. Averaging individually clipped prompt scores would
produce a different metric.

## Score a new experiment

The full runner computes these scores after each task and aggregates them at the
end of the selected experiment. Recompute its summary using the same configuration,
task selection, and repeat count:

```bash
python scripts/run_benchmark.py --config configs/benchmark.json \
  --output-dir outputs/benchmark --score-only
```


Both the full runner and the single-task CLI write `paper_scores.json` after judgment. To score existing
local outputs without new inference or API calls:

```bash
python scripts/score_paper_run.py \
  --activation outputs/cat/rep01/gemma_cat_001/activation.json \
  --judgment outputs/cat/rep01/gemma_cat_001/judgment_summary.json \
  --output outputs/cat/rep01/gemma_cat_001/paper_scores_recomputed.json
```

The activation input can also be a per-run JSON produced by
`score_agent_batch.py`. Use the judge summary for that run's selected feature,
task, and steering strength. For an exact Expert selection, use the matching
Expert replay summary. Retain this mapping when comparing repeated runs.

The scorer reads raw activation ranks, AUROC, and the three judge condition means.
It checks finite values and complete judge summary counts. It does not authenticate
the input files or audit the original transcript. The output records the input
paths and `scoring_version: rank-auroc-steering-v1`. It refuses to overwrite an
existing score file.

For a numerical example, with `r = r_expert`, `AUROC = 0.9`, and target
ratings of 3, 1, and 0.5 for feature, baseline, and random:

```json
{"Rank": 100.0, "Activation": 80.0, "Steering": 50.0, "Overall": 76.66666666666667}
```

This is an arithmetic example, not an experimental result.

## Steering strength

The README's single-task and full-benchmark commands both use candidate calibration.
The runner creates a five-value
grid at 0.5, 0.75, 1.0, 1.25, and 1.5 times the task's recorded Expert strength.
If no strength satisfies the calibration nondegeneration threshold, the evaluator
tries the provided lower fallback grid. It selects from strengths with at least
90% nondegenerate calibration outputs by target success rate, then target score,
then lower strength. The calibration target score uses the suite's cues.
The 4o judge evaluates the subsequent evaluation outputs.

If the Agent selects the Expert feature itself, the evaluator uses the recorded
Expert strength. Save `selected_alpha` and
calibration results with each feature. Different features can therefore use
different strengths. A strength of 160 or 80 does not itself add points to the
score.
