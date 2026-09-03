<div align="center">

# SAE-Bench

### Can AI agents conduct autonomous SAE interpretability research?

[Research blog](https://trae1oung.github.io/SAE-Bench/) · [Agent instruction](prompts/discover_feature.md)

</div>

SAE-Bench asks an AI agent to recover a semantically specified feature from an
official sparse autoencoder. The agent receives a restricted activation probe,
designs its own experiments, and submits one feature ID. A trusted evaluator
then measures localization, held-out activation selectivity, and causal steering.

![SAE-Bench system overview](assets/system-overview.svg)

## Benchmark

| | Setting |
|---|---|
| Model | Gemma 2 9B IT |
| SAE | Official Gemma Scope residual-stream SAE, 131K features |
| Agent input | An English description of the target feature |
| Agent access | Text-to-activation probe; no web, labels, expert IDs, or hidden cases |
| Submission | One feature ID |
| Evaluation | Overall Score, rank, activation, steering, exact match, causal control, usable control |

The agent controls its probe texts, analysis, and search strategy. The trusted
runtime controls the model and SAE, hidden evaluation cases, steering rollouts,
trace audit, and blinded GPT-4o judgment.

The released dataset is indexed by [`data/benchmark.json`](data/benchmark.json):
all 20 task descriptions are under `tasks/`, frozen activation and steering
cases are under `data/`, and the complete 160-run aggregate is available as
[`results/leaderboard.json`](results/leaderboard.json) with one compact record
per task under `results/by_task/`. Raw Agent traces and judge transcripts remain private.

## Results

The current 20-feature experiment uses the same hidden cases and expert directions
for every agent. Each task has three scores in `[0, 2]`, centered so the Expert
feature scores `1.0`: **Rank** measures positive-case rank recovery; **Activation**
averages AUROC, positive-versus-control contrast, and per-case activation-pattern
recovery; **Steering** averages control-adjusted effect recovery and per-instruction
steering-pattern recovery. **Overall** is their equal-weight mean. **Total** sums
Overall across 20 tasks, so the Expert baseline is `20.0 / 20`. A candidate can
score above `1.0` when its measured rank, separation, or steering effect exceeds
the Expert on the same hidden cases.

| Model | Overall ↑ | Total / 20 ↑ | Rank ↑ | Activation ↑ | Steering ↑ | Exact ↑ | Causal ↑ | Usable ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Expert feature baseline | **1.000** | **20.000** | **1.000** | **1.000** | **1.000** | **100%** | 60% | 0% |
| Kimi K3 High | **0.718** | **14.370** | **0.748** | 0.920 | 0.488 | **35%** | **15%** | 0% |
| Grok 4.6 High | 0.699 | 13.978 | 0.648 | **0.920** | **0.529** | **35%** | **15%** | 0% |
| Claude Sonnet 5 High | 0.696 | 13.916 | 0.710 | 0.917 | 0.461 | 30% | 10% | 0% |
| Claude Opus 4.8 High | 0.680 | 13.598 | 0.674 | 0.907 | 0.458 | **35%** | 10% | 0% |
| GPT-5.6 Sol High | 0.645 | 12.895 | 0.506 | 0.912 | 0.517 | **35%** | **15%** | 0% |
| GPT-5.5 High | 0.596 | 11.922 | 0.497 | 0.862 | 0.429 | 30% | 10% | 0% |
| GLM-5.2 High | 0.562 | 11.248 | 0.462 | 0.857 | 0.368 | 20% | **15%** | 0% |
| GPT-5.6 Luna High | 0.526 | 10.516 | 0.398 | 0.849 | 0.330 | 15% | 5% | 0% |

Across the 20 tasks, the raw Expert means are positive rank `53.74`, activation
AUROC `0.995`, activation contrast `19.99`, and control-adjusted steering effect
`0.577`. The normalized Expert row is therefore a reference point, not a claim
that no candidate can perform better on an individual task.

Target relevance is the judge's raw semantic-strength measurement. Causal requires
the steered output to beat both the unsteered and norm-matched-random controls;
Usable additionally requires task preservation and non-degenerate text. They remain
separate audit columns instead of being counted a second time in Overall.

These are single-run measurements. Three-run mean and variance, including
Claude Opus 5, will replace this table after the repeated evaluation completes.
Full analysis and per-feature visualizations are on the
[research blog](https://trae1oung.github.io/SAE-Bench/).

## Reproduce

Use Linux with an NVIDIA GPU and Python 3.10+. Accept the Gemma license on
Hugging Face, then install the benchmark and download the model and official SAE:

```bash
git clone https://github.com/Trae1ounG/SAE-Bench.git
cd SAE-Bench
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'

hf download google/gemma-2-9b-it --local-dir checkpoints/gemma-2-9b-it
hf download google/gemma-scope-9b-it-res \
  layer_9/width_131k/average_l0_121/params.npz \
  --local-dir checkpoints/gemma-scope-9b-it-res
```

Configure the Agent and GPT-4o endpoint in [`configs/cat.json`](configs/cat.json),
export the endpoint credentials, and run the complete experiment:

```bash
export AZURE_OPENAI_API_KEY='<your-key>'
export AZURE_OPENAI_ENDPOINT='https://<resource>.openai.azure.com/'
export OPENAI_API_VERSION='<supported-api-version>'

sae-bench run --config configs/cat.json
```

The command starts and stops the restricted probe runtime, runs the Agent,
scores the submitted feature on held-out activations, generates baseline / feature /
matched-random steering rollouts, and performs the blinded judge pass. Its complete
output is written under `outputs/cat-codex-run01/`.

Use `sae-bench serve --config configs/cat.json` only when running the probe as a
separate long-lived service. Developer-level batch commands remain in `scripts/`.
Run the protocol tests with `pytest -q`.
