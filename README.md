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
for every agent. Each task has three scores in `[0, 1]`, normalized so the Expert
feature scores `1.0`: **Rank** measures positive-case rank recovery; **Activation**
averages AUROC, positive-versus-control contrast, and per-case activation-pattern
recovery; **Steering** averages control-adjusted effect recovery and per-instruction
steering-pattern recovery. **Overall** is their equal-weight mean. **Total** sums
Overall across 20 tasks, so the Expert baseline is `20.0 / 20`.

| Model | Overall ↑ | Total / 20 ↑ | Rank ↑ | Activation ↑ | Steering ↑ | Exact ↑ | Causal ↑ | Usable ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Expert feature baseline | **1.000** | **20.000** | **1.000** | **1.000** | **1.000** | **100%** | 60% | 0% |
| Kimi K3 High | **0.641** | **12.813** | **0.603** | 0.858 | 0.461 | **35%** | **15%** | 0% |
| Grok 4.6 High | 0.640 | 12.803 | 0.558 | **0.862** | **0.500** | **35%** | **15%** | 0% |
| Claude Sonnet 5 High | 0.617 | 12.343 | 0.561 | 0.857 | 0.433 | 30% | 10% | 0% |
| Claude Opus 4.8 High | 0.617 | 12.341 | 0.560 | 0.848 | 0.443 | **35%** | 10% | 0% |
| GPT-5.6 Sol High | 0.597 | 11.947 | 0.461 | 0.849 | 0.482 | **35%** | **15%** | 0% |
| GPT-5.5 High | 0.540 | 10.796 | 0.431 | 0.785 | 0.403 | 30% | 10% | 0% |
| GLM-5.2 High | 0.512 | 10.241 | 0.408 | 0.787 | 0.341 | 20% | **15%** | 0% |
| GPT-5.6 Luna High | 0.453 | 9.063 | 0.315 | 0.755 | 0.290 | 15% | 5% | 0% |

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
