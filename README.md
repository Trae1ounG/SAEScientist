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
| Evaluation | Exact match, expert-normalized activation, causal steering, usable steering |

The agent controls its probe texts, analysis, and search strategy. The trusted
runtime controls the model and SAE, hidden evaluation cases, steering rollouts,
trace audit, and blinded GPT-4o judgment.

The released dataset is indexed by [`data/benchmark.json`](data/benchmark.json):
all 20 task descriptions are under `tasks/`, frozen activation and steering
cases are under `data/`, and the complete 160-run aggregate is available as
[`results/leaderboard.json`](results/leaderboard.json) with one compact record
per task under `results/by_task/`. Raw Agent traces and judge transcripts remain private.

## Results

The current 20-feature experiment uses the same hidden cases and expert
directions for every agent. Activation is normalized against the expert feature;
steering is evaluated against baseline and norm-matched random controls.

| Agent | Activation ↑ | Exact ↑ | GPT-4o target relevance ↑ | Causal steering ↑ |
|---|---:|---:|---:|---:|
| Kimi K3 | **0.794** | **35%** | 0.281 | **15%** |
| Grok 4.6 | 0.786 | **35%** | **0.303** | **15%** |
| Claude Sonnet 5 | 0.783 | 30% | 0.277 | 10% |
| Claude Opus 4.8 | 0.776 | **35%** | 0.226 | 10% |
| Codex Sol | 0.752 | **35%** | 0.298 | **15%** |
| GPT-5.5 | 0.697 | 30% | 0.217 | 10% |
| GLM-5.2 | 0.692 | 20% | 0.242 | **15%** |
| Codex Luna | 0.645 | 15% | 0.170 | 5% |

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
