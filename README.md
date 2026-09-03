<div align="center">

# SAEScientist-Bench

### Can AI agents conduct autonomous SAE interpretability research?

[Research blog](https://trae1oung.github.io/SAEScientist/) · [Agent instruction](prompts/discover_feature.md)

</div>

SAEScientist-Bench asks an AI agent to recover a semantically specified feature from an
official sparse autoencoder. The agent receives a restricted activation probe,
designs its own experiments, and submits one feature ID. A trusted evaluator
then measures localization, held-out activation selectivity, and causal steering.

![SAEScientist-Bench system overview](assets/system-overview.svg)

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
cases are under `data/`, and the three-run aggregate is available as
[`results/replicates.json`](results/replicates.json). The representative first-run
records remain in [`results/leaderboard.json`](results/leaderboard.json) and
`results/by_task/`. Raw Agent traces and judge transcripts remain private.

## Results

The current 20-feature experiment uses the same hidden cases and expert directions
for every agent. Each task has three scores in `[0, 2]`, centered so the Expert
feature scores `1.0`: **Rank** measures positive-case rank recovery; **Activation**
averages AUROC, positive-versus-control contrast, and per-case activation-pattern
recovery; **Steering** averages control-adjusted effect recovery and per-instruction
steering-pattern recovery. **Overall** is their equal-weight mean across tasks, with
the Expert baseline at `1.0`. A candidate can
score above `1.0` when its measured rank, separation, or steering effect exceeds
the Expert on the same hidden cases.

Each entry below is the mean ± population standard deviation over three independent
runs; every run contains the same 20 tasks (540 completed Agent episodes in total).

| Model | Overall ↑ | Rank ↑ | Activation ↑ | Steering ↑ | Exact ↑ | Causal ↑ | Usable ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Expert feature baseline | **1.000** | **1.000** | **1.000** | **1.000** | **100%** | 60% | 0% |
| Claude Opus 5 High | **0.738 ± 0.007** | **0.753 ± 0.001** | 0.935 ± 0.003 | **0.526 ± 0.023** | **41.7% ± 2.4** | **18.3% ± 2.4** | 0.0% ± 0.0 |
| Claude Sonnet 5 High | 0.727 ± 0.024 | 0.734 ± 0.023 | **0.939 ± 0.020** | 0.508 ± 0.033 | 40.0% ± 8.2 | 16.7% ± 4.7 | 0.0% ± 0.0 |
| Kimi K3 High | 0.724 ± 0.021 | 0.743 ± 0.046 | 0.927 ± 0.007 | 0.503 ± 0.016 | 31.7% ± 4.7 | 15.0% ± 0.0 | 0.0% ± 0.0 |
| Claude Opus 4.8 High | 0.705 ± 0.046 | 0.702 ± 0.052 | 0.929 ± 0.019 | 0.482 ± 0.071 | 35.0% ± 8.2 | 13.3% ± 4.7 | 0.0% ± 0.0 |
| Grok 4.6 High | 0.701 ± 0.005 | 0.672 ± 0.025 | 0.919 ± 0.001 | 0.513 ± 0.034 | 33.3% ± 6.2 | 16.7% ± 6.2 | 0.0% ± 0.0 |
| GPT-5.6 Sol High | 0.643 ± 0.006 | 0.516 ± 0.018 | 0.902 ± 0.007 | 0.511 ± 0.031 | 35.0% ± 0.0 | 16.7% ± 2.4 | **1.7% ± 2.4** |
| GPT-5.5 High | 0.611 ± 0.024 | 0.556 ± 0.043 | 0.876 ± 0.016 | 0.401 ± 0.046 | 25.0% ± 4.1 | 11.7% ± 2.4 | 0.0% ± 0.0 |
| GLM-5.2 High | 0.586 ± 0.028 | 0.477 ± 0.047 | 0.860 ± 0.020 | 0.420 ± 0.037 | 26.7% ± 6.2 | 13.3% ± 2.4 | 0.0% ± 0.0 |
| GPT-5.6 Luna High | 0.565 ± 0.063 | 0.469 ± 0.122 | 0.852 ± 0.014 | 0.373 ± 0.056 | 20.0% ± 7.1 | 10.0% ± 4.1 | 0.0% ± 0.0 |

Across the 20 tasks, the raw Expert means are positive rank `53.74`, activation
AUROC `0.995`, activation contrast `19.99`, and control-adjusted steering effect
`0.577`. The normalized Expert row is therefore a reference point, not a claim
that no candidate can perform better on an individual task.

Target relevance is the judge's raw semantic-strength measurement. Causal requires
the steered output to beat both the unsteered and norm-matched-random controls;
Usable additionally requires task preservation and non-degenerate text. They remain
separate audit columns instead of being counted a second time in Overall.

Across all 540 runs, 173 recover the exact expert ID, 79 pass the causal gate,
and one passes the stricter usable gate. Natural-activation recovery and steering
effect have Spearman correlation 0.685 overall and 0.322 among non-exact selections.
Full analysis and per-feature visualizations are on the
[research blog](https://trae1oung.github.io/SAEScientist/).

## Reproduce

Use Linux with an NVIDIA GPU and Python 3.10+. Accept the Gemma license on
Hugging Face, then install the benchmark and download the model and official SAE:

```bash
git clone https://github.com/Trae1ounG/SAEScientist.git
cd SAEScientist
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

sae-scientist run --config configs/cat.json
```

The command starts and stops the restricted probe runtime, runs the Agent,
scores the submitted feature on held-out activations, generates baseline / feature /
matched-random steering rollouts, and performs the blinded judge pass. Its complete
output is written under `outputs/cat-codex-run01/`.

Use `sae-scientist serve --config configs/cat.json` only when running the probe as a
separate long-lived service. Developer-level batch commands remain in `scripts/`.
Run the protocol tests with `pytest -q`.
