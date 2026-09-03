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

Start the restricted probe service:

```bash
PYTHONPATH=src python scripts/serve_probe.py \
  --model-path checkpoints/gemma-2-9b-it \
  --sae-path checkpoints/gemma-scope-9b-it-res/layer_9/width_131k/average_l0_121/params.npz \
  --layer 9 --workers 1 --host 127.0.0.1 --port 8765
```

Run an agent on the public Cat task:

```bash
PYTHONPATH=src python scripts/run_agent.py \
  --task examples/cat/task.json \
  --run-id cat-codex-run01 \
  --harness codex --model gpt-5.6-sol --reasoning-effort high \
  --cli-path "$(command -v codex)" \
  --probe-url http://127.0.0.1:8765
```

Score its feature on held-out activation cases:

```bash
PYTHONPATH=src python scripts/score_feature_submission.py \
  --model-path checkpoints/gemma-2-9b-it \
  --full-sae checkpoints/gemma-scope-9b-it-res/layer_9/width_131k/average_l0_121/params.npz \
  --suite examples/cat/suite.json \
  --submission runs/cat-codex-run01/workspace/submission.json \
  --expert-feature-id 62610 --layer 9 \
  --output outputs/cat_activation.json
```

Generate baseline, candidate-feature, and norm-matched-random steering rollouts:

```bash
FEATURE_ID="$(jq -r .feature_id runs/cat-codex-run01/workspace/submission.json)"

PYTHONPATH=src python scripts/fetch_gemma_feature.py \
  --layer 9 --width 131k --average-l0 121 --feature-id "$FEATURE_ID" \
  --output-dir artifacts

PYTHONPATH=src python scripts/evaluate_gemma_feature.py \
  --model-path checkpoints/gemma-2-9b-it \
  --feature "artifacts/gemma2_9b_it_l9_w131k_feature_${FEATURE_ID}.npz" \
  --suite examples/cat/suite.json --alphas 160 --max-new-tokens 192 \
  --output outputs/cat_steering.json
```

Judge the blinded rollouts with a configured GPT-4o endpoint:

```bash
PYTHONPATH=src python scripts/judge_feature_steering.py \
  --result outputs/cat_steering.json \
  --suite examples/cat/suite.json \
  --output-prefix outputs/cat_judgment \
  --provider azure-openai --model-name gpt-4o-2024-11-20 --repeats 2
```

Run the protocol tests with `pytest -q`. Raw traces, private evaluation cases,
and judge credentials are intentionally excluded from the public repository.
