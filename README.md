<div align="center">

# SAE-Bench

### Can AI agents conduct autonomous SAE interpretability research?

[中文](README.zh-CN.md) · [Research blog](https://trae1oung.github.io/SAE-Bench/) · [Agent instruction](prompts/discover_feature.md)

</div>

SAE-Bench gives an AI agent an English semantic target and a restricted probe
interface to an official sparse autoencoder. The agent must design diagnostic
texts, inspect activations and ranks, revise its hypothesis, and submit one
feature ID. A trusted evaluator then tests exact recovery, held-out activation
agreement, causal steering, and preservation of the original task.

[![SAE-Bench system overview](docs/assets/system-overview.svg)](docs/assets/system-overview.svg)

**Figure 1. SAE-Bench system overview.** The agent can access only its task,
writable workspace, and the local `probe_sae` tool. Expert IDs, evaluation
cases, model weights, and judge credentials remain outside the agent workspace.
The evaluator compares the submitted feature with the expert direction on
natural activation and on baseline, feature-steered, and norm-matched-random
generations.

## Benchmark contract

| Component | Setting |
|---|---|
| Base model | `google/gemma-2-9b-it` |
| SAE | Google DeepMind Gemma Scope residual-stream SAE |
| Agent input | English target description and SAE metadata |
| Agent tool | Text probes returning feature IDs, activations, and full-SAE ranks |
| Agent output | One `submission.json` containing one integer `feature_id` |
| Isolation | No web access and no access to evaluator code, hidden cases, expert IDs, or public labels |
| Evaluation | Exact match, expert-normalized activation, causal steering, and usable steering |

The agent writes the research procedure. The trusted runtime controls model
loading, GPU assignment, append-only traces, hidden evaluation, and final
scoring.

## Repository structure

```text
src/sae_bench/      activation, steering, scoring, suites, and episode logic
agents/             Codex, Cursor, Claude, and OpenCode harness adapters
scripts/            probe service, agent runner, scorer, steering, judge, audit
prompts/            shared agent instruction
examples/cat/       public end-to-end task and evaluation suite
tests/              unit and protocol tests
docs/assets/        framework figure
```

The public `main` branch contains executable benchmark code. The `website`
branch contains the research blog and sanitized aggregate visualizations. Raw
agent traces, complete experiment outputs, and per-prompt judge records are
stored in the private experiment repository.

## Reproduce one complete experiment

The public Cat case runs the same sequence as a scored task: serve the official
SAE, run an isolated agent, score its submitted feature, generate steering
rollouts, and judge the anonymized outputs.

### 1. Install

Use a Linux machine with an NVIDIA GPU, Python 3.10+, and enough memory for
Gemma 2 9B plus one 131K Gemma Scope SAE.

```bash
git clone https://github.com/Trae1ounG/SAE-Bench.git
cd SAE-Bench
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
```

Accept the Gemma license on Hugging Face, then download the exact model and
official SAE checkpoint:

```bash
hf download google/gemma-2-9b-it \
  --local-dir checkpoints/gemma-2-9b-it

hf download google/gemma-scope-9b-it-res \
  layer_9/width_131k/average_l0_121/params.npz \
  --local-dir checkpoints/gemma-scope-9b-it-res
```

### 2. Start the restricted probe service

```bash
ray start --head --num-gpus=1

PYTHONPATH=src python scripts/serve_probe.py \
  --model-path checkpoints/gemma-2-9b-it \
  --sae-path checkpoints/gemma-scope-9b-it-res/layer_9/width_131k/average_l0_121/params.npz \
  --layer 9 --workers 1 --host 127.0.0.1 --port 8765
```

The service loads the model and full SAE once. It returns measurements only;
it never returns feature labels or expert metadata.

### 3. Run the agent

In a second shell, point the adapter at an installed Agent CLI. This example
uses Codex; `agents/` contains the other adapters.

```bash
PYTHONPATH=src python scripts/run_agent.py \
  --task examples/cat/task.json \
  --run-id cat-codex-run01 \
  --harness codex \
  --model gpt-5.6-sol \
  --reasoning-effort high \
  --cli-path "$(command -v codex)" \
  --probe-url http://127.0.0.1:8765
```

The submitted ID is written to
`runs/cat-codex-run01/workspace/submission.json`; the complete agent trajectory
is stored in `runs/cat-codex-run01/logs/agent.jsonl`.

### 4. Score held-out activation

```bash
mkdir -p outputs

PYTHONPATH=src python scripts/score_feature_submission.py \
  --model-path checkpoints/gemma-2-9b-it \
  --full-sae checkpoints/gemma-scope-9b-it-res/layer_9/width_131k/average_l0_121/params.npz \
  --suite examples/cat/suite.json \
  --submission runs/cat-codex-run01/workspace/submission.json \
  --expert-feature-id 62610 \
  --layer 9 \
  --output outputs/cat_activation.json
```

This produces exact match, positive/hard-negative/neutral statistics, AUROC,
activation-pattern correlation, decoder cosine, and the expert-normalized
activation score.

### 5. Generate and judge steering rollouts

Extract the submitted decoder direction from the official checkpoint:

```bash
FEATURE_ID="$(jq -r .feature_id runs/cat-codex-run01/workspace/submission.json)"

PYTHONPATH=src python scripts/fetch_gemma_feature.py \
  --layer 9 --width 131k --average-l0 121 \
  --feature-id "$FEATURE_ID" \
  --output-dir artifacts
```

Run baseline, feature, and norm-matched-random generations:

```bash
PYTHONPATH=src python scripts/evaluate_gemma_feature.py \
  --model-path checkpoints/gemma-2-9b-it \
  --feature "artifacts/gemma2_9b_it_l9_w131k_feature_${FEATURE_ID}.npz" \
  --suite examples/cat/suite.json \
  --alphas 160 \
  --max-new-tokens 192 \
  --output outputs/cat_steering.json
```

Finally, configure an Azure OpenAI deployment of GPT-4o and run two blinded
judge passes over every held-out instruction:

```bash
export AZURE_OPENAI_API_KEY='<your-key>'
export AZURE_OPENAI_ENDPOINT='https://<resource>.openai.azure.com/'
export OPENAI_API_VERSION='<supported-api-version>'

PYTHONPATH=src python scripts/judge_feature_steering.py \
  --result outputs/cat_steering.json \
  --suite examples/cat/suite.json \
  --output-prefix outputs/cat_judgment \
  --provider azure-openai \
  --model-name gpt-4o-2024-11-20 \
  --repeats 2
```

The command writes the full per-prompt records to
`outputs/cat_judgment.jsonl` and the aggregate admission decision to
`outputs/cat_judgment_summary.json`.

## Verify the implementation

```bash
pytest -q
```

Tests cover feature admission, activation scoring, steering, judge aggregation,
agent-run isolation and audit, batch execution, and release validation. GPU
integration requires the downloaded model and SAE; unit tests do not download
checkpoints.
