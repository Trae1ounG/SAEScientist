# Running and evaluating an Agent

[Back to README](../README.md)

## Configure the experiment

Use `configs/benchmark.json` for the complete workflow. Set `model_path` to the
downloaded Gemma model and `sae_root` to the Gemma Scope directory containing
the two layer folders. The runner reads the task index to select the correct
layer, SAE revision, evaluation set, Expert ID, and generation length.

Under `agent`, choose `codex` or `cursor`, provide its executable name through
`cli`, and choose a model available to your account. Preserve the same model,
reasoning setting, and time budget when comparing repeated attempts. The vendor
CLI must be installed and authenticated before running the benchmark. Confirm
compatibility with its harness in `agents/`, including sandbox support.

Set `judge.model` to your Azure OpenAI deployment serving 4o. The judge currently
supports Azure OpenAI and uses two passes per evaluation prompt. Supply
`AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `OPENAI_API_VERSION`
through environment variables. The configuration stores the credential variable
name, not its value.

The runner uses one probe worker and processes tasks sequentially. Use Linux and
an NVIDIA GPU that can hold the base model and full SAE dictionary. There is no
need to start Ray or a probe server manually for this entry point. Keep the
default loopback probe address, since the probe service has no authentication.

## Check the plan and run one task

```bash
python scripts/run_benchmark.py --config configs/benchmark.json \
  --task-id gemma_cat_001 --replicates 1 --output-dir outputs/cat --dry-run
python scripts/run_benchmark.py --config configs/benchmark.json \
  --task-id gemma_cat_001 --replicates 1 --check
python scripts/run_benchmark.py --config configs/benchmark.json \
  --task-id gemma_cat_001 --replicates 1 --output-dir outputs/cat
```

The dry run validates the task data and configurations, prints the selected tasks
and planned attempt count, and creates no files. It does not check GPU memory,
download weights, authenticate the Agent, or call the judge.

`--check` inspects the local installation, CUDA availability, model architecture,
weight shard presence, tokenizer files, SAE paths, agent executable, judge settings,
and probe port. It reports setup problems together. A real run performs these
checks before creating its output directory. Service authentication, checkpoint
integrity, CLI sandbox compatibility, and peak GPU memory require the single-task run.

The real run loads the model and task's SAE. The Agent receives the concept
description and SAE specification, then uses `probe_sae` to measure its own
texts. Its submission is a JSON object containing a single integer `feature_id`.
The runner audits the local trace before evaluating the submitted feature.
Evaluation inputs and Expert metadata remain outside the Agent workspace.

After activation evaluation, the probe stops. The evaluator extracts the selected
feature, calibrates steering, generates answers, and obtains judge ratings.
A successful run produces `paper_scores.json` for the task and `summary.json`
for the experiment.

## Run all tasks and repeated attempts

Use a separate output directory for the full experiment:

```bash
python scripts/run_benchmark.py --config configs/benchmark.json \
  --output-dir outputs/benchmark
```

With the default `replicates: 3`, this executes 60 investigations for one Agent.
Use repeated `--task-id` options for a smaller selection, or `--replicates 1`
for one pass. A subset summary reports its actual task count and should not be
presented as a complete benchmark result.

Use a separate configuration and output directory for another Agent.
Each attempt has a fresh workspace and a distinct run ID. Tasks run sequentially,
including loading weights again for the next task. This entry point favors a
simple one-GPU setup; lower-level batch scripts remain available for deployments
that manage their own probe workers.

The runner refuses to start a new experiment in an existing output directory.
If a stage fails, inspect that task's local logs and outputs. It stops without
reporting an aggregate over incomplete tasks. The runner does not automatically
resume a failed experiment. Resolve the problem and use a new output directory
for a new experiment, or inspect individual stages with the lower-level scripts.

## How steering strength is chosen

For each task, the reference configuration supplies an Expert strength.
A candidate feature is tested at 0.5, 0.75, 1.0, 1.25, and 1.5 times that strength.
If none passes the calibration nondegeneration check, lower fallback strengths
are tried. Among strengths passing the check, the evaluator selects by target
success rate, then target score, then the smaller strength. These measurements
use the calibration prompts and the suite's target cues.

When the selected feature is Expert itself, the recorded Expert strength is used.
The chosen strength is then applied to both the feature direction and a
norm-matched random direction on the evaluation prompts. The judge rates those
answers and the unmodified baseline. [Evaluation](evaluation.md) explains how
these ratings become a Steering score.

The older `sae-scientist run --config configs/cat.json` command still runs an
individual fixed-strength example. Use `run_benchmark.py` for the task-specific
calibration described above.

## Inspect your outputs and recompute scores

The experiment directory contains one folder per repeat, with a task folder
inside it. For example, `rep01/gemma_cat_001/` holds the resolved `config.json`,
`activation.json`, `steering.json`, `judgment_summary.json`, and
`paper_scores.json`. Its `runs/` subdirectory holds that attempt's workspace
and local trace. The root `summary.json` contains per-task scores, per-repeat
means, and the mean and sample standard deviation across repeats.

`environment.json` records Python and package versions, the source revision, and
source file hashes. Each task configuration includes hashes of its task, Expert
configuration, activation suite, calibration prompts, and evaluation prompts.
These records contain no credential values. Record the installed agent CLI version
and the exact model versions served by your agent and judge providers as well.

Recompute the full experiment summary with:

```bash
python scripts/run_benchmark.py --config configs/benchmark.json \
  --output-dir outputs/benchmark --score-only
```

For the single-task command above, use the same selection:

```bash
python scripts/run_benchmark.py --config configs/benchmark.json \
  --task-id gemma_cat_001 --replicates 1 --output-dir outputs/cat --score-only
```

The score-only command reads the raw local measurements again. It verifies
configuration and input content consistency, task and run identities, the local eligibility audit,
the selected feature, and the judge count required by the evaluation set. It writes a new summary using
the current score formulas. No GPU or API call is required.

The summary is replaced atomically after the new scores have been validated.
An incomplete run leaves any existing summary intact.

With one repeat, sample standard deviation is undefined and is stored as
`null`. With multiple repeats, it is calculated over the repeat means.

## Common setup problems

If a weight download returns 401 or 403, check Gemma access approval and Hugging
Face authentication. If the Agent executable is missing or rejects an option,
check `agent.cli`, PATH, and harness compatibility.

If the probe exits or runs out of memory, check CUDA, the local weight paths,
and GPU availability. If judgment fails, check the Azure endpoint, API version,
key, and deployment name. A judge summary with missing or failed ratings cannot
be scored as a completed task.

A full experiment includes model inference and external model services, so new
runs can differ across model versions and service updates. Record the versions
and configurations used for your comparisons.
