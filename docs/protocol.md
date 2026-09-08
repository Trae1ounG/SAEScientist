# Reproducing the paper protocol

[Back to README](../README.md)

Run `scripts/run_benchmark.py` to discover and evaluate features with the current
paper scores. It resolves each task's dictionary, Expert feature, evaluation texts,
and generation settings from the benchmark index. The table below identifies
the settings used by this entry point.

| Component | Setting | Implementation |
|---|---|---|
| Tasks | 20 tasks, 17 concepts | `data/benchmark.json` |
| Model | Gemma-2-9B-IT, bfloat16 | `src/sae_scientist/probe.py`, `scripts/evaluate_gemma_feature.py` |
| SAE | 131,072 features, residual layers 9 and 20 | `data/official_cases/` |
| SAE revision | `e86af97a5b6fbbccca28ab654f2fda1b0768f770` | `data/official_cases/` |
| Agent input | Concept description and SAE specification | `tasks/`, `scripts/run_agent.py` |
| Probe | Up to 64 texts per request, 64 returned features by default | `src/sae_scientist/probe.py` |
| Text activation | Mean of the top three activations over nonspecial tokens | `src/sae_scientist/probe.py` |
| Dictionary rank | One plus the number of features with greater activation; inactive features receive the worst rank | `src/sae_scientist/probe.py` |
| Activation evaluation | Positive, hard-negative, and neutral texts; AUROC uses both control groups | `src/sae_scientist/scoring.py` |
| Calibration | Five prompts per task; task-specific Expert strength and candidate search | `data/eval/`, `src/sae_scientist/cli.py` |
| Generation | Greedy decoding; intervention at all token positions | `scripts/evaluate_gemma_feature.py` |
| Generation length | 192 new tokens for Cat, 128 for French/German/Portuguese/Spanish, 64 for the other tasks | `data/eval/` |
| Steering evaluation | 20 prompts, each with baseline, feature, and random-direction responses | `scripts/evaluate_gemma_feature.py` |
| Random control | Gaussian direction with the selected decoder vector's norm, seed 0 | `src/sae_scientist/steering.py` |
| Judge | 4o through Azure OpenAI, temperature 0, two passes, randomized condition order | `scripts/judge_feature_steering.py` |
| Judge rubric | Scores from 0 to 4; target relevance determines Steering | `src/sae_scientist/suites.py` |
| Task scores | Rank, Activation, Steering, and their mean | `src/sae_scientist/paper_scores.py` |
| Aggregation | Equal task means within each repeat, then mean and sample standard deviation over three repeats | `scripts/run_benchmark.py` |

## Calibration and scoring

The original calibration and evaluation prompts are retained, including the
single shared prompt in each of the four language tasks described in
[Dataset](dataset.md).

Candidate strengths are 0.5, 0.75, 1.0, 1.25, and 1.5 times the task's Expert
strength. If none reaches a 90% nondegenerate rate on calibration prompts, the
evaluator tries multipliers 0.0625, 0.125, 0.25, and 0.375. Passing strengths are
ordered by target success rate, then mean cue score, then smaller strength.
The cue score is 0, 0.5, or 1 for zero, one, or at least two matched cues.
The evaluator stops with an error when no tested strength passes calibration.

When the agent submits the Expert feature, calibration uses only the recorded
Expert strength. The selected strength is applied to both the feature and random
control. [Evaluation](evaluation.md) gives the final score formulas. Calibration
cue scores and legacy diagnostic fields such as `quality` and `gt_normalized`
do not enter those formulas.

## Run settings

The task files and SAE revision are fixed. The example download command does not
pin the base-model revision. Record the resolved Hugging Face revision and use
that revision for subsequent downloads. Agent CLI behavior and hosted agent/judge
models can also change. Record their versions, reasoning settings, and time budget
when reporting a reproduction. The runner's default agent budget is 60 minutes
per task and can be changed in the configuration.

Each new experiment records the installed Python packages and source hashes in
`environment.json`. Input hashes are saved with each task configuration and
checked when recomputing the experiment summary.
