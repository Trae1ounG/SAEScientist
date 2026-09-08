# Dataset and model weights

[Back to README](../README.md)

## Get the dataset

Clone this repository. The task descriptions and evaluation data are ordinary JSON
files, so no separate dataset download or API key is required:

```bash
python scripts/inspect_dataset.py
```

The inspector resolves all 20 tasks, their evaluation suites and Expert records.
It prints the layer, feature count, activation text counts, and steering prompt
counts for each task. It fails if a required file is missing.

## How the dataset is used

The runner starts from `data/benchmark.json`. Each entry points to an Agent task
file, an evaluation suite, and an Expert configuration. The task file under
`tasks/` contains the concept description and exact SAE specification.

The evaluation suite supplies labeled activation texts and steering prompts.
The Expert configuration supplies its feature metadata and intervention protocol.
Files under `data/official_cases/` identify the model, dictionary, feature ID,
and checkpoint revision. Expert configurations contain no measured experiment results.

A **task** pairs a concept with a model layer. The 20 tasks cover 17 concepts at
layers 9 and 20. Earnings reports, portfolio allocation, and tax filing each occur
at both layers.

An **evaluation suite** contains activation texts and steering prompts. Activation
texts carry `positive`, `hard_negative`, or `neutral` labels. Most tasks contain
8 positive, 8 hard-negative, and 4 neutral texts. Cat contains 12, 12, and 8.
Each task has 5 calibration prompts and 20 evaluation prompts for steering.
The shared steering prompt file retains its original name,
`data/steering/french_105738_v1.json`, but serves multiple concepts.

French, German, Portuguese, and Spanish each have one identical prompt appearing
in both calibration and evaluation. This overlap is present in the original task
files and is retained for reproduction. The inspector reports it as
`shared_calibration_evaluation_prompts`. The other tasks have no exact prompt
overlap between these sets.

Some suites store multiple concepts in one file. Use the benchmark index's
`concept_id` to select the correct concept. The loader resolves shared steering
prompt paths relative to the suite file. Do not concatenate every JSON file under
`data/`: that would mix configuration files with examples and duplicate suites.

For example, after installing the package:

```python
import json
from pathlib import Path
from sae_scientist.suites import load_suite, steering_sets

root = Path(".")
index = json.loads((root / "data/benchmark.json").read_text())
entry = index["tasks"][0]
task = json.loads((root / entry["task"]).read_text())
suite_path = root / entry["suite"]
suite = load_suite(suite_path, entry.get("concept_id"))
calibration, evaluation = steering_sets(suite, suite_path)
print(task["task_id"], len(suite["activation_cases"]), len(evaluation))
```

Public availability does not make the evaluation files Agent inputs. The runner
creates a separate workspace containing the task and instructions. Keep the
benchmark index, evaluation suites, Expert records, and checkpoint metadata outside
that workspace. Use the provided harness restrictions and audit your local runs.

## Where Expert features come from

Expert records identify fixed features in the Google DeepMind Gemma Scope release.
Their metadata includes the model, layer, feature ID, checkpoint path, and
resolved SAE revision. A Neuronpedia URL is retained where recorded in the source
metadata. These features provide a comparison baseline.

The Cat task originates from a Neuronpedia steering preset. Other tasks use concept suites under `data/discovery/`. The evaluation texts were
retained from benchmark construction. Reference files contain only the frozen
configuration needed to run new measurements.

Sources:
[Gemma Scope](https://huggingface.co/google/gemma-scope-9b-it-res),
[Neuronpedia Cat record](https://www.neuronpedia.org/gemma-2-9b-it/9-gemmascope-res-131k/62610).

## Download model and SAE weights

1. Request/accept access to [Gemma-2-9B-IT](https://huggingface.co/google/gemma-2-9b-it)
   using your Hugging Face account.
2. Run `hf auth login` in the evaluator environment.
3. Download the model and the two specified SAE dictionaries:

```bash
hf download google/gemma-2-9b-it --local-dir checkpoints/gemma-2-9b-it
hf download google/gemma-scope-9b-it-res \
  layer_9/width_131k/average_l0_121/params.npz \
  layer_20/width_131k/average_l0_81/params.npz \
  --revision e86af97a5b6fbbccca28ab654f2fda1b0768f770 \
  --local-dir checkpoints/gemma-scope-9b-it-res
```

| Layer | Dictionary path | Hook |
|---|---|---|
| 9 | `layer_9/width_131k/average_l0_121/params.npz` | `blocks.9.hook_resid_post` |
| 20 | `layer_20/width_131k/average_l0_81/params.npz` | `blocks.20.hook_resid_post` |

Both dictionaries contain 131,072 features. A feature ID is meaningful only with
its model, layer, width, and checkpoint revision. Record the downloaded base-model
revision as well. The example configuration pins the SAE revision but does not
pin a base-model revision. Model and SAE use remain subject to their publishers'
terms.

Weight files and original Agent traces are not included in this repository.
