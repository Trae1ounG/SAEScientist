# SAE feature discovery task

Read `task.json` and identify the single SAE feature that best matches its English description.

Network access and web search are disabled. Do not use public feature labels, explanations, dashboards, APIs, cached feature metadata, or prior knowledge of feature IDs. In particular, Neuronpedia and Hugging Face feature metadata are out of scope. The full task JSON is included below. Use only the benchmark-provided `probe_sae` tool to obtain measured activations. Construct contrasting positive, negative, and hard-negative texts, use top-feature intersections to discover candidates, then compare candidate activations and ranks across your probes. Do not train a new SAE. The target belongs to the exact official model, SAE release, and hook named in the task JSON; results from another model, layer, width, or SAE release are invalid. Write your report in English.

Do not delegate to a subagent or use any task, background-agent, web, browser, or non-benchmark MCP tool. Do not work around a failed file tool by invoking another tool. The trace auditor rejects the run if any such tool is called.

Before finishing, write `submission.json` at the workspace root using exactly the schema requested in `task.json`. Keep any analysis or evidence in `report.md`; do not add it to `submission.json`. If a direct file write fails or is unavailable, stop using tools and end the final response with exactly the same one-field JSON object; the runner records this fallback explicitly.

