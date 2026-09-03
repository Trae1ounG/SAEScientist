# SAEScientist-Bench leaderboard

Primary ordering: the equal-weight mean of Expert-centered Rank, Activation, and Steering scores. The Expert feature is the 1.0 reference point, not a ceiling or a competing model.

| Model | Runs | Overall | Rank | Activation | Steering | Exact | Causal | Usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert feature baseline | 20 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.600 | 0.000 |
| Claude Opus 5 High | 60 | 0.738 ± 0.007 | 0.753 | 0.935 | 0.526 | 0.417 | 0.183 | 0.000 |
| Claude Sonnet 5 High | 60 | 0.727 ± 0.024 | 0.734 | 0.939 | 0.508 | 0.400 | 0.167 | 0.000 |
| Kimi K3 High | 60 | 0.724 ± 0.021 | 0.743 | 0.927 | 0.503 | 0.317 | 0.150 | 0.000 |
| Claude Opus 4.8 High | 60 | 0.705 ± 0.046 | 0.702 | 0.929 | 0.482 | 0.350 | 0.133 | 0.000 |
| Grok 4.6 High | 60 | 0.701 ± 0.005 | 0.672 | 0.919 | 0.513 | 0.333 | 0.167 | 0.000 |
| Gemini 3.8 Flash High | 60 | 0.676 ± 0.012 | 0.695 | 0.898 | 0.436 | 0.333 | 0.100 | 0.000 |
| GPT-5.6 Sol High | 60 | 0.643 ± 0.006 | 0.516 | 0.902 | 0.511 | 0.350 | 0.167 | 0.017 |
| GPT-5.5 High | 60 | 0.611 ± 0.024 | 0.556 | 0.876 | 0.401 | 0.250 | 0.117 | 0.000 |
| GLM-5.2 High | 60 | 0.586 ± 0.028 | 0.477 | 0.860 | 0.420 | 0.267 | 0.133 | 0.000 |
| GPT-5.6 Luna High | 60 | 0.565 ± 0.063 | 0.469 | 0.852 | 0.373 | 0.200 | 0.100 | 0.000 |

The public JSON retains submitted and expert feature IDs plus metric-bearing run rows, while omitting hidden prompts, raw traces, evaluator payloads, and infrastructure metadata.
