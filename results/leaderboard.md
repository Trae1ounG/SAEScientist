# SAE-Bench leaderboard

Primary ordering: the equal-weight mean of Expert-centered Rank, Activation, and Steering scores. Total is the sum of 20 task scores. The Expert feature is the 20/20 reference point, not a ceiling or a competing model.

| Model | Coverage | Overall | Total / 20 | Rank | Activation | Steering | Exact | Causal | Usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert feature baseline | 20/20 | 1.000 | 20.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.600 | 0.000 |
| Kimi K3 High | 20/20 | 0.718 | 14.370 | 0.748 | 0.920 | 0.488 | 0.350 | 0.150 | 0.000 |
| Grok 4.6 High | 20/20 | 0.699 | 13.978 | 0.648 | 0.920 | 0.529 | 0.350 | 0.150 | 0.000 |
| Claude Sonnet 5 High | 20/20 | 0.696 | 13.916 | 0.710 | 0.917 | 0.461 | 0.300 | 0.100 | 0.000 |
| Claude Opus 4.8 High | 20/20 | 0.680 | 13.598 | 0.674 | 0.907 | 0.458 | 0.350 | 0.100 | 0.000 |
| GPT-5.6 Sol High | 20/20 | 0.645 | 12.895 | 0.506 | 0.912 | 0.517 | 0.350 | 0.150 | 0.000 |
| GPT-5.5 High | 20/20 | 0.596 | 11.922 | 0.497 | 0.862 | 0.429 | 0.300 | 0.100 | 0.000 |
| GLM-5.2 High | 20/20 | 0.562 | 11.248 | 0.462 | 0.857 | 0.368 | 0.200 | 0.150 | 0.000 |
| GPT-5.6 Luna High | 20/20 | 0.526 | 10.516 | 0.398 | 0.849 | 0.330 | 0.150 | 0.050 | 0.000 |

The public JSON retains submitted and expert feature IDs plus metric-bearing run rows, while omitting hidden prompts, raw traces, evaluator payloads, and infrastructure metadata.
