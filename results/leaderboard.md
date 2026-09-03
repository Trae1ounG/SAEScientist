# SAE-Bench leaderboard

Primary ordering: macro GT-normalized activation. Exact and steering outcomes are reported separately.

| Configuration | Coverage | GT activation | Exact | Causal | Usable | Median time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cursor/kimi-k3-high | 20/20 | 0.794 | 0.350 | 0.150 | 0.000 | 5.0 min |
| cursor/cursor-grok-4.6-high | 20/20 | 0.786 | 0.350 | 0.150 | 0.000 | 9.8 min |
| cursor/claude-sonnet-5-thinking-high | 20/20 | 0.783 | 0.300 | 0.100 | 0.000 | 8.2 min |
| cursor/claude-opus-4-8-thinking-high | 20/20 | 0.776 | 0.350 | 0.100 | 0.000 | 6.5 min |
| codex/gpt-5.6-sol (high) | 20/20 | 0.752 | 0.350 | 0.150 | 0.000 | 4.1 min |
| cursor/gpt-5.5-high | 20/20 | 0.697 | 0.300 | 0.100 | 0.000 | 7.6 min |
| cursor/glm-5.2-high | 20/20 | 0.692 | 0.200 | 0.150 | 0.000 | 5.7 min |
| codex/gpt-5.6-luna (high) | 20/20 | 0.645 | 0.150 | 0.050 | 0.000 | 4.0 min |

The public JSON retains submitted and expert feature IDs plus metric-bearing run rows, while omitting hidden prompts, raw traces, evaluator payloads, and infrastructure metadata.
