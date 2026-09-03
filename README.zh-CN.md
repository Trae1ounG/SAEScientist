# SAE-Bench

[English](README.md) · [交互式研究博客](https://trae1oung.github.io/SAE-Bench/) · [机器可读结果](results/leaderboard.json)

**SAE-Bench 评估 AI Agent 能否独立完成一次稀疏自编码器 Feature 的研究与定位。**

Agent 只获得一个英文语义目标和受限的 activation probe API。它需要自主编写诊断样例、比较 Feature 激活、修正假设，并最终提交一个 Feature ID。隐藏评测器随后检验该候选能否复现 Expert feature 的自然激活 pattern 与因果 steering 行为。

当前快照使用 Google 官方 Gemma Scope SAE 与 `google/gemma-2-9b-it`，包含 20 个 Expert feature/layer 任务、8 个 Agent 配置和 160 次通过 trace audit 的 discovery episode。Agent 不可联网，也不能读取 benchmark 代码、隐藏 case、Expert ID 或公开 feature label。

## 架构

```text
语义目标
   │
   ▼
隔离 Agent ── 编写 positive / hard-negative / neutral probes
   │
   ├── 查询：文本 + 候选 Feature IDs
   └── 返回：activations + ranks
   │
   ▼
提交一个 Feature ID
   │
   ▼
冻结隐藏评测器
   ├── Expert ID 精确恢复
   ├── held-out activation 一致性
   └── baseline / feature / 等范数随机方向 steering
                                      │
                                      ▼
                              盲化 GPT-4o Judge
```

Benchmark 分开汇报四项结果：

- **Exact**：提交 ID 是否等于冻结 Expert ID。
- **Activation**：候选对 Expert held-out rank、AUROC、激活对比度与逐 case pattern 的归一化恢复程度。
- **Causal**：Feature steering 是否比两个 control 更强地诱导目标行为。
- **Usable**：目标行为出现时，是否仍然保留用户的原始任务。

这四项不会被压缩成一个掩盖差异的总分。

## 当前结论

| 排名 | Agent 配置 | Activation | Exact | GPT-4o 目标分 | Causal | Usable |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Kimi K3 High · Cursor | 0.794 | 35% | 0.281 | 15% | 0% |
| 2 | Grok 4.6 High · Cursor | 0.786 | 35% | 0.303 | 15% | 0% |
| 3 | Claude Sonnet 5 High · Cursor | 0.783 | 30% | 0.277 | 10% | 0% |
| 4 | Claude Opus 4.8 High · Cursor | 0.776 | 35% | 0.226 | 10% | 0% |
| 5 | GPT-5.6 Sol High · Codex | 0.752 | 35% | 0.298 | 15% | 0% |
| 6 | GPT-5.5 High · Cursor | 0.697 | 30% | 0.217 | 10% | 0% |
| 7 | GLM-5.2 High · Cursor | 0.692 | 20% | 0.242 | 15% | 0% |
| 8 | GPT-5.6 Luna High · Codex | 0.645 | 15% | 0.170 | 5% | 0% |

当前单次运行快照支持三个观察：

1. Agent 经常能找到语义相关 Feature，但未必恢复准确的 Expert ID。
2. Activation 相似性有用，但不足以证明因果等价：在 GPT‑4o 重新打标后，全部运行的 activation–steering Spearman 相关为 0.690；只看非 exact 候选时降至 0.378。
3. 很强的 steering 可能直接抹掉用户任务，因此 causal target induction 和 usable control 必须分别评估。

GPT‑4o 将 19/160 条运行判为 causal，没有运行通过 usable gate；113 个非 exact 候选中只有 1 个通过 causal gate。20 个 Expert direction 中也有 8 个未达到冻结的 70% target-success 阈值。这是 Judge 敏感性结果，最终论文需要先重新校准 Expert steering，或把因果结论限制在重新准入的子集上。这些仍是快照结论，不是最终方差估计。

## Steering 评测

对提交的 Feature `k`，评测器在 SAE 所在层执行：

```text
h'_t = h_t + alpha × W_dec[:, k]
```

`alpha` 在 5 条 calibration prompts 上选择；正式评测使用另外 20 条 held-out instructions，以及 baseline、提交 Feature、等范数随机 decoder direction 三个条件。三个输出在交给 GPT‑4o 前会被打乱并匿名化。Judge 在 temperature 0 下独立打分两次，评价 target relevance、task preservation 与 degeneration。

完整 Judge Prompt、标签定义、聚合公式、逐 Feature 激活分布和 steering 前后案例均放在[研究博客](https://trae1oung.github.io/SAE-Bench/)正文中。

## 复现公开结果

```bash
git clone https://github.com/Trae1ounG/SAE-Bench.git
cd SAE-Bench
git checkout website
npm ci
npm test
npm run dev
```

`npm test` 会构建 GitHub Pages 静态站点，并执行数据与正文一致性检查；`npm run dev` 会启动本地交互博客。提交的结果快照可直接读取：

```bash
jq '.benchmark, .configurations' results/leaderboard.json
```

公开仓库可以复现当前分析与网站。隐藏 prompts 与脱敏后的原始 Agent traces 保存在私有研究仓库；credentials 不落盘，两边仓库都不保留机器相关的绝对路径。

## 引用

使用本 Benchmark 或文中结果时，请引用本仓库及所访问的 commit。
