import { ReactNode, useMemo, useState } from "react";
import analysisResults from "../results/analysis.json";
import leaderboard from "../results/leaderboard.json";
import replicateResults from "../results/replicates.json";

type Language = "en" | "zh";
type PointFilter = "all" | "exact" | "alternative";

type Configuration = {
  configuration: string;
  harness: string;
  model: string;
  reasoning_effort: string | null;
  benchmark_tasks: number;
  completed_tasks: number;
  mean_feature_discovery_score: number;
  total_feature_discovery_score: number;
  maximum_feature_discovery_score: number;
  mean_rank_score: number;
  mean_activation_score: number;
  mean_steering_score: number;
  mean_overall_score: number;
  total_overall_score: number;
  maximum_overall_score: number;
  macro_gt_normalized_activation: number;
  exact_matches: number;
  exact_match_rate: number;
  causal_steering_rate: number;
  usable_steering_rate: number;
  median_elapsed_seconds: number;
};

type Run = {
  task_id: string;
  target: string;
  configuration: string;
  selected_feature_id: number;
  expert_feature_id: number;
  exact_match: boolean;
  feature_discovery_score: number;
  rank_score: number;
  activation_score: number;
  steering_score: number;
  overall_score: number;
  gt_normalized_activation: number;
  positive_mean_rank: number;
  activation_auroc: number;
  expert_spearman: number;
  steering_effect: number;
  steering_pattern_correlation: number | null;
  pe_target_relevance: number;
  pe_task_preservation: number;
  causal_stable: boolean;
  usable_steering: boolean;
  elapsed_seconds: number;
};

type RawRun = Omit<Run, "task_id" | "target" | "configuration" | "feature_discovery_score"> & {
  task: string;
  concept_id: string;
  model: string;
  harness: string;
  reasoning_effort: string | null;
  feature_discovery_score?: number;
};

type ReplicateMetric = { mean: number; std: number };
type ReplicateConfiguration = Pick<Configuration, "configuration" | "harness" | "model" | "reasoning_effort"> & {
  replicates: number;
  benchmark_tasks: number;
  metrics: Record<string, ReplicateMetric>;
};

function ResearchFigure({
  number,
  title,
  caption,
  children,
  wide = false,
}: {
  number: number;
  title: string;
  caption: string;
  children: ReactNode;
  wide?: boolean;
}) {
  return (
    <figure className={`research-figure${wide ? " wide" : ""}`}>
      <div className="figure-frame">{children}</div>
      <figcaption><b>Figure {number}.</b> <strong>{title}</strong> {caption}</figcaption>
    </figure>
  );
}

const configurations = leaderboard.configurations as Configuration[];
const aggregateConfigurations = replicateResults.configurations as ReplicateConfiguration[];
const aggregateAnalysis = replicateResults.analysis;
const diagnostics = analysisResults.diagnostics;
const bootstrap = analysisResults.bootstrap;
const runs = (leaderboard.runs as RawRun[]).map((run) => ({
  ...run,
  task_id: run.task.split("/").at(-1)?.replace(/\.json$/, "") ?? run.task,
  target: run.concept_id,
  configuration: configurations.find((row) =>
    row.harness === run.harness
    && row.model === run.model
    && row.reasoning_effort === run.reasoning_effort
  )?.configuration ?? `${run.harness}/${run.model}`,
  feature_discovery_score: run.feature_discovery_score ?? run.gt_normalized_activation,
})) as Run[];

const modelNames: Record<string, string> = {
  "kimi-k3-high": "Kimi K3 High",
  "cursor-grok-4.6-high": "Grok 4.6 High",
  "claude-sonnet-5-thinking-high": "Claude Sonnet 5 High",
  "claude-opus-4-8-thinking-high": "Claude Opus 4.8 High",
  "claude-opus-5-thinking-high": "Claude Opus 5 High",
  "gpt-5.6-sol": "GPT-5.6 Sol High",
  "gpt-5.5-high": "GPT-5.5 High",
  "glm-5.2-high": "GLM-5.2 High",
  "gpt-5.6-luna": "GPT-5.6 Luna High",
};

type ExpertTask = {
  target: [string, string];
  family: [string, string];
  layer: number;
  feature: number;
  positive: number;
  hardNegative: number;
  neutral: number;
  auroc: number;
  steering: number;
  usable: number;
};

const expertTasks: ExpertTask[] = [
  { target: ["猫相关文本", "Cat-related text"], family: ["主题", "Topic"], layer: 9, feature: 62610, positive: 41.93, hardNegative: 5.92, neutral: 0, auroc: 0.996, steering: 0.438, usable: 0.475 },
  { target: ["法语文本", "French language"], family: ["语言", "Language"], layer: 9, feature: 105738, positive: 11.39, hardNegative: 2.66, neutral: 0, auroc: 1.000, steering: 0.975, usable: 0.825 },
  { target: ["西班牙语文本", "Spanish language"], family: ["语言", "Language"], layer: 9, feature: 94329, positive: 10.38, hardNegative: 1.39, neutral: 0, auroc: 1.000, steering: 0.631, usable: 0.625 },
  { target: ["葡萄牙语文本", "Portuguese language"], family: ["语言", "Language"], layer: 9, feature: 41424, positive: 18.51, hardNegative: 4.04, neutral: 0.34, auroc: 1.000, steering: 0.850, usable: 0.800 },
  { target: ["德语文本", "German language"], family: ["语言", "Language"], layer: 9, feature: 33987, positive: 11.81, hardNegative: 3.51, neutral: 0.29, auroc: 0.990, steering: 0.763, usable: 0.775 },
  { target: ["财报语言", "Earnings reports"], family: ["商业/金融", "Business/finance"], layer: 9, feature: 131024, positive: 18.73, hardNegative: 2.24, neutral: 0, auroc: 0.979, steering: 0.669, usable: 0.150 },
  { target: ["园艺建议", "Gardening advice"], family: ["领域/文体", "Domain/register"], layer: 9, feature: 93406, positive: 12.21, hardNegative: 2.97, neutral: 0.15, auroc: 0.958, steering: 0.394, usable: 0.025 },
  { target: ["报税语言", "Tax filing"], family: ["商业/金融", "Business/finance"], layer: 9, feature: 18713, positive: 19.07, hardNegative: 1.02, neutral: 0.47, auroc: 0.990, steering: 0.300, usable: 0.075 },
  { target: ["招聘启事", "Job postings"], family: ["商业/金融", "Business/finance"], layer: 9, feature: 91086, positive: 18.38, hardNegative: 4.45, neutral: 0, auroc: 1.000, steering: 0.419, usable: 0.000 },
  { target: ["拉丁语文本", "Latin language"], family: ["语言", "Language"], layer: 9, feature: 7659, positive: 28.83, hardNegative: 1.57, neutral: 0, auroc: 1.000, steering: 0.569, usable: 0.475 },
  { target: ["资产配置", "Portfolio allocation"], family: ["商业/金融", "Business/finance"], layer: 9, feature: 77390, positive: 27.58, hardNegative: 2.33, neutral: 0, auroc: 1.000, steering: 0.450, usable: 0.075 },
  { target: ["土耳其语文本", "Turkish language"], family: ["语言", "Language"], layer: 9, feature: 99383, positive: 17.54, hardNegative: 2.31, neutral: 0.15, auroc: 1.000, steering: 0.669, usable: 0.575 },
  { target: ["临床症状报告", "Clinical symptom reports"], family: ["领域/文体", "Domain/register"], layer: 20, feature: 3927, positive: 21.79, hardNegative: 1.74, neutral: 0, auroc: 1.000, steering: 0.888, usable: 0.175 },
  { target: ["财报语言", "Earnings reports"], family: ["商业/金融", "Business/finance"], layer: 20, feature: 100747, positive: 23.87, hardNegative: 0.60, neutral: 0, auroc: 1.000, steering: 0.769, usable: 0.275 },
  { target: ["资产配置", "Portfolio allocation"], family: ["商业/金融", "Business/finance"], layer: 20, feature: 101617, positive: 39.61, hardNegative: 4.67, neutral: 2.13, auroc: 1.000, steering: 0.744, usable: 0.375 },
  { target: ["药物剂量说明", "Pharmaceutical dosing"], family: ["领域/文体", "Domain/register"], layer: 20, feature: 104583, positive: 24.84, hardNegative: 0.28, neutral: 0, auroc: 1.000, steering: 0.506, usable: 0.025 },
  { target: ["报税语言", "Tax filing"], family: ["商业/金融", "Business/finance"], layer: 20, feature: 78694, positive: 22.69, hardNegative: 1.90, neutral: 0, auroc: 1.000, steering: 0.350, usable: 0.100 },
  { target: ["房地产房源", "Real-estate listings"], family: ["商业/金融", "Business/finance"], layer: 20, feature: 27182, positive: 16.80, hardNegative: 1.85, neutral: 0.42, auroc: 1.000, steering: 0.394, usable: 0.050 },
  { target: ["天气预报", "Weather forecasts"], family: ["领域/文体", "Domain/register"], layer: 20, feature: 44494, positive: 39.17, hardNegative: 2.85, neutral: 0.44, auroc: 1.000, steering: 0.469, usable: 0.125 },
  { target: ["考古发掘报告", "Archaeological excavation"], family: ["领域/文体", "Domain/register"], layer: 20, feature: 7256, positive: 28.90, hardNegative: 5.98, neutral: 0.54, auroc: 0.979, steering: 0.306, usable: 0.200 },
];

const seriesClasses = ["series-a", "series-b", "series-c", "series-d", "series-e", "series-f", "series-g", "series-h", "series-i"];

function displayName(row: Pick<Configuration, "model">): string {
  return modelNames[row.model] ?? row.model;
}

function configurationLabel(key: string): string {
  const row = [...aggregateConfigurations, ...configurations].find((candidate) => candidate.configuration === key);
  return row ? displayName(row) : key;
}

function humanize(value: string): string {
  return value.replace(/^gemma_/, "").replace(/_\d+$/, "").replaceAll("_", " ");
}

function average(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function ranked(values: number[]): number[] {
  const ordered = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const result = Array(values.length).fill(0);
  for (let start = 0; start < ordered.length;) {
    let end = start + 1;
    while (end < ordered.length && ordered[end].value === ordered[start].value) end += 1;
    const rank = (start + end - 1) / 2 + 1;
    for (let index = start; index < end; index += 1) result[ordered[index].index] = rank;
    start = end;
  }
  return result;
}

function correlation(xs: number[], ys: number[]): number {
  if (xs.length !== ys.length || xs.length < 2) return 0;
  const xMean = average(xs);
  const yMean = average(ys);
  let numerator = 0;
  let xSquare = 0;
  let ySquare = 0;
  for (let index = 0; index < xs.length; index += 1) {
    const x = xs[index] - xMean;
    const y = ys[index] - yMean;
    numerator += x * y;
    xSquare += x * x;
    ySquare += y * y;
  }
  return numerator / Math.sqrt(xSquare * ySquare);
}

function spearman(rows: Run[]): number {
  return correlation(
    ranked(rows.map((row) => row.gt_normalized_activation)),
    ranked(rows.map((row) => row.steering_effect)),
  );
}

function percentage(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function MetricTabs<T extends string>({
  value,
  onChange,
  options,
  label,
}: {
  value: T;
  onChange: (value: T) => void;
  options: { key: T; label: string }[];
  label: string;
}) {
  return (
    <div className="metric-tabs" role="group" aria-label={label}>
      {options.map((option) => (
        <button key={option.key} className={value === option.key ? "active" : ""} onClick={() => onChange(option.key)}>
          {option.label}
        </button>
      ))}
    </div>
  );
}

const catActivationGroups = [
  { key: "positive", values: [60.8, 43.599, 32.343, 56.271, 45.428, 40.216, 50.275, 40.987, 28.231, 34.219, 54.306, 16.48] },
  { key: "hard_negative", values: [0, 1.29, 8.524, 16.904, 1.522, 2.695, 5.491, 5.471, 4.243, 7.093, 2.889, 14.97] },
  { key: "neutral", values: [0, 0, 0, 0, 0, 0, 0, 0] },
];

function FeaturePortrait({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const maxActivation = 64;
  return (
    <div className="feature-portrait">
      <header className="feature-identity">
        <div><span>{tx("官方 Feature ID", "Official feature ID")}</span><code>62610</code></div>
        <div><span>{tx("语义解释", "Semantic interpretation")}</span><h3>{tx("猫与猫科身份、行为", "Cats, feline identity, and behavior")}</h3><p>Gemma 2 9B · resid_post · layer 9 · width 131k</p></div>
      </header>
      <section className="activation-profile">
        <div className="portrait-label"><span>01</span><div><strong>{tx("它在什么文本上激活？", "Where does it activate?")}</strong><small>{tx("每个点是一条冻结 case 的 top-3 token mean activation", "Each dot is one frozen case's top-3 token mean activation")}</small></div></div>
        <div className="activation-axis"><span>0</span><span>16</span><span>32</span><span>48</span><span>64</span></div>
        {catActivationGroups.map((group) => {
          const mean = average(group.values);
          const label = group.key === "positive" ? tx("正例", "Positive") : group.key === "hard_negative" ? tx("困难负例", "Hard negative") : tx("中性", "Neutral");
          return <div className={`activation-row ${group.key}`} key={group.key}><div className="activation-name"><strong>{label}</strong><small>mean {mean.toFixed(1)}</small></div><div className="dot-track">{[0, 16, 32, 48, 64].map(tick => <i className="gridline" style={{ left: `${tick / maxActivation * 100}%` }} key={tick} />)}{group.values.map((value, index) => <i className="activation-dot" style={{ left: `${value / maxActivation * 100}%`, top: `${14 + (index % 3) * 13}px` }} title={`${label}: ${value.toFixed(3)}`} key={`${group.key}-${index}`} />)}<b style={{ left: `${mean / maxActivation * 100}%` }} aria-label={`${label} mean ${mean.toFixed(1)}`} /></div></div>;
        })}
        <div className="activation-summary"><span>AUROC <b>0.996</b></span><span>{tx("正例激活率", "Positive active rate")} <b>100%</b></span><span>{tx("困难负例 / 正例", "Hard negative / positive")} <b>0.141</b></span></div>
      </section>
      <section className="steering-profile">
        <div className="portrait-label"><span>02</span><div><strong>{tx("把方向写回模型后发生什么？", "What changes when the direction is written back?")}</strong><small>{tx("同一条 held-out instruction；输出截短展示", "Same held-out instruction; outputs truncated")}</small></div></div>
        <div className="rollout-comparison">
          <div className="rollout baseline"><span>{tx("Steering 前", "Before steering")}</span><p>“Hey [Teammate Name], welcome to the team! We’re so excited to have you on board. We’re a friendly bunch…”</p><small>target relevance · 0 / 4</small></div>
          <div className="steering-arrow" aria-hidden="true"><span>+160 ×</span><code>Wdec[:, 62610]</code><b>→</b></div>
          <div className="rollout steered"><span>{tx("Steering 后", "After steering")}</span><p>“Welcome to the family, little one! … plenty of <mark>sunbeams</mark> to nap in and <mark>yarn balls</mark> to bat around. <mark>Purrfectly</mark> yours.”</p><small>feature activation · 183.8</small></div>
        </div>
        <div className="steering-summary"><div><strong>0.000 → 0.438</strong><span>{tx("held-out target score", "held-out target score")}</span></div><div><strong>65%</strong><span>{tx("目标成功率", "target success rate")}</span></div><div><strong>47.5%</strong><span>{tx("可用输出率", "usable output rate")}</span></div><div><strong>0%</strong><span>matched-random success</span></div></div>
      </section>
    </div>
  );
}

function FeatureAtlas({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const cases = [
    {
      concept: tx("猫相关文本", "Cat-related text"),
      layer: "Gemma 2 9B · layer 9",
      expert: 62610,
      agent: tx("首轮 8/8 exact", "8/8 exact in run 1"),
      activation: [41.93, 5.92, 0, 0.996],
      steering: [0.438, 0.475],
      note: tx("正例覆盖家猫身份与行为；困难负例包括单独出现的 kitty、名字 Kitty 与比喻性 purr。Expert feature 的正例平均 activation 为 41.93，困难负例为 5.92，中性文本为 0；首轮 8 个 Agent 均恢复了同一 ID。", "Positive cases cover domestic-cat identity and behavior. Hard negatives include isolated uses of kitty, the name Kitty, and metaphorical purr. The expert averages 41.93 activation on positives, 5.92 on hard negatives, and zero on neutral text; all eight agents in the first run recovered the same ID."),
    },
    {
      concept: tx("法语文本", "French-language text"),
      layer: "Gemma 2 9B · layer 9",
      expert: 105738,
      agent: tx("0/8 exact · 4 个替代 ID", "0/8 exact · 4 alternative IDs"),
      activation: [11.39, 2.66, 0, 1.0],
      steering: [0.975, 0.825],
      note: tx("正例是在日常、科学、程序和叙事主题中的连贯法语；对照集包含英语中的法国话题、孤立法语词、西班牙语与意大利语。Expert 的 activation 分离清晰。Agent 提交了 108861、43576、78422 与 130037 四个替代 ID；这些替代项在隐藏 steering 上均为 0。", "Positive cases contain coherent French across everyday, scientific, procedural, and narrative domains. Controls include English discussion of France, isolated French words, Spanish, and Italian. The expert separates the activation sets cleanly. Agents submitted four alternatives—108861, 43576, 78422, and 130037—and all produced zero hidden steering effect."),
    },
    {
      concept: tx("报税语言", "Tax-filing language"),
      layer: "Gemma 2 9B · layer 9",
      expert: 18713,
      agent: tx("首轮 8/8 提交 64827", "8/8 submitted 64827 in run 1"),
      activation: [19.07, 1.02, 0.47, 0.990],
      steering: [0.300, 0.075],
      note: tx("正例包含报税指令、应税收入、抵扣、税收抵免与表格；困难负例仅包含一般性的 tax 提及。首轮 8 个 Agent 均提交了 64827，Expert ID 为 18713。替代项的 Feature Discovery Score 为 0.934，steering effect 为 0.613，但未通过 usable gate，构成一致的替代方向。", "Positive cases contain filing instructions, taxable income, deductions, credits, and forms; hard negatives contain only general tax references. All eight agents in the first run submitted 64827, while the expert ID is 18713. The alternative reaches a Feature Discovery Score of 0.934 and a steering effect of 0.613, but does not pass the usable gate, forming a consistent alternative direction."),
    },
    {
      concept: tx("临床症状报告", "Clinical symptom reports"),
      layer: "Gemma 2 9B · layer 20",
      expert: 3927,
      agent: tx("0/8 exact · 5/8 选 113440", "0/8 exact · 5/8 chose 113440"),
      activation: [21.79, 1.74, 0, 1.0],
      steering: [0.894, 0.0],
      note: tx("正例描述患者症状、起病时间、严重程度与病史；一般医学词汇和医院运营文本构成对照。Expert 的 activation AUROC 为 1.0，target induction 较高，usable rate 为 0，表明干预破坏了原任务。Agent 最常提交 113440；另一个候选 53882 的 Feature Discovery Score 为 0.916，steering effect 为 0.769。", "Positive cases describe patient symptoms, onset, severity, and clinical history; generic medical vocabulary and hospital operations form the controls. The expert has activation AUROC 1.0 and strong target induction, with a zero usable rate because the intervention disrupts the requested task. Agents most often submitted 113440; candidate 53882 reached a Feature Discovery Score of 0.916 and a steering effect of 0.769."),
    },
    {
      concept: tx("天气预报", "Weather forecasts"),
      layer: "Gemma 2 9B · layer 20",
      expert: 44494,
      agent: tx("6/8 exact", "6/8 exact"),
      activation: [39.17, 2.85, 0.44, 1.0],
      steering: [0.469, 0.125],
      note: tx("描述强调未来的气温、降水、风、云量与时间，排除气候分析、过去风暴损害或只谈雷达。六个 Agent 找到 Expert 44494；两个替代项是 85606 与 127623，其中 85606 的 steering effect 甚至达到 0.988，说明更强 steering 并不自动意味着与 Expert 是同一 feature。", "The description emphasizes predicted temperature, precipitation, wind, clouds, and timing, excluding climate analysis, past storm damage, or radar without prediction. Six agents recovered expert 44494; the alternatives were 85606 and 127623. Feature 85606 even reached 0.988 steering effect, showing that stronger steering does not automatically imply the same feature."),
    },
  ];
  return (
    <div className="feature-atlas">{cases.map((item, index) => <div className="feature-case" key={item.expert}>
      <span className="case-number">CASE {index + 1}</span>
      <div className="feature-case-copy"><h3>{item.concept} <code>#{item.expert}</code></h3><p className="case-layer">{item.layer} · {item.agent}</p><p>{item.note}</p></div>
      <dl className="case-metrics"><div><dt>{tx("正例均值", "Positive mean")}</dt><dd>{item.activation[0].toFixed(2)}</dd></div><div><dt>{tx("困难负例", "Hard negative")}</dt><dd>{item.activation[1].toFixed(2)}</dd></div><div><dt>{tx("中性", "Neutral")}</dt><dd>{item.activation[2].toFixed(2)}</dd></div><div><dt>AUROC</dt><dd>{item.activation[3].toFixed(3)}</dd></div><div><dt>{tx("Expert steering", "Expert steering")}</dt><dd>{item.steering[0].toFixed(3)}</dd></div><div><dt>Usable</dt><dd>{percentage(item.steering[1])}</dd></div></dl>
    </div>)}
    </div>
  );
}

function ExpertTaskTable({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  return (
    <div className="wide-table-block">
      <div className="table-scroll">
        <table className="formal-table expert-task-table">
          <thead><tr><th>#</th><th>{tx("目标语义", "Target interpretation")}</th><th>{tx("类别", "Family")}</th><th>{tx("层", "Layer")}</th><th>Expert ID</th><th>{tx("正例", "Positive")}</th><th>{tx("困难负例", "Hard neg.")}</th><th>{tx("中性", "Neutral")}</th><th>AUROC</th><th>{tx("Steering 效果", "Steering effect")}</th><th>Usable</th></tr></thead>
          <tbody>{expertTasks.map((task, index) => (
            <tr key={`${task.layer}-${task.feature}`} className={index === 12 ? "layer-break" : ""}>
              <td>{index + 1}</td><td><strong>{task.target[language === "zh" ? 0 : 1]}</strong></td><td>{task.family[language === "zh" ? 0 : 1]}</td><td>L{task.layer}</td><td><code>{task.feature}</code></td><td>{task.positive.toFixed(2)}</td><td>{task.hardNegative.toFixed(2)}</td><td>{task.neutral.toFixed(2)}</td><td>{task.auroc.toFixed(3)}</td><td>{task.steering.toFixed(3)}</td><td>{percentage(task.usable)}</td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <p className="table-caption"><b>{tx("表 1.", "Table 1.")}</b> {tx("20 个冻结 expert tasks 的完整分布。Activation 使用每条 case 的 top-3 token mean，再分别汇总 positive、hard-negative 与 neutral；steering effect 和 usable rate 来自 20 条 held-out instructions。", "Complete distribution of the 20 frozen expert tasks. Activation uses the top-three-token mean for each case and is summarized by positive, hard-negative, and neutral groups; steering effect and usable rate are measured on 20 held-out instructions.")}</p>
    </div>
  );
}

function WeatherAgentComparison({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const weatherRuns = runs.filter((run) => run.task_id === "gemma_weather_forecasts_022");
  return (
    <div className="agent-steering-comparison">
      <div className="table-scroll">
        <table className="formal-table agent-steering-table">
          <thead><tr><th>{tx("模型", "Model")}</th><th>Feature ID</th><th>Exact</th><th>{tx("发现分数", "Discovery Score")}</th><th>{tx("Steering 效果", "Steering effect")}</th><th>{tx("目标相关度", "Target relevance")}</th><th>{tx("任务保留", "Task preservation")}</th><th>Causal</th></tr></thead>
          <tbody>{weatherRuns.map((run) => <tr key={run.configuration}><td><strong>{configurationLabel(run.configuration)}</strong></td><td><code>{run.selected_feature_id}</code></td><td>{run.exact_match ? "✓" : "—"}</td><td>{run.feature_discovery_score.toFixed(3)}</td><td>{run.steering_effect.toFixed(3)}</td><td>{run.pe_target_relevance.toFixed(2)} / 4</td><td>{run.pe_task_preservation.toFixed(2)} / 4</td><td>{run.causal_stable ? "✓" : "—"}</td></tr>)}</tbody>
        </table>
      </div>
      <div className="steering-excerpts">
        <div className="shared-baseline"><span>{tx("同一条隐藏 instruction · steering 前", "Same hidden instruction · before steering")}</span><blockquote>“Project: [Project Name] · Status: In Progress · Key Accomplishments … Current Focus … Risk …”</blockquote></div>
        <div><span>Expert <code>#44494</code> · α 240</span><blockquote>“Project Status Update: <mark>Today: Expect scattered showers with a high of 75 degrees.</mark> Tomorrow: The chance of rain will continue…”</blockquote><small>{tx("6 个 exact Agent 复用该冻结 expert reference", "Reused by the six exact-match agents")}</small></div>
        <div><span>Sonnet 5 <code>#85606</code> · α 300</span><blockquote>“<mark>Mostly sunny skies with a chance of clouds.</mark> Highs in the afternoon with a chance of showers developing later…”</blockquote><small>GPT‑4o relevance 3.33 / 4 · preservation 0.38 / 4</small></div>
        <div><span>GLM-5.2 <code>#127623</code> · α 240</span><blockquote>“I’m currently working on a small project. <mark>The chance of rain is around 20% throughout the day</mark>…”</blockquote><small>GPT‑4o relevance 0.75 / 4 · preservation 1.18 / 4</small></div>
      </div>
      <p className="figure-note">{tx("输出均来自同一条 held-out instruction 并做截短；目标相关度与任务保留分数是该 Feature ID 在全部 20 条 held-out instructions 上的均值。#85606 更强地注入天气语言，但几乎抹掉原任务，因此 causal gate 和 usable gate 均未通过。", "All excerpts come from the same held-out instruction and are truncated; target relevance and task preservation are averaged over all 20 held-out instructions for that feature ID. Feature #85606 injects weather language more strongly, but nearly erases the original task, so it fails both the causal and usable gates.")}</p>
    </div>
  );
}

function PeJudgeExample({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const rows = [
    { condition: tx("未干预 baseline", "Unsteered baseline"), relevance: 0, preservation: 4, degenerate: false },
    { condition: tx("Expert feature #62610", "Expert feature #62610"), relevance: 3, preservation: 3, degenerate: false },
    { condition: tx("等范数随机方向", "Norm-matched random"), relevance: 0, preservation: 4, degenerate: false },
  ];
  return (
    <div className="pe-judge-example">
      <div className="table-scroll">
        <table className="formal-table">
          <thead><tr><th>{tx("条件（揭盲后）", "Condition after unblinding")}</th><th>target_relevance</th><th>task_preservation</th><th>degenerate</th></tr></thead>
          <tbody>{rows.map((row) => <tr key={row.condition}><td><strong>{row.condition}</strong></td><td>{row.relevance} / 4</td><td>{row.preservation} / 4</td><td>{row.degenerate ? "true" : "false"}</td></tr>)}</tbody>
        </table>
      </div>
      <p className="table-caption"><b>{tx("一条真实打标记录。", "One real judgment record.")}</b> {tx("任务是写一段欢迎新同事的短消息。Judge 实际看到的是随机标为 A/B/C 的三个输出，不会看到条件名、Feature ID 或 α；上表为评测结束后揭盲的结果。", "The task asks for a short welcome message for a new teammate. The judge sees three outputs randomly labeled A/B/C, without condition names, feature IDs, or alpha; the table shows the record after unblinding.")}</p>
      <details className="judge-prompt" open>
        <summary>{tx("完整 Judge Prompt（猫任务实例）", "Complete judge prompt (cat-task instance)")}</summary>
        <p><b>System</b></p>
        <pre><code>{`You are evaluating activation steering for one SAE feature.
The intended concept is domestic-cat or feline identity and behavior.

For each candidate output, independently score:
- target_relevance (0-4): 0 none; 1 weak or ambiguous evidence; 2 implicit but recognizable target content; 3 explicit target content or multiple specific cues; 4 dominant and repeated target content.
- task_preservation (0-4): 0 ignores the user's task; 1 barely addresses it; 2 partially answers it; 3 mostly answers it; 4 fully answers it.
- degenerate: true only for broken, severely repetitive, incoherent, or unusably truncated output.

Strong target evidence: explicit cat or feline identity, or multiple specific cues such as meowing, purring, whiskers, litter boxes, paws, claws, tabby, or calico
Do not count as sufficient evidence: generic words such as animal, owner, elegant, mysterious, independent, queen, fur, or tail alone
Return strict JSON only:
{"ratings":[{"label":"A","target_relevance":0,"task_preservation":0,"degenerate":false,"reason":"short reason"}]}`}</code></pre>
        <p><b>User</b></p>
        <pre><code>{`User task:
{held-out instruction}

Output A:
{anonymized output}

Output B:
{anonymized output}

Output C:
{anonymized output}

Evaluate every output above exactly once. Return exactly three ratings with labels A, B, and C; do not omit, rename, or add labels.`}</code></pre>
      </details>
    </div>
  );
}

function FormalLeaderboard({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const ordered = [...aggregateConfigurations].sort((a, b) => b.metrics.mean_overall_score.mean - a.metrics.mean_overall_score.mean);
  const expert = replicateResults.expert_baseline;
  const score = (row: ReplicateConfiguration, key: string, scale = 1) => {
    const metric = row.metrics[key];
    return `${(metric.mean * scale).toFixed(3)} ± ${(metric.std * scale).toFixed(3)}`;
  };
  return (
    <div className="leaderboard-panel"><div className="table-scroll"><table className="formal-table"><thead><tr><th>#</th><th>{tx("模型", "Model")}</th><th>{tx("综合分", "Overall")}</th><th>Rank</th><th>Activation</th><th>Steering</th><th>Exact</th><th>{tx("目标相关度", "Target relevance")}</th><th>Causal</th><th>Usable</th><th>{tx("中位耗时", "Median time")}</th></tr></thead><tbody>
      <tr className="expert-baseline"><td>GT</td><td><strong>{tx("Expert Feature 基线", "Expert feature baseline")}</strong></td><td>{expert.mean_overall_score.toFixed(3)}</td><td>{expert.mean_rank_score.toFixed(3)}</td><td>{expert.mean_activation_score.toFixed(3)}</td><td>{expert.mean_steering_score.toFixed(3)}</td><td>{percentage(expert.exact_match_rate)}</td><td>{expert.mean_target_relevance.toFixed(3)}</td><td>{percentage(expert.causal_steering_rate)}</td><td>{percentage(expert.usable_steering_rate)}</td><td>—</td></tr>
      {ordered.map((row, index) => (
      <tr key={row.configuration}><td>{index + 1}</td><td><strong>{displayName(row)}</strong></td><td>{score(row, "mean_overall_score")}</td><td>{score(row, "mean_rank_score")}</td><td>{score(row, "mean_activation_score")}</td><td>{score(row, "mean_steering_score")}</td><td>{score(row, "exact_match_rate", 100)}%</td><td>{score(row, "macro_pe_target_relevance", 0.25)}</td><td>{score(row, "causal_steering_rate", 100)}%</td><td>{score(row, "usable_steering_rate", 100)}%</td><td>{score(row, "median_elapsed_seconds", 1 / 60)} min</td></tr>
    ))}</tbody></table></div><p className="figure-note">{tx(`GT Feature 在 20 题上的原始均值：positive mean rank ${expert.raw_metrics.mean_positive_rank.toFixed(2)}，AUROC ${expert.raw_metrics.mean_activation_auroc.toFixed(3)}，activation contrast ${expert.raw_metrics.mean_activation_contrast.toFixed(2)}，control-adjusted steering effect ${expert.raw_metrics.mean_steering_effect.toFixed(3)}。归一化后的 1.0 是参照点而非上限。`, `Raw GT-feature means across 20 tasks: positive mean rank ${expert.raw_metrics.mean_positive_rank.toFixed(2)}, AUROC ${expert.raw_metrics.mean_activation_auroc.toFixed(3)}, activation contrast ${expert.raw_metrics.mean_activation_contrast.toFixed(2)}, and control-adjusted steering effect ${expert.raw_metrics.mean_steering_effect.toFixed(3)}. The normalized value 1.0 is a reference point, not a ceiling.`)}</p></div>
  );
}

function ScatterExplorer({ language }: { language: Language }) {
  const [filter, setFilter] = useState<PointFilter>("all");
  const [selected, setSelected] = useState<Run | null>(null);
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const points = runs.filter((run) => filter === "all" || (filter === "exact" ? run.exact_match : !run.exact_match));
  const left = 76;
  const top = 24;
  const width = 820;
  const height = 390;
  const x = (value: number) => left + value * width;
  const y = (value: number) => top + (1 - Math.max(0, Math.min(1, value))) * height;
  return (
    <div className="interactive-panel">
      <div className="panel-head">
        <div>
          <span className="eyebrow">FIGURE EXPLORER</span>
          <h3>{tx("自然激活与 steering 效果的关系", "Association between natural activation and steering")}</h3>
          <p>{tx("每个点是一条 Agent × 题目运行。点击点查看发现分数、steering 和盲评结果。", "Each point is one agent-by-task run. Select a point to inspect its discovery score, steering, and blinded-judge outcomes.")}</p>
        </div>
        <MetricTabs
          value={filter}
          onChange={setFilter}
          label={tx("点过滤", "Point filter")}
          options={[
            { key: "all", label: tx("全部", "All") },
            { key: "exact", label: tx("Exact", "Exact") },
            { key: "alternative", label: tx("非精确", "Alternatives") },
          ]}
        />
      </div>
      <div className="scatter-wrap">
        <svg className="scatter" viewBox="0 0 940 470" role="img" aria-label={tx("Feature Discovery Score 与 steering 效果散点图", "Scatter plot of Feature Discovery Score and steering effect")}>
          <title>{tx("Feature Discovery Score 与 steering 效果", "Feature Discovery Score versus steering effect")}</title>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={`x-${tick}`}><line x1={x(tick)} x2={x(tick)} y1={top} y2={top + height} /><text x={x(tick)} y={top + height + 28} textAnchor="middle">{tick.toFixed(2)}</text></g>)}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={`y-${tick}`}><line x1={left} x2={left + width} y1={y(tick)} y2={y(tick)} /><text x={left - 16} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text></g>)}
          <text className="axis-title" x={left + width / 2} y={462} textAnchor="middle">Feature Discovery Score</text>
          <text className="axis-title" transform={`translate(20 ${top + height / 2}) rotate(-90)`} textAnchor="middle">{tx("因果 steering effect", "Causal steering effect")}</text>
          {points.map((run, index) => {
            const configIndex = configurations.findIndex((row) => row.configuration === run.configuration);
            const active = selected === run;
            return (
              <circle
                key={`${run.configuration}-${run.target}-${index}`}
                className={`${seriesClasses[configIndex]} ${run.exact_match ? "exact-point" : "alternative-point"} ${active ? "selected-point" : ""}`}
                cx={x(run.feature_discovery_score)}
                cy={y(run.steering_effect)}
                r={active ? 7 : run.exact_match ? 5 : 4}
                tabIndex={0}
                role="button"
                aria-label={`${configurationLabel(run.configuration)}, ${humanize(run.target)}, discovery score ${run.feature_discovery_score.toFixed(3)}, steering ${run.steering_effect.toFixed(3)}`}
                onClick={() => setSelected(run)}
                onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelected(run); }}
              >
                <title>{configurationLabel(run.configuration)} · {humanize(run.target)} · {run.feature_discovery_score.toFixed(3)} / {run.steering_effect.toFixed(3)}</title>
              </circle>
            );
          })}
        </svg>
        <div className="scatter-detail" aria-live="polite">
          {selected ? <>
            <span>{selected.exact_match ? "EXACT FEATURE" : "ALTERNATIVE FEATURE"} · ID {selected.selected_feature_id}</span>
            <h4>{configurationLabel(selected.configuration)}</h4>
            <p>{humanize(selected.target)}</p>
            <dl>
              <div><dt>Discovery Score</dt><dd>{selected.feature_discovery_score.toFixed(3)}</dd></div>
              <div><dt>Steering effect</dt><dd>{selected.steering_effect.toFixed(3)}</dd></div>
              <div><dt>Target relevance</dt><dd>{selected.pe_target_relevance.toFixed(2)} / 4</dd></div>
              <div><dt>Task preservation</dt><dd>{selected.pe_task_preservation.toFixed(2)} / 4</dd></div>
            </dl>
          </> : <>
            <span>{tx("选择一个点", "SELECT A POINT")}</span>
            <h4>{tx("查看单条运行", "Inspect one run")}</h4>
            <p>{tx("Exact feature 聚集在右侧，但非精确候选覆盖了从完全失效到强 steering 的整个区域。", "Exact features cluster at the right edge, while alternatives span the full range from no effect to strong steering.")}</p>
          </>}
        </div>
      </div>
    </div>
  );
}

function Home() {
  const [language, setLanguage] = useState<Language>("en");
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const summary = useMemo(() => {
    return {
      tasks: replicateResults.benchmark_tasks,
      runs: replicateResults.discovery_runs,
      exact: aggregateAnalysis.exact_runs,
      causal: aggregateAnalysis.causal_runs,
      usable: aggregateAnalysis.usable_runs,
      alternatives: aggregateAnalysis.alternative_runs,
      alternativeCausal: aggregateAnalysis.alternative_causal_runs,
      allCorrelation: aggregateAnalysis.activation_steering_spearman,
      alternativeCorrelation: aggregateAnalysis.alternative_activation_steering_spearman,
      exactEffect: aggregateAnalysis.exact_mean_steering_effect,
      alternativeEffect: aggregateAnalysis.alternative_mean_steering_effect,
    };
  }, []);

  return (
    <main lang={language === "zh" ? "zh-CN" : "en"}>
      <header className="site-header">
        <a className="wordmark" href="#top"><strong>SAE</strong><span>SCIENTIST</span></a>
        <nav><a href="#idea">{tx("核心想法", "Idea")}</a><a href="#features">Feature IDs</a><a href="#results">{tx("结果", "Results")}</a><a href="#method">{tx("方法", "Methods")}</a></nav>
        <div className="language-control" aria-label="Language"><button className={language === "zh" ? "active" : ""} onClick={() => setLanguage("zh")}>中文</button><button className={language === "en" ? "active" : ""} onClick={() => setLanguage("en")}>EN</button></div>
      </header>

      <article id="top" className="research-article">
        <section className="article-masthead">
          <h1>{tx("SAEScientist-Bench：评估 Agent 的自主 SAE 可解释性研究能力", "SAEScientist-Bench: Evaluating agents as autonomous SAE researchers")}</h1>
          <p className="dek">{tx("我们评估 Agent 能否像研究者一样提出对照、设计实验、搜索并解释 SAE feature，再用 activation 与 steering 证据验证自己的结论。", "We evaluate whether an agent can work like a researcher: form contrasts, design experiments, discover and interpret an SAE feature, then validate its claim with activation and steering evidence.")}</p>
          <div className="article-authors"><p className="author-names">SAEScientist Research Team</p></div>
          <p className="article-meta">{tx("2026 年 9 月", "September 2026")} <span>·</span> {tx("预计阅读时间：18 分钟", "18 min read")} <span>·</span> {summary.tasks} tasks <span>·</span> {summary.runs} runs</p>
          <p className="article-links"><a href="https://github.com/Trae1ounG/SAEScientist">Code &amp; data</a> <span>·</span> <a href="https://huggingface.co/google/gemma-scope-9b-it-res">Official SAE</a></p>
        </section>

        <section className="abstract-block">
          <h2>Abstract</h2>
          <p>{tx("稀疏自编码器把模型内部状态分解成大量 feature，但从语义假设走到可复现的 Feature ID，仍是一项需要构造数据、运行实验、解释证据并排除替代解释的科学研究工作。SAEScientist-Bench 测试 Agent 能否自主完成这条研究链。在 20 个官方 Gemma Scope expert features 上，Agent 经常能找到语义接近的方向，但 exact recovery 与等价 steering 明显更难。", "Sparse autoencoders decompose model states into a large feature dictionary, but moving from a semantic hypothesis to a reproducible feature ID remains a scientific workflow: construct data, run experiments, interpret evidence, and rule out alternatives. SAEScientist-Bench tests whether an agent can execute that research loop autonomously. Across 20 expert features from the official Gemma Scope release, semantic neighbors are common; exact recovery and equivalent steering are much harder.")}</p>
        </section>

        <aside className="table-of-contents"><details open><summary>{tx("目录", "Table of Contents")}</summary><ol><li><a href="#idea">{tx("我们究竟测什么能力", "What capability are we measuring?")}</a></li><li><a href="#features">{tx("Expert feature 与案例", "Expert features and cases")}</a></li><li><a href="#results">{tx("主要实验结果", "Main experimental results")}</a></li><li><a href="#analysis">{tx("语义—因果鸿沟", "The semantic–causal gap")}</a></li><li><a href="#method">{tx("有效性边界与局限", "Validity and limitations")}</a></li><li><a href="#conclusion">{tx("结论", "Conclusion")}</a></li></ol></details></aside>

        <aside className="claims-at-glance"><h2>{tx("一分钟了解", "In one minute")}</h2><div><p><b>{tx("任务", "Task")}</b><span>{tx("Agent 根据英文语义目标自主构造 probes，并从 131,072 个 SAE 方向中提交一个 Feature ID。", "From an English semantic target, an agent authors probes and submits one feature ID from 131,072 SAE directions.")}</span></p><p><b>{tx("证据", "Evidence")}</b><span>{tx("20 个官方 Gemma Scope expert features、9 个 Agent 配置、每个模型三次独立运行，共 540 条通过 trace audit 的运行。", "20 official Gemma Scope expert features, nine agent configurations, and three independent runs per model: 540 completed trace-audited episodes.")}</span></p><p><b>{tx("结果", "Result")}</b><span>{tx("语义近邻在候选中普遍存在。Exact recovery、自然激活与因果 steering 的结果需要分别报告。", "Semantic neighbors are common among candidate features. Exact recovery, natural activation, and causal steering require separate measurement.")}</span></p></div></aside>

        <section id="idea" className="article-section">
          <div className="article-copy">
            <h2>{tx("1. Benchmark 定义与评测协议", "1. Benchmark definition and evaluation protocol")}</h2>
            <p>{tx("SAE 中的 feature ID 只是某层字典里的列号，没有天然标签。Agent 从英文研究目标出发，自主写 positive、hard-negative 与 neutral probes，调用受限实验接口，观察激活并反复修正假设，最终提交一个 ID 和解释。", "A feature ID is only a column index in one layer's SAE dictionary; it has no intrinsic label. Starting from an English research target, the agent authors positive, hard-negative, and neutral probes, queries a restricted experiment interface, revises its hypothesis from activations, and finally submits one ID with an interpretation.")}</p>
            <p>{tx("Benchmark 随后在隐藏评测上复现实验，并把候选 decoder direction 写回残差流，与 expert 和 matched-random 对照。评分因此覆盖研究过程的三个层次：定位是否准确、解释是否吻合、干预是否沿相同方向改变行为。", "The benchmark then reproduces the experiment on hidden cases and writes the candidate decoder direction back into the residual stream against expert and matched-random controls. Scoring therefore covers three levels of research evidence: localization, interpretation, and causal intervention.")}</p>
            <h3>{tx("实验设置", "Experimental setting")}</h3>
            <dl className="setup-facts">
              <div><dt>{tx("基础模型", "Base model")}</dt><dd>Gemma 2 9B IT</dd></div>
              <div><dt>SAE</dt><dd>Official Gemma Scope · resid_post</dd></div>
              <div><dt>{tx("搜索空间", "Search space")}</dt><dd>131,072 features · layer 9 / 20</dd></div>
              <div><dt>{tx("Agent 可见", "Agent sees")}</dt><dd>{tx("英文研究目标 + 受限激活 probe", "English research target + restricted activation probe")}</dd></div>
              <div><dt>{tx("Agent 提交", "Agent submits")}</dt><dd>{tx("一个 Feature ID + 实验证据 + 解释", "One feature ID + evidence + interpretation")}</dd></div>
              <div><dt>{tx("隐藏评测", "Hidden evaluation")}</dt><dd>{tx("激活、Expert rank、steering 与盲评", "Activation, expert rank, steering, and blinded judging")}</dd></div>
            </dl>
          </div>
          <ResearchFigure wide number={1} title={tx("SAEScientist-Bench 的系统框架。", "System architecture of SAEScientist-Bench.")} caption={tx("上半部分展示 Expert task 的构造与准入；下半部分展示隔离的 Agent discovery episode、提交审计和可信隐藏评测。", "The upper lane constructs and admits expert tasks; the lower lane contains the isolated agent discovery episode, submission audit, and trusted hidden evaluation.")}><img className="mechanism-diagram" src={`${import.meta.env.BASE_URL}figures/feature-discovery-mechanism/diagram.svg?v=6`} alt={tx("SAEScientist-Bench 系统框架图", "SAEScientist-Bench system architecture")} /></ResearchFigure>
          <ResearchFigure wide number={2} title={tx("Feature 62610 的完整画像。", "A complete portrait of feature 62610.")} caption={tx("上半部分展示 feature 的语义与 32 条冻结 case 上的 activation 深度；下半部分展示同一 instruction 在 steering 前后的真实输出差异。", "The upper half shows the feature's interpretation and activation depth over 32 frozen cases; the lower half shows real before/after outputs for the same instruction.")}><FeaturePortrait language={language} /></ResearchFigure>
        </section>

        <section id="features" className="article-section">
          <div className="article-copy"><h2>{tx("2. Expert features 与候选方向的案例分析", "2. Case analysis of expert features and candidate directions")}</h2><p>{tx("20 个任务覆盖 1 个主题 feature、6 个语言 feature、8 个商业/金融 feature 与 5 个领域或文体 feature；12 个 anchor 位于 layer 9，8 个位于 layer 20。下面的逐题案例取自首轮可审计运行：猫特征 62610 被全部 8 个配置精确恢复；法语任务产生四个相近候选；报税任务的 8 个 Agent 一致提交 64827，而 Expert ID 为 18713。后两个任务展示了 SAE 字典中语义相近但索引不同的候选方向。", "The 20 tasks comprise one topic feature, six language features, eight business/finance features, and five domain or register features; 12 anchors are at layer 9 and eight at layer 20. The task-level examples below use the auditable first run: all eight configurations recover cat feature 62610 exactly, the French task produces four nearby candidates, and all eight agents submit 64827 for tax filing while the expert ID is 18713. The latter two tasks expose semantically related directions with distinct indices in the SAE dictionary.")}</p></div>
          <ResearchFigure number={3} title={tx("五个 Feature ID 案例。", "Five feature-ID case studies.")} caption={tx("首轮案例覆盖 8/8 exact recovery、稳定替代方向，以及强 steering 但低 usable 的方向。", "First-run cases span 8/8 exact recovery, stable alternative directions, and strong steering with low usability.")}><FeatureAtlas language={language} /></ResearchFigure>
          <div className="article-copy post-figure-copy">
            <p>{tx("French 与 clinical expert 在冻结数据上均呈现近乎完美的 activation 分离，Agent 仍会收敛到其他方向。这一结果表明，expert feature 的准入质量与 Agent 的精确恢复率需要独立评估；SAE 字典中的局部语义冗余会直接影响 Feature ID 定位。", "The French and clinical experts both show nearly perfect activation separation on frozen data, while agents converge on other directions. Expert admission quality and agent exact-recovery rate therefore require independent evaluation, with local semantic redundancy in the SAE dictionary directly affecting feature localization.")}</p>
            <p>{tx("Weather 候选 85606 的 target induction 高于 Expert 44494；clinical expert 则呈现较高 target induction 和 0 usable rate。两项观察共同支持分解式报告：exact、activation、causal effect 与 task preservation 分别对应不同的有效性证据。", "Weather candidate 85606 has higher target induction than expert 44494, while the clinical expert combines strong target induction with a zero usable rate. These observations support decomposed reporting: exact match, activation, causal effect, and task preservation measure distinct forms of validity.")}</p>
          </div>
          <div className="article-copy post-figure-copy">
            <h3>{tx("完整 20 题分布", "Distribution of all 20 tasks")}</h3>
            <p>{tx("下表逐题给出语义目标、官方 Feature ID、层位置、三类冻结 case 的平均激活，以及 expert direction 自身的 held-out steering 结果。20 个 anchor 均通过自然激活准入；更换为 GPT‑4o 后，未经重新校准的 expert steering 中有 12/20 通过既定 causal gate。我们因此保留 20 题的 discovery 结果，同时把 steering 主张视为需要重新准入的独立实验。", "The table reports each target, official feature ID, layer, mean activation on the three frozen case groups, and the expert direction's held-out steering result. All 20 anchors pass natural-activation admission. After switching to GPT-4o, 12/20 unrecalibrated expert directions pass the frozen causal gate. We therefore retain all 20 discovery tasks while treating steering claims as a separate experiment requiring re-admission.")}</p>
          </div>
          <ExpertTaskTable language={language} />
          <div className="article-copy post-figure-copy">
            <h3>{tx("同一道题，不同 Agent 提交的方向", "One task, different agent-submitted directions")}</h3>
            <p>{tx("天气预报任务提供了一个清晰的对照。六个 Agent 精确找到 44494，Sonnet 5 与 GLM-5.2 分别提交 85606 和 127623。每个非精确候选都在相同的 baseline、matched-random 和 held-out prompts 上单独执行 steering。Exact-match 运行使用 Expert direction 的同一份冻结生成输出；评分阶段，所有方向均由 GPT‑4o 使用同一 Prompt 重新打分。", "The weather-forecast task provides a controlled comparison. Six agents recover 44494 exactly, while Sonnet 5 and GLM-5.2 submit 85606 and 127623. Each non-exact candidate is steered against the same baseline, matched-random direction, and held-out prompts. Exact-match runs use the same frozen generations from the expert direction. At scoring time, GPT-4o re-scores every direction with the same prompt.")}</p>
          </div>
          <ResearchFigure wide number={4} title={tx("天气预报任务的 Agent 提交与 steering 输出。", "Agent submissions and steering outputs on the weather task.")} caption={tx("更高的 target relevance 可以来自更强但更破坏原任务的方向，因此需要同时报告 preservation 与 causal gate。", "Higher target relevance can come from a stronger direction that damages the original task, motivating the joint reporting of preservation and the causal gate.")}><WeatherAgentComparison language={language} /></ResearchFigure>
        </section>

        <section id="results" className="article-section">
          <div className="article-copy"><h2>{tx("3. 20 个任务上的主要实验结果", "3. Main results across 20 tasks")}</h2><p>{tx(`正式结果覆盖 ${summary.tasks} 道题、${aggregateConfigurations.length} 个 Agent 配置，每个配置独立运行三次，共 ${summary.runs} 条完整运行。每题先得到三个以同题 Expert 为 1.0 参照点的分数：Rank 衡量正例平均排名，Activation 汇总 AUROC、正负例激活对比度与逐 case 激活 pattern，Steering 汇总 control-adjusted effect 与逐 instruction steering pattern。Overall 是三项等权平均。候选若在同一 hidden set 上优于 Expert，可以得到高于 1.0 的单项或单题分数。`, `The formal results cover ${summary.tasks} tasks and ${aggregateConfigurations.length} agent configurations, with three independent runs per configuration and ${summary.runs} complete episodes. Each task receives three scores centered on its expert feature at 1.0: Rank measures positive-case ranking, Activation combines AUROC, positive-versus-control contrast, and per-case activation patterns, and Steering combines control-adjusted effect with per-instruction steering patterns. Overall is their equal-weight mean. A candidate can score above 1.0 when it exceeds the expert on the same hidden set.`)}</p></div>
          <ResearchFigure number={5} title={tx("SAEScientist-Bench 三轮主榜。", "Three-run SAEScientist-Bench leaderboard.")} caption={tx("表中为三次独立运行的均值 ± 总体标准差。GT 行是同一套 hidden cases 上的 Expert Feature 归一化基线，不参与模型排名。Exact 衡量 Expert ID 的精确恢复；Causal 要求目标效果显著强于 baseline 和等范数随机方向；Usable 还要求保留原始任务且输出不退化。GPT‑4o 仅作为 steering 输出的盲评模型。", "Entries are mean ± population standard deviation over three independent runs. The GT row is the expert-feature normalization baseline on the same hidden cases and is not a competing model. Exact measures expert-ID recovery; Causal requires a clear gain over baseline and a norm-matched random direction; Usable additionally requires task preservation and non-degenerate text. GPT-4o is used only as the blinded judge for steering outputs.")}><FormalLeaderboard language={language} /></ResearchFigure>
          <div className="article-copy post-figure-copy"><h3>{tx("盲评结果", "Blinded steering judgments")}</h3><p>{tx(`三轮实验共包含 ${aggregateAnalysis.evaluated_task_feature_pairs} 个不同的 task–feature 组合；每个组合使用 20 条 held-out instructions、三个匿名条件，并在 temperature 0 下独立判断两次。映射回 ${summary.runs} 条 Agent 运行后，${summary.causal} 条通过 causal gate，${summary.usable} 条通过 usable gate；${summary.alternatives} 个非 exact 选择中有 ${summary.alternativeCausal} 个通过 causal gate。连续 steering 分数进入 Overall，而 Causal 与 Usable 作为阈值型审计结果单独报告。`, `The three runs contain ${aggregateAnalysis.evaluated_task_feature_pairs} distinct task-feature pairs. Each pair is evaluated on 20 held-out instructions with three anonymized conditions and two independent judgments at temperature zero. Mapped back to ${summary.runs} agent episodes, ${summary.causal} pass the causal gate and ${summary.usable} passes the usable gate; ${summary.alternativeCausal} of ${summary.alternatives} non-exact selections pass the causal gate. The continuous Steering score enters Overall, while Causal and Usable remain separately reported thresholded audits.`)}</p></div>
        </section>

        <section id="analysis" className="article-section">
          <div className="article-copy"><h2>{tx("4. Activation fidelity 与因果 steering 的关系", "4. Relationship between activation fidelity and causal steering")}</h2><p>{tx("全部运行上的 activation—steering rank correlation 较高，部分原因是 exact features 会同时提高两个指标。将分析限制在非精确候选后，相关性显著降低。自然激活因此适合作为 discovery 指标，因果有效性仍需通过独立 steering 实验估计。", "Activation and steering have a relatively high rank correlation across all runs, partly because exact features increase both measures. The association is substantially weaker when the analysis is restricted to non-exact candidates. Natural activation is therefore suitable as a discovery metric, while causal validity still requires an independent steering experiment.")}</p></div>
          <div className="finding-grid"><div><span>Spearman · all</span><strong>{summary.allCorrelation.toFixed(3)}</strong><p>activation ↔ steering</p></div><div><span>Spearman · alternatives</span><strong>{summary.alternativeCorrelation.toFixed(3)}</strong><p>{tx("只看非精确候选", "non-exact candidates only")}</p></div><div><span>Mean causal effect</span><strong>{summary.exactEffect.toFixed(3)} <i>/</i> {summary.alternativeEffect.toFixed(3)}</strong><p>exact / alternative</p></div></div>
          <ResearchFigure wide number={6} title={tx("自然激活与因果效果。", "Natural activation versus causal effect.")} caption={tx("相关系数使用全部 540 条运行；可交互散点展示首轮的 160 条逐题记录，点击可查看 Feature ID 与盲评结果。", "Correlations use all 540 runs; the interactive scatter shows the 160 task-level records from the representative first run. Select a point to inspect its feature ID and blinded-judge outcomes.")}><ScatterExplorer language={language} /></ResearchFigure>
          <div className="article-copy post-figure-copy">
            <h3>{tx("难度主要来自 Expert anchor 的可发现性", "Difficulty is largely governed by expert-anchor discoverability")}</h3>
            <p>{tx(`20 道题的 exact recovery 并不只由语义类别决定。Expert feature 在正例上的平均 rank 与跨 Agent 的 exact rate 呈强负相关（Spearman ${diagnostics.task_discoverability_expert_rank_spearman.toFixed(3)}）：当 anchor 本身不在正例激活榜前列时，Agent 即使构造出合理 probes，也更容易稳定地收敛到另一个语义方向。报税任务 layer 9 是最清楚的例子：Expert 为 #18713，而 #64827 在 27 次运行中被提交 25 次。`, `Exact recovery across the 20 tasks is not determined by semantic category alone. The expert feature's mean rank on positive cases is strongly negatively associated with cross-agent exact rate (Spearman ${diagnostics.task_discoverability_expert_rank_spearman.toFixed(3)}). When the anchor itself is not near the top of the positive activation list, agents can construct sensible probes yet converge consistently on another semantic direction. The layer-9 tax task is the clearest example: the expert is #18713, while #64827 is submitted in 25 of 27 runs.`)}</p>
            <p>{tx(`模型级 exact rate 与 Overall 仍高度相关（Pearson ${diagnostics.model_exact_overall_pearson.toFixed(3)}），但并非同一个指标。只看 ${diagnostics.alternative_runs} 条非精确运行，Kimi K3 的平均 Overall 为 ${diagnostics.alternative_by_model[0].mean_overall_score.toFixed(3)}；这表明主榜同时包含“找到 anchor”与“找到有证据支持的替代方向”两种能力。`, `Model-level exact rate remains strongly correlated with Overall (Pearson ${diagnostics.model_exact_overall_pearson.toFixed(3)}), but the two are not identical. Restricting evaluation to the ${diagnostics.alternative_runs} non-exact runs, Kimi K3 attains an average Overall of ${diagnostics.alternative_by_model[0].mean_overall_score.toFixed(3)}. The main leaderboard therefore combines the ability to recover the anchor with the ability to find an evidence-supported alternative direction.`)}</p>
          </div>
          <ResearchFigure wide number={7} title={tx("题目难度与 activation–causality gap。", "Task difficulty and the activation–causality gap.")} caption={tx(`左图显示 Expert 正例 rank 与 exact recovery 的关系；右图显示 ${diagnostics.high_activation_alternatives}/${diagnostics.alternative_runs} 个替代方向有较高 activation score，但只有 ${diagnostics.high_activation_causal_alternatives} 个通过 causal gate。`, `The left panel relates expert positive rank to exact recovery. The right panel shows that ${diagnostics.high_activation_alternatives}/${diagnostics.alternative_runs} alternatives have high activation scores, while only ${diagnostics.high_activation_causal_alternatives} pass the causal gate.`)}><img className="mechanism-diagram diagnostic-figure" src={`${import.meta.env.BASE_URL}figures/diagnostic-results.png`} alt={tx("任务难度与 activation-causality gap 诊断图", "Diagnostic plots for task difficulty and the activation-causality gap")} /></ResearchFigure>
          <div className="article-copy post-figure-copy">
            <h3>{tx("前三名应视为一个统计簇", "The top three form a statistical cluster")}</h3>
            <p>{tx(`三次重复运行给出了配置内波动，但每次仍复用同一组 20 道题。我们因此按题目执行 10,000 次 paired bootstrap。Opus 5 相对 Sonnet 5 的 Overall 差值为 ${bootstrap.pairwise_task_bootstrap[0].mean_difference.toFixed(3)}，95% CI [${bootstrap.pairwise_task_bootstrap[0].ci95[0].toFixed(3)}, ${bootstrap.pairwise_task_bootstrap[0].ci95[1].toFixed(3)}]；相对 Kimi K3 的差值为 ${bootstrap.pairwise_task_bootstrap[1].mean_difference.toFixed(3)}，95% CI [${bootstrap.pairwise_task_bootstrap[1].ci95[0].toFixed(3)}, ${bootstrap.pairwise_task_bootstrap[1].ci95[1].toFixed(3)}]。两个区间都跨过 0，因此当前 20 题支持的是一个前三统计簇，而不是稳定的严格排序。`, `Three independent runs estimate within-configuration variation, but each run reuses the same 20 tasks. We therefore perform 10,000 paired task-bootstrap samples. Opus 5 exceeds Sonnet 5 by ${bootstrap.pairwise_task_bootstrap[0].mean_difference.toFixed(3)} Overall with a 95% CI of [${bootstrap.pairwise_task_bootstrap[0].ci95[0].toFixed(3)}, ${bootstrap.pairwise_task_bootstrap[0].ci95[1].toFixed(3)}], and exceeds Kimi K3 by ${bootstrap.pairwise_task_bootstrap[1].mean_difference.toFixed(3)} with a 95% CI of [${bootstrap.pairwise_task_bootstrap[1].ci95[0].toFixed(3)}, ${bootstrap.pairwise_task_bootstrap[1].ci95[1].toFixed(3)}]. Both intervals cross zero, so the current 20-task set supports a top-three statistical cluster rather than a stable strict ordering.`)}</p>
          </div>
        </section>

        <section id="method" className="article-section methods-section">
          <div className="article-copy">
            <h2>{tx("5. 冻结任务、评分协议与有效性边界", "5. Frozen tasks, scoring protocol, and validity")}</h2>
            <h3>{tx("Expert case 准入", "Expert-case admission")}</h3>
            <p>{tx("每个目标都绑定官方 Gemma Scope SAE 中的一个 expert feature。我们先在冻结的 positive、hard-negative 与 neutral cases 上计算 top-3 token mean activation。只有 AUROC 不低于 0.95、正例激活率不低于 80%、且困难负例均值与正例均值之比不高于 0.30 的 feature，才能进入 benchmark。随后还要通过 matched-random steering、独立 calibration prompts、20 条 held-out prompts 与盲评验证。", "Each target is anchored to an expert feature from the official Gemma Scope SAE. We first compute top-three-token mean activation on frozen positive, hard-negative, and neutral cases. A feature enters the benchmark only if AUROC is at least 0.95, positive active rate is at least 80%, and the ratio of hard-negative to positive mean activation is at most 0.30. It is then validated with matched-random steering, separate calibration prompts, 20 held-out prompts, and blinded judging.")}</p>
            <h3>{tx("Positive、hard negative 与 neutral", "Positive, hard negative, and neutral")}</h3>
            <p>{tx("每条 activation case 只有三种标签。Positive 是确实包含目标语义的文本；hard negative 是与目标共享词面、主题、文体或邻近概念、但缺少目标定义属性的近似反例；neutral 是与目标无关的背景文本。表中的 mean 只是该组 case activation 的算术平均，不是第四种标签。计算二分类指标时，negative 指 hard negative 与 neutral 的并集。", "Every activation case has one of three labels. A positive genuinely contains the target concept. A hard negative shares surface words, topic, register, or a neighboring concept but lacks the target's defining property. A neutral is unrelated background text. Mean in the tables is the arithmetic average activation within a group, not a fourth label. For binary metrics, negative denotes the union of hard negatives and neutrals.")}</p>
            <div className="definition-table table-scroll"><table className="formal-table"><thead><tr><th>{tx("字段", "Field")}</th><th>{tx("定义", "Definition")}</th><th>{tx("猫 feature 示例", "Cat-feature example")}</th></tr></thead><tbody>
              <tr><td><b>positive</b></td><td>{tx("目标语义成立", "Target concept is present")}</td><td>{tx("家猫在窗台打盹并发出呼噜声", "A domestic cat naps on a windowsill and purrs")}</td></tr>
              <tr><td><b>hard negative</b></td><td>{tx("很像目标，但关键语义不成立", "Close to the target, but its defining semantics are absent")}</td><td>{tx("发动机发出平稳的 purr；或狮子、老虎", "An engine settles into a purr; or lions and tigers")}</td></tr>
              <tr><td><b>neutral</b></td><td>{tx("无关背景", "Unrelated background")}</td><td>{tx("数据库索引降低查询延迟", "A database index reduces query latency")}</td></tr>
            </tbody></table></div>
            <p>{tx("每条 case 先取该 feature 在所有 token 上最大的 3 个激活并求平均，得到一个 case-level activation Aᵢ；positive mean、hard-negative mean 与 neutral mean 再分别对各组 Aᵢ 求平均。", "For each case, the three largest token activations of the feature are averaged into a case-level activation Aᵢ. Positive mean, hard-negative mean, and neutral mean then average Aᵢ within the corresponding group.")}</p>
            <h3>{tx("AUROC 如何计算", "How AUROC is computed")}</h3>
            <p>{tx("AUROC 不需要另选阈值。它枚举每一个 positive–negative case pair：positive 激活更高记 1 分，相等记 0.5 分，更低记 0 分，再对全部 pair 取平均。它等价于随机抽一条 positive 和一条 negative 时，positive activation 更高的概率。", "AUROC requires no chosen threshold. It enumerates every positive–negative case pair: a higher positive activation scores 1, a tie scores 0.5, and a lower positive activation scores 0, then averages over all pairs. Equivalently, it is the probability that a randomly drawn positive has a higher activation than a randomly drawn negative.")}</p>
            <div className="equation-block"><code>AUROC = [Σ I(A<sub>pos</sub> &gt; A<sub>neg</sub>) + ½ I(A<sub>pos</sub> = A<sub>neg</sub>)] / (N<sub>pos</sub>N<sub>neg</sub>)</code></div>
            <p>{tx("猫 feature 的冻结集包含 12 条 positive、12 条 hard negative 和 8 条 neutral，因此共有 12 × 20 = 240 个 pair；239 个 pair 的 positive 激活更高，AUROC = 239/240 = 0.9958。", "The frozen cat-feature set contains 12 positives, 12 hard negatives, and eight neutrals, giving 12 × 20 = 240 pairs. The positive activation is higher in 239 pairs, so AUROC = 239/240 = 0.9958.")}</p>
            <h3>{tx("Rank、Activation 与 Overall 如何评分", "How Rank, Activation, and Overall are scored")}</h3>
            <p>{tx("Agent 提交候选 Feature ID 后，评测器在未向 Agent 展示的冻结 cases 上同时运行候选与 Expert。不同 Feature 的原始 activation 尺度不可直接比较，因此每个原始量都相对同题 Expert 做对称重标定。对越大越好的量使用 2c/(c+e)，对越小越好的 rank 使用 2e/(c+e)。候选等于 Expert 时为 1；优于 Expert 时大于 1；弱于 Expert 时小于 1，取值范围为 0–2。", "After the agent submits a candidate feature ID, the evaluator runs both candidate and Expert on frozen cases never shown to the agent. Raw activation scales are not directly comparable across features, so every raw quantity is symmetrically rescaled against the same-task Expert. Higher-is-better quantities use 2c/(c+e), while lower-is-better rank uses 2e/(c+e). Equality with the Expert maps to 1, stronger candidates exceed 1, weaker candidates fall below 1, and the range is 0–2.")}</p>
            <div className="equation-block"><code>g↑(c,e) = 2c/(c+e)　　g↓(c,e) = 2e/(c+e)</code></div>
            <div className="equation-block"><code>S<sub>overall</sub> = ⅓ (S<sub>rank</sub> + S<sub>activation</sub> + S<sub>steering</sub>)</code></div>
            <ol className="method-list">
              <li><b>Rank Score</b>：{tx("对 candidate 与 Expert 的 positive mean rank 应用 g↓。rank 越靠前越好。", "apply g↓ to candidate and Expert positive mean rank; earlier ranks are better.")}</li>
              <li><b>Activation Score</b>：{tx("对 AUROC−0.5、positive mean−max(hard-negative mean, neutral mean)，以及非负的逐 case Spearman 分别重标定后取平均。", "average rescaled AUROC−0.5, positive mean minus the stronger hard-negative/neutral mean, and non-negative per-case Spearman agreement.")}</li>
              <li><b>Steering Score</b>：{tx("对 control-adjusted target effect 与逐 instruction steering-effect Spearman 分别重标定后取平均。", "average rescaled control-adjusted target effect and per-instruction steering-effect Spearman agreement.")}</li>
              <li><b>Overall</b>：{tx("Rank、Activation 与 Steering 三项等权平均，再对 20 道题取平均。", "take the equal-weight mean of Rank, Activation, and Steering, then average across the 20 tasks.")}</li>
            </ol>
            <p>{tx("Exact match 另行记录，不参与 Overall；Causal 与 Usable 也作为门槛型审计结果单列，避免把已经进入 Steering Score 的信号重复计权。", "Exact match is recorded separately rather than mixed into Overall. Causal and Usable are also shown as thresholded audit outcomes, avoiding double-counting signals already represented in the Steering Score.")}</p>
            <h3>{tx("Steering 如何执行与判分", "How steering is executed and judged")}</h3>
            <p>{tx("对候选 feature k，评测器在其 SAE 所在层、每个生成位置，把 decoder direction 加到残差流：", "For candidate feature k, the evaluator adds its decoder direction to the residual stream at the SAE layer and at every generated position:")}</p>
            <div className="equation-block"><code>h′<sub>t</sub> = h<sub>t</sub> + α · W<sub>dec</sub>[:, k]</code></div>
            <p>{tx("α 先在 5 条 calibration prompts 上，从围绕 expert 强度的固定网格中选择；正式分数在另外 20 条 held-out instructions 上计算。每条 instruction 同时生成 baseline、候选方向与等范数随机方向三个条件，标签打乱后交给 GPT‑4o-2024-11-20。Judge 分别给 target relevance 与 task preservation 打 0–4 分，并标记输出是否退化。", "Alpha is selected on five calibration prompts from a fixed grid centered on the expert strength; formal scoring uses a separate set of 20 held-out instructions. For every instruction, the evaluator generates baseline, candidate-direction, and norm-matched-random conditions, shuffles their labels, and sends them to GPT-4o-2024-11-20. The judge independently rates target relevance and task preservation from 0 to 4 and flags degenerate output.")}</p>
            <p>{tx("盲评输入由原始 user task 和三个匿名输出组成。target_relevance 的 0/1/2/3/4 分别表示无目标证据、微弱或含混、隐含但可识别、明确或包含多个具体线索、以及目标内容占主导且反复出现；task_preservation 的 0/1/2/3/4 分别表示忽略任务、几乎未完成、部分完成、大体完成、完全完成。degenerate 只在输出破碎、严重重复、不连贯或截断到不可用时为 true。temperature 固定为 0，每条输出独立打标两次。", "The blinded-judge input contains the original user task and three anonymous outputs. Target-relevance scores 0/1/2/3/4 mean no evidence, weak or ambiguous evidence, implicit but recognizable content, explicit or multiple specific cues, and dominant repeated target content. Task-preservation scores 0/1/2/3/4 mean ignored, barely addressed, partially answered, mostly answered, and fully answered. Degenerate is true only for broken, severely repetitive, incoherent, or unusably truncated output. Temperature is fixed at zero and each output is judged twice.")}</p>
            <PeJudgeExample language={language} />
            <div className="equation-block"><code>E<sub>causal</sub> = relevance(feature) / 4 − max(relevance(baseline), relevance(random)) / 4</code></div>
            <p>{tx("表里的 Expert steering 就是把 expert Feature ID 代入这个流程所得的 Ecausal。对每个条件，先对 20 条 held-out × 2 次 judge 的 target_relevance 求平均，再除以 4 归一化。GPT‑4o 对猫 feature 的 expert target score 为 0.438，baseline 与 matched-random 均为 0，因此 Expert steering = 0.438。它不是 decoder cosine，也不是内部 activation 大小。", "Expert steering is Ecausal obtained by running the expert feature ID through this protocol. For each condition, target relevance is averaged over 20 held-out instructions × two judge passes and divided by four. GPT-4o gives the cat expert a target score of 0.438 while baseline and matched random are both zero, so Expert steering is 0.438. It is neither decoder cosine nor internal activation magnitude.")}</p>
            <p>{tx("Target success rate 是 relevance ≥ 2 的比例；usable rate 是同时满足 relevance ≥ 2、preservation ≥ 2 且 degenerate = false 的比例。猫 expert 在 GPT‑4o 下分别为 65% 与 47.5%，因此未达到冻结的 70% causal gate。这一失败被保留在结果中，而没有通过放宽阈值来消除。", "Target success rate is the fraction with relevance at least two. Usable rate is the fraction simultaneously satisfying relevance at least two, preservation at least two, and degenerate = false. Under GPT-4o, the cat expert scores 65% and 47.5%, so it misses the frozen 70% causal gate. We retain this failure rather than removing it by weakening the threshold.")}</p>
            <p>{tx("Causal pass 要求 activation gate 通过、至少 20 条 held-out、相对 baseline 与 random 的 target effect 不低于 0.20、target success rate 不低于 70%、非退化率不低于 50%，并且两次独立 judge 的一致率不低于 80%。Usable pass 进一步要求 relevance ≥ 2 且 preservation ≥ 2 的输出不少于 50%，总体非退化率不少于 90%。", "A causal pass requires the activation gate, at least 20 held-out instructions, target effect of at least 0.20 over both baseline and random, target success rate of at least 70%, nondegenerate rate of at least 50%, and at least 80% agreement between two independent judge passes. A usable pass additionally requires at least 50% of outputs to have relevance of 2 or more and preservation of 2 or more, with an overall nondegenerate rate of at least 90%.")}</p>
            <h3>{tx("Agent 隔离与公开边界", "Agent isolation and release boundary")}</h3>
            <p>{tx("正式 discovery 禁止联网与搜索公开 feature labels，也不能读取 benchmark 仓库或 expert ID。Agent 唯一可用的外部反馈，是受限 probe 返回的激活与 rank。Feature ID、聚合指标与筛选后的输出案例可以公开复现；隐藏的是评测 prompts、原始轨迹与基础设施。", "Scored discovery disables internet access and public-label search and blocks access to the benchmark repository and expert IDs. The only external feedback is activation and rank returned by a restricted probe. Feature IDs, aggregate metrics, and screened output examples can be released for reproducibility; evaluation prompts, raw traces, and infrastructure remain hidden.")}</p>
            <h3>{tx("限制", "Limitations")}</h3>
            <p>{tx("当前榜单对每个配置执行三次独立运行，但仍只覆盖 Gemma 2 9B 的两个层与 20 个 Expert features。连续 target score 与 causal/usable 阈值回答不同问题，因此 exact recovery、activation quality、连续 steering effect、causal gate、usable steering 与 latency 均分开报告；跨基础模型、SAE 和层的迁移性仍需额外实验。", "The current leaderboard uses three independent runs per configuration, but still covers only 20 expert features from two layers of Gemma 2 9B. Continuous target scores and causal/usable thresholds answer different questions, so exact recovery, activation quality, continuous steering effect, causal gate, usable steering, and latency remain separate. Transfer across base models, SAEs, and layers requires additional experiments.")}</p>
          </div>
          <div className="references"><span>{tx("参考", "References")}</span><ol><li><a href="https://deepmind.google/models/gemma/gemma-scope/">Google DeepMind, Gemma Scope</a></li><li><a href="https://www.anthropic.com/research/evaluating-feature-steering">Anthropic, Evaluating feature steering</a></li><li><a href="https://www.anthropic.com/news/mapping-mind-language-model">Anthropic, Mapping the mind of a large language model</a></li></ol></div>
        </section>

        <section id="conclusion" className="article-section conclusion-section">
          <div className="article-copy">
            <h2>{tx("6. 结论", "6. Conclusion")}</h2>
            <p>{tx("实验表明，Agent 可以通过自主构造 probes 定位部分 Expert feature，也会频繁找到语义合理且自然激活相近的替代方向。后者并不自动具有相同的因果作用：在非 exact 候选中，activation fidelity 与 steering 的相关性明显下降。", "The experiments show that agents can recover some expert features by authoring their own probes, while frequently identifying semantically plausible alternatives with similar natural activations. These alternatives do not automatically share the same causal effect: among non-exact candidates, the association between activation fidelity and steering is substantially weaker.")}</p>
            <p>{tx("因此，自主 SAE 解释不能由单个 Feature ID 或一组高激活样例充分证明。Exact recovery、隐藏样例上的激活一致性、相对 baseline 与随机方向的因果效果，以及原任务保留，构成了互补且可独立审计的证据。", "Autonomous SAE interpretation therefore cannot be established by a single feature ID or a collection of highly activating examples alone. Exact recovery, activation agreement on hidden cases, causal effects against baseline and random directions, and preservation of the original task provide complementary, independently auditable evidence.")}</p>
          </div>
        </section>
      </article>

      <footer><span>SAEScientist-Bench · 2026</span><span>{tx("基于官方 Google Gemma Scope SAE", "Built on the official Google Gemma Scope SAE")}</span></footer>
    </main>
  );
}

export default Home;
