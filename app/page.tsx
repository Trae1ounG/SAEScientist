import { CSSProperties, ReactNode, useMemo, useState } from "react";
import leaderboard from "../results/leaderboard.json";

type Language = "en" | "zh";
type Metric = "activation" | "exact" | "causal" | "usable";
type MatrixMetric = "activation" | "steering" | "exact";
type PointFilter = "all" | "exact" | "alternative";

type Configuration = {
  configuration: string;
  harness: string;
  model: string;
  reasoning_effort: string | null;
  benchmark_tasks: number;
  completed_tasks: number;
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

function ResearchFigure({
  number,
  title,
  caption,
  children,
}: {
  number: number;
  title: string;
  caption: string;
  children: ReactNode;
}) {
  return (
    <figure className="research-figure">
      <div className="figure-frame">{children}</div>
      <figcaption><b>Figure {number}.</b> <strong>{title}</strong> {caption}</figcaption>
    </figure>
  );
}

const configurations = leaderboard.configurations as Configuration[];
const runs = leaderboard.runs as Run[];

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

const expertFeatures = [
  ["cat", 62610], ["French", 105738], ["Spanish", 94329], ["Portuguese", 41424], ["German", 33987],
  ["earnings · L9", 131024], ["gardening", 93406], ["tax · L9", 18713], ["job postings", 91086], ["Latin", 7659],
  ["portfolio · L9", 77390], ["Turkish", 99383], ["clinical", 3927], ["earnings · L20", 100747], ["portfolio · L20", 101617],
  ["pharma", 104583], ["tax · L20", 78694], ["real estate", 27182], ["weather", 44494], ["archaeology", 7256],
] as const;

const seriesClasses = ["series-a", "series-b", "series-c", "series-d", "series-e", "series-f", "series-g", "series-h", "series-i"];

function displayName(row: Pick<Configuration, "model">): string {
  return modelNames[row.model] ?? row.model;
}

function configurationLabel(key: string): string {
  const row = configurations.find((candidate) => candidate.configuration === key);
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

function metricValue(row: Configuration, metric: Metric): number {
  if (metric === "activation") return row.macro_gt_normalized_activation;
  if (metric === "exact") return row.exact_match_rate;
  if (metric === "causal") return row.causal_steering_rate;
  return row.usable_steering_rate;
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
        <div className="steering-summary"><div><strong>0.000 → 0.594</strong><span>{tx("held-out target score", "held-out target score")}</span></div><div><strong>70%</strong><span>{tx("目标成功率", "target success rate")}</span></div><div><strong>60%</strong><span>{tx("可用输出率", "usable output rate")}</span></div><div><strong>0%</strong><span>matched-random success</span></div></div>
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
      candidate: 62610,
      result: tx("8/8 Agent 精确命中", "8/8 agents found the exact ID"),
      note: tx("语义边界清晰，positive 与动物/品牌 hard negatives 容易分开。", "A crisp boundary separates animal positives from brand and metaphor hard negatives."),
    },
    {
      concept: tx("法语文本", "French-language text"),
      layer: "Gemma 2 9B · layer 9",
      expert: 105738,
      candidate: 108861,
      result: tx("0/8 精确；出现多个强替代项", "0/8 exact; several strong alternatives"),
      note: tx("108861、43576、78422 与 130037 都能表现出部分法语语义，但不是同一个 SAE 方向。", "108861, 43576, 78422, and 130037 capture parts of French semantics without reproducing the expert direction."),
    },
    {
      concept: tx("报税语言", "Tax-filing language"),
      layer: "Gemma 2 9B · layer 9",
      expert: 18713,
      candidate: 64827,
      result: tx("8/8 选择同一替代项", "8/8 agents chose the same alternative"),
      note: tx("替代项的 GT-normalized activation 约为 0.934，steering effect 为 0.613：看起来很像，也确实有因果作用，但方向仍不等价。", "The alternative reaches about 0.934 GT-normalized activation and 0.613 steering effect: semantically close and causal, yet still directionally distinct."),
    },
  ];
  return (
    <div className="feature-atlas">{cases.map((item, index) => <div className="feature-case" key={item.expert}>
      <span className="case-number">0{index + 1}</span>
      <div className="feature-case-copy"><span className="eyebrow">{item.layer}</span><h3>{item.concept}</h3><p>{item.note}</p><strong>{item.result}</strong></div>
      <div className="id-comparison"><div><span>Expert</span><code>{item.expert}</code></div><i aria-hidden="true">{item.expert === item.candidate ? "=" : "≠"}</i><div><span>Agent</span><code>{item.candidate}</code></div></div>
    </div>)}
    </div>
  );
}

function FormalLeaderboard({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const ordered = [...configurations].sort((a, b) => b.macro_gt_normalized_activation - a.macro_gt_normalized_activation);
  return (
    <div className="table-scroll"><table className="formal-table"><thead><tr><th>#</th><th>{tx("模型", "Model")}</th><th>{tx("GT 激活", "GT activation")}</th><th>Exact</th><th>Causal</th><th>Usable</th><th>{tx("中位耗时", "Median time")}</th></tr></thead><tbody>{ordered.map((row, index) => (
      <tr key={row.configuration}><td>{index + 1}</td><td><strong>{displayName(row)}</strong><small>{row.harness}</small></td><td>{row.macro_gt_normalized_activation.toFixed(3)}</td><td>{percentage(row.exact_match_rate)}</td><td>{percentage(row.causal_steering_rate)}</td><td>{percentage(row.usable_steering_rate)}</td><td>{(row.median_elapsed_seconds / 60).toFixed(1)} min</td></tr>
    ))}</tbody></table></div>
  );
}

function Leaderboard({ language }: { language: Language }) {
  const [metric, setMetric] = useState<Metric>("activation");
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const ordered = [...configurations].sort((a, b) => metricValue(b, metric) - metricValue(a, metric));
  const metricLabels: Record<Metric, string> = {
    activation: tx("GT 激活", "GT activation"),
    exact: tx("精确命中", "Exact match"),
    causal: tx("因果通过", "Causal pass"),
    usable: tx("可用通过", "Usable pass"),
  };
  return (
    <div className="interactive-panel">
      <div className="panel-head">
        <div>
          <span className="eyebrow">FIGURE EXPLORER</span>
          <h3>{tx("按不同指标重新查看模型", "Re-rank models by each metric")}</h3>
          <p>{tx("正式主榜固定按 GT activation 排序；这里可以观察不同指标怎样改变相对位置。", "The official ordering stays fixed to GT activation; this view reveals how other criteria change the order.")}</p>
        </div>
        <MetricTabs
          value={metric}
          onChange={setMetric}
          label={tx("榜单指标", "Leaderboard metric")}
          options={(["activation", "exact", "causal", "usable"] as Metric[]).map((key) => ({ key, label: metricLabels[key] }))}
        />
      </div>
      <div className="ranking-list">
        {ordered.map((row, index) => {
          const value = metricValue(row, metric);
          return (
            <div className="ranking-row" key={row.configuration}>
              <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
              <div className="model-label"><strong>{displayName(row)}</strong><small>{row.harness} · {row.completed_tasks}/{row.benchmark_tasks}</small></div>
              <div className="rank-track" aria-hidden="true"><i style={{ width: `${Math.max(value * 100, 1)}%` }} /></div>
              <strong className="rank-value">{metric === "activation" ? value.toFixed(3) : percentage(value)}</strong>
            </div>
          );
        })}
      </div>
    </div>
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
          <h3>{tx("自然激活相似，不等于 steering 相同", "Activation similarity is not steering equivalence")}</h3>
          <p>{tx("每个点是一条 Agent × 题目运行。点击点查看它的激活、steering 和 PE 结果。", "Each point is one agent-by-task run. Select a point to inspect activation, steering, and PE outcomes.")}</p>
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
        <svg className="scatter" viewBox="0 0 940 470" role="img" aria-label={tx("GT 激活与 steering 效果散点图", "Scatter plot of GT activation and steering effect")}>
          <title>{tx("GT 激活与 steering 效果", "GT activation versus steering effect")}</title>
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={`x-${tick}`}><line x1={x(tick)} x2={x(tick)} y1={top} y2={top + height} /><text x={x(tick)} y={top + height + 28} textAnchor="middle">{tick.toFixed(2)}</text></g>)}
          {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={`y-${tick}`}><line x1={left} x2={left + width} y1={y(tick)} y2={y(tick)} /><text x={left - 16} y={y(tick) + 4} textAnchor="end">{tick.toFixed(2)}</text></g>)}
          <text className="axis-title" x={left + width / 2} y={462} textAnchor="middle">{tx("GT-normalized activation", "GT-normalized activation")}</text>
          <text className="axis-title" transform={`translate(20 ${top + height / 2}) rotate(-90)`} textAnchor="middle">{tx("因果 steering effect", "Causal steering effect")}</text>
          {points.map((run, index) => {
            const configIndex = configurations.findIndex((row) => row.configuration === run.configuration);
            const active = selected === run;
            return (
              <circle
                key={`${run.configuration}-${run.task_id}`}
                className={`${seriesClasses[configIndex]} ${run.exact_match ? "exact-point" : "alternative-point"} ${active ? "selected-point" : ""}`}
                cx={x(run.gt_normalized_activation)}
                cy={y(run.steering_effect)}
                r={active ? 7 : run.exact_match ? 5 : 4}
                tabIndex={0}
                role="button"
                aria-label={`${configurationLabel(run.configuration)}, ${humanize(run.target)}, activation ${run.gt_normalized_activation.toFixed(3)}, steering ${run.steering_effect.toFixed(3)}`}
                onClick={() => setSelected(run)}
                onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") setSelected(run); }}
              >
                <title>{configurationLabel(run.configuration)} · {humanize(run.target)} · {run.gt_normalized_activation.toFixed(3)} / {run.steering_effect.toFixed(3)}</title>
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
              <div><dt>GT activation</dt><dd>{selected.gt_normalized_activation.toFixed(3)}</dd></div>
              <div><dt>Steering effect</dt><dd>{selected.steering_effect.toFixed(3)}</dd></div>
              <div><dt>PE relevance</dt><dd>{selected.pe_target_relevance.toFixed(2)} / 4</dd></div>
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

function quantileClass(value: number): string {
  return `q${Math.max(0, Math.min(5, Math.floor(value * 5.999)))}`;
}

function TaskMatrix({ language }: { language: Language }) {
  const [metric, setMetric] = useState<MatrixMetric>("activation");
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const orderedConfigs = [...configurations].sort((a, b) => b.macro_gt_normalized_activation - a.macro_gt_normalized_activation);
  const tasks = [...new Set(runs.map((run) => run.task_id))].sort();
  const lookup = new Map(runs.map((run) => [`${run.configuration}:${run.task_id}`, run]));
  const value = (run: Run) => metric === "activation" ? run.gt_normalized_activation : metric === "steering" ? run.steering_effect : Number(run.exact_match);
  return (
    <div className="interactive-panel matrix-panel">
      <div className="panel-head">
        <div>
          <span className="eyebrow">FIGURE EXPLORER</span>
          <h3>{tx("20 道题并不是同一种难度", "The 20 tasks are not equally difficult")}</h3>
          <p>{tx("行是 Agent，列是 task；颜色越深表示该指标越高。悬停方格查看精确数值。", "Rows are agents and columns are tasks; darker cells indicate higher values. Hover or focus a cell for the exact score.")}</p>
        </div>
        <MetricTabs
          value={metric}
          onChange={setMetric}
          label={tx("矩阵指标", "Matrix metric")}
          options={[
            { key: "activation", label: tx("GT 激活", "GT activation") },
            { key: "steering", label: "Steering" },
            { key: "exact", label: "Exact" },
          ]}
        />
      </div>
      <div className="matrix-scroll">
        <div className="task-matrix" style={{ "--task-count": tasks.length } as CSSProperties}>
          <div className="matrix-corner" />
          {tasks.map((task, index) => <div className="matrix-task" key={task}><span>{String(index + 1).padStart(2, "0")}</span><small>{humanize(task)}</small></div>)}
          {orderedConfigs.flatMap((config) => [
            <div className="matrix-model" key={`${config.configuration}-label`}><strong>{displayName(config)}</strong><small>{config.harness}</small></div>,
            ...tasks.map((task) => {
              const run = lookup.get(`${config.configuration}:${task}`)!;
              const score = value(run);
              return <button className={`matrix-cell ${quantileClass(score)}`} key={`${config.configuration}-${task}`} aria-label={`${displayName(config)}, ${humanize(task)}, ${score.toFixed(3)}`}><span>{metric === "exact" ? (run.exact_match ? "✓" : "·") : score.toFixed(2)}</span><small>{humanize(run.target)}</small></button>;
            }),
          ])}
        </div>
      </div>
      <div className="matrix-legend"><span>{tx("低", "Low")}</span>{[0, 1, 2, 3, 4, 5].map((level) => <i className={`q${level}`} key={level} />)}<span>{tx("高", "High")}</span></div>
    </div>
  );
}

function Home() {
  const [language, setLanguage] = useState<Language>("en");
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const summary = useMemo(() => {
    const alternatives = runs.filter((run) => !run.exact_match);
    const exact = runs.filter((run) => run.exact_match);
    return {
      tasks: new Set(runs.map((run) => run.task_id)).size,
      exact: exact.length,
      causal: runs.filter((run) => run.causal_stable).length,
      allCorrelation: spearman(runs),
      alternativeCorrelation: spearman(alternatives),
      exactEffect: average(exact.map((run) => run.steering_effect)),
      alternativeEffect: average(alternatives.map((run) => run.steering_effect)),
    };
  }, []);

  return (
    <main lang={language === "zh" ? "zh-CN" : "en"}>
      <header className="site-header">
        <a className="wordmark" href="#top"><strong>SAE</strong><span>BENCH</span></a>
        <nav><a href="#idea">{tx("核心想法", "Idea")}</a><a href="#features">Feature IDs</a><a href="#results">{tx("结果", "Results")}</a><a href="#method">{tx("方法", "Methods")}</a></nav>
        <div className="language-control" aria-label="Language"><button className={language === "zh" ? "active" : ""} onClick={() => setLanguage("zh")}>中文</button><button className={language === "en" ? "active" : ""} onClick={() => setLanguage("en")}>EN</button></div>
      </header>

      <article id="top" className="research-article">
        <section className="article-masthead">
          <span className="issue">SAE-BENCH · RESEARCH BLOG · V2</span>
          <h1>{tx("SAE-Bench：评估 Agent 的自主 SAE 可解释性研究能力", "SAE-Bench: Evaluating agents as autonomous SAE researchers")}</h1>
          <p className="dek">{tx("我们评估 Agent 能否像研究者一样提出对照、设计实验、搜索并解释 SAE feature，再用 activation 与 steering 证据验证自己的结论。", "We evaluate whether an agent can work like a researcher: form contrasts, design experiments, discover and interpret an SAE feature, then validate its claim with activation and steering evidence.")}</p>
          <div className="article-byline"><span>{tx("SAE-Bench 团队", "SAE-Bench team")}</span><span>September 2026</span><span>{summary.tasks} tasks · {runs.length} runs</span></div>
          <div className="hero-links"><a className="primary-link" href="#idea">{tx("阅读文章", "Read the article")}</a><a href="https://github.com/Trae1ounG/SAE-Bench">GitHub ↗</a><a href="https://huggingface.co/google/gemma-scope-9b-it-res">Gemma Scope ↗</a></div>
        </section>

        <section className="abstract-block">
          <span className="eyebrow">{tx("摘要", "Abstract")}</span>
          <p>{tx("稀疏自编码器把模型内部状态分解成大量 feature，但从语义假设走到可复现的 Feature ID，仍是一项需要构造数据、运行实验、解释证据并排除替代解释的科学研究工作。SAE-Bench 测试 Agent 能否自主完成这条研究链。在 20 个官方 Gemma Scope expert features 上，Agent 经常能找到语义接近的方向，但 exact recovery 与等价 steering 明显更难。", "Sparse autoencoders decompose model states into a large feature dictionary, but moving from a semantic hypothesis to a reproducible feature ID remains a scientific workflow: construct data, run experiments, interpret evidence, and rule out alternatives. SAE-Bench tests whether an agent can execute that research loop autonomously. Across 20 expert features from the official Gemma Scope release, semantic neighbors are common; exact recovery and equivalent steering are much harder.")}</p>
        </section>

        <aside className="table-of-contents"><span>{tx("目录", "Contents")}</span><ol><li><a href="#idea">{tx("从描述到可干预方向", "From description to intervention")}</a></li><li><a href="#features">{tx("Feature ID 到底代表什么", "What a feature ID represents")}</a></li><li><a href="#results">{tx("主榜结果", "Main results")}</a></li><li><a href="#analysis">{tx("语义—因果鸿沟", "The semantic–causal gap")}</a></li><li><a href="#method">{tx("方法与边界", "Methods and limits")}</a></li></ol></aside>

        <section id="idea" className="article-section">
          <div className="section-kicker">01 · {tx("核心想法", "The core idea")}</div>
          <div className="article-copy"><h2>{tx("我们测的不是检索，而是一条自主研究链。", "We evaluate a research loop, not retrieval.")}</h2><p>{tx("SAE 中的 feature ID 只是某层字典里的列号，没有天然标签。Agent 从英文研究目标出发，自主写 positive、hard-negative 与 neutral probes，调用受限实验接口，观察激活并反复修正假设，最终提交一个 ID 和解释。", "A feature ID is only a column index in one layer's SAE dictionary; it has no intrinsic label. Starting from an English research target, the agent authors positive, hard-negative, and neutral probes, queries a restricted experiment interface, revises its hypothesis from activations, and finally submits one ID with an interpretation.")}</p><p>{tx("Benchmark 随后在隐藏评测上复现实验，并把候选 decoder direction 写回残差流，与 expert 和 matched-random 对照。评分因此覆盖研究过程的三个层次：定位是否准确、解释是否吻合、干预是否沿相同方向改变行为。", "The benchmark then reproduces the experiment on hidden cases and writes the candidate decoder direction back into the residual stream against expert and matched-random controls. Scoring therefore covers three levels of research evidence: localization, interpretation, and causal intervention.")}</p></div>
          <ResearchFigure number={1} title={tx("SAE-Bench 的自主研究流程。", "The SAE-Bench autonomous research loop.")} caption={tx("公开任务只提供研究目标和受限 probe；Expert ID、隐藏 case 与 steering 结果在提交后才用于评估。", "The task exposes only a research target and a restricted probe; expert IDs, hidden cases, and steering outcomes are used only after submission.")}><img className="mechanism-diagram" src={`${import.meta.env.BASE_URL}figures/feature-discovery-mechanism/diagram.svg`} alt={tx("SAE-Bench 自主研究流程图", "SAE-Bench autonomous research workflow")} /></ResearchFigure>
          <ResearchFigure number={2} title={tx("Feature 62610 的完整画像。", "A complete portrait of feature 62610.")} caption={tx("上半部分展示 feature 的语义与 32 条冻结 case 上的 activation 深度；下半部分展示同一 instruction 在 steering 前后的真实输出差异。", "The upper half shows the feature's interpretation and activation depth over 32 frozen cases; the lower half shows real before/after outputs for the same instruction.")}><FeaturePortrait language={language} /></ResearchFigure>
        </section>

        <section id="features" className="article-section">
          <div className="section-kicker">02 · FEATURE IDS</div>
          <div className="article-copy"><h2>{tx("同一种语义，可以对应多个不同方向。", "One semantic idea can occupy several different directions.")}</h2><p>{tx("这正是 exact match 有意义的原因。猫特征 62610 是一个容易复现的锚点；法语则出现多个相近方向；报税任务更极端——所有 Agent 都选择了 64827，而 expert 是 18713。替代项既有高激活也能 steering，却仍不能被称为同一个方向。", "This is why exact match matters. Cat feature 62610 is an easy-to-recover anchor; French produces several nearby directions; tax filing is more striking—every agent selected 64827 while the expert is 18713. The alternative activates strongly and can steer, yet it is still not the same direction.")}</p></div>
          <ResearchFigure number={3} title={tx("三个 Feature ID 案例。", "Three feature-ID cases.")} caption={tx("同一版式比较精确命中、语义近邻、以及具有因果作用但方向不等价的替代 feature。", "The same layout compares an exact recovery, semantic neighbors, and a causal but directionally distinct alternative.")}><FeatureAtlas language={language} /></ResearchFigure>
          <div className="feature-index"><div className="feature-index-head"><span>{tx("20 个 Expert anchors", "20 expert anchors")}</span><small>{tx("官方 Gemma Scope · layer 9 / 20", "Official Gemma Scope · layers 9 / 20")}</small></div><div>{expertFeatures.map(([label, id]) => <span key={id}><small>{label}</small><code>{id}</code></span>)}</div></div>
        </section>

        <section id="results" className="article-section">
          <div className="section-kicker">03 · {tx("实验结果", "Results")}</div>
          <div className="article-copy"><h2>{tx("激活排名给出了一张主榜，但不是一个总分。", "Activation ranking gives us a leaderboard—not a single truth.")}</h2><p>{tx(`当前公开快照覆盖 ${summary.tasks} 道题、${configurations.length} 个 Agent 配置与 ${runs.length} 条完整运行。主排序使用 expert-normalized activation；Exact、causal 与 usable 独立报告，因为它们回答不同问题。`, `The current public snapshot covers ${summary.tasks} tasks, ${configurations.length} agent configurations, and ${runs.length} complete runs. The primary ordering uses expert-normalized activation; exact, causal, and usable outcomes remain separate because they answer different questions.`)}</p></div>
          <ResearchFigure number={4} title={tx("SAE-Bench v2 主榜。", "SAE-Bench v2 leaderboard.")} caption={tx("每个配置每题一次运行；该表是同协议观察结果，不是带置信区间的稳定能力结论。", "One run per configuration and task; this is a controlled-protocol snapshot, not a confidence-bounded capability estimate.")}><FormalLeaderboard language={language} /></ResearchFigure>
          <p className="status-note"><b>Claude Opus 5 High.</b> {tx("已接入同一离线协议，但完整 20 题结果尚未完成，因此当前不进入正式榜单。", "It is wired into the same offline protocol, but remains outside the formal table until all 20 tasks finish.")}</p>
        </section>

        <section id="analysis" className="article-section">
          <div className="section-kicker">04 · {tx("行为分析", "Behavioral analysis")}</div>
          <div className="article-copy"><h2>{tx("“找得像”与“推得动”之间有一道鸿沟。", "There is a gap between looking right and steering right.")}</h2><p>{tx("把所有运行混在一起时，activation 与 steering 的 rank correlation 看起来较强；但 exact features 会同时把两个指标推高。只看非精确候选，关系显著变弱。这意味着自然激活适合做 discovery 主指标，却不能替代因果验证。", "Across all runs, activation and steering rank correlation looks strong; exact features, however, raise both measures together. Among non-exact candidates alone, the relationship weakens markedly. Natural activation is therefore useful as a discovery metric, but cannot replace causal validation.")}</p></div>
          <div className="finding-grid"><div><span>Spearman · all</span><strong>{summary.allCorrelation.toFixed(3)}</strong><p>activation ↔ steering</p></div><div><span>Spearman · alternatives</span><strong>{summary.alternativeCorrelation.toFixed(3)}</strong><p>{tx("只看非精确候选", "non-exact candidates only")}</p></div><div><span>Mean causal effect</span><strong>{summary.exactEffect.toFixed(3)} <i>/</i> {summary.alternativeEffect.toFixed(3)}</strong><p>exact / alternative</p></div></div>
          <ResearchFigure number={5} title={tx("自然激活与因果效果。", "Natural activation versus causal effect.")} caption={tx("每个点是一条 Agent × task 运行；点击可查看 Feature ID 与 PE judge 结果。", "Each point is one agent-by-task run; select it to inspect the feature ID and PE judge outcomes.")}><ScatterExplorer language={language} /></ResearchFigure>
        </section>

        <section id="method" className="article-section methods-section">
          <div className="section-kicker">05 · {tx("方法与边界", "Methods and limits")}</div>
          <div className="article-copy"><h2>{tx("冻结任务，隐藏评测，公开 ID。", "Frozen tasks, hidden evaluation, public IDs.")}</h2><h3>{tx("Expert case 准入", "Expert-case admission")}</h3><p>{tx("每个目标都绑定官方 Gemma Scope SAE 中的一个 expert feature，并经过自然激活、matched-random steering、独立 calibration prompts、held-out prompts 与盲评验证。Feature ID 可以公开复现；隐藏的是评测 prompts、原始轨迹与基础设施。", "Each target is anchored to an expert feature from the official Gemma Scope SAE and admitted through natural-activation tests, matched-random steering, separate calibration prompts, held-out prompts, and blinded judging. Feature IDs are public for reproducibility; evaluation prompts, raw traces, and infrastructure remain hidden.")}</p><h3>{tx("Agent 隔离", "Agent isolation")}</h3><p>{tx("正式 discovery 禁止联网与搜索公开 feature labels，也不能读取 benchmark 仓库或 expert ID。Agent 唯一可用的外部反馈，是受限 probe 返回的激活与 rank。", "Scored discovery disables internet access and public-label search, and blocks access to the benchmark repository and expert IDs. The only external feedback is activation and rank returned by a restricted probe.")}</p><h3>{tx("限制", "Limitations")}</h3><p>{tx("当前榜单每个配置每题只有一次运行，且只覆盖 Gemma 2 9B 的两个层。Steering 的 judge 指标依然可能受 prompt 与解码随机性影响；因此 exact recovery、activation quality、causal stability、usable steering 与 latency 均分开报告。", "The current table has one run per configuration and task and covers only two layers of Gemma 2 9B. Steering judgments remain sensitive to prompts and decoding variance; exact recovery, activation quality, causal stability, usable steering, and latency are therefore reported separately.")}</p></div>
          <div className="references"><span>{tx("参考", "References")}</span><ol><li><a href="https://deepmind.google/models/gemma/gemma-scope/">Google DeepMind, Gemma Scope</a></li><li><a href="https://www.anthropic.com/research/evaluating-feature-steering">Anthropic, Evaluating feature steering</a></li><li><a href="https://www.anthropic.com/news/mapping-mind-language-model">Anthropic, Mapping the mind of a large language model</a></li></ol></div>
        </section>

        <section className="closing"><span className="eyebrow">THE CENTRAL RESULT</span><blockquote>{tx("Agent 常常能找到一个“像”的 feature；真正困难的是找到同一个方向。", "Agents often find a feature that looks right. The hard part is finding the same direction.")}</blockquote><div><a href="https://github.com/Trae1ounG/SAE-Bench">{tx("代码、Feature IDs 与结果", "Code, feature IDs, and results")} ↗</a></div></section>
      </article>

      <footer><span>SAE-Bench · 2026</span><span>{tx("基于官方 Google Gemma Scope SAE", "Built on the official Google Gemma Scope SAE")}</span></footer>
    </main>
  );
}

export default Home;
