import { ReactNode, useMemo, useState } from "react";
import leaderboard from "../results/leaderboard.json";

type Language = "en" | "zh";
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
      agent: tx("8/8 exact", "8/8 exact"),
      activation: [41.93, 5.92, 0, 0.996],
      steering: [0.594, 0.60],
      note: tx("正例覆盖家猫身份与行为；困难负例包括单独出现的 kitty、名字 Kitty 与比喻性 purr。Expert feature 的正例平均 activation 为 41.93，困难负例为 5.92，中性文本为 0；8 个 Agent 均恢复了同一 ID。", "Positive cases cover domestic-cat identity and behavior. Hard negatives include isolated uses of kitty, the name Kitty, and metaphorical purr. The expert averages 41.93 activation on positives, 5.92 on hard negatives, and zero on neutral text; all eight agents recovered the same ID."),
    },
    {
      concept: tx("法语文本", "French-language text"),
      layer: "Gemma 2 9B · layer 9",
      expert: 105738,
      agent: tx("0/8 exact · 4 个替代 ID", "0/8 exact · 4 alternative IDs"),
      activation: [11.39, 2.66, 0, 1.0],
      steering: [0.925, 0.90],
      note: tx("正例是在日常、科学、程序和叙事主题中的连贯法语；对照集包含英语中的法国话题、孤立法语词、西班牙语与意大利语。Expert 的 activation 分离清晰。Agent 提交了 108861、43576、78422 与 130037 四个替代 ID；这些替代项在隐藏 steering 上均为 0。", "Positive cases contain coherent French across everyday, scientific, procedural, and narrative domains. Controls include English discussion of France, isolated French words, Spanish, and Italian. The expert separates the activation sets cleanly. Agents submitted four alternatives—108861, 43576, 78422, and 130037—and all produced zero hidden steering effect."),
    },
    {
      concept: tx("报税语言", "Tax-filing language"),
      layer: "Gemma 2 9B · layer 9",
      expert: 18713,
      agent: tx("8/8 都提交 64827", "8/8 submitted 64827"),
      activation: [19.07, 1.02, 0.47, 0.990],
      steering: [0.644, 0.025],
      note: tx("正例包含报税指令、应税收入、抵扣、税收抵免与表格；困难负例仅包含一般性的 tax 提及。8 个 Agent 均提交了 64827，Expert ID 为 18713。替代项达到 0.934 GT-normalized activation 和 0.613 steering effect，但未通过 usable gate，构成一致的替代方向。", "Positive cases contain filing instructions, taxable income, deductions, credits, and forms; hard negatives contain only general tax references. All eight agents submitted 64827, while the expert ID is 18713. The alternative reaches 0.934 GT-normalized activation and 0.613 steering effect, but does not pass the usable gate, forming a consistent alternative direction."),
    },
    {
      concept: tx("临床症状报告", "Clinical symptom reports"),
      layer: "Gemma 2 9B · layer 20",
      expert: 3927,
      agent: tx("0/8 exact · 5/8 选 113440", "0/8 exact · 5/8 chose 113440"),
      activation: [21.79, 1.74, 0, 1.0],
      steering: [0.894, 0.0],
      note: tx("正例描述患者症状、起病时间、严重程度与病史；一般医学词汇和医院运营文本构成对照。Expert 的 activation AUROC 为 1.0，target induction 较高，usable rate 为 0，表明干预破坏了原任务。Agent 最常提交 113440；另一个候选 53882 达到 0.916 GT-normalized activation 与 0.769 steering effect。", "Positive cases describe patient symptoms, onset, severity, and clinical history; generic medical vocabulary and hospital operations form the controls. The expert has activation AUROC 1.0 and strong target induction, with a zero usable rate because the intervention disrupts the requested task. Agents most often submitted 113440; candidate 53882 reached 0.916 GT-normalized activation and 0.769 steering effect."),
    },
    {
      concept: tx("天气预报", "Weather forecasts"),
      layer: "Gemma 2 9B · layer 20",
      expert: 44494,
      agent: tx("6/8 exact", "6/8 exact"),
      activation: [39.17, 2.85, 0.44, 1.0],
      steering: [0.644, 0.175],
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

function FormalLeaderboard({ language }: { language: Language }) {
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;
  const ordered = [...configurations].sort((a, b) => b.macro_gt_normalized_activation - a.macro_gt_normalized_activation);
  return (
    <div className="table-scroll"><table className="formal-table"><thead><tr><th>#</th><th>{tx("模型", "Model")}</th><th>{tx("GT 激活", "GT activation")}</th><th>Exact</th><th>Causal</th><th>Usable</th><th>{tx("中位耗时", "Median time")}</th></tr></thead><tbody>{ordered.map((row, index) => (
      <tr key={row.configuration}><td>{index + 1}</td><td><strong>{displayName(row)}</strong><small>{row.harness}</small></td><td>{row.macro_gt_normalized_activation.toFixed(3)}</td><td>{percentage(row.exact_match_rate)}</td><td>{percentage(row.causal_steering_rate)}</td><td>{percentage(row.usable_steering_rate)}</td><td>{(row.median_elapsed_seconds / 60).toFixed(1)} min</td></tr>
    ))}</tbody></table></div>
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
          <h1>{tx("SAE-Bench：评估 Agent 的自主 SAE 可解释性研究能力", "SAE-Bench: Evaluating agents as autonomous SAE researchers")}</h1>
          <p className="dek">{tx("我们评估 Agent 能否像研究者一样提出对照、设计实验、搜索并解释 SAE feature，再用 activation 与 steering 证据验证自己的结论。", "We evaluate whether an agent can work like a researcher: form contrasts, design experiments, discover and interpret an SAE feature, then validate its claim with activation and steering evidence.")}</p>
          <div className="article-authors"><p className="author-names">SAE-Bench Research Team</p></div>
          <p className="article-meta">{tx("2026 年 9 月", "September 2026")} <span>·</span> {tx("预计阅读时间：18 分钟", "18 min read")} <span>·</span> {summary.tasks} tasks <span>·</span> {runs.length} runs</p>
          <p className="article-links"><a href="https://github.com/Trae1ounG/SAE-Bench">Code &amp; data</a> <span>·</span> <a href="https://huggingface.co/google/gemma-scope-9b-it-res">Official SAE</a></p>
        </section>

        <section className="abstract-block">
          <h2>Abstract</h2>
          <p>{tx("稀疏自编码器把模型内部状态分解成大量 feature，但从语义假设走到可复现的 Feature ID，仍是一项需要构造数据、运行实验、解释证据并排除替代解释的科学研究工作。SAE-Bench 测试 Agent 能否自主完成这条研究链。在 20 个官方 Gemma Scope expert features 上，Agent 经常能找到语义接近的方向，但 exact recovery 与等价 steering 明显更难。", "Sparse autoencoders decompose model states into a large feature dictionary, but moving from a semantic hypothesis to a reproducible feature ID remains a scientific workflow: construct data, run experiments, interpret evidence, and rule out alternatives. SAE-Bench tests whether an agent can execute that research loop autonomously. Across 20 expert features from the official Gemma Scope release, semantic neighbors are common; exact recovery and equivalent steering are much harder.")}</p>
        </section>

        <aside className="table-of-contents"><details open><summary>{tx("目录", "Table of Contents")}</summary><ol><li><a href="#idea">{tx("我们究竟测什么能力", "What capability are we measuring?")}</a></li><li><a href="#features">{tx("Expert feature 与案例", "Expert features and cases")}</a></li><li><a href="#results">{tx("主要实验结果", "Main experimental results")}</a></li><li><a href="#analysis">{tx("语义—因果鸿沟", "The semantic–causal gap")}</a></li><li><a href="#method">{tx("有效性边界与局限", "Validity and limitations")}</a></li><li><a href="#conclusion">{tx("结论", "Conclusion")}</a></li></ol></details></aside>

        <aside className="claims-at-glance"><h2>{tx("一分钟了解", "In one minute")}</h2><div><p><b>{tx("任务", "Task")}</b><span>{tx("Agent 根据英文语义目标自主构造 probes，并从 131,072 个 SAE 方向中提交一个 Feature ID。", "From an English semantic target, an agent authors probes and submits one feature ID from 131,072 SAE directions.")}</span></p><p><b>{tx("证据", "Evidence")}</b><span>{tx("20 个官方 Gemma Scope expert features、8 个 Agent 配置、160 条完成且通过 trace audit 的运行。", "20 official Gemma Scope expert features, eight agent configurations, and 160 completed trace-audited runs.")}</span></p><p><b>{tx("结果", "Result")}</b><span>{tx("语义近邻在候选中普遍存在。Exact recovery、自然激活与因果 steering 的结果需要分别报告。", "Semantic neighbors are common among candidate features. Exact recovery, natural activation, and causal steering require separate measurement.")}</span></p></div></aside>

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
              <div><dt>{tx("隐藏评测", "Hidden evaluation")}</dt><dd>{tx("激活、GT rank、steering、PE judge", "Activation, GT rank, steering, and PE judge")}</dd></div>
            </dl>
          </div>
          <ResearchFigure wide number={1} title={tx("SAE-Bench 的自主研究流程。", "The SAE-Bench autonomous research loop.")} caption={tx("公开任务只提供研究目标和受限 probe；Expert ID、隐藏 case 与 steering 结果在提交后才用于评估。", "The task exposes only a research target and a restricted probe; expert IDs, hidden cases, and steering outcomes are used only after submission.")}><img className="mechanism-diagram" src={`${import.meta.env.BASE_URL}figures/feature-discovery-mechanism/diagram.svg`} alt={tx("SAE-Bench 自主研究流程图", "SAE-Bench autonomous research workflow")} /></ResearchFigure>
          <ResearchFigure wide number={2} title={tx("Feature 62610 的完整画像。", "A complete portrait of feature 62610.")} caption={tx("上半部分展示 feature 的语义与 32 条冻结 case 上的 activation 深度；下半部分展示同一 instruction 在 steering 前后的真实输出差异。", "The upper half shows the feature's interpretation and activation depth over 32 frozen cases; the lower half shows real before/after outputs for the same instruction.")}><FeaturePortrait language={language} /></ResearchFigure>
        </section>

        <section id="features" className="article-section">
          <div className="article-copy"><h2>{tx("2. Expert features 与候选方向的案例分析", "2. Case analysis of expert features and candidate directions")}</h2><p>{tx("20 个任务覆盖语言、主题与文体 feature。猫特征 62610 在所有配置中均被精确恢复；法语任务产生四个相近候选；报税任务的 8 个 Agent 则一致提交 64827，而 Expert ID 为 18713。后两个任务展示了 SAE 字典中语义相近但索引不同的候选方向。", "The 20 tasks cover language, topic, and register features. Cat feature 62610 is recovered exactly by every configuration; the French task produces four nearby candidates; and all eight agents submit 64827 for tax filing, while the expert ID is 18713. The latter two tasks expose semantically related directions with distinct indices in the SAE dictionary.")}</p></div>
          <ResearchFigure number={3} title={tx("五个 Feature ID 案例。", "Five feature-ID case studies.")} caption={tx("从 8/8 exact recovery，到稳定替代方向，再到强 steering 但低 usable，五个案例展示不同的成功与失败模式。", "The five cases span 8/8 exact recovery, stable alternative directions, and strong steering with low usability.")}><FeatureAtlas language={language} /></ResearchFigure>
          <div className="article-copy post-figure-copy">
            <p>{tx("French 与 clinical expert 在冻结数据上均呈现近乎完美的 activation 分离，Agent 仍会收敛到其他方向。这一结果表明，expert feature 的准入质量与 Agent 的精确恢复率需要独立评估；SAE 字典中的局部语义冗余会直接影响 Feature ID 定位。", "The French and clinical experts both show nearly perfect activation separation on frozen data, while agents converge on other directions. Expert admission quality and agent exact-recovery rate therefore require independent evaluation, with local semantic redundancy in the SAE dictionary directly affecting feature localization.")}</p>
            <p>{tx("Weather 候选 85606 的 target induction 高于 Expert 44494；clinical expert 则呈现较高 target induction 和 0 usable rate。两项观察共同支持分解式报告：exact、activation、causal effect 与 task preservation 分别对应不同的有效性证据。", "Weather candidate 85606 has higher target induction than expert 44494, while the clinical expert combines strong target induction with a zero usable rate. These observations support decomposed reporting: exact match, activation, causal effect, and task preservation measure distinct forms of validity.")}</p>
          </div>
          <div className="feature-index"><div className="feature-index-head"><span>{tx("20 个 Expert anchors", "20 expert anchors")}</span><small>{tx("官方 Gemma Scope · layer 9 / 20", "Official Gemma Scope · layers 9 / 20")}</small></div><div>{expertFeatures.map(([label, id]) => <span key={id}><small>{label}</small><code>{id}</code></span>)}</div></div>
        </section>

        <section id="results" className="article-section">
          <div className="article-copy"><h2>{tx("3. 20 个任务上的主要实验结果", "3. Main results across 20 tasks")}</h2><p>{tx(`当前公开快照覆盖 ${summary.tasks} 道题、${configurations.length} 个 Agent 配置与 ${runs.length} 条完整运行。主排序使用 expert-normalized activation。Exact、causal 与 usable 作为独立指标报告，分别衡量 ID 恢复、干预方向和任务保留。`, `The current public snapshot covers ${summary.tasks} tasks, ${configurations.length} agent configurations, and ${runs.length} complete runs. The primary ordering uses expert-normalized activation. Exact, causal, and usable outcomes are reported separately to measure ID recovery, intervention direction, and task preservation.`)}</p></div>
          <ResearchFigure number={4} title={tx("SAE-Bench v2 主榜。", "SAE-Bench v2 leaderboard.")} caption={tx("每个配置在每个任务上运行一次。该表描述当前固定协议下的观测结果；重复运行与置信区间仍待补充。", "Each configuration is run once per task. The table reports observations under the current frozen protocol; repeated runs and confidence intervals remain future work.")}><FormalLeaderboard language={language} /></ResearchFigure>
          <p className="status-note"><b>Claude Opus 5 High.</b> {tx("已接入同一离线协议，但完整 20 题结果尚未完成，因此当前不进入正式榜单。", "It is wired into the same offline protocol, but remains outside the formal table until all 20 tasks finish.")}</p>
        </section>

        <section id="analysis" className="article-section">
          <div className="article-copy"><h2>{tx("4. Activation fidelity 与因果 steering 的关系", "4. Relationship between activation fidelity and causal steering")}</h2><p>{tx("全部运行上的 activation—steering rank correlation 较高，部分原因是 exact features 会同时提高两个指标。将分析限制在非精确候选后，相关性显著降低。自然激活因此适合作为 discovery 指标，因果有效性仍需通过独立 steering 实验估计。", "Activation and steering have a relatively high rank correlation across all runs, partly because exact features increase both measures. The association is substantially weaker when the analysis is restricted to non-exact candidates. Natural activation is therefore suitable as a discovery metric, while causal validity still requires an independent steering experiment.")}</p></div>
          <div className="finding-grid"><div><span>Spearman · all</span><strong>{summary.allCorrelation.toFixed(3)}</strong><p>activation ↔ steering</p></div><div><span>Spearman · alternatives</span><strong>{summary.alternativeCorrelation.toFixed(3)}</strong><p>{tx("只看非精确候选", "non-exact candidates only")}</p></div><div><span>Mean causal effect</span><strong>{summary.exactEffect.toFixed(3)} <i>/</i> {summary.alternativeEffect.toFixed(3)}</strong><p>exact / alternative</p></div></div>
          <ResearchFigure wide number={5} title={tx("自然激活与因果效果。", "Natural activation versus causal effect.")} caption={tx("每个点是一条 Agent × task 运行；点击可查看 Feature ID 与 PE judge 结果。", "Each point is one agent-by-task run; select it to inspect the feature ID and PE judge outcomes.")}><ScatterExplorer language={language} /></ResearchFigure>
        </section>

        <section id="method" className="article-section methods-section">
          <div className="article-copy"><h2>{tx("5. 冻结任务，隐藏评测，公开 ID", "5. Frozen tasks, hidden evaluation, public IDs")}</h2><h3>{tx("Expert case 准入", "Expert-case admission")}</h3><p>{tx("每个目标都绑定官方 Gemma Scope SAE 中的一个 expert feature，并经过自然激活、matched-random steering、独立 calibration prompts、held-out prompts 与盲评验证。Feature ID 可以公开复现；隐藏的是评测 prompts、原始轨迹与基础设施。", "Each target is anchored to an expert feature from the official Gemma Scope SAE and admitted through natural-activation tests, matched-random steering, separate calibration prompts, held-out prompts, and blinded judging. Feature IDs are public for reproducibility; evaluation prompts, raw traces, and infrastructure remain hidden.")}</p><h3>{tx("Agent 隔离", "Agent isolation")}</h3><p>{tx("正式 discovery 禁止联网与搜索公开 feature labels，也不能读取 benchmark 仓库或 expert ID。Agent 唯一可用的外部反馈，是受限 probe 返回的激活与 rank。", "Scored discovery disables internet access and public-label search, and blocks access to the benchmark repository and expert IDs. The only external feedback is activation and rank returned by a restricted probe.")}</p><h3>{tx("限制", "Limitations")}</h3><p>{tx("当前榜单每个配置每题只有一次运行，且只覆盖 Gemma 2 9B 的两个层。Steering 的 judge 指标依然可能受 prompt 与解码随机性影响；因此 exact recovery、activation quality、causal stability、usable steering 与 latency 均分开报告。", "The current table has one run per configuration and task and covers only two layers of Gemma 2 9B. Steering judgments remain sensitive to prompts and decoding variance; exact recovery, activation quality, causal stability, usable steering, and latency are therefore reported separately.")}</p></div>
          <div className="references"><span>{tx("参考", "References")}</span><ol><li><a href="https://deepmind.google/models/gemma/gemma-scope/">Google DeepMind, Gemma Scope</a></li><li><a href="https://www.anthropic.com/research/evaluating-feature-steering">Anthropic, Evaluating feature steering</a></li><li><a href="https://www.anthropic.com/news/mapping-mind-language-model">Anthropic, Mapping the mind of a large language model</a></li></ol></div>
        </section>

        <section id="conclusion" className="article-section conclusion-section">
          <div className="article-copy">
            <h2>{tx("6. 结论", "6. Conclusion")}</h2>
            <p>{tx("在这个第一版协议里，Agent 已经能够构造有信息量的 probes，并在一部分 feature 上精确复现 expert；更常见的结果则是找到语义合理、激活很高、甚至可以 steering 的替代方向。SAE-Bench 的价值正是把这些情况拆开，让“提出解释”和“证明解释”成为两个可以独立审计的能力。", "In this first protocol, agents can author informative probes and exactly reproduce the expert on some features. More often, they find an alternative direction that is semantically plausible, highly activating, and sometimes steerable. SAE-Bench separates these outcomes so that proposing an interpretation and proving it become independently auditable capabilities.")}</p>
            <p>{tx("后续实验将扩大经过准入验证的 expert set，对每个配置执行重复运行，并检验该研究流程在更多模型、层与 SAE 上的迁移性。", "Future experiments will expand the admitted expert set, repeat each configuration, and evaluate transfer across additional models, layers, and SAEs.")}</p>
          </div>
        </section>

        <section className="closing"><span className="eyebrow">SUMMARY</span><blockquote>{tx("自主 Feature Discovery 的有效性需要同时由语义定位、隐藏激活与因果干预支持。", "Valid autonomous feature discovery requires convergent evidence from semantic localization, hidden activation, and causal intervention.")}</blockquote><div><a href="https://github.com/Trae1ounG/SAE-Bench">{tx("代码、Feature IDs 与结果", "Code, feature IDs, and results")} ↗</a></div></section>
      </article>

      <footer><span>SAE-Bench · 2026</span><span>{tx("基于官方 Google Gemma Scope SAE", "Built on the official Google Gemma Scope SAE")}</span></footer>
    </main>
  );
}

export default Home;
