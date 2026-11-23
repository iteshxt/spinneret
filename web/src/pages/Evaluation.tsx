import { useQuery } from "@tanstack/react-query";
import { Fragment, useState } from "react";
import { api } from "../api/client";
import type { Scores, Summary, TierSummary } from "../api/types";
import { PageHead, Reveal } from "../components/Motion";
import { params, pct, secs } from "../format";

const TARGET = { em: 0.65, ex: 0.72 }; // T3 targets (PRD)
const LEVELS = ["easy", "medium", "hard", "extra"] as const;

const short = (tier: string) => tier.split(" ")[0] + (tier.includes("served") ? "*" : "");

function Delta({ from, to }: { from: number | undefined; to: number | undefined }) {
  if (from == null || to == null) return <>—</>;
  const d = (to - from) * 100;
  const cls = d > 0.05 ? "delta-pos" : d < -0.05 ? "delta-neg" : "muted";
  return <span className={cls}>{`${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}`}</span>;
}

function TierBars({ tiers }: { tiers: TierSummary[] }) {
  const scale = (x: number) => `${Math.max(0, Math.min(1, x)) * 100}%`;
  return (
    <div className="panel">
      <div className="bars">
        {tiers.map((t) => {
          const s = t.scores;
          const lat = t.latency;
          const total = lat?.latency_ms.total;
          const size = lat ? lat.generator_params_m + (lat.ranker_params_m ?? 0) : null;
          return (
            <div className="bar-row" key={t.tier}>
              <div className="name">
                {t.tier}
                <small>{t.description}</small>
              </div>
              <div className="bar-track" aria-hidden>
                <div className="bar em" style={{ width: scale(s?.em ?? 0) }}>
                  <span className="bar-label">EM {pct(s?.em)}</span>
                </div>
                <div className="bar" style={{ width: scale(s?.ex ?? 0) }}>
                  <span className="bar-label">EX {pct(s?.ex)}</span>
                </div>
                <div className="target-line" style={{ left: scale(TARGET.ex) }} />
              </div>
              <div className="meta">
                {params(size)} · p50 {secs(total?.p50)}
                <br />
                {lat?.peak_gpu_memory_mb ? `${(lat.peak_gpu_memory_mb / 1024).toFixed(2)} GB GPU` : ""}
              </div>
            </div>
          );
        })}
      </div>
      <div className="legend">
        <span>
          <i style={{ background: "color-mix(in srgb, var(--accent) 45%, var(--surface-2))" }} />
          exact match (EM)
        </span>
        <span>
          <i style={{ background: "var(--accent)" }} />
          execution accuracy (EX)
        </span>
        <span>┊ T3 target {pct(TARGET.ex, 0)} EX</span>
      </div>
    </div>
  );
}

function ablationRows(s: Summary) {
  const run = (id: string) =>
    [...s.tiers.map((t) => ({ run: t.run, scores: t.scores })),
     ...s.e2_decoding, ...s.training_ablations.flatMap((a) => a.variants)].find(
      (x) => x.run === id,
    )?.scores ?? undefined; // prettier-ignore
  const e2 = s.e2_decoding;
  const row = (name: string, change: string, from?: Scores, to?: Scores) => ({ name, change, from, to });
  return [
    row("E2 decoding", "greedy → execution-guided beam 8 (T3)", e2[0]?.scores ?? undefined, e2.at(-1)?.scores ?? undefined),
    row("E3 values", "off → on (T1)", run("ablations/t5-small-baseline-novalues-dev-eg4"), run("t5-small-baseline-dev-eg4")),
    row("E4 ranker", "full schema → ranked schema (T5-small, SQL)", run("t5-small-baseline-dev-eg4"), run("ablations/t5-small-ranked-noskel-dev-eg4")),
    row("E4 skeleton", "SQL → skeleton + SQL (T5-small, ranked)", run("ablations/t5-small-ranked-noskel-dev-eg4"), run("t5-small-ranked-dev-eg4")),
    row("E7 size", "T5-small → T5-base (ranked, skeleton)", run("t5-small-ranked-dev-eg4"), run("t5-base-ranked-dev-eg8")),
    row("E6 train_others", "off → on (T1)", run("ablations/t5-small-baseline-nothers-dev-eg4"), run("t5-small-baseline-dev-eg4")),
  ]; // prettier-ignore
}

function Taxonomy({ tiers }: { tiers: TierSummary[] }) {
  const withTax = tiers.filter((t) => t.taxonomy);
  const [tier, setTier] = useState(withTax.at(-1)?.tier ?? "");
  const [open, setOpen] = useState<string | null>(null);
  const t = withTax.find((x) => x.tier === tier);
  const tax = t?.taxonomy;
  if (!tax) return null;
  const rows = Object.entries(tax.primary)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1]);
  const wrong = tax.ex_wrong;
  return (
    <>
      <div className="section-head">
        <div>
          <h2>Failure taxonomy (primary label)</h2>
          <p className="muted" style={{ margin: 0 }}>
            {wrong} questions with the wrong result
            {tax.ranker_miss != null && ` · ${tax.ranker_miss} where the ranker dropped a needed item`}
            . Expand a row for an example.
          </p>
        </div>
        <div className="seg" role="group" aria-label="Tier">
          {withTax.map((x) => (
            <button key={x.tier} type="button" aria-pressed={x.tier === tier} onClick={() => setTier(x.tier)}>
              {short(x.tier)}
            </button>
          ))}
        </div>
      </div>
      <div className="panel scroll-x">
        <table className="data">
          <thead>
            <tr>
              <th scope="col">Category</th>
              <th scope="col" className="num">Count</th>
              <th scope="col" className="num">Share</th>
              <th scope="col">Example</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([cat, n]) => {
              const ex = tax.examples_per_category?.[cat]?.[0];
              const isOpen = open === cat;
              return (
                <Fragment key={cat}>
                  <tr className={cat === "semantically_equivalent" ? "ref" : undefined}>
                    <td className="mono">{cat}</td>
                    <td className="num">{n}</td>
                    <td className="num">{cat === "semantically_equivalent" ? "—" : pct(n / wrong)}</td>
                    <td>
                      {ex && (
                        <button
                          type="button"
                          className="btn"
                          aria-expanded={isOpen}
                          onClick={() => setOpen(isOpen ? null : cat)}
                          style={{ maxWidth: 360, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", display: "block" }}
                        >
                          {isOpen ? "▾" : "▸"} {ex.question}
                        </button>
                      )}
                    </td>
                  </tr>
                  {isOpen && ex && (
                    <tr>
                      <td colSpan={4} style={{ padding: 0 }}>
                        <div className="example-pair">
                          <div>
                            <span className="label">Gold</span>
                            <pre className="mono">{ex.gold_sql}</pre>
                          </div>
                          <div>
                            <span className="label">Predicted</span>
                            <pre className="mono">{ex.pred_sql}</pre>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ fontSize: 12 }}>
        Labels come from comparing the parsed predicted and gold SQL, first difference in the
        order shown in the specs (§8.4). <code>semantically_equivalent</code> counts predictions written
        differently from the gold SQL (exact match fails) that still return the right result; they
        are not errors.
      </p>
    </>
  );
}

export function Evaluation() {
  const summary = useQuery({ queryKey: ["summary"], queryFn: api.summary });
  if (summary.isPending) return <div className="page"><div className="loader" /></div>;
  if (summary.error || !summary.data)
    return (
      <div className="page">
        <h1>How well it works</h1>
        <p className="page-lead">No evaluation summary yet: run <code>spinneret summarize</code>.</p>
      </div>
    );
  const s = summary.data;
  const tiers = s.tiers.filter((t) => t.scores);
  const t3 = tiers.find((t) => t.tier.startsWith("T3"));
  const cov = s.e5_ranker_k["4x5"]?.dev_coverage;
  const k2 = ["4", "5", "6"];

  return (
    <div className="page">
      <PageHead title="How well it works">
        <p>
          Every model was graded on Spider's development set: 1,034 questions over 20 databases
          none of them saw in training, scored with the official evaluation scripts. Speed was
          measured question by question on a laptop GPU (GTX 1650).
        </p>
      </PageHead>

      <Reveal className="tiles">
        <div className="panel tile">
          <div className="label">T3 execution accuracy</div>
          <div className="value">{pct(t3?.scores?.ex)}</div>
          <div className="sub">target {pct(TARGET.ex, 0)}</div>
        </div>
        <div className="panel tile">
          <div className="label">T3 exact match</div>
          <div className="value">{pct(t3?.scores?.em)}</div>
          <div className="sub">target {pct(TARGET.em, 0)}</div>
        </div>
        <div className="panel tile">
          <div className="label">Ranker coverage</div>
          <div className="value">{pct(cov, 1)}</div>
          <div className="sub">all gold items kept, top 4 tables × 5 columns</div>
        </div>
        <div className="panel tile">
          <div className="label">T3 latency p50</div>
          <div className="value">{secs(t3?.latency?.latency_ms.total.p50)}</div>
          <div className="sub">p95 {secs(t3?.latency?.latency_ms.total.p95)} · laptop GPU</div>
        </div>
      </Reveal>

      <Reveal as="section">
        <h2>Accuracy, size and speed</h2>
        <TierBars tiers={tiers} />
        {tiers.some((t) => t.tier.includes("served")) && (
          <p className="muted" style={{ fontSize: 12 }}>
            * The demo serves Lite without the skeleton: on dev it matches the planned Lite on EM,
            beats it on EX and valid SQL, and generates faster (ablation E4).
          </p>
        )}
      </Reveal>

      <Reveal as="section">
        <h2>By difficulty (EM / EX)</h2>
        <div className="panel scroll-x">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Tier</th>
                {LEVELS.map((l) => <th key={l} scope="col" className="num">{l}</th>)}
                <th scope="col" className="num">valid SQL</th>
              </tr>
            </thead>
            <tbody>
              {tiers.map((t) => (
                <tr key={t.tier}>
                  <td>{t.tier}</td>
                  {LEVELS.map((l) => (
                    <td key={l} className="num mono">
                      {pct(t.scores!.em_by_level[l], 0)} / {pct(t.scores!.ex_by_level[l], 0)}
                    </td>
                  ))}
                  <td className="num mono">{pct(t.scores!.valid_sql, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>

      <Reveal as="section">
        <h2>By question type (EX)</h2>
        <div className="panel scroll-x">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Type</th>
                <th scope="col" className="num">n</th>
                {tiers.map((t) => <th key={t.tier} scope="col" className="num">{short(t.tier)}</th>)}
              </tr>
            </thead>
            <tbody>
              {Object.keys(tiers[0]?.question_types?.types ?? {}).map((qt) => (
                <tr key={qt}>
                  <td>{qt}</td>
                  <td className="num">{tiers[0].question_types!.types[qt].count}</td>
                  {tiers.map((t) => (
                    <td key={t.tier} className="num mono">{pct(t.question_types?.types[qt]?.ex)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>

      <Reveal as="section">
        <h2>Ablations</h2>
        <p className="muted" style={{ margin: "0 0 8px" }}>
          Each row changes one thing; the difference is in percentage points.
        </p>
        <div className="panel scroll-x">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Experiment</th>
                <th scope="col">Change</th>
                <th scope="col" className="num">EM</th>
                <th scope="col" className="num">EX</th>
                <th scope="col" className="num">Δ EM</th>
                <th scope="col" className="num">Δ EX</th>
              </tr>
            </thead>
            <tbody>
              {ablationRows(s).map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  <td className="muted">{r.change}</td>
                  <td className="num mono">{pct(r.from?.em)} → {pct(r.to?.em)}</td>
                  <td className="num mono">{pct(r.from?.ex)} → {pct(r.to?.ex)}</td>
                  <td className="num mono"><Delta from={r.from?.em} to={r.to?.em} /></td>
                  <td className="num mono"><Delta from={r.from?.ex} to={r.to?.ex} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>

      <Reveal as="section">
        <h2>E5 · ranker top-k at inference (T3, greedy): EX (dev coverage)</h2>
        <div className="panel scroll-x">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">tables \ columns</th>
                {k2.map((c) => <th key={c} scope="col" className="num">{c}</th>)}
              </tr>
            </thead>
            <tbody>
              {["3", "4", "5"].map((k1) => (
                <tr key={k1}>
                  <td>{k1}</td>
                  {k2.map((c) => {
                    const v = s.e5_ranker_k[`${k1}x${c}`];
                    const main = k1 === "4" && c === "5";
                    return (
                      <td key={c} className="num mono" style={main ? { fontWeight: 600 } : undefined}>
                        {pct(v?.scores?.ex)} <span className="muted">({pct(v?.dev_coverage)})</span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Reveal>

      <Reveal as="section">
        <Taxonomy tiers={tiers} />
      </Reveal>
    </div>
  );
}
