import { Link } from "react-router-dom";
import type { ModelInfo } from "../api/types";
import { params, pct, secs } from "../format";

const REPO = "https://github.com/iteshxt/spinneret";

// What each served model is (design §11.2). Numbers come from the API (latest evaluation).
const CARDS: Record<string, { base: string; input: string; target: string; purpose: string }> = {
  t1: {
    purpose: "Reference point: the smallest model reading the whole schema.",
    base: "T5-small (60M)",
    input: "question | database | every table and column, with matched values",
    target: "SQL",
  },
  t2: {
    purpose: "Fast mode: the small model, helped by the schema ranker.",
    base: "T5-small (60M) + RoBERTa-base ranker (126M)",
    input: "question | database | the ranker's top 4 tables × 5 columns, with matched values",
    target: "SQL (the skeleton-first variant did not help this size; ablation E4)",
  },
  t3: {
    purpose: "Best accuracy: a larger generator, the ranker and skeleton-first decoding.",
    base: "T5-base (223M) + RoBERTa-base ranker (126M)",
    input: "question | database | the ranker's top 4 tables × 5 columns, with matched values",
    target: "skeleton | SQL (the query's shape first, then the full query)",
  },
};

const hubUrl = (ref: string | null) =>
  ref?.startsWith("hub:") ? `https://huggingface.co/${ref.slice(4)}` : null;

export function About({ models }: { models: ModelInfo[] }) {
  return (
    <div className="page">
      <h1>About Spinneret</h1>
      <div className="prose">
        <p className="lead">
          Spinneret turns English questions into SQL for databases it has never seen, with models
          small enough to run on a laptop GPU. It was trained on the Spider benchmark and is
          evaluated on Spider's 20 held-out development databases — the same ones in this demo.
        </p>
        <p>
          Each answer goes through the same steps you see on the Query page: database values that
          match words in the question are looked up; for Lite and Spinneret a schema ranker scores
          every table and column and keeps the most relevant ones; the T5 model writes several
          candidate queries with beam search; and the first candidate that runs is shown
          (execution-guided decoding). Queries run read-only, with a 5-second limit and at most
          500 rows.
        </p>
      </div>

      <h2>Models</h2>
      <div className="cards">
        {models.map((m) => {
          const card = CARDS[m.id];
          const gen = hubUrl(m.generator);
          const rk = hubUrl(m.ranker);
          return (
            <article key={m.id} className="panel card">
              <h3>{m.label}</h3>
              {card && <p className="muted" style={{ margin: 0 }}>{card.purpose}</p>}
              <dl>
                {card && (
                  <>
                    <dt>Model</dt>
                    <dd>{card.base}</dd>
                    <dt>Input</dt>
                    <dd className="mono" style={{ fontSize: 12 }}>{card.input}</dd>
                    <dt>Output</dt>
                    <dd>{card.target}</dd>
                  </>
                )}
                <dt>Size</dt>
                <dd>{params(m.params_m)} parameters</dd>
                <dt>Spider dev</dt>
                <dd>
                  {pct(m.dev_ex)} execution · {pct(m.dev_em)} exact match
                </dd>
                <dt>Latency</dt>
                <dd>
                  {secs(m.p50_ms)} median · {secs(m.p95_ms)} p95 (laptop GPU)
                </dd>
                <dt>Weights</dt>
                <dd>
                  {gen ? <a href={gen}>{m.generator.slice(4)}</a> : <code>{m.generator}</code>}
                  {rk && (
                    <>
                      <br />
                      <a href={rk}>{m.ranker!.slice(4)}</a>
                    </>
                  )}
                </dd>
              </dl>
            </article>
          );
        })}
      </div>
      <p className="muted" style={{ fontSize: 12 }}>
        Full numbers, ablations and failure analysis are on the <Link to="/evaluation">Evaluation</Link> page.
      </p>

      <h2>Limitations</h2>
      <ul className="prose">
        <li>English questions only; the models were trained on Spider's style of questions.</li>
        <li>Unseen schemas are the main source of errors: wrong joins and wrong columns.</li>
        <li>Queries with three or more joins and nested or set queries are much weaker than simple lookups.</li>
        <li>Values must be spelled as in the database; near misses ("JetBlue" for "JetBlue Airways") fail.</li>
        <li>With Lite and Spinneret, an item the ranker drops cannot be used by the generator.</li>
        <li>Read-only: only SELECT queries run.</li>
      </ul>

      <h2>Data and licences</h2>
      <ul className="prose">
        <li>
          Training and evaluation data: Spider (Yu et al., 2018), CC BY-SA 4.0, including the
          <code> train_others</code> split.
        </li>
        <li>Evaluation: the official test-suite-sql-eval scripts (Apache-2.0).</li>
        <li>Base models: T5 (Apache-2.0) and RoBERTa (MIT).</li>
        <li>
          Code and docs: <a href={REPO}>github.com/iteshxt/spinneret</a>.
        </li>
      </ul>
    </div>
  );
}
