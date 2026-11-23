import { motion, type Variants } from "framer-motion";
import { Link, Navigate, useSearchParams } from "react-router-dom";
import type { ModelInfo } from "../api/types";
import { inView, item, Reveal, stagger } from "../components/Motion";
import { SilkHero } from "../components/SilkHero";
import { pct, secs } from "../format";
import { REPO, TRY_QUESTIONS } from "../site";
import "./landing.css";

const EASE = [0.2, 0.8, 0.2, 1] as const;
const line: Variants = {
  hidden: { scaleX: 0 },
  shown: { scaleX: 1, transition: { duration: 0.9, ease: EASE, delay: 0.25 } },
};

/** One headline line that slides up from behind its own baseline. */
function Line({ children, delay, cursor }: { children: string; delay: number; cursor?: boolean }) {
  return (
    <span className="line-mask">
      <motion.span
        initial={{ y: "105%" }}
        animate={{ y: 0 }}
        transition={{ duration: 0.9, ease: EASE, delay }}
      >
        {children}
        {cursor && <span className="cursor" aria-hidden />}
      </motion.span>
    </span>
  );
}

const askLink = (db: string, question: string) =>
  `/query?${new URLSearchParams({ db, q: question }).toString()}`;

export function Landing({ models }: { models: ModelInfo[] }) {
  const [params] = useSearchParams();
  // old demo links lived at "/?db=…"; send them to the demo
  if (params.has("db") || params.has("model") || params.has("q")) {
    return <Navigate to={{ pathname: "/query", search: params.toString() }} replace />;
  }
  const best = models.find((m) => m.default) ?? models[models.length - 1];

  return (
    <div className="landing">
      <section className="hero" aria-labelledby="hero-title">
        <div className="hero-copy">
          <h1 id="hero-title" className="hero-title">
            <Line delay={0.05}>Ask in English.</Line>
            <Line delay={0.18} cursor>
              Get SQL that runs.
            </Line>
          </h1>
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: EASE, delay: 0.4 }}
          >
            <p className="hero-lead">
              Spinneret writes SQL for databases it has never seen. Pick one of 20, ask a
              question, and it writes the query, runs it and shows you the rows. Free to use, and
              small enough to run on a laptop.
            </p>
            <div className="hero-actions">
              <Link to="/query" className="cta cta-primary">
                Open the demo
              </Link>
              <Link to="/evaluation" className="cta">
                See the scores
              </Link>
            </div>
            <p className="hero-note">No sign-up, and the server keeps no copy of your questions.</p>
          </motion.div>
        </div>
        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, y: 40, scale: 0.98 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 1, ease: EASE, delay: 0.45 }}
        >
          <SilkHero />
        </motion.div>
        <motion.a
          href="#how-title"
          className="scroll-cue"
          aria-label="Scroll to how it works"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.6, duration: 0.8 }}
          onClick={(e) => {
            e.preventDefault();
            document.getElementById("how-title")?.scrollIntoView({ behavior: "smooth", block: "start" });
          }}
        >
          <span className="scroll-thread" aria-hidden>
            <motion.span
              className="scroll-drop"
              animate={{ y: [0, 30, 30], opacity: [0, 1, 0] }}
              transition={{ duration: 2, times: [0, 0.7, 1], repeat: Infinity, ease: "easeInOut" }}
            />
          </span>
          <span>Scroll</span>
        </motion.a>
      </section>

      <section className="l-section" aria-labelledby="how-title">
        <Reveal>
          <h2 id="how-title" className="l-title">
            From question to answer in four steps
          </h2>
          <p className="l-lead">
            One question, <q>What is the average, minimum, and maximum age of all singers from
            France?</q>, followed through the pipeline. Every number is from that one run.
          </p>
        </Reveal>
        <motion.ol className="steps" variants={stagger} {...inView}>
          <motion.li className="step" variants={item}>
            <motion.span className="step-line" variants={line} aria-hidden />
            <h3>Find the values</h3>
            <p>
              Words that match stored data are looked up, so <q>France</q> is spelled the way the
              database spells it.
            </p>
            <div className="step-demo">
              <span className="pill-val">France</span>
              <span className="muted">found in</span>
              <code>singer.country</code>
            </div>
            <span className="step-time">0.3 ms</span>
          </motion.li>
          <motion.li className="step" variants={item}>
            <motion.span className="step-line" variants={line} aria-hidden />
            <h3>Rank the schema</h3>
            <p>
              A RoBERTa ranker scores every table and column. The model reads only the top four
              tables and the top five columns of each.
            </p>
            <div className="step-demo scores">
              {[
                ["singer", 96],
                ["singer.age", 95],
                ["singer.country", 92],
                ["stadium.average", 24],
              ].map(([name, v]) => (
                <div key={name} className="score-row">
                  <code>{name}</code>
                  <span className="score-track">
                    <motion.span
                      initial={{ width: 0 }}
                      whileInView={{ width: `${v}%` }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.9, ease: EASE, delay: 0.4 }}
                    />
                  </span>
                  <span className="score-num">{(Number(v) / 100).toFixed(2)}</span>
                </div>
              ))}
            </div>
            <span className="step-time">80 ms</span>
          </motion.li>
          <motion.li className="step" variants={item}>
            <motion.span className="step-line" variants={line} aria-hidden />
            <h3>Write candidates</h3>
            <p>
              T5 writes eight queries with beam search, each starting from the query's shape before
              the names and values.
            </p>
            <div className="step-demo">
              <code className="shape-line">select avg(_), min(_), max(_) from _ where _ = _</code>
            </div>
            <span className="step-time">1.27 s</span>
          </motion.li>
          <motion.li className="step" variants={item}>
            <motion.span className="step-line" variants={line} aria-hidden />
            <h3>Run the first that works</h3>
            <p>
              Candidates run read-only against the database. The first one without an error is the
              answer, and the others stay one click away.
            </p>
            <div className="step-demo">
              <span className="pill-val">34.5</span>
              <span className="pill-val">25</span>
              <span className="pill-val">43</span>
            </div>
            <span className="step-time">1.2 ms</span>
          </motion.li>
        </motion.ol>
      </section>

      <section className="l-section tradeoff" aria-labelledby="models-title">
        <Reveal>
          <h2 id="models-title" className="l-title">
            Three models: pick speed or accuracy
          </h2>
          <p className="l-lead">
            All three were graded on Spider's 1,034 development questions over 20 databases they
            never saw in training. Switch between them at any time in the demo.
          </p>
          <motion.ul className="model-list" variants={stagger} {...inView}>
            {models.map((m) => {
              const [name, desc] = m.label.split(" · ");
              return (
                <motion.li key={m.id} variants={item}>
                  <div>
                    <strong>{name}</strong>
                    <span className="muted">{desc}</span>
                  </div>
                  <span className="model-stat">
                    {pct(m.dev_ex)} correct in {secs(m.p50_ms)}
                  </span>
                  <Link to={`/query?model=${m.id}`} className="cta cta-small">
                    Try {name}
                  </Link>
                </motion.li>
              );
            })}
          </motion.ul>
        </Reveal>
        {models.length > 0 && (
          <Reveal delay={0.15}>
            <TradeoffChart models={models} />
          </Reveal>
        )}
      </section>

      <section className="l-section" aria-labelledby="try-title">
        <Reveal>
          <h2 id="try-title" className="l-title">
            Ask one of these
          </h2>
          <p className="l-lead">
            Each opens the demo on its database and asks
            {best ? ` ${best.label.split(" · ")[0]}` : " the model"}. Or write your own.
          </p>
        </Reveal>
        <motion.ul className="try-list" variants={stagger} {...inView}>
          {TRY_QUESTIONS.map((t) => (
            <motion.li key={t.question} variants={item}>
              <Link to={askLink(t.db, t.question)}>
                <span>{t.question}</span>
                <code>{t.db}</code>
              </Link>
            </motion.li>
          ))}
        </motion.ul>
      </section>

      <section className="l-section closing" aria-labelledby="run-title">
        <Reveal>
          <h2 id="run-title" className="l-title">
            Run it on your own machine
          </h2>
          <p className="l-lead">
            All three models fit together in about 1.6 GB of GPU memory, so a laptop with a 4 GB
            GTX 1650 serves them all. The code, the training scripts and the full evaluation are
            on GitHub.
          </p>
          <div className="hero-actions">
            <a href={REPO} className="cta cta-primary">
              View on GitHub
            </a>
            <Link to="/about" className="cta">
              How it was built
            </Link>
          </div>
        </Reveal>
        <Reveal delay={0.15}>
        <pre className="terminal" aria-label="Commands to run Spinneret locally">
          <code>
            <span className="muted"># Python 3.12 and uv</span>
            {"\n"}uv sync --extra train --extra api
            {"\n"}uv run spinneret download
            {"\n"}(cd web &amp;&amp; npm ci &amp;&amp; npm run build)
            {"\n"}uv run spinneret serve
          </code>
        </pre>
        </Reveal>
      </section>

    </div>
  );
}

/** Accuracy against median latency, one dot per model (live from /api/models). */
function TradeoffChart({ models }: { models: ModelInfo[] }) {
  const W = 460;
  const H = 320;
  const pad = { l: 48, r: 24, t: 20, b: 48 };
  const xMax = Math.max(1.5, ...models.map((m) => (m.p50_ms ?? 0) / 1000)) * 1.05;
  const yMin = 0.4;
  const yMax = 0.75;
  const x = (s: number) => pad.l + (s / xMax) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);
  const xTicks = [0, 0.5, 1, 1.5].filter((t) => t <= xMax);
  const yTicks = [0.4, 0.5, 0.6, 0.7];

  return (
    <figure className="tradeoff-chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby="tradeoff-desc">
        <desc id="tradeoff-desc">
          {models
            .map((m) => `${m.label}: ${pct(m.dev_ex)} correct, median ${secs(m.p50_ms)}`)
            .join(". ")}
        </desc>
        {yTicks.map((t) => (
          <g key={t}>
            <line className="grid" x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} />
            <text className="tick" x={pad.l - 8} y={y(t) + 4} textAnchor="end">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} className="tick" x={x(t)} y={H - pad.b + 18} textAnchor="middle">
            {t === 0 ? "0" : `${t} s`}
          </text>
        ))}
        <text className="axis" x={pad.l} y={H - 8}>
          Median time per question, laptop GPU
        </text>
        {models.map((m, i) => {
          if (m.dev_ex == null || m.p50_ms == null) return null;
          const cx = x(m.p50_ms / 1000);
          const cy = y(m.dev_ex);
          const right = cx < W * 0.6;
          const name = m.label.split(" · ")[0];
          return (
            <motion.g
              key={m.id}
              className="dot"
              initial={{ opacity: 0, scale: 0.4 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ type: "spring", stiffness: 260, damping: 18, delay: 0.3 + i * 0.15 }}
              style={{ transformOrigin: `${cx}px ${cy}px` }}
            >
              <title>{`${m.label}: ${pct(m.dev_ex)} correct, median ${secs(m.p50_ms)}`}</title>
              <circle cx={cx} cy={cy} r={14} className="hit" />
              <circle cx={cx} cy={cy} r={6} />
              <text x={right ? cx + 14 : cx - 14} y={cy - 2} textAnchor={right ? "start" : "end"} className="dot-name">
                {name}
              </text>
              <text x={right ? cx + 14 : cx - 14} y={cy + 14} textAnchor={right ? "start" : "end"} className="dot-val">
                {pct(m.dev_ex)}
              </text>
            </motion.g>
          );
        })}
      </svg>
      <figcaption className="muted">Questions answered correctly (execution accuracy)</figcaption>
    </figure>
  );
}
