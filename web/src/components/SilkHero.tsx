import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { HERO_EXAMPLES, plain, spans, type HeroExample } from "../site";

const WORD = 0.07; // seconds between the question's words appearing
const STEP = 0.22; // seconds between threads
const HOLD_MS = 9500; // time on each example before moving to the next

function useReducedMotion() {
  const q = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState(() => window.matchMedia?.(q).matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.(q);
    const on = () => setReduced(mq.matches);
    mq?.addEventListener("change", on);
    return () => mq?.removeEventListener("change", on);
  }, []);
  return reduced;
}

/**
 * The hero: a real question spun down into SQL and then into its answer. Threads join each
 * word to the SQL it became, and the selected SQL to the value the database returned.
 */
export function SilkHero() {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const reduced = useReducedMotion();
  const auto = !reduced && !paused;

  useEffect(() => {
    if (!auto) return;
    const t = setTimeout(() => setIndex((i) => (i + 1) % HERO_EXAMPLES.length), HOLD_MS);
    return () => clearTimeout(t);
  }, [index, auto]);

  return (
    <figure
      className="silk"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className="silk-tabs" role="tablist" aria-label="Example database">
        {HERO_EXAMPLES.map((ex, i) => (
          <button
            key={ex.db}
            type="button"
            role="tab"
            aria-selected={i === index}
            className="silk-tab"
            onClick={() => setIndex(i)}
          >
            {ex.db}
            {i === index && (
              <span
                key={`${index}-${auto}`}
                className={`silk-timer${auto ? " running" : ""}`}
                style={{ animationDuration: `${HOLD_MS}ms` }}
                aria-hidden
              />
            )}
          </button>
        ))}
      </div>
      <Strands key={index} ex={HERO_EXAMPLES[index]} />
      <figcaption className="silk-caption">
        Unedited output of the Spinneret model. The grey text is the query's shape, which it
        writes before filling in names and values.
      </figcaption>
    </figure>
  );
}

interface Thread {
  d: string;
  x: number;
  y: number;
  key: string;
  delay: number;
  stage: 1 | 2;
}

const curve = (x1: number, y1: number, x2: number, y2: number) => {
  const sag = (y2 - y1) * 0.6;
  const f = (n: number) => n.toFixed(1);
  return `M${f(x1)},${f(y1)} C${f(x1)},${f(y1 + sag)} ${f(x2)},${f(y2 - sag)} ${f(x2)},${f(y2)}`;
};

function Strands({ ex }: { ex: HeroExample }) {
  const box = useRef<HTMLDivElement>(null);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [hot, setHot] = useState<string | null>(null);
  const q = spans(ex.question);
  const s = spans(ex.sql);

  // timeline: words appear, the SQL's shape appears, threads drop word by word into the
  // SQL, then a second set of threads carries the SQL down to the answer
  const words: { text: string; key?: string }[] = q.flatMap((sp) =>
    sp.key ? [sp] : sp.text.split(/(?<=\s)/).map((t) => ({ text: t })),
  );
  const shapeAt = words.length * WORD + 0.25;
  let n = 0;
  const sqlOrder = s.map((sp) => (sp.key ? n++ : -1));
  const threadAt = (i: number) => shapeAt + 0.35 + i * STEP;
  const answerAt = threadAt(n) + 0.5;

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const out: Thread[] = [];
      const tokens = [...el.querySelectorAll<HTMLElement>("[data-s]")];
      const words = [...el.querySelectorAll<HTMLElement>("[data-q]")];
      for (const t of tokens) {
        const key = t.dataset.s!;
        const b = t.getBoundingClientRect();
        for (const w of words.filter((w) => w.dataset.q === key)) {
          const a = w.getBoundingClientRect();
          const x1 = a.left + a.width / 2 - r.left;
          const y1 = a.bottom - r.top + 4;
          out.push({
            key,
            stage: 1,
            x: x1,
            y: y1,
            delay: threadAt(Number(t.dataset.i)),
            d: curve(x1, y1, b.left + b.width / 2 - r.left, b.top - r.top - 4),
          });
        }
      }
      for (const v of el.querySelectorAll<HTMLElement>("[data-r]")) {
        const key = v.dataset.r!;
        const c = v.getBoundingClientRect();
        for (const t of tokens.filter((t) => t.dataset.s === key)) {
          const b = t.getBoundingClientRect();
          const x1 = b.left + b.width / 2 - r.left;
          const y1 = b.bottom - r.top + 4;
          out.push({
            key,
            stage: 2,
            x: x1,
            y: y1,
            delay: answerAt - 0.1,
            d: curve(x1, y1, c.left + 14 - r.left, c.top - r.top - 6),
          });
        }
      }
      setThreads(out);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    document.fonts?.ready.then(measure).catch(() => {});
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const dim = (key?: string) => (hot && key !== hot ? " dim" : "");
  const hover = (key?: string) =>
    key ? { onMouseEnter: () => setHot(key), onMouseLeave: () => setHot(null) } : {};
  const at = (t: number) => ({ "--d": `${t.toFixed(2)}s` }) as React.CSSProperties;

  return (
    <div className="strands" ref={box}>
      <svg className="threads" aria-hidden>
        {threads.map((t, i) => (
          <g key={i} className={`thread stage-${t.stage}${dim(t.key)}`} style={at(t.delay)}>
            <path d={t.d} pathLength={1} />
            <circle r={2.5} cx={t.x} cy={t.y} />
          </g>
        ))}
      </svg>

      <span className="strand-step" style={at(0)}>
        You ask
      </span>
      <p className="strand-q" aria-label={plain(ex.question)}>
        {words.map((w, i) =>
          w.key ? (
            <span key={i} data-q={w.key} className={`w${dim(w.key)}`} style={at(i * WORD)} {...hover(w.key)}>
              {w.text}
            </span>
          ) : (
            <span key={i} className="word" style={at(i * WORD)} aria-hidden>
              {w.text}
            </span>
          ),
        )}
      </p>

      <span className="strand-step step-sql" style={at(shapeAt)}>
        Spinneret writes
      </span>
      <code className="strand-sql" aria-label={plain(ex.sql)}>
        {s.map((sp, i) =>
          sp.key ? (
            <span
              key={i}
              data-s={sp.key}
              data-i={sqlOrder[i]}
              className={`tok${dim(sp.key)}`}
              style={at(threadAt(sqlOrder[i]) + 0.45)}
              {...hover(sp.key)}
            >
              {sp.text}
            </span>
          ) : (
            <span key={i} className="shape" style={at(shapeAt)} aria-hidden>
              {sp.text}
            </span>
          ),
        )}
      </code>

      <span className="strand-step step-answer" style={at(answerAt)}>
        The database answers
      </span>
      <div className="strand-answer">
        {ex.answers.map((a) => (
          <div key={a.label} className="answer" data-r={a.from} style={at(answerAt + 0.4)}>
            <span className="mono">{a.label}</span>
            <strong>{a.value}</strong>
          </div>
        ))}
        <p style={at(answerAt + 0.7)}>
          from <span className="mono">{ex.db}</span> in {(ex.ms / 1000).toFixed(1)} s on a laptop GPU
        </p>
      </div>
    </div>
  );
}
