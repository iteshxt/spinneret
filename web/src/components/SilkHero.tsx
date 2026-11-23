import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { HERO_EXAMPLES, plain, spans, type HeroExample } from "../site";

const STEP = 0.24; // seconds between threads
const START = 0.55; // the query's shape shows first, then the threads drop in
const HOLD_MS = 9000; // time on each example before moving to the next

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

/** The hero: a real question, threaded word by word to the SQL the model wrote for it. */
export function SilkHero() {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const reduced = useReducedMotion();

  useEffect(() => {
    if (reduced || paused) return;
    const t = setTimeout(() => setIndex((i) => (i + 1) % HERO_EXAMPLES.length), HOLD_MS);
    return () => clearTimeout(t);
  }, [index, paused, reduced]);

  return (
    <figure
      className="silk"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocus={() => setPaused(true)}
      onBlur={() => setPaused(false)}
    >
      <div className="silk-top">
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
            </button>
          ))}
        </div>
        <span className="silk-model">Spinneret model, unedited output</span>
      </div>
      <Strands key={index} ex={HERO_EXAMPLES[index]} />
      <figcaption className="silk-caption">
        Each thread joins a word to the part of the query it became. The grey text is the
        query's shape, which the model writes before filling in names and values.
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
}

function Strands({ ex }: { ex: HeroExample }) {
  const box = useRef<HTMLDivElement>(null);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [hot, setHot] = useState<string | null>(null);
  const q = spans(ex.question);
  const s = spans(ex.sql);

  // order of each linked SQL span, left to right: threads and tokens arrive in that order
  let n = 0;
  const sqlOrder = s.map((sp) => (sp.key ? n++ : -1));
  const lastDelay = START + (n - 1) * STEP;

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      const words = [...el.querySelectorAll<HTMLElement>("[data-q]")];
      const tokens = [...el.querySelectorAll<HTMLElement>("[data-s]")];
      const out: Thread[] = [];
      for (const t of tokens) {
        const key = t.dataset.s!;
        const b = t.getBoundingClientRect();
        for (const w of words.filter((w) => w.dataset.q === key)) {
          const a = w.getBoundingClientRect();
          const x1 = a.left + a.width / 2 - r.left;
          const y1 = a.bottom - r.top + 3;
          const x2 = b.left + b.width / 2 - r.left;
          const y2 = b.top - r.top - 3;
          const sag = (y2 - y1) * 0.6;
          out.push({
            key,
            x: x1,
            y: y1,
            delay: START + Number(t.dataset.i) * STEP,
            d: `M${x1.toFixed(1)},${y1.toFixed(1)} C${x1.toFixed(1)},${(y1 + sag).toFixed(1)} ${x2.toFixed(1)},${(y2 - sag).toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}`,
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
  }, []);

  const dim = (key?: string) => (hot && key !== hot ? " dim" : "");
  const hover = (key?: string) =>
    key ? { onMouseEnter: () => setHot(key), onMouseLeave: () => setHot(null) } : {};

  return (
    <div className="strands" ref={box}>
      <svg className="threads" aria-hidden>
        {threads.map((t, i) => (
          <g key={i} className={`thread${dim(t.key)}`} style={{ "--d": `${t.delay}s` } as React.CSSProperties}>
            <path d={t.d} pathLength={1} />
            <circle r={2.5} cx={t.x} cy={t.y} />
          </g>
        ))}
      </svg>
      <p className="strand-q" aria-label={plain(ex.question)}>
        {q.map((sp, i) =>
          sp.key ? (
            <span key={i} data-q={sp.key} className={`w${dim(sp.key)}`} {...hover(sp.key)}>
              {sp.text}
            </span>
          ) : (
            <span key={i} aria-hidden>
              {sp.text}
            </span>
          ),
        )}
      </p>
      <code className="strand-sql" aria-label={plain(ex.sql)}>
        {s.map((sp, i) =>
          sp.key ? (
            <span
              key={i}
              data-s={sp.key}
              data-i={sqlOrder[i]}
              className={`tok${dim(sp.key)}`}
              style={{ "--d": `${START + sqlOrder[i] * STEP + 0.45}s` } as React.CSSProperties}
              {...hover(sp.key)}
            >
              {sp.text}
            </span>
          ) : (
            <span key={i} className="shape" aria-hidden>
              {sp.text}
            </span>
          ),
        )}
      </code>
      <div className="strand-result" style={{ "--d": `${lastDelay + 0.9}s` } as React.CSSProperties}>
        <table>
          <thead>
            <tr>
              {ex.columns.map((c) => (
                <th key={c} className="mono">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ex.rows.map((row, i) => (
              <tr key={i}>
                {row.map((v, j) => (
                  <td key={j}>{v}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          Ran on <span className="mono">{ex.db}</span> and answered in{" "}
          {(ex.ms / 1000).toFixed(1)} s on a laptop GPU.
        </p>
      </div>
    </div>
  );
}
