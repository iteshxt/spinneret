import { useMemo, useState } from "react";
import type { SchemaInfo, SchemaItems } from "../api/types";

interface Props {
  schema: SchemaInfo;
  focus: SchemaItems | null; // ranker scores from the last answer (Lite / Spinneret)
  onColumn?: (name: string) => void;
}

const fmt = (s: number) => s.toFixed(2).replace(/^0/, "");

export function SchemaExplorer({ schema, focus, onColumn }: Props) {
  const [filter, setFilter] = useState("");
  const [focusOn, setFocusOn] = useState(true);
  const [open, setOpen] = useState<Record<string, boolean>>({});

  const scores = useMemo(() => {
    const t = new Map<string, { score: number; kept: boolean }>();
    const c = new Map<string, { score: number; kept: boolean }>();
    for (const x of focus?.tables ?? []) t.set(x.name.toLowerCase(), x);
    for (const x of focus?.columns ?? []) c.set(`${x.table}.${x.name}`.toLowerCase(), x);
    return { t, c };
  }, [focus]);
  const showFocus = focusOn && focus != null;

  const fks = useMemo(() => {
    const m = new Map<string, string>();
    for (const fk of schema.foreign_keys) m.set(fk.source.toLowerCase(), fk.target);
    return m;
  }, [schema]);

  const q = filter.trim().toLowerCase();
  let tables = schema.tables.filter(
    (t) =>
      !q ||
      t.name.toLowerCase().includes(q) ||
      t.columns.some((c) => c.name.toLowerCase().includes(q)),
  );
  if (showFocus) {
    tables = [...tables].sort(
      (a, b) =>
        (scores.t.get(b.name.toLowerCase())?.score ?? 0) -
        (scores.t.get(a.name.toLowerCase())?.score ?? 0),
    );
  }

  return (
    <section aria-label="Database schema">
      <div className="schema-head">
        <span className="label">Schema</span>
        <input
          className="input"
          placeholder="Filter"
          aria-label="Filter tables and columns"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />
        {focus && (
          <label className="focus-toggle">
            <input type="checkbox" checked={focusOn} onChange={(e) => setFocusOn(e.target.checked)} />
            focus
          </label>
        )}
      </div>
      <ul className="schema-list">
        {tables.map((t) => {
          const ts = scores.t.get(t.name.toLowerCase());
          const kept = showFocus && ts?.kept;
          const expanded = open[t.name] ?? (q ? true : showFocus ? !!kept : false);
          const state = showFocus ? (kept ? "kept by model" : "filtered out by model") : "";
          return (
            <li key={t.name}>
              <button
                type="button"
                className={`table-row ${kept ? "kept-bar" : ""} ${showFocus && !kept ? "not-kept" : ""}`}
                aria-expanded={expanded}
                aria-label={`Table ${t.name}${state ? `, ${state}` : ""}${showFocus && ts ? `, score ${fmt(ts.score)}` : ""}`}
                onClick={() => setOpen({ ...open, [t.name]: !expanded })}
              >
                <span className="caret" aria-hidden>
                  {expanded ? "▾" : "▸"}
                </span>
                {t.name}
                {showFocus && ts && <span className="score">{fmt(ts.score)}</span>}
              </button>
              {expanded && (
                <ul className="columns">
                  {t.columns.map((c) => {
                    const cs = scores.c.get(`${t.name}.${c.name}`.toLowerCase());
                    const ck = showFocus && cs?.kept;
                    const fk = fks.get(`${t.name}.${c.name}`.toLowerCase());
                    return (
                      <li key={c.name}>
                        <button
                          type="button"
                          className={`column-row ${ck ? "kept" : ""} ${showFocus && !ck ? "not-kept" : ""}`}
                          title={fk ? `→ ${fk}` : `Insert ${c.name} into the question`}
                          aria-label={`Column ${c.name}, ${c.type}${c.is_pk ? ", primary key" : ""}${fk ? `, references ${fk}` : ""}${showFocus ? (ck ? ", kept by model" : ", filtered out by model") : ""}`}
                          onClick={() => onColumn?.(c.name)}
                        >
                          {c.is_pk && <span className="key">PK</span>}
                          {fk && <span className="key">FK</span>}
                          <span>{c.name}</span>
                          <span className="col-type">{c.type}</span>
                          {showFocus && cs && <span className="score">{fmt(cs.score)}</span>}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
