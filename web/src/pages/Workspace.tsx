import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { Execution, ModelInfo, QueryResponse } from "../api/types";
import { CandidatesList } from "../components/CandidatesList";
import { HistoryList, type HistoryEntry } from "../components/HistoryList";
import { ModelSwitcher } from "../components/ModelSwitcher";
import { ResultTable } from "../components/ResultTable";
import { Select } from "../components/Select";
import { SchemaExplorer } from "../components/SchemaExplorer";
import { SqlPanel } from "../components/SqlPanel";
import { StatusLine } from "../components/StatusLine";
import { copyText, toCsv } from "../format";

const MAX_Q = 500;

const formatValue = (v: unknown) =>
  v == null ? "NULL" : typeof v === "number" && !Number.isInteger(v) ? String(+v.toFixed(4)) : String(v);
const HISTORY_KEY = "spinneret.history";
const DEFAULT_DB = "concert_singer";

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(sessionStorage.getItem(HISTORY_KEY) ?? "[]");
  } catch {
    return [];
  }
}

interface Props {
  models: ModelInfo[];
  model: string;
  setModel: (id: string) => void;
  ready: boolean;
  offline: boolean;
  dark: boolean;
  sheetOpen: boolean;
  setSheetOpen: (open: boolean) => void;
}

export function Workspace(props: Props) {
  const { models, model, setModel, ready, offline, dark, sheetOpen, setSheetOpen } = props;
  const queryClient = useQueryClient();
  // the hosted site's server went away mid-visit: let the header and banner say so
  const onError = (e: Error) => {
    if (e instanceof ApiError && e.code === "offline")
      queryClient.invalidateQueries({ queryKey: ["health"] });
  };
  const [params, setParams] = useSearchParams();
  const databases = useQuery({ queryKey: ["databases"], queryFn: api.databases });
  const dbList = databases.data ?? [];
  const wanted = params.get("db");
  const db = dbList.some((d) => d.db_id === wanted)
    ? wanted!
    : (dbList.find((d) => d.db_id === DEFAULT_DB) ?? dbList[0])?.db_id ?? "";

  const schema = useQuery({ queryKey: ["schema", db], queryFn: () => api.schema(db), enabled: !!db });
  const examples = useQuery({
    queryKey: ["examples", db],
    queryFn: () => api.examples(db),
    enabled: !!db,
  });

  const [question, setQuestion] = useState("");
  const [inputError, setInputError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<QueryResponse | null>(null);
  const [sql, setSql] = useState("");
  const [execution, setExecution] = useState<Execution | null>(null);
  const [ranEdited, setRanEdited] = useState(false);
  const [showCandidates, setShowCandidates] = useState(false);
  const [selected, setSelected] = useState(0);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [copied, setCopied] = useState<string | null>(null);
  const [version, setVersion] = useState(0); // bumps on every new answer or run, to animate it in
  const questionRef = useRef<HTMLTextAreaElement>(null);
  const resultRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    try {
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      /* private mode: history lasts for this page only */
    }
  }, [history]);

  const clearResult = () => {
    setAnswer(null);
    setExecution(null);
    setSql("");
  };

  const setDb = (id: string) => {
    const next = new URLSearchParams(params);
    next.set("db", id);
    setParams(next, { replace: true });
    clearResult();
  };

  const ask = useMutation({
    mutationFn: (q: string) => api.query({ db_id: db, question: q, options: { model } }),
    onError,
    onSuccess: (res, q) => {
      setAnswer(res);
      setSql(res.sql);
      setExecution(res.execution);
      setRanEdited(false);
      setSelected(res.selected_rank);
      setShowCandidates(res.execution.status !== "ok");
      setVersion((v) => v + 1);
      setHistory((h) =>
        [{ question: q, db_id: db, model: res.model, sql: res.sql }, ...h].slice(0, 20),
      );
      requestAnimationFrame(() => resultRef.current?.focus());
    },
  });

  const run = useMutation({
    mutationFn: (s: string) => api.execute(db, s),
    onError,
    onSuccess: (res, s) => {
      setExecution(res);
      setVersion((v) => v + 1);
      setRanEdited(answer == null || s.trim() !== answer.candidates[selected]?.sql.trim());
    },
  });

  const submit = (q = question) => {
    const text = q.trim();
    if (!text) return setInputError("Type a question first.");
    if (text.length > MAX_Q) return setInputError(`Questions are limited to ${MAX_Q} characters.`);
    setInputError(null);
    ask.mutate(text);
  };

  const runSql = useCallback(() => {
    if (sql.trim()) run.mutate(sql);
  }, [sql, run]);

  // A question passed in the URL (?q=, from the landing page) is asked once the models are ready.
  const pending = params.get("q");
  const askedFromUrl = useRef(false);
  useEffect(() => {
    if (!pending || askedFromUrl.current || !ready || !db || (wanted && db !== wanted)) return;
    askedFromUrl.current = true;
    setQuestion(pending);
    submit(pending);
    const next = new URLSearchParams(params);
    next.delete("q");
    setParams(next, { replace: true });
  });

  // "/" focuses the question box (design §10.3)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.key === "Escape") setSheetOpen(false);
      if (t.closest("input, textarea, select, [contenteditable]")) return;
      if (e.key === "/") {
        e.preventDefault();
        questionRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setSheetOpen]);

  // auto-grow the question box
  useEffect(() => {
    const el = questionRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight + 2}px`;
  }, [question]);

  const insertColumn = (name: string) => {
    const el = questionRef.current;
    const at = el?.selectionStart ?? question.length;
    const next = `${question.slice(0, at)}${name}${question.slice(el?.selectionEnd ?? at)}`;
    setQuestion(next);
    requestAnimationFrame(() => {
      el?.focus();
      el?.setSelectionRange(at + name.length, at + name.length);
    });
  };

  const pickHistory = (h: HistoryEntry) => {
    if (h.db_id !== db) setDb(h.db_id);
    if (models.some((m) => m.id === h.model)) setModel(h.model);
    setQuestion(h.question);
    setAnswer(null);
    setSql(h.sql);
    setRanEdited(false);
    run.mutate(h.sql);
    setSheetOpen(false);
  };

  const flash = (what: string) => {
    setCopied(what);
    setTimeout(() => setCopied(null), 1500);
  };

  const askError = ask.error instanceof ApiError ? ask.error : null;
  const busy = ask.isPending;
  const dbInfo = dbList.find((d) => d.db_id === db);
  const activeModel = models.find((m) => m.id === model);

  const askButton = (
    <button
      type="button"
      className="btn btn-primary"
      onClick={() => submit()}
      disabled={!ready || busy || !db}
      title={ready ? "Ask (Enter)" : offline ? "The live demo is offline" : "The models are still loading"}
    >
      {busy ? (
        <>
          <span className="spinner" aria-hidden /> Spinning…
        </>
      ) : ready ? (
        "Ask"
      ) : offline ? (
        "Offline"
      ) : (
        "Warming up…"
      )}
    </button>
  );

  const sidebar = (
    <aside className={`sidebar ${sheetOpen ? "open" : ""}`} aria-label="Database and schema">
      <button
        type="button"
        className="btn mobile-only"
        style={{ alignSelf: "flex-end" }}
        onClick={() => setSheetOpen(false)}
      >
        Done
      </button>
      <div className="mobile-only side-block">
        <label className="label" htmlFor="model-mobile">
          Model
        </label>
        {models.length > 0 && (
          <ModelSwitcher id="model-mobile" models={models} value={model} onChange={setModel} />
        )}
      </div>
      <div className="side-block">
        <label className="label" htmlFor="db-picker">
          Database
        </label>
        <Select
          id="db-picker"
          className="db-picker"
          value={db}
          onChange={setDb}
          options={dbList.map((d) => ({
            value: d.db_id,
            label: d.db_id,
            detail: `${d.n_tables} tables, ${d.n_columns} columns`,
          }))}
        />
        {dbInfo && (
          <div className="db-meta">
            {dbInfo.n_tables} tables, {dbInfo.n_columns} columns. Click a column to add it to your
            question.
          </div>
        )}
      </div>
      {schema.data && (
        <SchemaExplorer
          key={db}
          schema={schema.data}
          focus={answer?.schema_items ?? null}
          onColumn={insertColumn}
        />
      )}
      <HistoryList items={history} onPick={pickHistory} />
    </aside>
  );

  return (
    <div className="workspace-layout">
      {sidebar}
      <main className="main">
        <div className="workspace">
          <button
            type="button"
            className="btn mobile-only"
            style={{ marginBottom: 16, alignSelf: "flex-start" }}
            onClick={() => setSheetOpen(!sheetOpen)}
            aria-expanded={sheetOpen}
          >
            Database and schema: {db}
          </button>

          <motion.header
            className="ws-head"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.2, 0.8, 0.2, 1] }}
          >
            <h1>Ask a database</h1>
            <p>
              Asking <code>{db || "…"}</code>
              {activeModel ? ` with ${activeModel.label.split(" · ")[0]}` : ""}. Spinneret writes
              the SQL, runs it read-only and shows you the rows.
            </p>
          </motion.header>

          <section className="stage" aria-labelledby="q-head">
            <span className="stage-num" aria-hidden>
              1
            </span>
            <div className="stage-head">
              <h2 id="q-head" className="stage-title">
                Question
              </h2>
              {question.length > MAX_Q - 50 && (
                <span className={`counter ${question.length > MAX_Q ? "over" : ""}`}>
                  {question.length} / {MAX_Q}
                </span>
              )}
            </div>
            <div className="composer">
            <textarea
              ref={questionRef}
              className="question-box"
              rows={1}
              value={question}
              placeholder="Ask a question in plain English…"
              aria-label="Question"
              aria-invalid={!!inputError}
              onChange={(e) => {
                setQuestion(e.target.value);
                if (inputError) setInputError(null);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (!busy && ready) submit();
                }
              }}
            />
              <div className="composer-bar">
                <span className="composer-hint">
                  <kbd>Enter</kbd> to ask, <kbd>Shift</kbd> + <kbd>Enter</kbd> for a new line
                </span>
                {askButton}
              </div>
            </div>
            {inputError && <div className="field-error">{inputError}</div>}
            <div className="question-row">
              <span className="label">Try one of these</span>
              <div className="chips" aria-label="Example questions">
                {(examples.data ?? []).map((ex) => (
                  <button
                    key={ex.question}
                    type="button"
                    className="chip"
                    disabled={!ready || busy}
                    onClick={() => {
                      setQuestion(ex.question);
                      submit(ex.question);
                    }}
                  >
                    {ex.question}
                  </button>
                ))}
              </div>
            </div>
            {askError && (
              <div className="field-error" role="alert">
                {askError.code === "model_loading"
                  ? "The models are still loading — try again in a moment."
                  : askError.message}
              </div>
            )}
          </section>

          <section className="stage" aria-labelledby="sql-head">
            <span className="stage-num" aria-hidden>
              2
            </span>
            <div className="stage-head">
              <h2 id="sql-head" className="stage-title">
                SQL
              </h2>
              {activeModel && answer && (
                <span className="stage-sub">
                  {models.find((m) => m.id === answer.model)?.label ?? answer.model}
                </span>
              )}
              <div className="actions">
                <button
                  type="button"
                  className="btn"
                  disabled={!sql}
                  onClick={async () => (await copyText(sql)) && flash("sql")}
                >
                  {copied === "sql" ? "Copied" : "⧉ Copy"}
                </button>
              </div>
            </div>
            {busy ? (
              <div className="loader" aria-hidden />
            ) : sql || answer ? (
              <motion.div
                key={`sql-${version}`}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.45, ease: [0.2, 0.8, 0.2, 1] }}
              >
                {answer?.skeleton && (
                  <p className="skeleton-line" title="The query's shape, which the model wrote first">
                    <span className="tag">Shape</span>
                    {answer.skeleton}
                  </p>
                )}
                <SqlPanel
                  value={sql}
                  generated={answer ? answer.candidates[selected]?.sql ?? answer.sql : null}
                  onChange={setSql}
                  onRun={runSql}
                  onReset={() => answer && setSql(answer.candidates[selected]?.sql ?? answer.sql)}
                  dark={dark}
                  running={run.isPending}
                />
                {answer && (
                  <>
                    <CandidatesList
                      candidates={answer.candidates}
                      selected={selected}
                      open={showCandidates}
                      onToggle={setShowCandidates}
                      onPick={(c) => {
                        setSelected(c.rank);
                        setSql(c.sql);
                        run.mutate(c.sql);
                      }}
                    />
                    {Object.keys(answer.matched_values).length > 0 && (
                      <details className="details">
                        <summary>
                          matched values:{" "}
                          {Object.entries(answer.matched_values)
                            .map(([col, vals]) => `${col} = ${vals.join(", ")}`)
                            .join(" · ")}
                        </summary>
                        <p className="muted" style={{ margin: "4px 0 0" }}>
                          Database values that match words in the question are shown to the model
                          next to their column.
                        </p>
                      </details>
                    )}
                  </>
                )}
              </motion.div>
            ) : (
              <p className="hint">The SQL the model writes will appear here, ready to edit.</p>
            )}
          </section>

          <section className="stage" aria-labelledby="res-head">
            <span className="stage-num" aria-hidden>
              3
            </span>
            <div className="stage-head">
              <h2 id="res-head" className="stage-title" tabIndex={-1} ref={resultRef}>
                Result
              </h2>
              {execution?.status === "ok" && (
                <span className="stage-sub">
                  {execution.row_count} row{execution.row_count === 1 ? "" : "s"}
                </span>
              )}
              <div className="actions">
                <button
                  type="button"
                  className="btn"
                  disabled={!execution?.row_count}
                  onClick={async () =>
                    execution &&
                    (await copyText(toCsv(execution.columns, execution.rows))) &&
                    flash("csv")
                  }
                >
                  {copied === "csv" ? "Copied" : "⧉ CSV"}
                </button>
              </div>
            </div>
            <div aria-live="polite" className={busy || run.isPending ? "dimmed" : undefined}>
              {execution ? (
                <motion.div
                  key={`res-${version}`}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, ease: [0.2, 0.8, 0.2, 1], delay: 0.08 }}
                >
                  <StatusLine
                    execution={execution}
                    selectedRank={selected}
                    candidates={answer?.candidates.length}
                    timings={
                      answer && !ranEdited && selected === answer.selected_rank
                        ? answer.timings_ms
                        : undefined
                    }
                    edited={ranEdited}
                  />
                  {execution.status === "ok" &&
                    (execution.row_count === 1 && execution.columns.length <= 4 ? (
                      // a single row reads best as the answer itself, not as a table
                      <div className="answer-values">
                        {execution.columns.map((c, i) => (
                          <div key={i} className="answer-value">
                            <span className="mono">{c}</span>
                            <strong>{formatValue(execution.rows[0][i])}</strong>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="panel" style={{ marginTop: 10 }}>
                        <ResultTable execution={execution} />
                      </div>
                    ))}
                </motion.div>
              ) : (
                <p className="hint">The rows the query returns will appear here.</p>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
