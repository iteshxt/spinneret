import { useMutation, useQuery } from "@tanstack/react-query";
import { useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { Execution, ModelInfo, QueryResponse } from "../api/types";
import { CandidatesList } from "../components/CandidatesList";
import { HistoryList, type HistoryEntry } from "../components/HistoryList";
import { ModelSwitcher } from "../components/ModelSwitcher";
import { ResultTable } from "../components/ResultTable";
import { SchemaExplorer } from "../components/SchemaExplorer";
import { SqlPanel } from "../components/SqlPanel";
import { StatusLine } from "../components/StatusLine";
import { copyText, toCsv } from "../format";

const MAX_Q = 500;
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
  dark: boolean;
  sheetOpen: boolean;
  setSheetOpen: (open: boolean) => void;
}

export function Workspace({ models, model, setModel, ready, dark, sheetOpen, setSheetOpen }: Props) {
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
    onSuccess: (res, q) => {
      setAnswer(res);
      setSql(res.sql);
      setExecution(res.execution);
      setRanEdited(false);
      setSelected(res.selected_rank);
      setShowCandidates(res.execution.status !== "ok");
      setHistory((h) =>
        [{ question: q, db_id: db, model: res.model, sql: res.sql }, ...h].slice(0, 20),
      );
      requestAnimationFrame(() => resultRef.current?.focus());
    },
  });

  const run = useMutation({
    mutationFn: (s: string) => api.execute(db, s),
    onSuccess: (res, s) => {
      setExecution(res);
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
      <div className="mobile-only" style={{ flexDirection: "column", gap: 6 }}>
        <label className="label" htmlFor="model-mobile">
          Model
        </label>
        {models.length > 0 && (
          <ModelSwitcher id="model-mobile" models={models} value={model} onChange={setModel} />
        )}
      </div>
      <div>
        <label className="label" htmlFor="db-picker">
          Database
        </label>
        <select
          id="db-picker"
          className="select"
          value={db}
          onChange={(e) => setDb(e.target.value)}
          style={{ marginTop: 6 }}
        >
          {dbList.map((d) => (
            <option key={d.db_id} value={d.db_id}>
              {d.db_id}
            </option>
          ))}
        </select>
        {dbInfo && (
          <div className="muted" style={{ fontSize: 12, marginTop: 4 }}>
            {dbInfo.n_tables} tables · {dbInfo.n_columns} columns
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
            Database & schema · {db}
          </button>

          <section className="stage" aria-labelledby="q-head">
            <span className="stage-num" aria-hidden>
              1
            </span>
            <div className="stage-head">
              <h2 id="q-head" className="label" style={{ margin: 0 }}>
                Question
              </h2>
              {question.length > MAX_Q - 50 && (
                <span className={`counter ${question.length > MAX_Q ? "over" : ""}`}>
                  {question.length} / {MAX_Q}
                </span>
              )}
            </div>
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
            {inputError && <div className="field-error">{inputError}</div>}
            <div className="question-row">
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
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => submit()}
                disabled={!ready || busy || !db}
                title={ready ? "Ask (Enter)" : "The models are still loading"}
              >
                {busy ? (
                  <>
                    <span className="spinner" aria-hidden /> Spinning…
                  </>
                ) : ready ? (
                  "Ask ⏎"
                ) : (
                  "Warming up…"
                )}
              </button>
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
              <h2 id="sql-head" className="label" style={{ margin: 0 }}>
                SQL
              </h2>
              {activeModel && answer && (
                <span className="muted" style={{ fontSize: 12 }}>
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
              <>
                {answer?.skeleton && (
                  <p className="skeleton-line" title="The skeleton the model wrote first">
                    skeleton&nbsp;&nbsp;{answer.skeleton}
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
              </>
            ) : (
              <p className="panel hint">Your SQL will appear here.</p>
            )}
          </section>

          <section className="stage" aria-labelledby="res-head">
            <span className="stage-num" aria-hidden>
              3
            </span>
            <div className="stage-head">
              <h2 id="res-head" className="label" style={{ margin: 0 }} tabIndex={-1} ref={resultRef}>
                Result
                {execution?.status === "ok" &&
                  ` · ${execution.row_count} row${execution.row_count === 1 ? "" : "s"}`}
              </h2>
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
                <>
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
                  {execution.status === "ok" && (
                    <div className="panel" style={{ marginTop: 8 }}>
                      <ResultTable execution={execution} />
                    </div>
                  )}
                </>
              ) : (
                <p className="panel hint">Results will appear here.</p>
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
