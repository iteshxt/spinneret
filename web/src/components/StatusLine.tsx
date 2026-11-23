import type { Execution } from "../api/types";
import { secs } from "../format";

const TEXT: Record<string, [string, string, string]> = {
  // status → [css class, icon, words]
  ok: ["status-ok", "✓", "Ran"],
  error: ["status-error", "✕", "The SQL didn't run."],
  failed: ["status-error", "✕", "The generated SQL didn't run."],
  timeout: ["status-warn", "◷", "Query took longer than 5 s and was stopped."],
  rejected: ["status-warn", "⚠", "Only read-only SELECT queries can run here."],
};

/** Plain-language reason for a SQLite error (the raw message stays one click away). */
export function explain(error: string | null): string | null {
  if (!error) return null;
  const e = error.toLowerCase();
  if (e.includes("no such column")) return "It refers to a column that isn't in this database.";
  if (e.includes("no such table")) return "It refers to a table that isn't in this database.";
  if (e.includes("ambiguous column")) return "A column name matches more than one table.";
  if (e.includes("syntax error") || e.includes("incomplete input")) return "It isn't valid SQL.";
  if (e.includes("misuse of aggregate")) return "An aggregate (COUNT, SUM…) is used in the wrong place.";
  return null;
}

interface Props {
  execution: Execution;
  selectedRank?: number;
  candidates?: number;
  timings?: Record<string, number>;
  edited?: boolean;
}

export function StatusLine({ execution, selectedRank, candidates, timings, edited }: Props) {
  const [cls, icon, words] = TEXT[execution.status] ?? TEXT.error;
  const why = execution.status === "ok" ? null : explain(execution.error);
  return (
    <div className={cls}>
      <div className="status-line" role="status">
        <span className="state">
          <span aria-hidden>{icon} </span>
          {words}
        </span>
        {execution.status === "ok" && edited && <span className="muted">your edited SQL</span>}
        {execution.status === "ok" && !edited && selectedRank != null && candidates != null && (
          <span className="muted">
            candidate {selectedRank + 1} of {candidates}
          </span>
        )}
        {timings && !edited && (
          <span className="muted mono">
            {secs(timings.total)}
            {" ("}
            {timings.ranking ? `rank ${Math.round(timings.ranking)} · ` : ""}
            gen {Math.round(timings.inference)} · sql {Math.round(timings.execution)}
            {")"}
          </span>
        )}
      </div>
      {why && <p className="muted" style={{ margin: "4px 0 0" }}>{why}</p>}
      {execution.error && (
        <details className="details">
          <summary>SQLite message</summary>
          <pre className="raw-error mono">{execution.error}</pre>
        </details>
      )}
    </div>
  );
}
