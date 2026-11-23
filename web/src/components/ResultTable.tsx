import type { Execution } from "../api/types";

const isNumber = (v: unknown) => typeof v === "number";

export function ResultTable({ execution }: { execution: Execution }) {
  const { columns, rows, row_count, truncated } = execution;
  if (row_count === 0) {
    return <p className="hint">The query ran but no rows matched.</p>;
  }
  // a column is right-aligned when every non-null value in it is a number
  const numeric = columns.map((_, j) =>
    rows.every((r) => r[j] == null || isNumber(r[j])) && rows.some((r) => isNumber(r[j])),
  );
  return (
    <>
      {truncated && <div className="notice">Showing the first {row_count} rows.</div>}
      <div className="table-wrap" tabIndex={0} aria-label="Query result">
        <table className="result">
          <thead>
            <tr>
              {columns.map((c, j) => (
                <th key={j} scope="col">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((v, j) => (
                  <td key={j} className={numeric[j] ? "num" : undefined}>
                    {v == null ? <span className="null">NULL</span> : String(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
