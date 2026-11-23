export interface HistoryEntry {
  question: string;
  db_id: string;
  model: string;
  sql: string;
}

export function HistoryList({
  items,
  onPick,
}: {
  items: HistoryEntry[];
  onPick: (h: HistoryEntry) => void;
}) {
  if (!items.length) return null;
  return (
    <section aria-label="History">
      <div className="label" style={{ marginBottom: 6 }}>
        History (this session)
      </div>
      <ul className="history">
        {items.map((h, i) => (
          <li key={i}>
            <button type="button" onClick={() => onPick(h)} title={`${h.question} — ${h.model}`}>
              <span className="muted mono" style={{ fontSize: 11 }}>
                {h.model}
              </span>{" "}
              {h.question}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
