import type { Candidate } from "../api/types";

const LABEL: Record<Candidate["status"], string> = {
  ok: "ran ✓",
  error: "error",
  timeout: "timeout",
  rejected: "rejected",
  not_run: "not run",
};

interface Props {
  candidates: Candidate[];
  selected: number;
  open: boolean;
  onToggle: (open: boolean) => void;
  onPick: (c: Candidate) => void;
}

export function CandidatesList({ candidates, selected, open, onToggle, onPick }: Props) {
  return (
    <details
      className="details"
      open={open}
      onToggle={(e) => onToggle((e.currentTarget as HTMLDetailsElement).open)}
    >
      <summary>{candidates.length} candidates</summary>
      <ul className="candidates">
        {candidates.map((c) => (
          <li key={c.rank}>
            <button
              type="button"
              className={`candidate ${c.rank === selected ? "selected" : ""}`}
              onClick={() => onPick(c)}
              aria-label={`Candidate ${c.rank + 1}, ${LABEL[c.status]}, score ${c.score.toFixed(2)}: ${c.sql}`}
            >
              <span className="muted mono">#{c.rank + 1}</span>
              <code>{c.sql}</code>
              <span className="muted mono" style={{ fontSize: 12 }}>
                {c.score.toFixed(2)}
              </span>
              <span className={`cand-status ${c.status}`}>{LABEL[c.status]}</span>
            </button>
          </li>
        ))}
      </ul>
    </details>
  );
}
