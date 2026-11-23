import { useState } from "react";

interface Props {
  /** Ask the live server again; resolves once the answer (or the failure) is in. */
  onCheck: () => Promise<unknown>;
}

const MIN_CHECK_MS = 900; // an off laptop fails in milliseconds; keep "Checking…" long enough to see

const time = (d: Date) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

/** The hosted site's live server (the author's laptop) is off: what is off, what still works,
 * and a manual check on top of the automatic one every 30 s. When a check finds the server,
 * the app replaces this notice with the live demo. */
export function OfflineNotice({ onCheck }: Props) {
  const [checking, setChecking] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);

  const check = async () => {
    setChecking(true);
    const started = performance.now();
    await onCheck().catch(() => null);
    const left = MIN_CHECK_MS - (performance.now() - started);
    if (left > 0) await new Promise((r) => setTimeout(r, left));
    setCheckedAt(new Date());
    setChecking(false);
  };

  return (
    <section className="offline-alert" aria-label="Live demo status">
      <span className="offline-alert-icon" aria-hidden>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <path d="M12 3v8" />
          <path d="M6.3 6.8a8 8 0 1 0 11.4 0" />
        </svg>
      </span>
      <div>
        <p className="offline-alert-kicker">Live demo offline</p>
        <p className="offline-alert-title">Questions can't be answered right now</p>
        <p className="offline-alert-body">
          Spinneret's models run on the author's laptop GPU, and it's switched off. Schemas, example
          questions and the evaluation still work here; asking comes back the moment the laptop
          does.
        </p>
      </div>
      <div className="offline-alert-actions">
        <span className="offline-alert-auto" aria-live="polite">
          {checking
            ? "Contacting the laptop…"
            : checkedAt
              ? `Still offline · checked ${time(checkedAt)}`
              : "Checks every 30 s"}
        </span>
        <button
          type="button"
          className="btn btn-danger"
          onClick={check}
          disabled={checking}
          aria-busy={checking}
        >
          {checking ? (
            <>
              <span className="spinner" aria-hidden /> Checking…
            </>
          ) : (
            "Check again"
          )}
        </button>
      </div>
    </section>
  );
}
