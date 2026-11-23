interface Props {
  checking: boolean;
  onCheck: () => void;
}

/** The hosted site's live server (the author's laptop) is off: what is off, what still works,
 * and a manual check on top of the automatic one every 30 s. */
export function OfflineNotice({ checking, onCheck }: Props) {
  return (
    <section className="offline-alert" role="status" aria-label="Live demo status">
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
        <span className="offline-alert-auto">Checks every 30 s</span>
        <button type="button" className="btn btn-danger" onClick={onCheck} disabled={checking}>
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
