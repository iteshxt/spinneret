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
        <svg viewBox="0 0 20 20" fill="currentColor">
          <path
            fillRule="evenodd"
            d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm.75-11.5a.75.75 0 0 0-1.5 0v4a.75.75 0 0 0 1.5 0v-4ZM10 15a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
            clipRule="evenodd"
          />
        </svg>
      </span>
      <div>
        <p className="offline-alert-title">The live demo is offline</p>
        <p className="offline-alert-body">
          Spinneret's models run on the author's laptop GPU, which is switched off right now, so
          questions can't be answered. Schemas, example questions and the evaluation are still
          available.
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
