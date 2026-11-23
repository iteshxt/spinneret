import { motion } from "framer-motion";

interface Props {
  checking: boolean;
  onCheck: () => void;
}

/** The hosted site's live server (the author's laptop) is off. An alert that can't be missed:
 * what is off, what still works, and a manual check on top of the automatic one every 30 s. */
export function OfflineNotice({ checking, onCheck }: Props) {
  return (
    <motion.section
      className="offline-alert"
      role="status"
      aria-label="Live demo status"
      initial={{ opacity: 0, y: -10, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <span className="offline-alert-icon" aria-hidden>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
          <path d="M12 3v8" />
          <path d="M6.3 6.8a8 8 0 1 0 11.4 0" />
        </svg>
      </span>
      <div>
        <p className="offline-alert-kicker">
          <span className="pulse-dot" aria-hidden />
          Live demo offline
        </p>
        <p className="offline-alert-title">Questions can't be answered right now</p>
        <p className="offline-alert-body">
          Spinneret's models run on the author's laptop GPU, and it's switched off. Schemas, example
          questions and the evaluation still work here; asking comes back the moment the laptop
          does.
        </p>
      </div>
      <div className="offline-alert-actions">
        <button type="button" className="btn btn-danger" onClick={onCheck} disabled={checking}>
          {checking ? (
            <>
              <span className="spinner" aria-hidden /> Checking…
            </>
          ) : (
            "Check again"
          )}
        </button>
        <span className="offline-alert-auto">Auto-checks every 30 s</span>
      </div>
    </motion.section>
  );
}
