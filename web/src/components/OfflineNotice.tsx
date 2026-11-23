import { motion } from "framer-motion";

interface Props {
  checking: boolean;
  onCheck: () => void;
}

/** The hosted site's live server (the author's laptop) is off: say so calmly, say what still
 * works, and offer a manual check on top of the automatic one every 30 s. */
export function OfflineNotice({ checking, onCheck }: Props) {
  return (
    <motion.section
      className="offline-notice"
      role="status"
      aria-label="Live demo status"
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <span className="offline-icon" aria-hidden>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
          <path d="M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5Z" strokeLinejoin="round" />
        </svg>
      </span>
      <div className="offline-text">
        <p className="offline-title">The live demo is asleep</p>
        <p className="offline-body">
          Spinneret's models run on the author's laptop GPU, which is off right now. You can still
          browse every database, its schema and example questions; asking works again as soon as
          the laptop is back.
        </p>
      </div>
      <div className="offline-actions">
        <button type="button" className="btn" onClick={onCheck} disabled={checking}>
          {checking ? (
            <>
              <span className="spinner" aria-hidden /> Checking…
            </>
          ) : (
            "Check again"
          )}
        </button>
        <span className="offline-auto">checks every 30 s</span>
      </div>
    </motion.section>
  );
}
