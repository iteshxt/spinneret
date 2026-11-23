import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";

export interface Option {
  value: string;
  label: ReactNode;
  /** Shown under the label in the menu (not on the button). */
  detail?: ReactNode;
  /** Plain text for type-ahead and screen readers when `label` is not a string. */
  text?: string;
}

interface Props {
  options: Option[];
  value: string;
  onChange: (value: string) => void;
  id?: string;
  className?: string;
  "aria-label"?: string;
  /** What the closed button shows; defaults to the selected option's label. */
  display?: ReactNode;
  align?: "start" | "end";
}

/**
 * A styled dropdown that behaves like a native select: a button that opens a listbox.
 * Keyboard: Enter, Space or the arrows open it; arrows, Home and End move; Enter or Space
 * picks; Escape or Tab closes; typing letters jumps to the next matching option.
 */
export function Select({ options, value, onChange, id, className, display, align = "start", ...rest }: Props) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const typed = useRef({ text: "", at: 0 });
  const listId = useId();
  const selected = Math.max(0, options.findIndex((o) => o.value === value));
  const textOf = (o: Option) => o.text ?? (typeof o.label === "string" ? o.label : o.value);

  const show = () => {
    setActive(selected);
    setOpen(true);
  };
  const close = (refocus = true) => {
    setOpen(false);
    if (refocus) button.current?.focus();
  };
  const pick = (i: number) => {
    const o = options[i];
    if (o && o.value !== value) onChange(o.value);
    close();
  };

  // close on a click outside
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) close(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [open]);

  // move focus into the list when it opens, and keep the active option in view
  useEffect(() => {
    if (open) list.current?.focus();
  }, [open]);
  useEffect(() => {
    if (open) list.current?.querySelector<HTMLElement>(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" });
  }, [open, active]);

  const typeAhead = (key: string) => {
    const now = Date.now();
    const t = typed.current;
    t.text = now - t.at > 700 ? key : t.text + key;
    t.at = now;
    const from = (active + (t.text.length === 1 ? 1 : 0)) % options.length;
    for (let k = 0; k < options.length; k++) {
      const i = (from + k) % options.length;
      if (textOf(options[i]).toLowerCase().startsWith(t.text.toLowerCase())) return setActive(i);
    }
  };

  const onButtonKey = (e: React.KeyboardEvent) => {
    if (["ArrowDown", "ArrowUp", "Enter", " "].includes(e.key)) {
      e.preventDefault();
      show();
    }
  };

  const onListKey = (e: React.KeyboardEvent) => {
    const last = options.length - 1;
    if (e.key === "ArrowDown") setActive((i) => Math.min(last, i + 1));
    else if (e.key === "ArrowUp") setActive((i) => Math.max(0, i - 1));
    else if (e.key === "Home") setActive(0);
    else if (e.key === "End") setActive(last);
    else if (e.key === "Enter" || e.key === " ") pick(active);
    else if (e.key === "Escape") close();
    else if (e.key === "Tab") return close(false);
    else if (e.key.length === 1 && /\S/.test(e.key)) typeAhead(e.key);
    else return;
    e.preventDefault();
  };

  return (
    <div className={`dropdown${open ? " open" : ""}`} ref={root}>
      <button
        ref={button}
        id={id}
        type="button"
        className={`dropdown-button ${className ?? ""}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-label={rest["aria-label"]}
        onClick={() => (open ? close() : show())}
        onKeyDown={onButtonKey}
      >
        <span className="dropdown-value">{display ?? options[selected]?.label}</span>
        <svg className="dropdown-chevron" viewBox="0 0 12 12" aria-hidden>
          <path d="M3 4.5 6 7.5 9 4.5" />
        </svg>
      </button>
      <AnimatePresence>
        {open && (
          <motion.ul
            ref={list}
            id={listId}
            role="listbox"
            tabIndex={-1}
            aria-label={rest["aria-label"]}
            aria-activedescendant={`${listId}-${active}`}
            className={`dropdown-menu align-${align}`}
            onKeyDown={onListKey}
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.16, ease: [0.2, 0.8, 0.2, 1] }}
          >
            {options.map((o, i) => (
              <li
                key={o.value}
                id={`${listId}-${i}`}
                data-i={i}
                role="option"
                aria-selected={i === selected}
                className={`dropdown-option${i === active ? " active" : ""}`}
                onPointerMove={() => setActive(i)}
                onClick={() => pick(i)}
              >
                <span className="dropdown-label">{o.label}</span>
                {o.detail && <span className="dropdown-detail">{o.detail}</span>}
                {i === selected && (
                  <svg className="dropdown-check" viewBox="0 0 12 12" aria-hidden>
                    <path d="M2.5 6.5 5 9l4.5-6" />
                  </svg>
                )}
              </li>
            ))}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
}
