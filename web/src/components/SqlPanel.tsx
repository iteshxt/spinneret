import { sql as sqlLang } from "@codemirror/lang-sql";
import { EditorView, keymap } from "@codemirror/view";
import CodeMirror from "@uiw/react-codemirror";
import { useMemo } from "react";

interface Props {
  value: string;
  generated: string | null; // the model's SQL; null for none yet
  onChange: (sql: string) => void;
  onRun: () => void;
  onReset: () => void;
  dark: boolean;
  running?: boolean;
}

export function SqlPanel({ value, generated, onChange, onRun, onReset, dark, running }: Props) {
  const edited = generated != null && value.trim() !== generated.trim();
  const extensions = useMemo(
    () => [
      sqlLang(),
      EditorView.lineWrapping,
      keymap.of([{ key: "Mod-Enter", run: () => (onRun(), true) }]),
    ],
    [onRun],
  );
  return (
    <div>
      <div className="panel sql-editor">
        <CodeMirror
          value={value}
          onChange={onChange}
          extensions={extensions}
          theme={dark ? "dark" : "light"}
          basicSetup={{ foldGutter: false, highlightActiveLine: false }}
          aria-label="Generated SQL, editable"
          minHeight="44px"
        />
      </div>
      <div className="status-line">
        {edited && <span className="badge accent">Edited</span>}
        {edited && (
          <button type="button" className="btn" onClick={onReset}>
            Reset
          </button>
        )}
        <button
          type="button"
          className="btn"
          onClick={onRun}
          disabled={!value.trim() || running}
          title="Run SQL (Ctrl/Cmd + Enter)"
        >
          ▶ Run
        </button>
      </div>
    </div>
  );
}
