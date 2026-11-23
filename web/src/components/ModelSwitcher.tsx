import type { ModelInfo } from "../api/types";
import { params, pct, secs } from "../format";

interface Props {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
  id?: string;
}

export const modelOption = (m: ModelInfo) =>
  `${m.label} · ${params(m.params_m)} · ${pct(m.dev_ex, 0)} EX · ${secs(m.p50_ms)}`;

export function ModelSwitcher({ models, value, onChange, id }: Props) {
  return (
    <select
      id={id}
      className="select model-switcher"
      aria-label="Model"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{ width: "auto", maxWidth: 420 }}
    >
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {modelOption(m)}
        </option>
      ))}
    </select>
  );
}
