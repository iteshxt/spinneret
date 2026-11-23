import type { ModelInfo } from "../api/types";
import { params, pct, secs } from "../format";
import { Select } from "./Select";

interface Props {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
  id?: string;
  align?: "start" | "end";
}

export const modelOption = (m: ModelInfo) =>
  `${m.label} · ${params(m.params_m)} · ${pct(m.dev_ex, 0)} EX · ${secs(m.p50_ms)}`;

const split = (m: ModelInfo) => {
  const [name, desc] = m.label.split(" · ");
  return { name, desc: desc ?? "" };
};

export function ModelSwitcher({ models, value, onChange, id, align }: Props) {
  const current = models.find((m) => m.id === value);
  return (
    <Select
      id={id}
      className="model-switcher"
      aria-label="Model"
      value={value}
      onChange={onChange}
      align={align}
      display={
        current && (
          <>
            <strong>{split(current).name}</strong>
            <span className="dropdown-meta">
              {pct(current.dev_ex, 0)} in {secs(current.p50_ms)}
            </span>
          </>
        )
      }
      options={models.map((m) => ({
        value: m.id,
        text: split(m).name,
        label: <strong>{split(m).name}</strong>,
        detail: (
          <>
            {split(m).desc}
            <br />
            {params(m.params_m)} parameters, {pct(m.dev_ex)} correct, {secs(m.p50_ms)} median
          </>
        ),
      }))}
    />
  );
}
