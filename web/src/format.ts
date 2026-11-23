export const pct = (x: number | null | undefined, digits = 1): string =>
  x == null ? "—" : `${(x * 100).toFixed(digits)}%`;

export const secs = (ms: number | null | undefined): string =>
  ms == null ? "—" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;

export const params = (m: number | null | undefined): string =>
  m == null ? "—" : m >= 1000 ? `${(m / 1000).toFixed(1)}B` : `${Math.round(m)}M`;

export function toCsv(columns: string[], rows: unknown[][]): string {
  const cell = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [columns, ...rows].map((r) => r.map(cell).join(",")).join("\n");
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}
