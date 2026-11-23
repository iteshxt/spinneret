// Mirrors the API models in spinneret/api/models.py (specs §9).

export interface Health {
  status: "ok" | "loading" | "error";
  device: string;
  databases: number;
  models_loaded: string[];
  error: string | null;
}

export interface ModelInfo {
  id: string;
  label: string;
  generator: string;
  ranker: string | null;
  default: boolean;
  loaded: boolean;
  skeleton: boolean | null;
  params_m: number | null;
  dev_em: number | null;
  dev_ex: number | null;
  p50_ms: number | null;
  p95_ms: number | null;
  run: string | null;
}

export interface DatabaseSummary {
  db_id: string;
  n_tables: number;
  n_columns: number;
}

export interface ColumnInfo {
  name: string;
  type: string;
  is_pk: boolean;
}

export interface TableInfo {
  name: string;
  columns: ColumnInfo[];
}

export interface SchemaInfo {
  db_id: string;
  tables: TableInfo[];
  foreign_keys: { source: string; target: string }[];
}

export interface Example {
  question: string;
  hardness: string;
}

export type ExecStatus = "ok" | "error" | "timeout" | "rejected" | "failed";

export interface Execution {
  status: ExecStatus;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  error: string | null;
}

export interface Candidate {
  rank: number;
  sql: string;
  score: number;
  status: "ok" | "error" | "timeout" | "rejected" | "not_run";
}

export interface SchemaItems {
  tables: { name: string; score: number; kept: boolean }[];
  columns: { table: string; name: string; score: number; kept: boolean }[];
}

export interface QueryResponse {
  model: string;
  sql: string;
  skeleton: string | null;
  selected_rank: number;
  candidates: Candidate[];
  schema_items: SchemaItems | null;
  matched_values: Record<string, string[]>;
  execution: Execution;
  timings_ms: Record<string, number>;
}

export interface QueryRequest {
  db_id: string;
  question: string;
  options?: { model?: string; beam_size?: number; decoding?: "eg" | "beam"; max_rows?: number };
}

// --- evaluation summary (reports/summary/summary.json, spinneret/eval/summary.py) ---

export interface Scores {
  em: number;
  ex: number;
  valid_sql: number | null;
  ex_plug: number | null;
  em_by_level: Record<string, number>;
  ex_by_level: Record<string, number>;
}

export interface LatencyStage {
  p50: number;
  p95: number;
  mean: number;
}

export interface TierSummary {
  tier: string;
  run: string;
  description: string;
  scores: Scores | null;
  latency: {
    latency_ms: Record<string, LatencyStage>;
    generator_params_m: number;
    ranker_params_m?: number;
    peak_gpu_memory_mb: number | null;
    device: string;
  } | null;
  taxonomy: {
    ex_wrong: number;
    primary: Record<string, number>;
    ranker_miss?: number;
    examples_per_category?: Record<
      string,
      { id: string; question: string; gold_sql: string; pred_sql: string }[]
    >;
  } | null;
  question_types: { types: Record<string, { count: number; ex: number }> } | null;
}

export interface Summary {
  tiers: TierSummary[];
  e2_decoding: { mode: string; run: string; scores: Scores | null; generation_seconds: number | null }[];
  training_ablations: {
    id: string;
    question: string;
    variants: { label: string; run: string; scores: Scores | null }[];
  }[];
  e5_ranker_k: Record<string, { scores: Scores | null; dev_coverage: number | null }>;
}

export interface ApiErrorBody {
  error: { code: string; message: string };
}
