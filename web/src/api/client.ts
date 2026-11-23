import type {
  DatabaseSummary,
  Example,
  Execution,
  Health,
  ModelInfo,
  QueryRequest,
  QueryResponse,
  SchemaInfo,
  Summary,
} from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "unreachable", "Can't reach the Spinneret server.");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = body?.error;
    const detail = Array.isArray(body?.detail) ? body.detail[0]?.msg : undefined;
    throw new ApiError(res.status, err?.code ?? "error", err?.message ?? detail ?? res.statusText);
  }
  return body as T;
}

export const api = {
  health: () => request<Health>("/api/health"),
  models: () => request<ModelInfo[]>("/api/models"),
  databases: () => request<DatabaseSummary[]>("/api/databases"),
  schema: (db: string) => request<SchemaInfo>(`/api/databases/${encodeURIComponent(db)}/schema`),
  examples: (db: string, limit = 6) =>
    request<Example[]>(`/api/databases/${encodeURIComponent(db)}/examples?limit=${limit}`),
  query: (req: QueryRequest) =>
    request<QueryResponse>("/api/query", { method: "POST", body: JSON.stringify(req) }),
  execute: (db_id: string, sql: string) =>
    request<Execution>("/api/execute", { method: "POST", body: JSON.stringify({ db_id, sql }) }),
  summary: () => request<Summary>("/api/summary"),
};
