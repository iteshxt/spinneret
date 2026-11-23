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

export interface ClientOptions {
  /** Static site: read-only data from /snapshot, and an offline state instead of an error. */
  hosted: boolean;
  /** Origin of the live server ("" for the page's own origin). */
  apiUrl: string;
}

export const OFFLINE_MESSAGE =
  "The live demo is offline right now: it runs on the author's laptop GPU, which isn't on.";
const STATUS_TIMEOUT_MS = 5000; // health and models: give up quickly and show the offline state
const QUERY_TIMEOUT_MS = 60_000; // a full queue of questions on the GPU takes well under this
const EXECUTE_TIMEOUT_MS = 30_000; // SQL is stopped after 5 s on the server
const OFFLINE_HEALTH: Health = {
  status: "offline",
  device: "",
  databases: 0,
  models_loaded: [],
  error: null,
};

export function createApi({ hosted, apiUrl }: ClientOptions) {
  const base = apiUrl.replace(/\/+$/, "");

  async function request<T>(path: string, init?: RequestInit, timeoutMs?: number): Promise<T> {
    let res: Response;
    try {
      res = await fetch(base + path, {
        ...init,
        headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
        signal: timeoutMs ? AbortSignal.timeout(timeoutMs) : undefined,
      });
    } catch {
      throw hosted
        ? new ApiError(0, "offline", OFFLINE_MESSAGE)
        : new ApiError(0, "unreachable", "Can't reach the Spinneret server.");
    }
    const body = await res.json().catch(() => null);
    // A hosted site's live server sits behind a tunnel: when the laptop is off, the tunnel
    // answers for it with its own error page, which is not the API's JSON.
    if (hosted && (body === null || (!res.ok && !body.error && !body.detail))) {
      throw new ApiError(0, "offline", OFFLINE_MESSAGE);
    }
    if (!res.ok) {
      const err = body?.error;
      const detail = Array.isArray(body?.detail) ? body.detail[0]?.msg : undefined;
      throw new ApiError(res.status, err?.code ?? "error", err?.message ?? detail ?? res.statusText);
    }
    return body as T;
  }

  async function snapshot<T>(file: string): Promise<T> {
    const res = await fetch(`/snapshot/${file}`).catch(() => null);
    if (!res?.ok) throw new ApiError(res?.status ?? 0, "not_found", "This page's data is missing.");
    return (await res.json()) as T;
  }

  const offline = (e: unknown) => e instanceof ApiError && e.code === "offline";

  return {
    health: async (): Promise<Health> => {
      if (!hosted) return request<Health>("/api/health");
      try {
        return await request<Health>("/api/health", undefined, STATUS_TIMEOUT_MS);
      } catch (e) {
        if (offline(e)) return OFFLINE_HEALTH;
        throw e;
      }
    },
    models: async (): Promise<ModelInfo[]> => {
      if (!hosted) return request<ModelInfo[]>("/api/models");
      try {
        return await request<ModelInfo[]>("/api/models", undefined, STATUS_TIMEOUT_MS);
      } catch (e) {
        if (offline(e)) return snapshot<ModelInfo[]>("models.json");
        throw e;
      }
    },
    // read-only data never changes while the site is up, so a hosted site reads its own copy
    databases: () =>
      hosted
        ? snapshot<DatabaseSummary[]>("databases.json")
        : request<DatabaseSummary[]>("/api/databases"),
    schema: (db: string) =>
      hosted
        ? snapshot<SchemaInfo>(`schema/${encodeURIComponent(db)}.json`)
        : request<SchemaInfo>(`/api/databases/${encodeURIComponent(db)}/schema`),
    examples: async (db: string, limit = 6): Promise<Example[]> =>
      hosted
        ? (await snapshot<Example[]>(`examples/${encodeURIComponent(db)}.json`)).slice(0, limit)
        : request<Example[]>(`/api/databases/${encodeURIComponent(db)}/examples?limit=${limit}`),
    summary: () => (hosted ? snapshot<Summary>("summary.json") : request<Summary>("/api/summary")),
    query: (req: QueryRequest) =>
      request<QueryResponse>(
        "/api/query",
        { method: "POST", body: JSON.stringify(req) },
        QUERY_TIMEOUT_MS,
      ),
    execute: (db_id: string, sql: string) =>
      request<Execution>(
        "/api/execute",
        { method: "POST", body: JSON.stringify({ db_id, sql }) },
        EXECUTE_TIMEOUT_MS,
      ),
  };
}

export const HOSTED = import.meta.env.VITE_HOSTED === "1";

export const api = createApi({ hosted: HOSTED, apiUrl: import.meta.env.VITE_API_URL ?? "" });
