import { ApiError, createApi } from "./client";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

function stubFetch(handler: (url: string) => Response | Promise<Response>) {
  const calls: string[] = [];
  vi.stubGlobal("fetch", async (url: string) => {
    calls.push(url);
    return handler(url);
  });
  return calls;
}

afterEach(() => vi.unstubAllGlobals());

describe("hosted client", () => {
  const api = createApi({ hosted: true, apiUrl: "https://laptop.example.ts.net/" });

  it("reads read-only data from the static snapshot, never the live server", async () => {
    const calls = stubFetch((url) =>
      json(url.includes("examples") ? [1, 2, 3, 4, 5, 6, 7, 8].map((i) => ({ question: `q${i}` })) : []),
    );
    await api.databases();
    await api.schema("concert_singer");
    await api.summary();
    expect(await api.examples("pets_1", 3)).toHaveLength(3);
    expect(calls).toEqual([
      "/snapshot/databases.json",
      "/snapshot/schema/concert_singer.json",
      "/snapshot/summary.json",
      "/snapshot/examples/pets_1.json",
    ]);
  });

  it("asks the live server for health and models while it is up", async () => {
    const calls = stubFetch((url) =>
      url.endsWith("/api/health") ? json({ status: "ok", models_loaded: ["t1"] }) : json([{ id: "t1" }]),
    );
    expect((await api.health()).status).toBe("ok");
    expect(await api.models()).toEqual([{ id: "t1" }]);
    expect(calls).toEqual([
      "https://laptop.example.ts.net/api/health",
      "https://laptop.example.ts.net/api/models",
    ]);
  });

  it("turns an unreachable server into the offline state, with models from the snapshot", async () => {
    const calls = stubFetch((url) => {
      if (url.startsWith("/snapshot/")) return json([{ id: "t3" }]);
      throw new TypeError("Failed to fetch");
    });
    expect((await api.health()).status).toBe("offline");
    expect(await api.models()).toEqual([{ id: "t3" }]);
    expect(calls.at(-1)).toBe("/snapshot/models.json");
  });

  it("treats the tunnel's own error page as offline, but keeps the API's errors", async () => {
    stubFetch(() => new Response("<html>502 Bad Gateway</html>", { status: 502 }));
    const err = await api.query({ db_id: "x", question: "q" }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe("offline");

    stubFetch(() => json({ error: { code: "busy", message: "the demo is busy" } }, 503));
    const busy = await api.query({ db_id: "x", question: "q" }).catch((e) => e);
    expect([busy.status, busy.code, busy.message]).toEqual([503, "busy", "the demo is busy"]);
  });
});

describe("local client", () => {
  const api = createApi({ hosted: false, apiUrl: "" });

  it("uses the page's own server for everything and reports it unreachable", async () => {
    const calls = stubFetch(() => {
      throw new TypeError("Failed to fetch");
    });
    const err = await api.health().catch((e) => e);
    expect([err.status, err.code]).toEqual([0, "unreachable"]);
    await api.databases().catch(() => null);
    expect(calls).toEqual(["/api/health", "/api/databases"]);
  });
});
