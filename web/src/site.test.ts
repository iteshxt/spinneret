import { HERO_EXAMPLES, plain, spans } from "./site";

describe("hero example markup", () => {
  it("splits marked spans", () => {
    expect(spans("How [many|h] rows?")).toEqual([
      { text: "How " },
      { text: "many", key: "h" },
      { text: " rows?" },
    ]);
    expect(plain("select [count(*)|h] from t")).toBe("select count(*) from t");
  });

  it("links every question word to some SQL and back", () => {
    for (const ex of HERO_EXAMPLES) {
      const q = new Set(spans(ex.question).flatMap((s) => (s.key ? [s.key] : [])));
      const s = new Set(spans(ex.sql).flatMap((s) => (s.key ? [s.key] : [])));
      expect([...q].sort()).toEqual([...s].sort());
    }
  });
});
