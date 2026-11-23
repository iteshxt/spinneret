import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { NavLink, Route, Routes, useSearchParams } from "react-router-dom";
import { api, ApiError } from "./api/client";
import { ModelSwitcher } from "./components/ModelSwitcher";
import { About } from "./pages/About";
import { Evaluation } from "./pages/Evaluation";
import { Workspace } from "./pages/Workspace";
import { useTheme } from "./theme";

export function App() {
  const { dark, toggle } = useTheme();
  const [params, setParams] = useSearchParams();
  const [sheetOpen, setSheetOpen] = useState(false);

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    retry: false,
    refetchInterval: (q) => (q.state.data?.status === "ok" ? false : 2000),
  });
  const models = useQuery({
    queryKey: ["models", health.data?.models_loaded.length ?? 0],
    queryFn: api.models,
    enabled: health.isSuccess,
  });

  const list = models.data ?? [];
  const fallback = list.find((m) => m.default)?.id ?? list[0]?.id ?? "";
  const model = list.some((m) => m.id === params.get("model")) ? params.get("model")! : fallback;
  const setModel = (id: string) => {
    const next = new URLSearchParams(params);
    next.set("model", id);
    setParams(next, { replace: true });
  };

  // "M" focuses the model switcher (design §10.3)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [contenteditable]")) return;
      if (e.key === "m" || e.key === "M") {
        const el = document.querySelector<HTMLSelectElement>(
          sheetOpen ? ".sidebar .model-switcher" : ".header .model-switcher",
        );
        el?.focus();
        e.preventDefault();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sheetOpen]);

  const unreachable = health.error instanceof ApiError && health.error.status === 0;
  const h = health.data;
  const dot =
    h?.status === "ok"
      ? ["var(--ok)", h.device === "cpu" ? "ready · CPU" : "ready · GPU"]
      : h?.status === "error"
        ? ["var(--error)", "models failed to load"]
        : h
          ? ["var(--warn)", "warming up"]
          : ["var(--text-muted)", "connecting"];

  return (
    <div className="app">
      <header className="header">
        <NavLink to={{ pathname: "/", search: params.toString() }} className="brand">
          <span className="brand-mark" aria-hidden>
            ◈
          </span>
          Spinneret
        </NavLink>
        <nav className="nav" aria-label="Main">
          <NavLink to={{ pathname: "/", search: params.toString() }} end>
            Query
          </NavLink>
          <NavLink to="/evaluation">Evaluation</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
        <div className="header-right">
          {list.length > 0 && (
            <span className="model-switcher-wrap">
              <ModelSwitcher models={list} value={model} onChange={setModel} />
            </span>
          )}
          <span
            className="status-dot"
            style={{ "--dot": dot[0] } as React.CSSProperties}
            title={h?.device ?? ""}
          >
            <span className="model-switcher-wrap">{dot[1]}</span>
          </span>
          <button
            type="button"
            className="icon-btn"
            onClick={toggle}
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
          >
            {dark ? "☀" : "☾"}
          </button>
        </div>
      </header>
      {unreachable && (
        <div className="banner" role="alert">
          Can't reach the Spinneret server. Is <code>spinneret serve</code> running?
          <button type="button" className="btn" onClick={() => health.refetch()}>
            Retry
          </button>
        </div>
      )}
      {h?.status === "error" && (
        <div className="banner" role="alert">
          The models failed to load: {h.error}
        </div>
      )}
      <Routes>
        <Route
          path="/"
          element={
            <Workspace
              models={list}
              model={model}
              setModel={setModel}
              ready={h?.status === "ok"}
              dark={dark}
              sheetOpen={sheetOpen}
              setSheetOpen={setSheetOpen}
            />
          }
        />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/about" element={<About models={list} />} />
        <Route path="*" element={<div className="page">Page not found.</div>} />
      </Routes>
    </div>
  );
}
