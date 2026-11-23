import { useQuery } from "@tanstack/react-query";
import { MotionConfig, motion, useScroll, useSpring } from "framer-motion";
import { lazy, Suspense, useEffect, useState } from "react";
import { NavLink, Route, Routes, useLocation, useSearchParams } from "react-router-dom";
import { api, ApiError, OFFLINE_MESSAGE } from "./api/client";
import { Backdrop } from "./components/Backdrop";
import { GitHubIcon } from "./components/icons";
import { ModelSwitcher } from "./components/ModelSwitcher";
import { SiteFooter } from "./components/SiteFooter";
import { About } from "./pages/About";
import { Evaluation } from "./pages/Evaluation";
import { Landing } from "./pages/Landing";
import { Privacy } from "./pages/Privacy";
import { REPO } from "./site";
import { useTheme } from "./theme";

// The demo carries the SQL editor (CodeMirror); load it only when someone opens the demo.
const Workspace = lazy(() => import("./pages/Workspace").then((m) => ({ default: m.Workspace })));

export function App() {
  const { dark, toggle } = useTheme();
  const [params, setParams] = useSearchParams();
  const [sheetOpen, setSheetOpen] = useState(false);
  const { pathname, hash } = useLocation();
  const inApp = pathname === "/query";

  // flat bar at the top of the page; a floating pill once the page scrolls
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 16);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);
  // new page: start at the top (or at the #section asked for)
  useEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [pathname, hash]);
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 200, damping: 30, restDelta: 0.001 });

  const health = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    retry: false,
    // loading: watch it warm up; offline (hosted site, laptop off): look again now and then
    refetchInterval: (q) => {
      const status = q.state.data?.status;
      return status === "ok" ? false : status === "offline" ? 30_000 : 2000;
    },
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
        const el = document.querySelector<HTMLElement>(
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
        : h?.status === "offline"
          ? ["var(--text-muted)", "offline"]
        : h
          ? ["var(--warn)", "warming up"]
          : ["var(--text-muted)", "connecting"];

  return (
    <MotionConfig reducedMotion="user">
    <Backdrop />
    {!inApp && <motion.div className="progress" style={{ scaleX: progress }} aria-hidden />}
    <div className={`app${inApp ? " wide" : ""}`}>
      <div className={`header-bar${scrolled ? " scrolled" : ""}`}>
        <header className="header">
          <NavLink to="/" className="brand" aria-label="Spinneret home">
            <span className="brand-mark" aria-hidden>
              ◈
            </span>
            Spinneret
          </NavLink>
          <nav className="nav" aria-label="Main">
            <NavLink to={{ pathname: "/query", search: params.toString() }}>Demo</NavLink>
            <NavLink to="/evaluation">Evaluation</NavLink>
            <NavLink to="/about">About</NavLink>
          </nav>
          <div className="header-right">
            {inApp && list.length > 0 && (
              <span className="model-switcher-wrap">
                <ModelSwitcher models={list} value={model} onChange={setModel} align="end" />
              </span>
            )}
            {inApp && (
              <span
                className="status-dot"
                style={{ "--dot": dot[0] } as React.CSSProperties}
                title={h?.device ?? ""}
              >
                <span className="model-switcher-wrap">{dot[1]}</span>
              </span>
            )}
            {!inApp && (
              <a className="icon-btn hide-sm" href={REPO} aria-label="Source code on GitHub">
                <GitHubIcon />
              </a>
            )}
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
      </div>
      {inApp && unreachable && (
        <div className="banner" role="alert">
          Can't reach the Spinneret server. Is <code>spinneret serve</code> running?
          <button type="button" className="btn" onClick={() => health.refetch()}>
            Retry
          </button>
        </div>
      )}
      {inApp && h?.status === "offline" && (
        <div className="banner" role="status">
          {OFFLINE_MESSAGE} You can still browse the databases and example questions; asking
          works again once it's back.
          <button type="button" className="btn" onClick={() => health.refetch()}>
            Check again
          </button>
        </div>
      )}
      {inApp && h?.status === "error" && (
        <div className="banner" role="alert">
          The models failed to load: {h.error}
        </div>
      )}
      <Suspense
        fallback={
          <div className="page">
            <div className="loader" />
          </div>
        }
      >
      <Routes>
        <Route path="/" element={<Landing models={list} />} />
        <Route
          path="/query"
          element={
            <Workspace
              models={list}
              model={model}
              setModel={setModel}
              ready={h?.status === "ok"}
              offline={h?.status === "offline"}
              dark={dark}
              sheetOpen={sheetOpen}
              setSheetOpen={setSheetOpen}
            />
          }
        />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/about" element={<About models={list} />} />
        <Route path="/privacy" element={<Privacy />} />
        <Route
          path="*"
          element={
            <div className="page">
              <h1>Page not found</h1>
              <p className="page-lead">
                There's nothing at this address. Go to the <NavLink to="/">home page</NavLink> or{" "}
                <NavLink to="/query">open the demo</NavLink>.
              </p>
            </div>
          }
        />
      </Routes>
      </Suspense>
      <SiteFooter />
    </div>
    </MotionConfig>
  );
}

