"""Figures for the final report, written as SVG/HTML and rendered to PNG by capture.mjs.

Run from the repo root:  python3 docs/report/build/figures.py
Every number comes from data/processed/stats.json or reports/summary/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/report/build/html"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1c1c1a"
MUTED = "#5f5f58"
LINE = "#8a8a82"
ACCENT = "#b45309"
SOFT = "#fdf3e7"
BOX = "#ffffff"
GREY = "#f1f1ee"
FONT = "'Liberation Sans', Arial, sans-serif"
MONO = "'DejaVu Sans Mono', monospace"


def page(name: str, w: int, h: int, svg: str) -> None:
    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#fff;}} svg{{display:block}}
text{{font-family:{FONT};fill:{INK}}} .m{{font-family:{MONO}}}
</style></head><body><svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
viewBox="0 0 {w} {h}">
<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>
<marker id="aa" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{ACCENT}"/></marker></defs>
{svg}</svg></body></html>"""
    (OUT / f"{name}.html").write_text(html, encoding="utf-8")
    (OUT / f"{name}.size").write_text(f"{w} {h}")


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def box(x, y, w, h, title, lines=(), *, fill=BOX, stroke=LINE, accent=False, mono=False):
    st = ACCENT if accent else stroke
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
           f'stroke="{st}" stroke-width="{1.6 if accent else 1.2}"/>']
    ty = y + 24 if lines else y + h / 2 + 5
    out.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="15" '
               f'font-weight="700">{esc(title)}</text>')
    for i, ln in enumerate(lines):
        cls = ' class="m"' if mono else ""
        out.append(f'<text{cls} x="{x + w / 2}" y="{y + 44 + i * 17}" text-anchor="middle" '
                   f'font-size="12.5" fill="{MUTED}">{esc(ln)}</text>')
    return "\n".join(out)


def diamond(cx, cy, w, h, text, fill=SOFT):
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    return (f'<polygon points="{pts}" fill="{fill}" stroke="{ACCENT}" stroke-width="1.4"/>'
            f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-size="13" '
            f'font-weight="700">{esc(text)}</text>')


def arrow(pts, label=None, lx=None, ly=None, accent=False, dashed=False):
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    col = ACCENT if accent else LINE
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    s = (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.5"{dash} '
         f'marker-end="url(#{"aa" if accent else "a"})"/>')
    if label:
        s += (f'<text x="{lx}" y="{ly}" font-size="12" fill="{MUTED}" '
              f'text-anchor="middle">{esc(label)}</text>')
    return s


def lane(x, y, w, h, title):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{GREY}" '
            f'stroke="none"/><text x="{x + 14}" y="{y + 22}" font-size="12.5" '
            f'font-weight="700" fill="{MUTED}">{esc(title)}</text>')


# --- Figure 1: inference pipeline -----------------------------------------------------------
def fig_inference():
    W, H = 1000, 900
    s = []
    s.append(lane(20, 10, 960, 110, "Web app (React)"))
    s.append(lane(20, 135, 960, 620, "API server (FastAPI) - one request at a time on the GPU"))
    s.append(lane(20, 770, 960, 120, "Web app (React)"))
    s.append(box(330, 40, 340, 66, "User asks a question",
                 ["picks one of 20 databases and a model"], accent=True, fill=SOFT))
    s.append(arrow([(500, 106), (500, 170)], "POST /api/query", 570, 145))
    s.append(box(330, 172, 340, 66, "1  Value matching",
                 ["question n-grams vs. stored text values", "(exact, or fuzzy ratio >= 85)"]))
    s.append(arrow([(500, 238), (500, 268)]))
    s.append(diamond(500, 300, 250, 62, "Model uses the ranker?"))
    # yes branch
    s.append(arrow([(625, 300), (720, 300), (720, 330)], "yes (Lite, Spinneret)", 700, 290))
    s.append(box(580, 332, 280, 82, "2  Schema ranker",
                 ["RoBERTa-base cross-encoder", "scores every table and column",
                  ], accent=True))
    s.append(arrow([(720, 414), (720, 440)]))
    s.append(box(580, 442, 280, 50, "Keep top 4 tables x 5 columns"))
    # no branch
    s.append(arrow([(375, 300), (280, 300), (280, 330)], "no (Baseline)", 300, 290))
    s.append(box(140, 332, 280, 82, "Full schema", ["every table and column"]))
    # merge into serializer
    s.append(arrow([(280, 414), (280, 520), (400, 520)]))
    s.append(arrow([(720, 492), (720, 520), (600, 520)]))
    s.append(box(400, 500, 200, 40, "3  Serialise input", fill=BOX))
    s.append(arrow([(500, 540), (500, 566)]))
    s.append(box(300, 568, 400, 66, "4  T5 generator, beam search",
                 ["k = 4 (Baseline, Lite) or 8 (Spinneret) candidates",
                  ], accent=True))
    s.append(arrow([(500, 634), (500, 660)]))
    s.append(diamond(500, 692, 300, 62, "5  Next candidate runs?"))
    s.append(arrow([(650, 692), (900, 692), (900, 600), (700, 600)]))
    s.append(f'<text x="905" y="640" font-size="12" fill="{MUTED}">no: try the</text>'
             f'<text x="905" y="655" font-size="12" fill="{MUTED}">next candidate</text>')
    s.append(box(70, 650, 160, 84, "Read-only SQLite",
                 ["one SELECT/WITH", "5 s limit, 500 rows"], fill=BOX))
    s.append(arrow([(350, 692), (232, 692)], "runs on", 290, 684, dashed=True))
    s.append(arrow([(500, 723), (500, 790)], "yes", 520, 760, accent=True))
    s.append(box(250, 792, 500, 84, "Answer",
                 ["SQL and skeleton, all candidates, ranker scores,",
                  "matched values, result rows and timings"], accent=True, fill=SOFT))
    page("fig-inference", W, H, "\n".join(s))


# --- Figure 2: training and evaluation pipeline --------------------------------------------
def fig_training():
    W, H = 1000, 560
    s = []
    s.append(lane(20, 10, 300, 540, "Data (laptop)"))
    s.append(lane(350, 10, 300, 540, "Training (free Colab / Kaggle T4)"))
    s.append(lane(680, 10, 300, 540, "Evaluation and release (laptop)"))
    s.append(box(40, 45, 260, 66, "Download Spider 1.0", ["SHA-256 verified zip"]))
    s.append(arrow([(170, 111), (170, 135)]))
    s.append(box(40, 137, 260, 118, "Prepare",
                 ["normalise SQL, build skeletons", "match database values",
                  "split train / valid by database", "tokenizer audit (+ <, <=, <>)"]))
    s.append(arrow([(170, 255), (170, 280)]))
    s.append(box(40, 282, 260, 100, "Splits",
                 ["train 7,941 questions (130 DBs)", "valid 718 (16 DBs)",
                  "dev 1,034 (20 unseen DBs)"], fill=SOFT))
    s.append(arrow([(300, 330), (370, 330), (370, 95), (390, 95)]))
    s.append(box(390, 45, 240, 84, "Train schema ranker",
                 ["RoBERTa-base, focal loss", "best by coverage@(4,5)"], accent=True))
    s.append(arrow([(510, 129), (510, 152)]))
    s.append(box(390, 154, 240, 84, "Build ranked inputs",
                 ["train: missed gold items added", "valid/dev: ranker only"]))
    s.append(arrow([(510, 238), (510, 262)]))
    s.append(box(390, 264, 240, 118, "Train generators",
                 ["T1 T5-small, full schema", "T2 T5-small + ranker", "T3 T5-base + ranker",
                  "Adafactor, early stop on valid EX"], accent=True))
    s.append(arrow([(510, 382), (510, 405)]))
    s.append(box(390, 407, 240, 84, "Hub checkpoint each epoch",
                 ["resume after session limits", "(--resume hub)"]))
    s.append(arrow([(630, 330), (700, 330), (700, 95), (720, 95)]))
    s.append(box(720, 45, 240, 66, "Predict on dev", ["execution-guided beam"]))
    s.append(arrow([(840, 111), (840, 135)]))
    s.append(box(720, 137, 240, 66, "Evaluate", ["official test-suite: EM, EX"]))
    s.append(arrow([(840, 203), (840, 227)]))
    s.append(box(720, 229, 240, 84, "Analyse",
                 ["failure taxonomy, question types,", "latency, ablations E2-E7"]))
    s.append(arrow([(840, 313), (840, 337)]))
    s.append(box(720, 339, 240, 66, "Summarise + model cards"))
    s.append(arrow([(840, 405), (840, 429)]))
    s.append(box(720, 431, 240, 84, "Release",
                 ["public Hugging Face models", "FastAPI + React demo"], accent=True,
                 fill=SOFT))
    page("fig-training", W, H, "\n".join(s))


# --- Figure 3: model input/output for the three tiers ----------------------------------------
def fig_ranker():
    W, H = 1000, 330
    s = []
    s.append(box(30, 20, 300, 66, "Question + whole schema",
                 ["text pair, max 512 tokens"], fill=SOFT))
    s.append(arrow([(330, 53), (380, 53)]))
    s.append(box(380, 20, 240, 66, "RoBERTa-base encoder", ["125.8M parameters"], accent=True))
    s.append(arrow([(620, 53), (670, 53)]))
    s.append(box(670, 20, 300, 66, "Item vectors",
                 ["mean of each name's tokens;", "column vector + its table's vector"]))
    s.append(arrow([(820, 86), (820, 120), (680, 120), (680, 145)]))
    s.append(arrow([(820, 120), (920, 120), (920, 145)]))
    s.append(box(560, 147, 240, 56, "Table head (sigmoid)"))
    s.append(box(820, 147, 150, 56, "Column head"))
    s.append(arrow([(680, 203), (680, 235)]))
    s.append(arrow([(895, 203), (895, 235)]))
    s.append(box(560, 237, 410, 72, "Score for every table and column",
                 ["keep top 4 tables x 5 columns (+ linking keys)"], accent=True, fill=SOFT))
    s.append(box(30, 130, 470, 180, "Training", [
        "labels: tables and columns used by the gold SQL",
        "loss: focal loss (gamma = 2), tables + columns",
        "AdamW, learning rate 3e-5, up to 16 epochs",
        "selection: validation coverage@(4,5)",
        "result: table AUC 0.984, column AUC 0.978",
        "coverage@(4,5): 98.1% valid, 98.5% dev",
    ]))
    page("fig-ranker", W, H, "\n".join(s))


# --- Figure 4: EDA -------------------------------------------------------------------------
def bars(x0, y0, w, h, title, groups, series, colors, ymax, fmt="{:.0f}", unit="",
         labels=False):
    """Grouped vertical bars; groups = labels; series = {name: [values]}."""
    s = [f'<text x="{x0}" y="{y0 - 14}" font-size="14" font-weight="700">{esc(title)}</text>']
    for t in range(0, ymax + 1, ymax // 4):
        y = y0 + h - h * t / ymax
        s.append(f'<line x1="{x0 + 34}" x2="{x0 + w}" y1="{y}" y2="{y}" stroke="#e3e3df"/>')
        s.append(f'<text x="{x0 + 28}" y="{y + 4}" font-size="11" text-anchor="end" '
                 f'fill="{MUTED}">{t}{unit}</text>')
    n = len(groups)
    k = len(series)
    gw = (w - 44) / n
    bw = min(26, (gw - 14) / k)
    for gi, g in enumerate(groups):
        gx = x0 + 40 + gi * gw + (gw - bw * k - 2 * (k - 1)) / 2
        for si, (name, vals) in enumerate(series.items()):
            v = vals[gi]
            bh = h * v / ymax
            bx = gx + si * (bw + 2)
            s.append(f'<rect x="{bx}" y="{y0 + h - bh}" width="{bw}" height="{bh}" '
                     f'fill="{colors[si]}" rx="2"/>')
            if labels:
                s.append(f'<text x="{bx + bw / 2}" y="{y0 + h - bh - 5}" font-size="11" '
                         f'text-anchor="middle" fill="{MUTED}">{fmt.format(v)}</text>')
        s.append(f'<text x="{x0 + 40 + gi * gw + gw / 2}" y="{y0 + h + 18}" font-size="12" '
                 f'text-anchor="middle">{esc(g)}</text>')
    lx = x0 + 40
    for si, name in enumerate(series):
        s.append(f'<rect x="{lx}" y="{y0 + h + 32}" width="11" height="11" '
                 f'fill="{colors[si]}" rx="2"/>')
        s.append(f'<text x="{lx + 16}" y="{y0 + h + 42}" font-size="12">{esc(name)}</text>')
        lx += 16 + 8 * len(name) + 22
    return "\n".join(s)


def fig_eda():
    st = json.loads((ROOT / "data/processed/stats.json").read_text())["splits"]
    lv = ["easy", "medium", "hard", "extra"]
    pct = lambda sp, k: [100 * st[sp]["hardness"][x] / st[sp]["questions"] for x in k]  # noqa: E731
    colors = ["#d6b38a", ACCENT, "#5f5f58"]
    s = [bars(20, 40, 470, 250, "Difficulty mix (% of questions)", lv,
              {"train": pct("train", lv), "valid": pct("valid", lv), "dev": pct("dev", lv)},
              colors, 48, unit="%")]
    feats = ["WHERE", "JOIN", "aggregate", "GROUP BY", "ORDER BY", "nested", "set op"]
    short = ["WHERE", "JOIN", "aggregate", "GROUP", "ORDER", "nested", "set op"]
    vals = {  # share of queries containing each construct (computed in eda.json)
        k: [v[f] for f in feats]
        for k, v in json.loads((OUT.parent / "eda.json").read_text()).items()
        if k in ("train", "dev")
    }
    s.append(bars(520, 40, 470, 250, "SQL constructs (% of queries)", short, vals,
                  [colors[0], colors[2]], 60, unit="%"))
    page("fig-eda", 1000, 350, "\n".join(s))


# --- Figure 5: results by difficulty ---------------------------------------------------------
def fig_results():
    summ = json.loads((ROOT / "reports/summary/summary.json").read_text())
    tiers = [t for t in summ["tiers"] if t.get("scores")]
    lv = ["easy", "medium", "hard", "extra", "all"]
    keep = {"T1 Baseline": "Baseline", "T2 Lite · served": "Lite (served)",
            "T3 Spinneret": "Spinneret"}
    series = {keep[t["tier"]]: [100 * (t["scores"]["ex_by_level"].get(x) if x != "all"
                                       else t["scores"]["ex"]) for x in lv]
              for t in tiers if t["tier"] in keep}
    s = [bars(20, 40, 960, 250, "Execution accuracy on Spider dev by difficulty (%)", lv,
              series, ["#c9c9c2", "#d6b38a", ACCENT], 100, fmt="{:.1f}", unit="%",
              labels=True)]
    page("fig-results", 1000, 350, "\n".join(s))


if __name__ == "__main__":
    fig_inference()
    fig_training()
    fig_ranker()
    fig_eda()
    fig_results()
    print("written:", ", ".join(sorted(p.stem for p in OUT.glob("*.html"))))
