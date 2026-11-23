"""Writes docs/report/source/report.html, the designed final report (printed to PDF by
print.mjs). Run from the repo root:  python3 docs/report/build/report.py

Every number comes from reports/summary/summary.json, data/processed/stats.json, the model
cards or the configs. Title-page details are in TITLE below; empty ones print as blanks.
"""

from __future__ import annotations

import base64
import re
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import figures  # noqa: E402

OUT = ROOT / "docs/report/source/report.html"

TITLE = {
    "course": "",  # e.g. "CA-2 of INT-344"
    "batch": "",  # e.g. "2023 – 2027"
    "to_name": "",
    "to_uid": "",
    "by_name": "",
    "by_reg": "",
    "by_roll": "",
}


def blank(v: str, width: str = "9em") -> str:
    return v if v else f'<span class="blank" style="width:{width}"></span>'


def img(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def svg_fig(name: str) -> str:
    w, h, inner = figures.SVGS[name]
    defs = (
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        'fill="#8a8a82"/></marker><marker id="aa" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 '
        'L0,10 z" fill="#b45309"/></marker></defs>'
    )
    return (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
            f'preserveAspectRatio="xMidYMid meet">{defs}{inner}</svg>')


# --- the cover's spider web (same construction as the site's background) --------------------
def web_svg(seed: int, spokes: int, rings: int, size: float) -> str:
    rnd = random.Random(seed)
    angles = [((i + rnd.random() * 0.35) / spokes) * 2 * math.pi for i in range(spokes)]
    reach = [size * (0.82 + rnd.random() * 0.18) for _ in angles]
    paths = [f"M0,0 L{math.cos(a) * r:.1f},{math.sin(a) * r:.1f}" for a, r in zip(angles, reach)]
    for k in range(rings):
        radius = size * 0.07 * 1.17**k + k * 6
        d, prev = "", 0.0
        for i in range(spokes + 1):
            j = i % spokes
            a = angles[j] + (2 * math.pi if i == spokes else 0)
            rr = min(radius * (0.96 + rnd.random() * 0.08), reach[j] * 0.98)
            x, y = math.cos(a) * rr, math.sin(a) * rr
            if i == 0:
                d = f"M{x:.1f},{y:.1f}"
            else:
                mid = (prev + a) / 2
                d += f" Q{math.cos(mid) * rr * 0.9:.1f},{math.sin(mid) * rr * 0.9:.1f} {x:.1f},{y:.1f}"
            prev = a
        paths.append(d)
    return "".join(f'<path d="{p}"/>' for p in paths)


# --- small builders ------------------------------------------------------------------------
def table(cls: str, head: list[str], rows: list[list[str]], caption: str, key: str) -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<figure class="tbl {cls}"><figcaption><b>Table @@T:{key}@@</b>{caption}</figcaption>'
            f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></figure>")


def fig(content: str, caption: str, key: str, cls: str = "") -> str:
    return (f'<figure class="fig {cls}">{content}<figcaption><b>Figure @@F:{key}@@</b>{caption}'
            "</figcaption></figure>")


def shot(path: Path, caption: str, key: str) -> str:
    return fig(f'<img class="shot" src="{img(path)}" alt="">', caption, key, "screenshot")


def opener(num: str, title: str, lede: str) -> str:
    return (f'<header class="opener"><span class="num">{num}</span><h1>{title}</h1>'
            f'<p class="lede">{lede}</p></header>')


# --- build ----------------------------------------------------------------------------------
def main() -> None:
    figures.fig_inference()
    figures.fig_training()
    figures.fig_ranker()
    figures.fig_eda()
    figures.fig_results()
    figures.fig_curves()
    fonts = ROOT / "docs/report/source/fonts"
    font = lambda f: "data:font/woff2;base64," + base64.b64encode((fonts / f).read_bytes()).decode()  # noqa: E731
    logo = img(ROOT / "docs/img/logo.png")
    T = TITLE

    cover = f"""
<section class="cover">
  <svg class="cover-web" viewBox="-620 -620 1240 1240">{web_svg(7, 17, 14, 620)}</svg>
  <svg class="cover-web small" viewBox="-560 -560 1120 1120">{web_svg(23, 15, 13, 560)}</svg>
  <div class="cover-top">
    <div class="uni"><img src="{logo}" alt=""><span>Lovely Professional University<br>
      <small>Department of Computer Science and Engineering</small></span></div>
    <span class="kind">Project report</span>
  </div>
  <div class="cover-title">
    <p class="pre">Submitted for partial fulfillment for {blank(T['course'], '10em')}</p>
    <h1>Spinneret</h1>
    <p class="tag">Ask in English.<br>Get SQL that runs.<span class="cursor"></span></p>
    <p class="sub">Small, fast text-to-SQL query generation with fine-tuned transformers</p>
  </div>
  <div class="cover-cascade" id="cascade">
    <span class="step">you ask</span>
    <p class="q">What is the <span data-q="a">average</span> <span data-q="g">age</span> of
      <span data-q="s">singers</span> from <span data-q="f">France</span>?</p>
    <span class="step">Spinneret writes</span>
    <p class="sql"><span class="sh">select</span> <span data-s="a">avg</span><span class="sh">(</span><span
      data-s="g">age</span><span class="sh">) from</span> <span data-s="s">singer</span>
      <span class="sh">where</span> <span data-s="f">country</span> <span class="sh">=</span>
      <span data-s="f">'France'</span></p>
    <span class="step">the database answers</span>
    <p class="ans" data-r="a">34.5</p>
    <svg class="threads"></svg>
  </div>
  <div class="cover-bottom">
    <div class="degree">Bachelor of Technology in Computer Science &amp; Engineering
      <span>Batch {blank(T['batch'], '7em')}</span></div>
    <div class="people">
      <div><small>Submitted to</small>{blank(T['to_name'], '12em')}<br>UID {blank(T['to_uid'], '6em')}<br>
        Computer Science &amp; Engineering</div>
      <div><small>Submitted by</small>{blank(T['by_name'], '12em')}<br>Reg. No. {blank(T['by_reg'], '7em')}<br>
        Roll No. {blank(T['by_roll'], '3em')}</div>
    </div>
    <p class="addr">Lovely Professional University, Jalandhar-Delhi G.T. Road, Phagwara, Punjab</p>
  </div>
</section>"""

    abstract = f"""
<section class="page abstract">
  <h1 class="plain">Abstract</h1>
  <div class="stats">
    <div><b>68.8%</b><span>of Spider's unseen-database questions answered correctly by Spinneret</span></div>
    <div><b>0.39 s</b><span>median answer time for the Lite model on a 4&nbsp;GB laptop GPU</span></div>
    <div><b>7</b><span>public model checkpoints, MIT-licensed code and a live demo</span></div>
  </div>
  <p>Text-to-SQL systems let people query a relational database in plain English. The hard
  version of the task is <i>cross-database</i>: the model must answer questions about
  databases it has never seen, reading an unfamiliar schema and mapping each word of the
  question to the right table, column and value. Large language models do this well but are
  expensive to run; this project asks how far small, open, fine-tuned models can go when they
  must run on a student laptop.</p>
  <p><b>Spinneret</b> builds and compares three tiers of models on the Spider benchmark.
  <b>Baseline</b> fine-tunes T5-small (60M parameters) on the question and the whole schema.
  <b>Lite</b> adds a RoBERTa-base schema ranker (126M) that scores every table and column and
  passes only the top four tables and five columns of each to T5-small. <b>Spinneret</b> uses
  the same ranker with T5-base (223M) and skeleton-first decoding, writing the query's shape
  before its names and values. All three use database-value matching and execution-guided beam
  search: candidate queries are run read-only, and the first one that executes is returned.</p>
  <p>The models were trained on free Colab and Kaggle T4 GPUs and evaluated with the official
  test-suite scripts on Spider's development set (1,034 questions over 20 unseen databases).
  Spinneret reaches <b>62.6% exact match and 68.8% execution accuracy</b> with a median latency
  of 1.29&nbsp;s on a laptop GTX 1650; Lite reaches 55.8% in 0.39&nbsp;s; Baseline reaches 49.2%,
  above the published T5-small result of 46.2%. Ablations show that execution-guided decoding
  (+6.6 points), the schema ranker (+6.6) and database values (+2.5) all help, while the
  skeleton does not help the small model. The remaining errors are mostly wrong joins and wrong
  columns.</p>
  <p>The models are served by a FastAPI back end and a React web application that shows the
  ranker's scores, the query's skeleton, every candidate and the result. The code is released
  under the MIT licence and all seven model checkpoints are public on the Hugging Face Hub.</p>
  <p class="keywords"><b>Keywords</b> text-to-SQL · semantic parsing · T5 · schema linking ·
  execution-guided decoding · Spider</p>
</section>"""

    problem = f"""
<section class="page">
  {opener("01", "Problem statement", "Translate English questions into executable SQL for databases the model has never seen, with models small enough for a laptop.")}
  <p>Most structured data sits in relational databases, but querying them needs SQL, which most
  users do not know. This project addresses the problem statement <b>“Text-to-SQL Query
  Generation”</b>: translate English questions into executable SQL for a given database with
  fine-tuned sequence-to-sequence transformer models.</p>
  <h2>Why the task is hard</h2>
  <div class="grid2">
    <div class="card"><h3>Unseen schemas</h3><p>The evaluation databases never appear in training, so
    the model must read each schema and link words to it: “singers” to <code>singer</code>, “from
    France” to <code>country = 'France'</code>.</p></div>
    <div class="card"><h3>Long, noisy inputs</h3><p>A serialised schema can exceed a small model's
    512-token input, and most of it is irrelevant to any one question.</p></div>
    <div class="card"><h3>Exact values, compositional SQL</h3><p>Literals must match the stored
    spelling, and joins, grouping, nesting and set operations must all be right; one wrong
    column makes the answer wrong.</p></div>
    <div class="card"><h3>Correctness is semantic</h3><p>A query can differ from the reference and
    still be right, or look right and fail to run, so it must be judged by executing it.</p></div>
  </div>
  <h2>Constraints</h2>
  <p>The project was built solo in five weeks with no paid compute: training on free Colab and
  Kaggle T4 GPUs whose sessions end without warning, and serving every model on a laptop GTX
  1650 with 4&nbsp;GB of memory in about a second per question. Large language models were
  therefore out of scope.</p>
  <h2>Objectives</h2>
  <ol class="objectives">
    <li>Measure how accurate small fine-tuned models can be on Spider's unseen databases, and
    what each technique adds, at a known size and speed.</li>
    <li>Train three tiers (a required T5-small baseline, a fast Lite model and a best-accuracy
    Spinneret model) and evaluate them with the official scripts, ablations and an error
    analysis.</li>
    <li>Ship a safe, transparent demo that answers questions on all 20 evaluation databases.</li>
  </ol>
</section>"""

    proposed = f"""
<section class="page">
  {opener("02", "Proposed work", "A fine-tuned T5 at the centre, with four cheap techniques around it, each of which can be switched off and measured.")}
  <p>Spinneret does not rely on model size. It adds four techniques from the literature to a
  fine-tuned T5 encoder-decoder: database-value matching (after BRIDGE [8]), a schema-item
  ranker and skeleton-first decoding (after RESDSQL [9]), and execution-guided decoding [10].
  Figure @@REF:F:f1@@ follows one question through the served system.</p>
  {fig(svg_fig("fig-inference"), "Inference pipeline: from an English question to an executed answer.", "f1", "tall")}
  <h2>Three model tiers</h2>
  <p>The tiers share the data pipeline, value matching, decoding and evaluation, and differ only
  in the generator, its schema input and its target (Table @@REF:T:t1@@), so each technique can be isolated
  by an ablation.</p>
  {table("tiers", ["Tier", "Generator", "Schema input", "Target", "Beam", "Parameters"], [
      ["Baseline", "T5-small", "whole schema", "SQL", "4", "60.5M"],
      ["Lite", "T5-small", "ranker top 4 × 5", "SQL*", "4", "60.5M + 125.8M"],
      ["Spinneret", "T5-base", "ranker top 4 × 5", "skeleton | SQL", "8", "222.9M + 125.8M"],
  ], "The three model tiers. *Planned with a skeleton; ablation E4 showed it does not help T5-small, so the served Lite writes plain SQL.", "t1")}
  <p>The three tiers were chosen from five options weighed before any training (Table
  @@REF:T:topt@@). Published results show steep diminishing returns from size alone: T5-3B, fifty
  times larger than T5-small, stays near 71% EX, while a base-size generator with a schema ranker
  reaches about 78%. Technique is cheap; parameters are not.</p>
  {table("", ["Option", "Models", "Size", "Expected EX", "Decision"], [
      ["Baseline", "T5-small", "60M", "about 46% (published)", "<b>Baseline</b>, the required reference"],
      ["Lite", "T5-small + RoBERTa-base ranker", "about 185M", "55–62% (estimate)", "<b>Lite</b>, the fast mode"],
      ["Balanced", "T5-base + RoBERTa-base ranker", "about 345M", "68–75% (estimate)", "<b>Spinneret</b>, the main model"],
      ["Max", "T5-base + RoBERTa-large ranker", "about 575M", "72–78% (RESDSQL-Base)", "only if ranker coverage fell below 95%"],
      ["Large generators", "T5-large, T5-3B", "0.8–3B", "67–71% (published)", "rejected: too big for the laptop"],
  ], "Options considered before training.", "topt")}
  <h2>Four techniques</h2>
  <div class="grid2">
    <div class="card"><h3>Database-value matching</h3><p>Question n-grams are matched against the
    text values stored in the database, and matches are written next to their column
    (<code>country ( France )</code>), so the model can copy the exact spelling.</p></div>
    <div class="card"><h3>Schema ranker</h3><p>A RoBERTa-base cross-encoder scores every table and
    column. Only the top four tables and five columns per table, plus the keys that link them,
    reach the generator, ordered by score.</p></div>
    <div class="card"><h3>Skeleton-first decoding</h3><p>The target is <code>skeleton | SQL</code>, e.g.
    <code>select avg ( _ ) from _ where _ = _ | select avg ( age ) …</code>: the model commits to
    the query's shape before choosing names and values.</p></div>
    <div class="card"><h3>Execution-guided decoding</h3><p>Beam search returns k candidates; each is
    checked by a SQL guard and run read-only, and the first that executes is the answer. A
    query that cannot run is never shown as an answer.</p></div>
  </div>
  <h2>Training and evaluation plan</h2>
  <p>Figure @@REF:F:f2@@ shows the offline pipeline. The data is prepared once; the ranker is trained
  first, because its scores build the inputs of Lite and Spinneret; the generators are trained
  on free GPUs, pushing a checkpoint to the Hugging Face Hub after every epoch so that a run can
  resume when a session ends. Every model is scored on Spider dev with the official test-suite
  scripts [11]: exact match (EM, same SQL structure) and execution accuracy (EX, same result),
  per difficulty level and per question type.</p>
  {fig(svg_fig("fig-training"), "Training, evaluation and release pipeline.", "f2", "mid")}
  <p>Seven experiments measure the design: <b>E1</b> compares the tiers, <b>E2</b> greedy, beam and
  execution-guided decoding, <b>E3</b> removes database values, <b>E4</b> separates the ranker from
  the skeleton, <b>E5</b> varies the ranker's top-k, <b>E6</b> removes the <code>train_others</code>
  split and <b>E7</b> compares T5-small with T5-base. A rule-based taxonomy labels every wrong
  answer, and a sample of 100 errors is labelled a second time to check the rules.</p>
  <h2>System architecture</h2>
  <p>Five parts, kept small so one person can build and maintain them (Table @@REF:T:t2@@).</p>
  {table("", ["Component", "Role", "Technology"], [
      ["<code>spinneret</code> package and CLI", "data, ranker, generator, decoding, evaluation, serving", "Python 3.12, PyTorch, Transformers"],
      ["Model hub", "checkpoints during training, public models after release", "Hugging Face Hub"],
      ["API server", "loads the models once, answers questions, serves results", "FastAPI, Uvicorn, Pydantic"],
      ["Web application", "landing, demo, evaluation, about and privacy pages", "React 18, TypeScript, Vite"],
      ["Databases", "the 20 Spider dev databases, opened read-only", "SQLite"],
  ], "System components.", "t2")}
  <h2>Success criteria and plan</h2>
  <p>Targets were fixed before training from published results for each model size (Table @@REF:T:t3@@),
  together with two component targets: ranker coverage@(4,5) of at least 95% on validation and
  at least 95% executable SQL for Spinneret.</p>
  {table("", ["Tier", "Published reference", "Minimum", "Target", "Stretch"], [
      ["Baseline", "T5-small 45.8 / 46.2 [12]", "42 / 42", "46 / 48", "50 / 52"],
      ["Lite", "none published", "50 / 52", "55 / 58", "60 / 63"],
      ["Spinneret", "RESDSQL-Base 71.7 / 77.9 [9]", "60 / 66", "65 / 72", "70 / 78"],
  ], "Targets on Spider dev, EM / EX %.", "t3")}
  <ol class="timeline">
    <li><b>Week 1</b>Spider download, preparation, tokenizer audit, remote-training setup</li>
    <li><b>Week 2</b>Baseline, schema ranker, evaluation wrapper</li>
    <li><b>Week 3</b>Ranked inputs; Lite and Spinneret trained and scored</li>
    <li><b>Week 4</b>Evaluation, ablations E2–E7, failure taxonomy, latency</li>
    <li><b>Week 5</b>API, web application, model cards, public release</li>
  </ol>
  <h2>Risks and how they were handled</h2>
  {table("", ["Risk", "Mitigation", "Outcome"], [
      ["Free GPU sessions end mid-run", "checkpoint to the Hub every epoch; resume; Kaggle as a second source", "every run finished"],
      ["The ranker drops needed items", "measure coverage first; generous k; gold items added to training inputs", "98.5% dev coverage; large ranker not needed"],
      ["T5 cannot write <code>&lt;</code>", "tokenizer round-trip audit; four tokens added; unit test", "comparisons decode correctly"],
      ["Schemas exceed 512 tokens", "ranked inputs for Lite and Spinneret; truncated share reported", "4% of training inputs truncated, none on dev"],
      ["Spinneret misses its target", "ablations show where points are lost; honest comparison with published results", "68.8% EX: above the 66% minimum, below 72%"],
      ["Solo schedule slips", "Baseline and ranker by week 2; training runs in parallel with other work", "all five milestones met"],
  ], "Risks identified in the plan, and what happened.", "trisk")}
</section>"""

    detailed = f"""
<section class="page">
  {opener("03", "Detailed description", "How each stage is built: one command per stage, every behaviour fixed by a YAML config, so any result can be reproduced from the repository.")}
  <h2>3.1 Data preparation</h2>
  <p><code>spinneret download</code> fetches the Spider 1.0 archive and verifies its SHA-256
  checksum. <code>spinneret prepare</code> writes one JSON line per question with the model
  input, the targets and metadata.</p>
  <dl class="defs">
    <dt>Splits</dt><dd>Spider's <code>train_spider</code> and <code>train_others</code> form the
    training data. A validation set is carved out <i>by database</i> (10%, 16 databases), so it
    also measures unseen-schema performance; Spider dev stays untouched.</dd>
    <dt>Normalisation</dt><dd>Gold SQL is lower-cased except string literals, double quotes become
    single quotes, and every token is separated by one space (<code>select count ( * ) from
    singer</code>). A test runs gold and normalised SQL and checks the results are identical.</dd>
    <dt>Skeletons</dt><dd>Tables, columns, aliases, <code>*</code> and literals become
    <code>_</code> and runs such as <code>_ , _</code> are collapsed:
    <code>select count ( _ ) from _</code>.</dd>
    <dt>Value matching</dt><dd>Distinct text values of every column are cached (skipping values
    over 64 characters and columns with more than 5,000 values). A question n-gram of 1–6 words
    matches a value when equal ignoring case, or when its fuzzy ratio is at least 85 for values
    of three or more characters; at most two values per column are kept.</dd>
    <dt>Tokenizer audit</dt><dd>T5's vocabulary cannot write <code>&lt;</code>, so
    <code>&lt;=</code> and <code>&lt;&gt;</code> would be lost; four tokens (<code> &lt;</code>,
    <code>&lt;</code>, <code> &lt;=</code>, <code> &lt;&gt;</code>) were added to the tokenizer and
    the embedding matrix.</dd>
  </dl>
  <p>The input lists the question, the database and each table with its columns; matched values
  appear in brackets. For the ranked models the order follows the ranker's scores and the
  foreign keys are appended:</p>
  <pre class="code"><span class="c">input</span>  What is the average, minimum, and maximum age of all singers from France?
       | concert_singer
       | singer : age , country ( <b>France</b> ) , singer_id , name , is_male
       | singer_in_concert : singer_id , concert_id
       | stadium : average , stadium_id , highest , lowest , name
       | concert : year , stadium_id , concert_id , concert_name , theme
       | fk : concert.stadium_id = stadium.stadium_id , …
<span class="c">target</span> select avg ( _ ) , min ( _ ) , max ( _ ) from _ where _ = _
       | select avg ( age ) , min ( age ) , max ( age ) from singer where country = 'France'</pre>
  <h2>3.2 Schema ranker</h2>
  <p>The ranker (Figure @@REF:F:f3@@) is a RoBERTa-base cross-encoder. The question and the serialised
  schema are encoded together; the character offsets of each table and column name are mapped to
  tokens and averaged into one vector per item, and each column vector is joined with its
  table's vector. Two linear heads with a sigmoid give the probability that the gold SQL uses
  each item. Labels are the tables and columns the gold query references, and since most items
  are negatives the loss is focal loss (γ = 2). Schemas over 512 tokens are split into chunks of
  tables, each paired with the question.</p>
  {fig(svg_fig("fig-ranker"), "Schema ranker: architecture and training.", "f3")}
  <p>The ranked input is built in four steps: keep the four highest-scoring tables and the five
  highest-scoring columns in each; keep the keys that link two kept tables, so joins can still be
  written; order tables and columns by score, so the order itself signals importance; and append
  the foreign-key links. The checkpoint is chosen by <i>coverage@(4,5)</i>, the share of
  questions whose every needed item survives the cut: 98.1% on validation and 98.5% on dev, so
  the larger RoBERTa-large ranker in the plan was not needed. For training inputs, any gold item
  the ranker missed is inserted at its scored position, so the generator never learns to write
  what it cannot see; validation, dev and live inputs use the ranker's output only.</p>
  <p>The cut-off of four tables and five columns was checked by ablation E5 (Table @@REF:T:t4@@): Spinneret
  was decoded greedily on dev with nine different cuts. Keeping more items raises coverage but
  lengthens the input; accuracy is flat within about one point from four tables onward, so the
  4 × 5 cut used in training was kept.</p>
  {table("grid", ["Tables kept", "4 columns", "5 columns", "6 columns"], [
      ["3", "61.2 (95.9)", "61.9 (97.6)", "62.5 (98.4)"],
      ["4", "62.0 (96.8)", "<b>62.2 (98.5)</b>", "62.2 (99.2)"],
      ["5", "62.4 (96.8)", "62.3 (98.5)", "62.2 (99.2)"],
  ], "Ablation E5: Spinneret's EX % with greedy decoding on dev, and ranker coverage % in brackets, for each cut-off.", "t4")}
  <h2>3.3 Generator training</h2>
  <p>Generators are trained with Hugging Face Transformers [14] and the Adafactor optimiser [4]:
  effective batch 32, linear schedule with 3% warm-up, inputs cut at 512 and targets at 256
  tokens, full fp32 precision (T5 overflows in fp16). After each epoch the validation set is
  decoded greedily and training stops when validation EX has not improved for five epochs
  (Table @@REF:T:t5@@, Figure @@REF:F:f4@@).</p>
  {table("", ["Model", "Base", "Learning rate", "Best epoch", "Valid EM / EX", "Trained on"], [
      ["Baseline", "T5-small", "1e-3", "18", "44.7 / 49.3", "Colab T4 + laptop"],
      ["Lite, skeleton", "T5-small", "1e-3", "12", "50.8 / 52.6", "Colab T4"],
      ["Lite, served (E4)", "T5-small", "1e-3", "17", "50.1 / 54.5", "Kaggle T4"],
      ["Spinneret", "T5-base", "1e-4", "18", "59.7 / 67.0", "Colab T4"],
  ], "Generator training runs.", "t5")}
  {fig(svg_fig("fig-curves"), "Validation execution accuracy after each epoch (greedy decoding), from the checkpoints' training logs; dots mark the kept epoch.", "f4")}
  <p>Spinneret overtakes the small models after four epochs and peaks at epoch 18; Baseline and
  Spinneret then ran five more epochs without improving and stopped. Including validation, the
  runs took about 1.4 hours (Baseline), 2.2 hours (Lite with skeleton), 0.9 hours (served Lite,
  whose Kaggle run ended at epoch 17) and 5.2 hours (Spinneret) of T4 time, and the ranker
  0.3 hours.</p>
  <p>Free sessions end without warning, so training was driven from the laptop with Google's
  Colab command-line tool, or the Kaggle CLI when Colab had no GPU free. A run pushes its latest
  checkpoint, optimiser and scheduler state to a private Hub repository every epoch, and
  <code>--resume hub</code> continues exactly where it stopped. Small scripts open a session,
  upload the code, start, follow and stop a run; on Kaggle, where command-line notebooks cannot
  read secrets, checkpoints are saved locally and pushed from the laptop. The Hugging Face token
  travels as a file and never appears in commands, logs or the repository.</p>
  <h2>3.4 Decoding and safe execution</h2>
  <p>The generator returns k beam candidates (4 for Baseline and Lite, 8 for Spinneret); for
  skeleton models the SQL is the text after the first <code> | </code>. A guard accepts a single
  <code>SELECT</code> or <code>WITH</code> statement and refuses anything that changes data or
  settings (<code>INSERT</code>, <code>UPDATE</code>, <code>DELETE</code>, <code>DROP</code>,
  <code>ATTACH</code>, <code>PRAGMA</code> and others). Queries run on a SQLite connection opened
  read-only with <code>query_only</code> on, a 5-second limit and at most 500 rows. The first
  candidate that runs is the answer; if none does, the top candidate is shown with its error
  explained in plain language.</p>
  <pre class="code"><span class="c"># execution-guided selection (spinneret/decoding)</span>
for cand in beam_candidates:                    <span class="c"># best log-probability first</span>
    if not guard.allows(cand.sql):
        cand.status = "rejected"; continue      <span class="c"># not a single read-only query</span>
    cand.status = run_read_only(db, cand.sql, timeout=5.0, max_rows=500)
    if cand.status == "ok":                     <span class="c"># an empty result also counts</span>
        return cand
return beam_candidates[0]                       <span class="c"># nothing ran: show the error</span></pre>
  <p>Ablation E2 (Table @@REF:T:t6@@) measures what this buys on Spinneret. A plain beam of eight barely
  helps (+0.5 EX), because its top candidate is usually the greedy one; running the candidates
  and skipping those that fail adds 6.6 points, most of the gain coming from the first four
  beams.</p>
  {table("", ["Decoding (Spinneret)", "EM %", "EX %", "Dev generation time"], [
      ["greedy", "57.6", "62.2", "218 s"],
      ["beam 8, top candidate", "58.0", "62.7", "1,282 s"],
      ["execution-guided, beam 4", "61.6", "67.4", "868 s"],
      ["<b>execution-guided, beam 8</b>", "<b>62.6</b>", "<b>68.8</b>", "1,282 s"],
  ], "Ablation E2: decoding strategies for Spinneret on Spider dev (1,034 questions, laptop GPU).", "t6")}
  <h2>3.5 Evaluation and error analysis</h2>
  <p><code>spinneret evaluate</code> scores predictions with the official test-suite evaluation
  [11], and <code>spinneret analyze</code> groups questions by type (simple lookup, aggregation,
  two-table join, multi-join, nested or set operation) and gives every wrong answer a primary
  label from a rule-based taxonomy: the first rule that matches wins (Table @@REF:T:t7@@). Predictions that
  miss EM but return the right result are counted separately as semantically equivalent.</p>
  {table("taxonomy", ["#", "Label", "Rule: prediction vs. gold"], [
      ["1", "invalid_sql", "fails to parse or to execute"],
      ["2", "wrong_table", "different tables in FROM, same number"],
      ["3", "join_error", "missing or extra join, or different join conditions"],
      ["4", "nesting_setop_error", "sub-query or INTERSECT / UNION / EXCEPT differs"],
      ["5", "wrong_column", "different columns in SELECT or WHERE"],
      ["6", "aggregation_error", "different, missing or extra aggregate"],
      ["7", "group_by_error", "GROUP BY or HAVING differs"],
      ["8", "condition_error", "same WHERE columns, different operator, AND/OR or negation"],
      ["9", "order_limit_error", "ORDER BY column or direction, or LIMIT, differs"],
      ["10", "value_error", "only a literal value is wrong"],
      ["11", "other", "none of the above, e.g. DISTINCT"],
  ], "Failure taxonomy.", "t7")}
  <p>To check the rules, a seeded sample of 100 Spinneret errors was labelled independently: a
  blind review agreed with the automatic labels on 93 (86 before one rule was fixed) and a second
  check on 94. <b>Both reviews were done by an AI assistant (Claude), not by a human
  annotator.</b> Grouping by question type (Table @@REF:T:t8@@) shows where each model is strong: lookups and
  aggregations are mostly solved, while joins across three or more tables remain hard.</p>
  {table("num", ["Question type", "n", "Baseline", "Lite", "Lite served", "Spinneret"], [
      ["simple lookup", "252", "65.5", "71.8", "73.8", "83.7"],
      ["aggregation", "292", "70.9", "75.7", "76.4", "85.3"],
      ["two-table join", "288", "25.3", "36.5", "37.5", "55.2"],
      ["multi-join", "43", "14.0", "0.0", "0.0", "27.9"],
      ["nested or set operation", "159", "36.5", "34.6", "37.7", "50.3"],
  ], "Execution accuracy (%) by question type on Spider dev.", "t8")}
  <p>Latency is measured stage by stage at batch size 1 on the laptop GPU (Table @@REF:T:t9@@):
  generation dominates, the ranker adds about 60&nbsp;ms, and peak memory is 0.41, 0.62 and
  1.66&nbsp;GB, so all three models fit together on a 4&nbsp;GB GPU.</p>
  {table("", ["Model", "Values", "Ranking", "Generation", "Execution", "Total"], [
      ["Baseline", "&lt; 1", "–", "371", "1", "374"],
      ["Lite, skeleton", "&lt; 1", "59", "550", "1", "623"],
      ["Lite, served", "&lt; 1", "62", "307", "1", "393"],
      ["Spinneret", "&lt; 1", "63", "1,220", "1", "1,286"],
  ], "Median latency per stage in ms (laptop GTX 1650, batch 1, Spider dev).", "t9")}
  <h2>3.6 Serving: API and web application</h2>
  <p><code>spinneret serve</code> starts a FastAPI application that also serves the built web
  app. Models load once in a background thread and are warmed up with one question each, cutting
  the first answer from 3.4&nbsp;s to about 0.5&nbsp;s; meanwhile the API serves schemas and
  metrics and reports “warming up”. One GPU lock runs a single generation at a time, so concurrent
  requests queue instead of exhausting the 4&nbsp;GB of memory.</p>
  {table("", ["Endpoint", "Purpose"], [
      ["<code>GET /api/health</code>", "status, device and loaded models"],
      ["<code>GET /api/models</code>", "served models with size, accuracy and median latency"],
      ["<code>GET /api/databases/…</code>", "databases, schemas with keys, example questions"],
      ["<code>POST /api/query</code>", "question to answer: SQL, skeleton, candidates, ranker scores, rows, timings"],
      ["<code>POST /api/execute</code>", "edited SQL through the same guard, time limit and row cap"],
      ["<code>GET /api/summary</code>", "evaluation results for the Evaluation page"],
  ], "Main API endpoints.", "t10")}
  <p>The web application (React 18, TypeScript, Vite, TanStack Query, CodeMirror, Framer
  Motion) has a landing page that threads a real question word by word to its SQL and answer; a
  demo page with the schema highlighted by the ranker's scores, the question box, the skeleton,
  an editable SQL editor, every candidate and the result; and evaluation, about and privacy pages.
  It works in light and dark themes, on phones and from the keyboard, loads nothing from other
  sites, and the server keeps no record of the questions asked.</p>
  <h2>3.7 Testing and release</h2>
  <p>The Python package has 101 tests (pytest) for normalisation, the guard, value matching,
  serialisation, evaluation, the taxonomy and the API; the web app has 15 component tests
  (Vitest); linting uses ruff. Model cards are generated from the reports and published with each
  checkpoint. All seven checkpoints are public on the Hugging Face Hub and the code is on GitHub
  under the MIT licence, so the demo runs with four commands:</p>
  <pre class="code">uv sync --extra train --extra api      <span class="c"># Python 3.12 environment</span>
uv run spinneret download              <span class="c"># Spider databases, checksum verified</span>
(cd web &amp;&amp; npm ci &amp;&amp; npm run build)    <span class="c"># web application</span>
uv run spinneret serve                 <span class="c"># http://localhost:8000</span></pre>
  <h2>3.8 Limitations and future work</h2>
  <p>The models answer English questions in Spider's style and are weakest on queries joining
  three or more tables (27.9% EX for Spinneret) and on nested or set queries (50.3%). Values must
  be spelled as stored, and numbers written in words are often wrong. Lite and Spinneret cannot
  recover an item the ranker drops (8 of Spinneret's errors). The clearest next steps follow from
  the ablations: a RoBERTa-large ranker as in RESDSQL [9], where most of the gap to 77.9% EX
  lies; constrained decoding such as PICARD [6] to rule out invalid SQL during generation;
  reinforcement-learning objectives that reward execution accuracy [12]; a code-pretrained
  generator such as CodeT5; and 8-bit quantisation for faster answers on small GPUs.</p>
</section>"""

    datasets = f"""
<section class="page">
  {opener("04", "Dataset and exploratory analysis", "Spider 1.0: 200 databases across 138 domains, with training and development databases that never overlap.")}
  <p>The project uses <b>Spider 1.0</b> [1], a human-labelled, cross-domain text-to-SQL dataset
  (CC BY-SA 4.0). Because its training and development databases are disjoint, Spider dev tests
  generalisation to unseen schemas (Table @@REF:T:t11@@); the hidden test set was not used.</p>
  {table("", ["Split", "Questions", "Databases", "easy / medium / hard / extra", "Tables per DB"], [
      ["train: <code>train_spider</code> + <code>train_others</code>", "7,941", "130", "1,815 / 2,723 / 1,746 / 1,657", "mean 5.4, max 26"],
      ["valid: 10% of databases", "718", "16", "168 / 276 / 176 / 98", ""],
      ["dev: the benchmark", "1,034", "20", "248 / 446 / 174 / 166", "mean 4.0, max 11"],
  ], "Splits used in this project.", "t11")}
  {fig(svg_fig("fig-eda"), "Difficulty mix per split, and the share of queries using each SQL construct.", "f5")}
  <div class="grid2 facts">
    <p><b>Unseen schemas.</b> None of the 20 dev databases appears in training. Dev schemas average
    4.0 tables and 22.1 columns; training schemas 5.4 tables and 27.8 columns, up to 26 tables and
    352 columns.</p>
    <p><b>Difficulty.</b> Dev has more medium questions (43%) and fewer extra-hard ones (16%) than
    training (34% and 21%).</p>
    <p><b>SQL constructs.</b> 40% of dev queries join tables, 53% aggregate, 27% group, 8% nest a
    sub-query and 8% use a set operation; 56% need only one table.</p>
    <p><b>Input length.</b> Training inputs average 200 T5 tokens; 319 of 7,941 (4%) exceed 512 and
    are truncated, which motivated the ranker. Targets average 45 tokens, 71 with the skeleton.</p>
    <p><b>Values.</b> 28% of training and 34% of dev questions mention a value stored in the
    database.</p>
  </div>
</section>"""

    results = f"""
<section class="page">
  {opener("05", "Results", "Spinneret answers 68.8% of questions on unseen databases correctly, 8.6 points above a published T5-base.")}
  {table("results", ["Model", "EM %", "EX %", "Valid SQL %", "Parameters", "p50 / p95"], [
      ["Baseline: T5-small, full schema", "48.5", "49.2", "67.5", "60.5M", "0.37 / 0.74 s"],
      ["Lite: T5-small + ranker + skeleton", "52.4", "54.4", "78.3", "186M", "0.62 / 1.18 s"],
      ["Lite served: no skeleton (E4)", "52.5", "55.8", "82.2", "186M", "0.39 / 0.91 s"],
      ["Spinneret: T5-base + ranker + skeleton", "62.6", "68.8", "91.7", "349M", "1.29 / 2.50 s"],
  ], "Spider dev, official test-suite evaluation, 1,034 questions; latency per question on a laptop GTX 1650.", "t12")}
  {fig(svg_fig("fig-results"), "Execution accuracy by difficulty.", "f6")}
  <div class="grid3 findings">
    <div><h3>Against published models</h3><p>Baseline beats the published T5-small (46.2% EX) and
    Spinneret beats T5-base (60.2%) [12]; RESDSQL-Base [9], with a larger ranker, reaches 77.9%.
    Spinneret passes its 66% minimum but misses its 72% target; Lite misses 58% by 2.2 points.</p></div>
    <div><h3>What each technique adds</h3><p>Execution-guided decoding +6.6 EX (62.2 → 68.8); the
    ranker +6.6 on T5-small; database values +2.5; T5-base +14.4 over T5-small. The skeleton costs
    T5-small 1.4, and <code>train_others</code> makes no difference (49.0 vs 49.2).</p></div>
    <div><h3>Where it still fails</h3><p>Invalid SQL fell from 336 (Baseline) to 88. Spinneret's 323
    wrong answers are mostly invalid SQL (88), wrong joins (88) and wrong columns (59); 92 more
    differ from the gold text but return the right result.</p></div>
  </div>
</section>"""

    shots = f"""
<section class="page">
  {opener("06", "Screenshots", "The running system: real model output on the Spider dev databases.")}
  {shot(ROOT / "docs/img/landing.png", "Landing page: a real question, its generated SQL and the answer, joined by threads.", "f7")}
  {shot(ROOT / "docs/img/demo.png", "Demo page: ranker scores on the schema, the skeleton and SQL, and the answer.", "f8")}
  {shot(ROOT / "docs/report/figures/shot-candidates.png", "Execution-guided decoding: candidates 1–3 fail, candidate 4 runs and is returned; the model menu is open.", "f9")}
  {shot(ROOT / "docs/img/evaluation.png", "Evaluation page: accuracy, size and speed of each model, and accuracy by difficulty.", "f10")}
</section>"""

    refs = [
        "T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z. Li, J. Ma, I. Li, Q. Yao, S. Roman, Z. Zhang and D. Radev, “Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task,” in <i>Proc. EMNLP</i>, 2018, pp. 3911–3921.",
        "C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li and P. J. Liu, “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,” <i>J. Mach. Learn. Res.</i>, vol. 21, no. 140, pp. 1–67, 2020.",
        "Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer and V. Stoyanov, “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” arXiv:1907.11692, 2019.",
        "N. Shazeer and M. Stern, “Adafactor: Adaptive Learning Rates with Sublinear Memory Cost,” in <i>Proc. ICML</i>, 2018.",
        "T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, “Focal Loss for Dense Object Detection,” in <i>Proc. ICCV</i>, 2017.",
        "T. Scholak, N. Schucher and D. Bahdanau, “PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models,” in <i>Proc. EMNLP</i>, 2021.",
        "T. Xie et al., “UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models,” in <i>Proc. EMNLP</i>, 2022.",
        "X. V. Lin, R. Socher and C. Xiong, “Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing,” in <i>Findings of EMNLP</i>, 2020.",
        "H. Li, J. Zhang, C. Li and H. Chen, “RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL,” in <i>Proc. AAAI</i>, 2023.",
        "C. Wang, K. Tatwawadi, M. Brockschmidt, P.-S. Huang, Y. Mao, O. Polozov and R. Singh, “Robust Text-to-SQL Generation with Execution-Guided Decoding,” arXiv:1807.03100, 2018.",
        "R. Zhong, T. Yu and D. Klein, “Semantic Evaluation for Text-to-SQL with Distilled Test Suites,” in <i>Proc. EMNLP</i>, 2020.",
        "X.-B. Nguyen, X.-H. Phan and M. Piccardi, “Fine-tuning text-to-SQL models with reinforcement-learning training objectives,” <i>Natural Language Processing Journal</i>, vol. 10, Art. 100135, 2025.",
        "P. Shaw, M.-W. Chang, P. Pasupat and K. Toutanova, “Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?,” in <i>Proc. ACL</i>, 2021.",
        "T. Wolf et al., “Transformers: State-of-the-Art Natural Language Processing,” in <i>Proc. EMNLP: System Demonstrations</i>, 2020.",
    ]
    references = (
        '<section class="page refs">'
        + opener("07", "References", "Datasets, models and methods this work builds on.")
        + "<ol>" + "".join(f"<li>{r}</li>" for r in refs) + "</ol>"
        + '<p class="links"><b>Code</b> github.com/iteshxt/spinneret (MIT) &nbsp; <b>Models</b> '
        "huggingface.co/iteshxt (CC BY-SA 4.0, derived from Spider)</p></section>"
    )

    css = (HERE / "report.css").read_text(encoding="utf-8")
    css = (css.replace("FONT_SANS", font("inter-latin-wght-normal.woff2"))
              .replace("FONT_ITALIC", font("inter-latin-wght-italic.woff2"))
              .replace("FONT_CODE", font("jetbrains-mono-latin-wght-normal.woff2")))
    js = (HERE / "report.js").read_text(encoding="utf-8")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Spinneret: project report</title><style>{css}</style></head><body>
{cover}{abstract}{problem}{proposed}{detailed}{datasets}{results}{shots}{references}
<script>{js}</script></body></html>"""
    for kind in ("T", "F"):
        order: dict[str, int] = {}
        for key in re.findall(rf"@@{kind}:([\w-]+)@@", html):
            order.setdefault(key, len(order) + 1)
        html = re.sub(rf"@@{kind}:([\w-]+)@@", lambda m: str(order[m.group(1)]), html)
        html = re.sub(rf"@@REF:{kind}:([\w-]+)@@", lambda m: str(order[m.group(1)]), html)
    assert "@@" not in html, "unresolved table or figure reference"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), f"({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
