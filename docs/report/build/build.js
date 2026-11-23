// Builds docs/report/Spinneret-Report.docx. Run: node docs/report/build/build.js
// Every number in the text comes from reports/summary/summary.json, data/processed/stats.json,
// the model cards or the configs; see docs/report/README.md.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, ShadingType, HeadingLevel, LevelFormat, PageNumber, Footer,
  VerticalAlign, TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "../../..");
const FIG = path.join(ROOT, "docs/report/figures");
const IMG = path.join(ROOT, "docs/img");
const OUT = path.join(ROOT, "docs/report/Spinneret-Report.docx");

const SERIF = "Times New Roman";
const MONO = "Courier New";
const TEXT_W = 9026; // A4 width 11906 - 2 x 1440 margins (DXA)

// --- helpers -----------------------------------------------------------------------------------

/** "plain **bold** `code` *italic*" -> TextRuns */
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|\[\[[^\]]+\]\])/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("`"))
      out.push(new TextRun({ text: t.slice(1, -1), font: MONO, size: 21, ...base }));
    else if (t.startsWith("[["))
      out.push(new TextRun({ text: t.slice(2, -2), highlight: "yellow", ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const P = (text, opts = {}) =>
  new Paragraph({
    alignment: opts.align ?? AlignmentType.JUSTIFIED,
    spacing: { after: opts.after ?? 120, before: opts.before ?? 0 },
    keepNext: opts.keepNext,
    children: runs(text, opts.run),
  });

const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(text)] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] });

const bullets = (items, ref = "bullets") =>
  items.map(
    (t) =>
      new Paragraph({
        numbering: { reference: ref, level: 0 },
        alignment: AlignmentType.JUSTIFIED,
        spacing: { after: 60 },
        children: runs(t),
      }),
  );

const code = (lines) =>
  lines.map(
    (l, i) =>
      new Paragraph({
        spacing: { line: 240, before: i === 0 ? 60 : 0, after: i === lines.length - 1 ? 160 : 0 },
        keepNext: i < lines.length - 1,
        keepLines: true,
        shading: { type: ShadingType.CLEAR, fill: "F2F2EF", color: "auto" },
        indent: { left: 180, right: 180 },
        children: [new TextRun({ text: l, font: MONO, size: 17 })],
      }),
  );

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

let figN = 0;
let tabN = 0;
function figure(file, caption, widthIn = 6.2, maxHeightIn = 9) {
  const { w, h, data } = pngSize(file);
  let width = widthIn * 96;
  let height = (width * h) / w;
  if (height > maxHeightIn * 96) {
    height = maxHeightIn * 96;
    width = (height * w) / h;
  }
  figN += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60, line: 240 },
      keepNext: true,
      children: [new ImageRun({ type: "png", data, transformation: { width: Math.round(width), height: Math.round(height) } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 200, line: 276 },
      children: [new TextRun({ text: `Figure ${figN}: `, bold: true, size: 21 }), ...runs(caption, { size: 21 })],
    }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "8A8A82" };
const borders = { top: border, bottom: border, left: border, right: border };

function table(caption, header, rows, widths, opts = {}) {
  tabN += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (text, i, head, keep) =>
    new TableCell({
      borders,
      width: { size: widths[i], type: WidthType.DXA },
      shading: head ? { type: ShadingType.CLEAR, fill: "EFEFEA", color: "auto" } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      verticalAlign: VerticalAlign.CENTER,
      children: [
        new Paragraph({
          alignment: i === 0 || opts.leftAll ? AlignmentType.LEFT : AlignmentType.CENTER,
          spacing: { line: 240 },
          keepNext: keep,
          children: runs(String(text), { size: opts.size ?? 20, bold: head || undefined }),
        }),
      ],
    });
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 80, line: 276 },
      keepNext: true,
      children: [new TextRun({ text: `Table ${tabN}: `, bold: true, size: 21 }), ...runs(caption, { size: 21 })],
    }),
    new Table({
      width: { size: total, type: WidthType.DXA },
      columnWidths: widths,
      alignment: AlignmentType.CENTER,
      rows: [
        new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, i, true, true)) }),
        ...rows.map((r, ri) =>
          new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, ri < rows.length - 1)) }),
        ),
      ],
    }),
    new Paragraph({ spacing: { after: 80 }, children: [] }),
  ];
}

// --- title page (layout of the department's example front page) ------------------------------

const C = (text, size, opts = {}) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: opts.after ?? 0, before: opts.before ?? 0, line: 276 },
    children: runs(text, { size, bold: opts.bold, font: SERIF }),
  });

const logo = pngSize(path.join(IMG, "logo.png"));
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const sideCell = (lines, align) =>
  new TableCell({
    borders: noBorders,
    width: { size: TEXT_W / 2, type: WidthType.DXA },
    children: lines.map(
      (l, i) =>
        new Paragraph({
          alignment: align,
          spacing: { line: 300 },
          children: runs(l, { size: 23, bold: i === 0 || undefined }),
        }),
    ),
  });

const titlePage = [
  C("[[Submitted for partial fulfillment for CA-_ of INT-___]]", 24, { before: 200, after: 360 }),
  C("A Project Report on", 24, { after: 120 }),
  C("SPINNERET", 40, { bold: true, after: 80 }),
  C("Small, Fast Text-to-SQL Query Generation with Fine-Tuned Transformers", 26, { bold: true, after: 420 }),
  C("BACHELOR'S OF TECHNOLOGY", 30, { bold: true }),
  C("IN", 22, { before: 40, after: 40 }),
  C("COMPUTER SCIENCE & ENGINEERING", 30, { bold: true }),
  C("[[Batch (2023 – 2027)]]", 24, { bold: true, before: 60, after: 300 }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 360 },
    children: [new ImageRun({ type: "png", data: logo.data, transformation: { width: 200, height: 200 } })],
  }),
  new Table({
    width: { size: TEXT_W, type: WidthType.DXA },
    columnWidths: [TEXT_W / 2, TEXT_W / 2],
    rows: [
      new TableRow({
        children: [
          sideCell(["Submitted to:", "[[Name of the faculty]]", "[[UID: _____]]", "Computer Science & Engineering"], AlignmentType.LEFT),
          sideCell(["Submitted by:", "[[Name of the student]]", "[[Reg No: _________]]", "[[Roll Number: __]]"], AlignmentType.RIGHT),
        ],
      }),
    ],
  }),
  C("Department of Computer Science and Engineering", 20, { before: 700 }),
  C("Lovely Professional University, Jalandhar-Delhi G.T Road", 20),
  C("Phagwara, Punjab", 20),
];

// --- abstract -----------------------------------------------------------------------------------

const abstract = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Abstract")] }),
  P("Text-to-SQL systems let people query a relational database in plain English. The hard version of the task is *cross-database*: the model must answer questions about databases it has never seen, reading an unfamiliar schema and mapping each word of the question to the right table, column and value. Large language models do this well but are expensive to run; this project asks how far small, open, fine-tuned models can go when they must run on a student laptop."),
  P("**Spinneret** builds and compares three tiers of models on the Spider benchmark. **Baseline** fine-tunes T5-small (60M parameters) on the question and the whole schema. **Lite** adds a RoBERTa-base schema ranker (126M) that scores every table and column and passes only the top four tables and five columns of each to T5-small. **Spinneret** uses the same ranker with T5-base (223M) and skeleton-first decoding, in which the model writes the query's shape before filling in names and values. All three use database-value matching and execution-guided beam search: candidate queries are run read-only, and the first one that executes is returned."),
  P("All models were trained on free Colab and Kaggle T4 GPUs and evaluated with the official test-suite scripts on Spider's development set (1,034 questions over 20 unseen databases). Spinneret reaches **62.6% exact match and 68.8% execution accuracy** with a median latency of 1.29 s on a laptop GTX 1650; Lite reaches 55.8% execution accuracy in 0.39 s; Baseline reaches 49.2%, above the published T5-small result of 46.2%. Ablations show that execution-guided decoding (+6.6 points), the schema ranker (+6.6) and database values (+2.5) all help, while the skeleton does not help the small model. A failure taxonomy shows that the remaining errors are mostly wrong joins and wrong columns."),
  P("The models are served by a FastAPI back end and a React web application that shows the ranker's scores, the query's skeleton, every candidate and the result. The code is released under the MIT licence and all seven model checkpoints are public on the Hugging Face Hub."),
  P("**Keywords:** text-to-SQL, semantic parsing, T5, schema linking, execution-guided decoding, Spider.", { before: 120 }),
];

// --- 1. problem statement -------------------------------------------------------------------------

const problem = [
  H1("1. Problem Statement"),
  P("Most structured data sits in relational databases, but querying them needs SQL, which most users do not know. This project addresses the problem statement **\"Text-to-SQL Query Generation\"**: translate English questions into executable SQL for a given database with fine-tuned sequence-to-sequence transformer models."),
  H2("1.1 Why the task is hard"),
  ...bullets([
    "**Unseen schemas.** The evaluation databases never appear in training, so the model must read each schema and link words to it (\"singers\" to `singer`, \"from France\" to `country = 'France'`).",
    "**Long, noisy inputs.** A serialised schema can exceed a small model's 512-token input, and most of it is irrelevant to any one question.",
    "**Exact values and compositional SQL.** Literals must match the stored spelling, and joins, grouping, nesting and set operations must all be right; one wrong column makes the answer wrong.",
    "**Correctness is semantic.** A query can differ from the reference and still be right, or look right and fail to run; it must be judged by executing it.",
  ]),
  H2("1.2 Constraints"),
  P("The project was built solo in five weeks with no paid compute: training on free Colab and Kaggle T4 GPUs whose sessions end without warning, and serving all models on a laptop GTX 1650 (4 GB) in about a second per question. Large language models were therefore out of scope."),
  H2("1.3 Objectives"),
  ...bullets([
    "Measure how accurate small fine-tuned models can be on Spider's unseen databases, and what each technique adds, at a known size and speed.",
    "Train three tiers (a required T5-small baseline, a fast Lite model and a best-accuracy Spinneret model) and evaluate them with the official scripts, ablations and an error analysis.",
    "Ship a safe, transparent demo that answers questions on all 20 evaluation databases.",
  ]),
];

// --- 2. proposed work ---------------------------------------------------------------------------------

const proposed = [
  H1("2. Proposed Work"),
  P("Spinneret is a pipeline around a fine-tuned T5 encoder-decoder. Instead of relying on model size, it adds four cheap techniques from the literature, each of which can be switched on or off and measured: database-value matching (after BRIDGE [8]), a schema-item ranker and skeleton-first decoding (after RESDSQL [9]), and execution-guided decoding [10]. Figure 1 shows how one question flows through the served system."),
  ...figure(path.join(FIG, "fig-inference.png"), "Inference pipeline: from an English question to an executed answer.", 5.6, 5.0),
  H2("2.1 Three model tiers"),
  P("The three tiers share the same data pipeline, value matching, decoding and evaluation, and differ only in the generator, its schema input and its target format (Table 1). This makes their results directly comparable and lets each technique be isolated by an ablation."),
  ...table("The three model tiers.", ["Tier", "Generator", "Schema input", "Target", "Beam", "Parameters"], [
    ["Baseline (T1)", "T5-small", "whole schema", "SQL", "4", "60.5M"],
    ["Lite (T2)", "T5-small", "ranker top 4 x 5", "SQL*", "4", "60.5M + 125.8M"],
    ["Spinneret (T3)", "T5-base", "ranker top 4 x 5", "skeleton | SQL", "8", "222.9M + 125.8M"],
  ], [1700, 1250, 1700, 1700, 800, 1876]),
  P("*Lite was planned with skeleton-first targets; ablation E4 showed the skeleton does not help T5-small, so the served Lite writes plain SQL (both variants are reported in Section 5).", { run: { size: 21 } }),
  H2("2.2 Techniques"),
  ...bullets([
    "**Database-value matching.** Question n-grams are matched against the text values stored in the database, and matches are written next to their column in the input (`country ( France )`), so the model can copy the exact spelling.",
    "**Schema ranker.** A RoBERTa-base cross-encoder reads the question with the whole schema and scores every table and column. Only the top four tables and five columns per table, plus the keys that link them, are given to the generator, ordered by score. The input becomes shorter and the distracting items disappear.",
    "**Skeleton-first decoding.** The target is `skeleton | SQL`, e.g. `select avg ( _ ) from _ where _ = _ | select avg ( age ) from singer where country = 'France'`: the model commits to the query's structure before choosing names and values.",
    "**Execution-guided decoding.** Beam search returns k candidates; each is checked by a SQL guard and run read-only on the database, and the first that executes without error is the answer. Queries that cannot run are never shown as answers.",
  ]),
  H2("2.3 Training and evaluation plan"),
  P("Figure 2 shows the offline pipeline. The data is prepared once; the ranker is trained first, because its scores build the inputs of Lite and Spinneret; the generators are then trained on free GPUs, pushing a checkpoint to the Hugging Face Hub after every epoch so that a training run can resume when a session ends. Each model is evaluated on Spider dev with the official test-suite scripts [11], reporting exact match (EM, same SQL structure) and execution accuracy (EX, same result), per difficulty level and per question type."),
  ...figure(path.join(FIG, "fig-training.png"), "Training, evaluation and release pipeline.", 5.8, 3.3),
  P("Seven experiments measure the design: E1 compares the tiers; E2 compares greedy, beam and execution-guided decoding; E3 removes database values; E4 separates the ranker from the skeleton; E5 varies the ranker's top-k; E6 removes the `train_others` split; and E7 compares T5-small with T5-base. A rule-based failure taxonomy labels every wrong answer, and a sample of 100 errors is checked by a second labelling pass."),
  H2("2.4 System architecture"),
  P("The system has five parts, kept deliberately small so that one person can build and maintain them (Table 2). The Python package holds every stage of the pipeline behind one command-line tool; trained checkpoints live on the Hugging Face Hub and are downloaded on first use; the API server loads them once and answers requests; the web application is a static bundle served by the same process; and the 20 Spider databases are read with read-only SQLite connections."),
  ...table("System components.", ["Component", "Role", "Technology"], [
    ["`spinneret` package + CLI", "data, ranker, generator, decoding, evaluation, serving", "Python 3.12, PyTorch, Transformers, Typer"],
    ["Model hub", "checkpoints during training, public models after release", "Hugging Face Hub"],
    ["API server", "loads models, answers questions, serves results", "FastAPI, Uvicorn, Pydantic"],
    ["Web application", "landing, demo, evaluation, about and privacy pages", "React 18, TypeScript, Vite"],
    ["Databases", "the 20 Spider dev databases, read-only", "SQLite"],
  ], [2500, 3600, 2926], { leftAll: true }),
  H2("2.5 Success criteria"),
  P("Targets were fixed before training, from published results for each model size (Table 3). Exact match (EM) requires the same SQL structure as the reference; execution accuracy (EX) requires the same result, which is the measure a user cares about. Two further targets were set for the components: ranker coverage@(4,5) of at least 95% on validation (otherwise a larger ranker would be trained) and at least 95% executable SQL for Spinneret."),
  ...table("Targets on Spider dev (EM / EX %).", ["Tier", "Published reference", "Minimum", "Target", "Stretch"], [
    ["Baseline", "T5-small 45.8 / 46.2 [12]", "42 / 42", "46 / 48", "50 / 52"],
    ["Lite", "none published", "50 / 52", "55 / 58", "60 / 63"],
    ["Spinneret", "RESDSQL-Base 71.7 / 77.9 [9]", "60 / 66", "65 / 72", "70 / 78"],
  ], [1500, 3000, 1500, 1500, 1526]),
  H2("2.6 Work plan"),
  ...table("Five-week plan.", ["Week", "Work", "Milestone"], [
    ["1", "Spider download, preparation, tokenizer audit, remote-training setup", "processed splits, smoke test on Colab"],
    ["2", "Baseline training, schema ranker, evaluation wrapper", "Baseline scored; ranker coverage decided"],
    ["3", "Ranked inputs, Lite and Spinneret training", "all three tiers scored on dev"],
    ["4", "Evaluation, ablations E2-E7, failure taxonomy, latency", "summary report"],
    ["5", "API, web application, model cards, release", "public demo and models"],
  ], [900, 4900, 3226], { leftAll: true }),
  P("The deliverables are seven trained checkpoints with model cards on the Hugging Face Hub (the three served models, the ranker and three evaluation-only checkpoints), the evaluation reports and ablation tables, the API and web application, and the open-source code (MIT licence), documentation and this report."),
];

// --- 3. detailed description -------------------------------------------------------------------------------

const detailed = [
  H1("3. Detailed Description"),
  P("The implementation is a Python 3.12 package (`spinneret`) with a single command-line entry point, plus a TypeScript web application. Every stage below is one CLI command whose behaviour is fixed by a YAML config, so each result can be reproduced from the repository."),
  H2("3.1 Data preparation"),
  P("`spinneret download` fetches the Spider 1.0 archive and verifies its SHA-256 checksum. `spinneret prepare` then produces one JSON line per question with the model input, the targets and metadata:"),
  ...bullets([
    "**Splits.** Spider's `train_spider` and `train_others` form the training data. A validation set is carved out of it *by database* (10%, 16 databases), so validation also measures unseen-schema performance; Spider dev remains the untouched benchmark.",
    "**SQL normalisation.** Gold SQL is lower-cased except string literals, double quotes become single quotes, and every token is separated by one space (`select count ( * ) from singer`). A unit test runs gold and normalised SQL on the database and checks that the results are identical.",
    "**Skeletons.** Every table, column, alias, `*` and literal is replaced by `_`, and runs such as `_ , _` are collapsed, giving targets like `select count ( _ ) from _`.",
    "**Value matching.** For each database the distinct values of text columns are cached (skipping values over 64 characters and columns with more than 5,000 values). Question n-grams of 1-6 words match a value when equal ignoring case, or when their fuzzy ratio (rapidfuzz) is at least 85 for values of three or more characters; at most two values per column are kept.",
    "**Tokenizer audit.** T5's SentencePiece vocabulary cannot write `<`, so `<=` and `<>` would be lost. `spinneret audit-tokenizer` found the missing pieces and four tokens (` <`, `<`, ` <=`, ` <>`) were added to the tokenizer and embedding matrix.",
  ]),
  P("The serialised input lists the question, the database name and each table with its columns; matched values appear in brackets. For the ranked models the order of tables and columns follows the ranker's scores and the foreign keys are appended:"),
  ...code([
    "What is the average, minimum, and maximum age of all singers from France?",
    "  | concert_singer",
    "  | singer : age , country ( France ) , singer_id , name , is_male",
    "  | singer_in_concert : singer_id , concert_id",
    "  | stadium : average , stadium_id , highest , lowest , name",
    "  | concert : year , stadium_id , concert_id , concert_name , theme",
    "  | fk : concert.stadium_id = stadium.stadium_id , ...",
    "target: select avg ( _ ) , min ( _ ) , max ( _ ) from _ where _ = _",
    "  | select avg ( age ) , min ( age ) , max ( age ) from singer",
    "    where country = 'France'",
  ]),
  H2("3.2 Schema ranker"),
  P("The ranker (Figure 3) is a RoBERTa-base cross-encoder. The question and the serialised schema are encoded together; the character offsets of every table and column name are mapped to tokens and averaged into one vector per item, and each column vector is concatenated with its table's vector. Two linear heads with a sigmoid give the probability that the gold SQL uses each table and each column. Labels come from the tables and columns referenced anywhere in the gold query. Because most items are negatives, the loss is focal loss (gamma = 2). Schemas longer than 512 tokens are split into chunks of tables, each paired with the question."),
  ...figure(path.join(FIG, "fig-ranker.png"), "Schema ranker: architecture and training.", 6.2, 2.3),
  P("The ranked input for Lite and Spinneret is then built in four steps: (1) keep the four highest-scoring tables and, in each, the five highest-scoring columns; (2) also keep the primary and foreign keys that link two kept tables, so join conditions can still be written; (3) order tables, and columns within each table, by descending score, so the order itself tells the generator what matters most; (4) append the kept foreign-key links after the tables."),
  P("The checkpoint is selected by *coverage@(4,5)*: the share of questions for which every needed table and column survives the top-4-tables, top-5-columns cut. The best epoch reached 98.1% on validation and 98.5% on Spider dev, so the larger RoBERTa-large ranker considered in the plan was not needed. When building training inputs for the generators, any gold item the ranker missed is inserted at its scored position, so the generator never learns to write an item it cannot see; validation, dev and live inputs use the ranker's output only."),
  H2("3.3 Generator training"),
  P("All generators are trained with Hugging Face Transformers using the Adafactor optimiser, an effective batch of 32, a linear learning-rate schedule with 3% warm-up, inputs truncated at 512 and targets at 256 tokens, and full fp32 precision (T5 overflows in fp16). After every epoch the model decodes the validation set greedily, and training stops when validation EX has not improved for five epochs; the best epoch is kept (Table 5)."),
  ...table("Generator training runs.", ["Model", "Base", "Learning rate", "Best epoch", "Valid EM / EX", "Trained on"], [
    ["Baseline", "T5-small", "1e-3", "18", "44.7 / 49.3", "Colab T4 + laptop"],
    ["Lite (skeleton)", "T5-small", "1e-3", "12", "50.8 / 52.6", "Colab T4"],
    ["Lite (served, E4)", "T5-small", "1e-3", "17", "50.1 / 54.5", "Kaggle T4"],
    ["Spinneret", "T5-base", "1e-4", "18", "59.7 / 67.0", "Colab T4"],
  ], [1900, 1100, 1300, 1150, 1650, 1926]),
  P("Free GPU sessions end without warning, so training was driven from the laptop with Google's Colab command-line tool (and the Kaggle CLI when Colab had no GPU free). A training run pushes its latest checkpoint, optimiser and scheduler state to a private Hub repository every epoch; `--resume hub` downloads it and continues exactly where it stopped. The Hugging Face token is copied to the remote machine as a file and never appears in commands, logs or the repository."),
  P("Small shell scripts wrap this workflow. `scripts/colab/start.sh` opens a T4 session, uploads the code and the token file and installs the locked dependencies; `train.sh` starts a training run in the background on the remote machine; `logs.sh` follows its output; and `stop.sh` releases the GPU. On Kaggle, `dataset.sh` packs the code and the prepared data into a private dataset and `push.sh` submits one run as a GPU notebook; because notebooks pushed from the command line cannot read secrets, those runs save checkpoints locally and the best one is downloaded and pushed to the Hub from the laptop. Training logs are written to TensorBoard."),
  H2("3.4 Decoding and safe execution"),
  P("At inference the generator returns k beam candidates (k = 4 for Baseline and Lite, 8 for Spinneret). For skeleton models the SQL is the text after the first ` | `. Each candidate passes a guard that accepts a single `SELECT` or `WITH` statement and refuses data- or schema-changing keywords (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `PRAGMA` and others). It then runs on a SQLite connection opened read-only with `query_only` set, a 5-second time limit and at most 500 returned rows. The first candidate that runs is the answer; if none runs, the top candidate is shown with its error explained in plain language."),
  H2("3.5 Evaluation and error analysis"),
  P("`spinneret predict` decodes Spider dev, `spinneret evaluate` scores it with the official test-suite evaluation [11] (EM and EX per difficulty) and `spinneret analyze` adds two analyses. Questions are grouped by type (simple lookup, aggregation, two-table join, multi-join, nested or set operation), and every wrong answer receives a primary error label from a rule-based taxonomy that compares the parsed prediction with the parsed gold query in a fixed order: invalid SQL, wrong table, join error, nesting or set-operation error, wrong column, aggregation, GROUP BY, condition, ORDER BY/LIMIT and value errors. Predictions that miss EM but return the right result are counted separately as semantically equivalent. A seeded sample of 100 Spinneret errors was labelled independently to check the rules: a blind review agreed with the automatic labels on 93 of 100 (86 before one rule was fixed), and a second check agreed on 94. Both reviews were done by an AI assistant (Claude), not by a human annotator, and are reported as such. `spinneret latency` measures each stage of the pipeline question by question at batch size 1 on the laptop GPU (Table 7)."),
  ...table("Failure taxonomy: the first rule that matches gives the primary label.", ["#", "Label", "Rule (prediction vs. gold)"], [
    ["1", "invalid_sql", "fails to parse or fails to execute"],
    ["2", "wrong_table", "different tables in FROM (same number)"],
    ["3", "join_error", "missing or extra join, or different join conditions"],
    ["4", "nesting_setop_error", "sub-query or INTERSECT / UNION / EXCEPT differs"],
    ["5", "wrong_column", "different columns in SELECT or WHERE"],
    ["6", "aggregation_error", "different, missing or extra COUNT / SUM / AVG / MIN / MAX"],
    ["7", "group_by_error", "GROUP BY or HAVING differs"],
    ["8", "condition_error", "same WHERE columns, different operator, AND/OR or negation"],
    ["9", "order_limit_error", "ORDER BY column or direction, or LIMIT, differs"],
    ["10", "value_error", "only a literal value is wrong"],
    ["11", "other", "none of the above (e.g. DISTINCT)"],
  ], [600, 2500, 5926], { leftAll: true, size: 19 }),
  ...table("Median latency per pipeline stage (ms, laptop GTX 1650, batch 1, Spider dev).", ["Model", "Values", "Ranking", "Generation", "Execution", "Total"], [
    ["Baseline", "< 1", "-", "371", "1", "374"],
    ["Lite (skeleton)", "< 1", "59", "550", "1", "623"],
    ["Lite (served)", "< 1", "62", "307", "1", "393"],
    ["Spinneret", "< 1", "63", "1,220", "1", "1,286"],
  ], [2500, 1200, 1200, 1500, 1200, 1426]),
  P("Generation dominates the time; the ranker adds about 60 ms, and value matching and query execution are negligible. Peak GPU memory is 0.41 GB for Baseline, 0.62 GB for Lite and 1.66 GB for Spinneret, so all three fit together on a 4 GB laptop GPU."),
  H2("3.6 Serving: API and web application"),
  P("`spinneret serve` starts a FastAPI application that also serves the built web app. Models load once in a background thread and are warmed up with one question each (cutting the first answer from 3.4 s to about 0.5 s); until they are ready the API answers schema and metrics requests and reports \"warming up\". A single GPU lock runs one generation at a time, so concurrent requests queue instead of exhausting the 4 GB of memory. The main endpoints are `POST /api/query` (question to answer), `POST /api/execute` (run edited SQL through the same guard), and read-only endpoints for databases, schemas, example questions, models and evaluation results."),
  ...table("Main API endpoints.", ["Method and path", "Purpose"], [
    ["GET /api/health", "status (loading, ok or error), device and loaded models"],
    ["GET /api/models", "served models with size, accuracy and median latency"],
    ["GET /api/databases, .../{db}/schema, .../{db}/examples", "databases, their tables and keys, sample questions"],
    ["POST /api/query", "question to answer: SQL, skeleton, candidates, ranker scores, rows, timings"],
    ["POST /api/execute", "run edited SQL through the same guard, time limit and row cap"],
    ["GET /api/summary, /api/metrics/{run}", "evaluation results for the Evaluation page"],
  ], [4000, 5026], { leftAll: true, size: 19 }),
  P("The web application is written in React 18 and TypeScript with Vite, TanStack Query and CodeMirror, with Framer Motion for animation. It has a landing page that shows a real question threaded word by word to its SQL and answer; a demo page with the database's schema (highlighted with the ranker's scores), the question box, the skeleton, an editable SQL editor, all candidates and the result; an evaluation page; an about page with the model cards; and a privacy page. The interface works in light and dark themes, on phones, and from the keyboard, and the page makes no third-party requests. The server keeps no record of the questions asked."),
  H2("3.7 Testing and release"),
  P("The Python package has 101 tests (pytest), covering SQL normalisation, the guard, value matching, serialisation, evaluation wrappers, the taxonomy rules and the API, and the web app has 15 component tests (Vitest); linting uses ruff. Model cards are generated from the evaluation reports by `spinneret model-cards` and published with each checkpoint. All seven checkpoints were made public on the Hugging Face Hub and the code was released on GitHub under the MIT licence, so the demo can be reproduced with four commands:"),
  ...code([
    "uv sync --extra train --extra api      # Python 3.12 environment",
    "uv run spinneret download              # Spider databases (checksum verified)",
    "(cd web && npm ci && npm run build)    # web application",
    "uv run spinneret serve                 # http://localhost:8000",
  ]),
  H2("3.8 Limitations and future work"),
  P("The models answer English questions in the style of Spider and are weakest on queries that join three or more tables (27.9% EX for Spinneret) and on nested or set queries (50.3%). Values must be spelled as stored; numbers written in words are often wrong. Because Lite and Spinneret only see what the ranker keeps, a missed table or column cannot be recovered (8 of Spinneret's errors). Spinneret's 1.3 s median answer time is acceptable for a demo but not for heavy use on a laptop GPU."),
  P("The clearest next steps follow from the ablations and the error analysis: a RoBERTa-large ranker as in RESDSQL [9], which is where most of the remaining gap to 77.9% EX lies; constrained decoding such as PICARD [6] to rule out invalid SQL during generation rather than after it; reinforcement-learning objectives that reward execution accuracy directly [12]; CodeT5 as a code-pretrained generator of the same size; and 8-bit quantisation to make Spinneret faster on small GPUs. Evaluating on harder benchmarks with realistic, messy databases would test how far these results carry beyond Spider."),
];

// --- 4. datasets ------------------------------------------------------------------------------------

const datasets = [
  H1("4. Dataset and Exploratory Data Analysis"),
  P("The project uses **Spider 1.0** [1], a large human-labelled, cross-domain text-to-SQL dataset (CC BY-SA 4.0) of 200 SQLite databases covering 138 domains. Its training databases and development databases do not overlap, which makes Spider dev a test of generalisation to unseen schemas. Table 9 shows the splits used here; the hidden test set was not used."),
  ...table("Splits used in this project (after the by-database validation split).", ["Split", "Questions", "Databases", "easy / medium / hard / extra", "Tables per DB (mean / max)"], [
    ["train (train_spider + train_others)", "7,941", "130", "1,815 / 2,723 / 1,746 / 1,657", "5.4 / 26 (incl. valid)"],
    ["valid (from train_spider)", "718", "16", "168 / 276 / 176 / 98", ""],
    ["dev (benchmark)", "1,034", "20", "248 / 446 / 174 / 166", "4.0 / 11"],
  ], [2700, 1050, 1050, 2350, 1876]),
  ...figure(path.join(FIG, "fig-eda.png"), "Difficulty mix per split and share of queries using each SQL construct.", 6.2, 2.3),
  ...bullets([
    "**Unseen schemas.** None of the 20 dev databases appears in training; dev schemas average 4.0 tables and 22.1 columns, training schemas 5.4 tables and 27.8 columns (up to 26 tables and 352 columns).",
    "**Difficulty.** Dev has more medium questions (43%) and fewer extra-hard ones (16%) than training (34% and 21%).",
    "**SQL constructs.** 40% of dev queries join tables, 53% aggregate, 27% group, 8% nest a sub-query and 8% use a set operation; 56% of dev questions need only one table.",
    "**Input length.** Training inputs average 200 T5 tokens (median 162); 319 of 7,941 (4%) exceed 512 tokens and are truncated, which motivated the schema ranker. Dev inputs never exceed 468 tokens. Targets average 45 tokens (71 with the skeleton).",
    "**Values.** 28% of training and 34% of dev questions mention a value that matches a stored database value.",
  ]),
];

// --- 5. results -------------------------------------------------------------------------------------

const results = [
  H1("5. Results"),
  ...table("Spider dev results (official test-suite evaluation, 1,034 questions; latency per question on a laptop GTX 1650, batch 1).", ["Model", "EM %", "EX %", "Valid SQL %", "Parameters", "p50 / p95"], [
    ["Baseline (T5-small, full schema)", "48.5", "49.2", "67.5", "60.5M", "0.37 / 0.74 s"],
    ["Lite (T5-small + ranker + skeleton)", "52.4", "54.4", "78.3", "186M", "0.62 / 1.18 s"],
    ["Lite served (no skeleton, E4)", "52.5", "55.8", "82.2", "186M", "0.39 / 0.91 s"],
    ["**Spinneret (T5-base + ranker + skeleton)**", "**62.6**", "**68.8**", "**91.7**", "349M", "1.29 / 2.50 s"],
  ], [3300, 800, 800, 1150, 1300, 1676], { size: 19 }),
  ...figure(path.join(FIG, "fig-results.png"), "Execution accuracy by difficulty.", 6.2, 2.15),
  ...bullets([
    "**Against published models:** Baseline beats the published T5-small result (46.2% EX) and Spinneret beats T5-base (60.2% EX) [12] by 8.6 points; RESDSQL-Base [9], which uses a larger ranker, reaches 77.9%. Spinneret passes its minimum (66% EX) but misses its 72% target; Lite misses its 58% target by 2.2 points.",
    "**Ablations:** execution-guided decoding adds 6.6 EX points over greedy (62.2% to 68.8%); the ranker adds 6.6 to T5-small (E4); database values add 2.5 (E3); T5-base adds 14.4 over T5-small (E7); the skeleton costs T5-small 1.4 points; `train_others` makes no measurable difference (E6: 49.0% without it, 49.2% with it).",
    "**Errors:** Spinneret's invalid SQL fell from 336 (Baseline) to 88; its remaining 323 wrong answers are mostly invalid SQL (88), wrong joins (88) and wrong columns (59). Another 92 predictions differ from the gold text but return the right result. Multi-table joins remain weakest (27.9% EX).",
  ]),
];

// --- 6. screenshots ---------------------------------------------------------------------------------

const shots = [
  H1("6. Screenshots"),
  ...figure(path.join(IMG, "landing.png"), "Landing page: a real question, its generated SQL and the answer, joined by threads.", 6.2, 3.9),
  ...figure(path.join(IMG, "demo.png"), "Demo page: ranker scores on the schema, the skeleton and SQL, and the answer (34.5, 25, 43).", 6.2, 3.9),
  ...figure(path.join(FIG, "shot-candidates.png"), "Execution-guided decoding: candidates 1-3 fail, candidate 4 runs and is returned; the model menu is open.", 6.2, 3.9),
  ...figure(path.join(IMG, "evaluation.png"), "Evaluation page: accuracy, size and speed of each model, and accuracy by difficulty.", 6.2, 3.9),
];

// --- references ------------------------------------------------------------------------------------

const refs = [
  "T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z. Li, J. Ma, I. Li, Q. Yao, S. Roman, Z. Zhang and D. Radev, \"Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task,\" in *Proc. EMNLP*, 2018, pp. 3911-3921.",
  "C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li and P. J. Liu, \"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer,\" *J. Mach. Learn. Res.*, vol. 21, no. 140, pp. 1-67, 2020.",
  "Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer and V. Stoyanov, \"RoBERTa: A Robustly Optimized BERT Pretraining Approach,\" arXiv:1907.11692, 2019.",
  "N. Shazeer and M. Stern, \"Adafactor: Adaptive Learning Rates with Sublinear Memory Cost,\" in *Proc. ICML*, 2018.",
  "T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollar, \"Focal Loss for Dense Object Detection,\" in *Proc. ICCV*, 2017.",
  "T. Scholak, N. Schucher and D. Bahdanau, \"PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models,\" in *Proc. EMNLP*, 2021.",
  "T. Xie et al., \"UnifiedSKG: Unifying and Multi-Tasking Structured Knowledge Grounding with Text-to-Text Language Models,\" in *Proc. EMNLP*, 2022.",
  "X. V. Lin, R. Socher and C. Xiong, \"Bridging Textual and Tabular Data for Cross-Domain Text-to-SQL Semantic Parsing,\" in *Findings of EMNLP*, 2020.",
  "H. Li, J. Zhang, C. Li and H. Chen, \"RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL,\" in *Proc. AAAI*, 2023.",
  "C. Wang, K. Tatwawadi, M. Brockschmidt, P.-S. Huang, Y. Mao, O. Polozov and R. Singh, \"Robust Text-to-SQL Generation with Execution-Guided Decoding,\" arXiv:1807.03100, 2018.",
  "R. Zhong, T. Yu and D. Klein, \"Semantic Evaluation for Text-to-SQL with Distilled Test Suites,\" in *Proc. EMNLP*, 2020.",
  "X.-B. Nguyen, X.-H. Phan and M. Piccardi, \"Fine-tuning text-to-SQL models with reinforcement-learning training objectives,\" *Natural Language Processing Journal*, vol. 10, Art. 100135, 2025, doi: 10.1016/j.nlp.2025.100135.",
  "P. Shaw, M.-W. Chang, P. Pasupat and K. Toutanova, \"Compositional Generalization and Natural Language Variation: Can a Semantic Parsing Approach Handle Both?,\" in *Proc. ACL*, 2021.",
  "T. Wolf et al., \"Transformers: State-of-the-Art Natural Language Processing,\" in *Proc. EMNLP: System Demonstrations*, 2020.",
];

const references = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun("References")] }),
  ...refs.map(
    (r, i) =>
      new Paragraph({
        alignment: AlignmentType.LEFT,
        spacing: { after: 100, line: 276 },
        indent: { left: 540, hanging: 540 },
        tabStops: [{ type: TabStopType.LEFT, position: 540 }],
        children: [new TextRun({ text: `[${i + 1}]\t`, size: 22 }), ...runs(r, { size: 22 })],
      }),
  ),
  new Paragraph({
    spacing: { before: 200, line: 276 },
    children: runs("Code: https://github.com/iteshxt/spinneret (MIT). Models: https://huggingface.co/iteshxt (CC BY-SA 4.0, derived from Spider).", { size: 21, italics: true }),
  }),
];

// --- document ---------------------------------------------------------------------------------------

const doc = new Document({
  creator: "Spinneret",
  title: "Spinneret: Small, Fast Text-to-SQL Query Generation",
  styles: {
    default: { document: { run: { font: SERIF, size: 24 }, paragraph: { spacing: { line: 360 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: SERIF, size: 32, bold: true },
        paragraph: { spacing: { before: 0, after: 240, line: 276 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: SERIF, size: 26, bold: true },
        paragraph: { spacing: { before: 200, after: 100, line: 276 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
    ],
  },
  sections: [
    {
      properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      children: titlePage,
    },
    {
      properties: {
        page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 }, pageNumbers: { start: 1 } },
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })],
        }),
      },
      children: [...abstract, ...problem, ...proposed, ...detailed, ...datasets, ...results, ...shots, ...references],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", path.relative(ROOT, OUT));
});
