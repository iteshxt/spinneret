# Final report

`Spinneret-Report.pdf` is the designed report (18 pages, A4). `Spinneret-Report.docx` holds the
same pages as full-page images, for places that require a Word file; its text is not editable.

The report is written as HTML and CSS and printed to PDF with headless Chrome. Rebuild from the
repository root (needs Python 3, Node, Google Chrome and Poppler):

```bash
python3 docs/report/build/report.py      # figures + docs/report/source/report.html
node docs/report/build/print.mjs         # docs/report/Spinneret-Report.pdf
(cd docs/report/build && npm ci)         # docx library, for the Word version
pdftoppm -png -r 170 docs/report/Spinneret-Report.pdf /tmp/pages/p
node docs/report/build/docx.js /tmp/pages
```

Title-page details (course, batch, names, registration and roll numbers) are read from
`build/title.json`, which is git-ignored so personal details stay out of the repository;
missing fields print as blanks. The cover uses `docs/img/logo.png`.

Numbers come from `reports/summary/summary.json`, `data/processed/stats.json` (EDA, saved in
`build/eda.json`), the checkpoints' training logs (`build/training_history.json`), the model
cards and the configs. Fonts: Inter and JetBrains Mono (SIL Open Font License).
