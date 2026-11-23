# Final report

`Spinneret-Report.docx` / `.pdf` — the project report (18 pages, A4).

Rebuild from the repository root (needs Node, Google Chrome and LibreOffice Writer):

```bash
python3 docs/report/build/figures.py            # flowcharts and charts as HTML/SVG
node docs/report/build/capture.mjs              # render them to docs/report/figures/*.png
(cd docs/report/build && npm ci)                # docx library
node docs/report/build/build.js                 # docs/report/Spinneret-Report.docx
soffice --headless --convert-to pdf --outdir docs/report docs/report/Spinneret-Report.docx
```

Numbers come from `reports/summary/summary.json`, `data/processed/stats.json` (EDA, saved
in `build/eda.json`), the model cards and the configs. The title page uses `docs/img/logo.png`;
highlighted fields on it are to be filled in by hand.
