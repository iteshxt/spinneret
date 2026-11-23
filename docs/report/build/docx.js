// Wraps the designed PDF in a DOCX: one full-page image per page, so the file looks exactly
// like the PDF in Word. Text is not editable in this DOCX; edit report.py and rebuild instead.
// Usage: node docs/report/build/docx.js <folder of page PNGs from pdftoppm>
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, ImageRun } = require("docx");

const dir = process.argv[2];
const out = path.resolve(__dirname, "../Spinneret-Report.docx");
const pages = fs.readdirSync(dir).filter((f) => f.endsWith(".png")).sort();
const A4 = { width: 11906, height: 16838 }; // twips
const px = { width: Math.round((210 / 25.4) * 96), height: Math.round((297 / 25.4) * 96) };

const doc = new Document({
  title: "Spinneret: project report",
  sections: pages.map((f) => ({
    properties: { page: { size: A4, margin: { top: 0, bottom: 0, left: 0, right: 0, header: 0, footer: 0 } } },
    children: [
      new Paragraph({
        spacing: { before: 0, after: 0, line: 240 },
        children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(dir, f)), transformation: px })],
      }),
    ],
  })),
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(out, buf);
  console.log("wrote", path.relative(process.cwd(), out), `(${pages.length} pages)`);
});
