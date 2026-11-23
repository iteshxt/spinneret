// Prints docs/report/source/report.html to docs/report/Spinneret-Report.pdf with headless
// Chrome (CSS paged media: A4, page-margin boxes). Usage: node docs/report/build/print.mjs
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = join(here, "..", "source", "report.html");
const out = process.argv[2] ?? join(here, "..", "Spinneret-Report.pdf");
const chrome = spawn("google-chrome", ["--headless=new", "--disable-gpu", "--remote-debugging-port=9335",
  `--user-data-dir=${mkdtempSync(join(tmpdir(), "rep-"))}`, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let list;
for (let i = 0; i < 50 && !list; i++) {
  await sleep(200);
  list = await fetch("http://127.0.0.1:9335/json/list").then((r) => r.json()).catch(() => null);
}
const ws = new WebSocket(list.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const wait = new Map();
ws.addEventListener("message", (e) => {
  const m = JSON.parse(e.data);
  if (m.id && wait.has(m.id)) { wait.get(m.id)(m); wait.delete(m.id); }
});
const send = (method, params = {}) =>
  new Promise((r) => { wait.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
await send("Page.enable");
await send("Page.navigate", { url: `file://${src}` });
await sleep(2500);
const pdf = await send("Page.printToPDF", { printBackground: true, preferCSSPageSize: true });
if (!pdf.result) { console.error(pdf); process.exit(1); }
writeFileSync(out, Buffer.from(pdf.result.data, "base64"));
console.log("wrote", out);
ws.close();
chrome.kill();
