// Render each figure page (build/html/*.html) to a 2x PNG in docs/report/figures/ via the
// Chrome DevTools Protocol. Usage: node docs/report/build/capture.mjs
import { spawn } from "node:child_process";
import { readdirSync, readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const html = join(here, "html");
const out = join(here, "..", "figures");
const chrome = spawn("google-chrome", ["--headless=new", "--disable-gpu", "--hide-scrollbars",
  "--remote-debugging-port=9334", `--user-data-dir=${mkdtempSync(join(tmpdir(), "fig-"))}`,
  "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let list;
for (let i = 0; i < 50 && !list; i++) {
  await sleep(200);
  list = await fetch("http://127.0.0.1:9334/json/list").then((r) => r.json()).catch(() => null);
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
for (const f of readdirSync(html).filter((f) => f.endsWith(".html"))) {
  const name = f.replace(".html", "");
  const [w, h] = readFileSync(join(html, `${name}.size`), "utf8").split(" ").map(Number);
  await send("Emulation.setDeviceMetricsOverride", { width: w, height: h, deviceScaleFactor: 2, mobile: false });
  await send("Page.navigate", { url: `file://${join(html, f)}` });
  await sleep(700);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(out, `${name}.png`), Buffer.from(shot.result.data, "base64"));
  console.log("saved", name);
}
ws.close();
chrome.kill();
