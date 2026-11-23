// Draws the cover's threads: each question word to the SQL it became, and the selected
// column down to the answer (same idea as the web app's hero).
(function () {
  const box = document.getElementById("cascade");
  const svg = box.querySelector("svg.threads");
  const r = box.getBoundingClientRect();
  const ns = "http://www.w3.org/2000/svg";
  const at = (el) => el.getBoundingClientRect();
  const path = (x1, y1, x2, y2, cls) => {
    const sag = (y2 - y1) * 0.6;
    const p = document.createElementNS(ns, "path");
    p.setAttribute("d", `M${x1},${y1} C${x1},${y1 + sag} ${x2},${y2 - sag} ${x2},${y2}`);
    if (cls) p.setAttribute("class", cls);
    svg.appendChild(p);
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", x1); c.setAttribute("cy", y1); c.setAttribute("r", 2.2);
    svg.appendChild(c);
  };
  for (const t of box.querySelectorAll("[data-s]")) {
    const b = at(t);
    for (const w of box.querySelectorAll(`[data-q="${t.dataset.s}"]`)) {
      const a = at(w);
      path(a.left + a.width / 2 - r.left, a.bottom - r.top + 2, b.left + b.width / 2 - r.left, b.top - r.top - 2);
    }
  }
  const ans = box.querySelector("[data-r]");
  const src = box.querySelector(`[data-s="${ans.dataset.r}"]`);
  const a = at(src), c = at(ans);
  path(a.left + a.width / 2 - r.left, a.bottom - r.top + 2, c.left + 12 - r.left, c.top - r.top + 4, "strong");
  document.body.dataset.ready = "1";
})();
