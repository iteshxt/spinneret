import { motion, useScroll, useTransform } from "framer-motion";
import { useMemo } from "react";

// A small deterministic random source, so the web looks the same on every visit.
function rng(seed: number) {
  return () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
}

interface Web {
  spokes: string[];
  rings: string[];
  dew: { x: number; y: number; delay: number }[];
}

/** An orb web: spokes from a hub, a spiral of sagging strands between them, a few dew drops. */
function buildWeb(seed: number, spokes: number, rings: number, size: number): Web {
  const rand = rng(seed);
  const angles = Array.from({ length: spokes }, (_, i) => ((i + rand() * 0.35) / spokes) * Math.PI * 2);
  const reach = angles.map(() => size * (0.82 + rand() * 0.18));
  const spokePaths: string[] = [];
  const ringPaths: string[] = [];
  const dew: Web["dew"] = [];

  angles.forEach((a, i) => {
    spokePaths.push(`M0,0 L${(Math.cos(a) * reach[i]).toFixed(1)},${(Math.sin(a) * reach[i]).toFixed(1)}`);
  });

  for (let r = 0; r < rings; r++) {
    const radius = size * 0.07 * Math.pow(1.17, r) + r * 6;
    let d = "";
    let prevA = 0;
    for (let i = 0; i <= spokes; i++) {
      const k = i % spokes;
      const a = angles[k] + (i === spokes ? Math.PI * 2 : 0);
      const rr = Math.min(radius * (0.96 + rand() * 0.08), reach[k] * 0.98);
      const x = Math.cos(a) * rr;
      const y = Math.sin(a) * rr;
      if (i === 0) {
        d = `M${x.toFixed(1)},${y.toFixed(1)}`;
      } else {
        // the strand sags towards the hub between two spokes
        const mid = (prevA + a) / 2;
        const sag = rr * 0.9;
        d += ` Q${(Math.cos(mid) * sag).toFixed(1)},${(Math.sin(mid) * sag).toFixed(1)} ${x.toFixed(1)},${y.toFixed(1)}`;
        if (rand() < 0.07) dew.push({ x, y, delay: rand() * 7 });
      }
      prevA = a;
    }
    ringPaths.push(d);
  }
  return { spokes: spokePaths, rings: ringPaths, dew };
}

const SPOKE_STEP = 0.06; // seconds between spokes
const RING_STEP = 0.32; // seconds between spiral turns

/**
 * The web spins itself in: spokes shoot out from the hub, then the spiral is laid turn by
 * turn from the inside out. Afterwards a glint of light keeps running along a few strands.
 */
function WebSvg({ web, size, start }: { web: Web; size: number; start: number }) {
  const ringsAt = start + web.spokes.length * SPOKE_STEP * 0.6 + 0.5;
  const done = ringsAt + web.rings.length * RING_STEP;
  const glints = [3, 7, 10].filter((i) => i < web.rings.length);
  return (
    <g className="web">
      {web.spokes.map((d, i) => (
        <path
          key={`s${i}`}
          className="strand"
          d={d}
          pathLength={1}
          style={{ "--d": `${(start + i * SPOKE_STEP * 0.6).toFixed(2)}s`, "--len": "0.7s" } as React.CSSProperties}
        />
      ))}
      {web.rings.map((d, i) => (
        <path
          key={`r${i}`}
          className="strand"
          d={d}
          pathLength={1}
          style={{ "--d": `${(ringsAt + i * RING_STEP).toFixed(2)}s`, "--len": "0.9s" } as React.CSSProperties}
        />
      ))}
      {glints.map((i, k) => (
        <path
          key={`g${i}`}
          className="glint"
          d={web.rings[i]}
          pathLength={1}
          style={{ animationDelay: `${(done + k * 4).toFixed(1)}s` }}
        />
      ))}
      {web.dew.map((p, i) => (
        <circle
          key={i}
          className="dew"
          cx={p.x}
          cy={p.y}
          r={size / 260}
          style={{ animationDelay: `${(done + p.delay).toFixed(2)}s` }}
        />
      ))}
    </g>
  );
}

/** Two faint webs, top right and bottom left, that sway a little and drift with the scroll. */
export function Backdrop() {
  const top = useMemo(() => buildWeb(7, 17, 14, 620), []);
  const bottom = useMemo(() => buildWeb(23, 15, 13, 560), []);
  const { scrollY } = useScroll();
  const slow = useTransform(scrollY, (y) => y * -0.1);
  const slower = useTransform(scrollY, (y) => y * -0.03);

  return (
    <div className="backdrop" aria-hidden>
      <motion.svg
        style={{ y: slow, right: -560, top: -470, width: 1240, height: 1240 }}
        viewBox="-620 -620 1240 1240"
      >
        <g className="sway" style={{ transformOrigin: "0 0" }}>
          <WebSvg web={top} size={620} start={0.2} />
        </g>
      </motion.svg>
      <motion.svg
        style={{ y: slower, left: -500, bottom: -520, width: 1120, height: 1120 }}
        viewBox="-560 -560 1120 1120"
      >
        <g className="sway" style={{ transformOrigin: "0 0", animationDelay: "-6s" }}>
          <WebSvg web={bottom} size={560} start={1.2} />
        </g>
      </motion.svg>
    </div>
  );
}
