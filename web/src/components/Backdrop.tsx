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
  paths: string[];
  dew: { x: number; y: number; delay: number }[];
}

/** An orb web: spokes from a hub, a spiral of sagging strands between them, a few dew drops. */
function buildWeb(seed: number, spokes: number, rings: number, size: number): Web {
  const rand = rng(seed);
  const angles = Array.from({ length: spokes }, (_, i) => ((i + rand() * 0.35) / spokes) * Math.PI * 2);
  const reach = angles.map(() => size * (0.82 + rand() * 0.18));
  const paths: string[] = [];
  const dew: Web["dew"] = [];

  angles.forEach((a, i) => {
    paths.push(`M0,0 L${(Math.cos(a) * reach[i]).toFixed(1)},${(Math.sin(a) * reach[i]).toFixed(1)}`);
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
    paths.push(d);
  }
  return { paths, dew };
}

function WebSvg({ web, size }: { web: Web; size: number }) {
  return (
    <g className="web">
      {web.paths.map((d, i) => (
        <path key={i} d={d} />
      ))}
      {web.dew.map((p, i) => (
        <circle
          key={i}
          className="dew"
          cx={p.x}
          cy={p.y}
          r={size / 260}
          style={{ animationDelay: `${p.delay.toFixed(2)}s` }}
        />
      ))}
    </g>
  );
}

/** Two faint webs behind the page that sway a little and drift slower than the content. */
export function Backdrop() {
  const big = useMemo(() => buildWeb(7, 17, 14, 620), []);
  const small = useMemo(() => buildWeb(23, 13, 11, 380), []);
  const { scrollY } = useScroll();
  const slow = useTransform(scrollY, (y) => y * -0.12);
  const slower = useTransform(scrollY, (y) => y * -0.05);

  return (
    <div className="backdrop" aria-hidden>
      <motion.svg
        style={{ y: slow, right: -560, top: -470, width: 1240, height: 1240 }}
        viewBox="-620 -620 1240 1240"
      >
        <g className="sway" style={{ transformOrigin: "0 0" }}>
          <WebSvg web={big} size={620} />
        </g>
      </motion.svg>
      <motion.svg
        style={{ y: slower, left: -430, top: "58vh", width: 760, height: 760 }}
        viewBox="-380 -380 760 760"
      >
        <g className="sway" style={{ transformOrigin: "0 0", animationDelay: "-6s" }}>
          <WebSvg web={small} size={380} />
        </g>
      </motion.svg>
    </div>
  );
}
