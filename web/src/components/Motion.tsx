import { motion, type Variants } from "framer-motion";
import type { ReactNode } from "react";

const EASE = [0.2, 0.8, 0.2, 1] as const;

/** Fades and lifts its content in the first time it scrolls into view. */
export function Reveal({
  children,
  delay = 0,
  className,
  as = "div",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
  as?: "div" | "section";
}) {
  const M = as === "section" ? motion.section : motion.div;
  return (
    <M
      className={className}
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.7, ease: EASE, delay }}
    >
      {children}
    </M>
  );
}

/** Parent/child variants for lists whose items arrive one after another. */
export const stagger: Variants = {
  hidden: {},
  shown: { transition: { staggerChildren: 0.08 } },
};

export const item: Variants = {
  hidden: { opacity: 0, y: 18 },
  shown: { opacity: 1, y: 0, transition: { duration: 0.55, ease: EASE } },
};

export const inView = {
  initial: "hidden",
  whileInView: "shown",
  viewport: { once: true, amount: 0.2 },
} as const;

/** The heading block every inner page starts with. */
export function PageHead({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <motion.header
      className="page-head"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, ease: EASE }}
    >
      <h1>{title}</h1>
      {children && <div className="page-lead">{children}</div>}
    </motion.header>
  );
}
