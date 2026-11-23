// Landing-page content. Every question, query and result below is real output from the
// Spinneret model (T5-base + ranker) on Spider's dev databases, recorded 2026-11-23.

export const REPO = "https://github.com/iteshxt/spinneret";
export const LICENSE_URL = `${REPO}/blob/main/LICENSE`;

/**
 * A worked example for the hero. In `question` and `sql`, `[text|key]` marks a span; spans
 * that share a key are joined by a thread (question word → the SQL it became).
 */
export interface HeroExample {
  db: string;
  question: string;
  sql: string;
  columns: string[];
  rows: (string | number)[][];
  ms: number;
}

export const HERO_EXAMPLES: HeroExample[] = [
  {
    db: "concert_singer",
    question: "What is the [average|a] [age|g] of [singers|s] from [France|f]?",
    sql: "select [avg|a]([age|g]) from [singer|s] where [country|f] = ['France'|f]",
    columns: ["avg(age)"],
    rows: [[34.5]],
    ms: 1115,
  },
  {
    db: "battle_death",
    question: "[How many|h] [ships|s] ended up being [captured|c]?",
    sql: "select [count(*)|h] from [ship|s] where [disposition_of_ship|c] = ['Captured'|c]",
    columns: ["count(*)"],
    rows: [[4]],
    ms: 815,
  },
  {
    db: "world_1",
    question: "What is the [population|p] of [Japan|j]?",
    sql: "select [population|p] from country where [name|j] = ['Japan'|j]",
    columns: ["Population"],
    rows: [["126,714,000"]],
    ms: 679,
  },
];

/** Questions the model answers correctly; each opens the demo and asks it. */
export const TRY_QUESTIONS: { db: string; question: string }[] = [
  { db: "orchestra", question: "What are the names of conductors, ordered by age?" },
  { db: "network_1", question: "How many high schoolers are in grade 9 or 10?" },
  { db: "dog_kennels", question: "What is the name of the breed with the most dogs?" },
  { db: "employee_hire_evaluation", question: "Which city do the most employees live in?" },
  { db: "world_1", question: "What is the total population of all countries in Asia?" },
  { db: "tvshow", question: "List the title of all cartoons ordered by their air dates." },
  { db: "singer", question: "What are the names of singers whose citizenship is not France?" },
  {
    db: "museum_visit",
    question: "What is the average age of visitors with membership level higher than 4?",
  },
];

export interface Span {
  text: string;
  key?: string;
}

/** "a [b|k] c" → [{text:"a "}, {text:"b", key:"k"}, {text:" c"}] */
export function spans(src: string): Span[] {
  const out: Span[] = [];
  const re = /\[([^\]|]+)\|(\w+)\]/g;
  let last = 0;
  for (const m of src.matchAll(re)) {
    if (m.index! > last) out.push({ text: src.slice(last, m.index) });
    out.push({ text: m[1], key: m[2] });
    last = m.index! + m[0].length;
  }
  if (last < src.length) out.push({ text: src.slice(last) });
  return out;
}

export const plain = (src: string) => spans(src).map((s) => s.text).join("");
