/**
 * LAB-426 analysis: turns measure.ts output into the Markdown tables of
 * docs/recall-floor.md (signal distributions and the floor trade-off grid).
 * Reads one file and prints to stdout; touches no database.
 *
 * Usage (from ts/): bun run tests/recall-floor/analyze.ts <signals.json>
 */
import { readFileSync } from "node:fs";

interface Row {
  id: string;
  rrf: number | null;
  ft_pos: number | null;
  sim: number | null;
  ts_rank: number;
  coverage: number;
  idf_coverage: number;
}
interface Case {
  set: "gold" | "model" | "negative";
  id: string;
  kind?: string;
  query: string;
  gold_ids: string[];
  fused: Row[];
  gold: Row[];
}
type Signal = "sim" | "idf_coverage" | "coverage" | "ts_rank";

const SIGNALS: Array<[Signal, string]> = [
  ["sim", "cosine similarity"],
  ["idf_coverage", "IDF-weighted coverage"],
  ["coverage", "unweighted coverage"],
  ["ts_rank", "ts_rank"],
];
const PCTS = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1];

function quantiles(xs: number[]): string {
  const s = [...xs].sort((a, b) => a - b);
  return PCTS.map((p) => s[Math.min(s.length - 1, Math.round(p * (s.length - 1)))]!.toFixed(3)).join(" | ");
}

function recallAt5(c: Case, keep: (r: Row) => boolean, fulltextOnly: boolean): number | null {
  if (c.gold_ids.length === 0) return null;
  let list = c.fused;
  if (fulltextOnly) list = list.filter((r) => r.ft_pos !== null).sort((a, b) => a.ft_pos! - b.ft_pos!);
  const top = list.filter(keep).slice(0, 5).map((r) => r.id);
  return c.gold_ids.filter((id) => top.includes(id)).length / c.gold_ids.length;
}

function evaluate(cases: Case[], keep: (r: Row) => boolean, fulltextOnly = false) {
  const mean = (set: string) => {
    const xs = cases.filter((c) => c.set === set).map((c) => recallAt5(c, keep, fulltextOnly)).filter((x): x is number => x !== null);
    return xs.reduce((s, x) => s + x, 0) / xs.length;
  };
  const survivors = (c: Case) => c.fused.filter((r) => keep(r) && (!fulltextOnly || r.ft_pos !== null)).length;
  const negatives = cases.filter((c) => c.set === "negative").map(survivors);
  const abstain = cases.filter((c) => c.set !== "negative" && c.gold_ids.length === 0).map(survivors);
  return {
    hook: mean("gold"),
    model: mean("model"),
    negZero: negatives.filter((n) => n === 0).length,
    negFull: negatives.filter((n) => n >= 5).length,
    negTotal: negatives.length,
    abstainZero: abstain.filter((n) => n === 0).length,
    abstainTotal: abstain.length,
  };
}

const path = process.argv[2];
if (!path) {
  console.error("usage: bun run tests/recall-floor/analyze.ts <signals.json>");
  process.exit(2);
}
const data = JSON.parse(readFileSync(path, "utf8")) as { memories: number; embedded: number; model: string; cases: Case[] };
const cases = data.cases;
const sets = { gold: cases.filter((c) => c.set === "gold"), model: cases.filter((c) => c.set === "model"), negative: cases.filter((c) => c.set === "negative") };

console.log(`Corpus: ${data.memories} memories, ${data.embedded} embedded with ${data.model}.`);
console.log(`Queries: ${sets.gold.length} gold hooks, ${sets.model.length} model queries, ${sets.negative.length} negatives.\n`);

console.log("### Distributions\n");
console.log(`| signal | population | n | ${PCTS.map((p) => (p === 0 ? "min" : p === 1 ? "max" : `p${Math.round(p * 100)}`)).join(" | ")} |`);
console.log(`|---|---|---|${PCTS.map(() => "---").join("|")}|`);
for (const [key, label] of SIGNALS) {
  const value = (r: Row) => (r[key] ?? 0) as number;
  const populations: Array<[string, number[]]> = [
    ["gold memories, hook queries", sets.gold.flatMap((c) => c.gold.map(value))],
    ["gold memories, model queries", sets.model.flatMap((c) => c.gold.map(value))],
    ["non-gold in the unfloored top 5, hook and model queries", [...sets.gold, ...sets.model].flatMap((c) => c.fused.slice(0, 5).filter((r) => !c.gold_ids.includes(r.id)).map(value))],
    ["unfloored top 5, negative queries", sets.negative.flatMap((c) => c.fused.slice(0, 5).map(value))],
    ["best candidate per negative query", sets.negative.map((c) => Math.max(...c.fused.map(value)))],
  ];
  for (const [name, xs] of populations) console.log(`| ${label} | ${name} | ${xs.length} | ${quantiles(xs)} |`);
}

const base = evaluate(cases, () => true);
const pp = (x: number, b: number) => `${(100 * (x - b) >= 0 ? "+" : "")}${(100 * (x - b)).toFixed(1)}`;
const line = (label: string, e: ReturnType<typeof evaluate>, b: ReturnType<typeof evaluate>) => {
  const ok = e.negZero / e.negTotal >= 0.9 && e.negFull === 0 && e.hook >= b.hook - 0.02 && e.model >= b.model - 0.02;
  return `| ${label} | ${e.hook.toFixed(4)} (${pp(e.hook, b.hook)}) | ${e.model.toFixed(4)} (${pp(e.model, b.model)}) | ${e.negZero}/${e.negTotal} (${((100 * e.negZero) / e.negTotal).toFixed(1)}%) | ${e.negFull} | ${e.abstainZero}/${e.abstainTotal} | ${ok ? "yes" : "no"} |`;
};
const header = (first: string) => {
  console.log(`| ${first} | recall@5 hooks (pp vs no floor) | recall@5 model queries (pp vs no floor) | negatives returning 0 | negatives returning a full page | abstain queries returning 0 | meets both targets |`);
  console.log("|---|---|---|---|---|---|---|");
};

console.log("\n### Trade-off curve: similarity floor OR IDF-weighted coverage floor\n");
header("similarity / coverage floor");
console.log(line("0 / 0 (no floor)", base, base));
for (let s = 50; s <= 66; s += 2) {
  for (const f of [0.4, 0.5, 0.6, 0.7]) {
    console.log(line(`${(s / 100).toFixed(2)} / ${f.toFixed(2)}`, evaluate(cases, (r) => (r.sim ?? 0) >= s / 100 || r.idf_coverage >= f), base));
  }
}

console.log("\n### Similarity floor near the chosen value, coverage floor 0.50\n");
header("similarity floor");
for (let s = 55; s <= 63; s++) {
  console.log(line((s / 100).toFixed(2), evaluate(cases, (r) => (r.sim ?? 0) >= s / 100 || r.idf_coverage >= 0.5), base));
}

console.log("\n### Full-text criterion compared at similarity floor 0.58\n");
header("full-text criterion");
const criteria: Array<[string, (r: Row) => boolean]> = [
  ["none (similarity only)", () => false],
  ["unweighted coverage >= 0.50", (r) => r.coverage >= 0.5],
  ["unweighted coverage >= 0.67", (r) => r.coverage >= 0.67],
  ["unweighted coverage >= 0.75", (r) => r.coverage >= 0.75],
  ["ts_rank >= 0.04", (r) => r.ts_rank >= 0.04],
  ["ts_rank >= 0.05", (r) => r.ts_rank >= 0.05],
  ["IDF-weighted coverage >= 0.40", (r) => r.idf_coverage >= 0.4],
  ["IDF-weighted coverage >= 0.50", (r) => r.idf_coverage >= 0.5],
  ["IDF-weighted coverage >= 0.60", (r) => r.idf_coverage >= 0.6],
];
for (const [label, ft] of criteria) console.log(line(label, evaluate(cases, (r) => (r.sim ?? 0) >= 0.58 || ft(r)), base));

console.log("\n### Full-text-only fallback: IDF-weighted coverage floor alone\n");
const ftBase = evaluate(cases, () => true, true);
header("coverage floor");
console.log(line("0 (no floor)", ftBase, ftBase));
for (const f of [0.3, 0.4, 0.5, 0.6]) console.log(line(f.toFixed(2), evaluate(cases, (r) => r.idf_coverage >= f, true), ftBase));

console.log("\n### Negative queries\n");
console.log("| id | kind | best cosine | best IDF-weighted coverage | query |");
console.log("|---|---|---|---|---|");
for (const c of sets.negative) {
  console.log(`| ${c.id} | ${c.kind} | ${Math.max(...c.fused.map((r) => r.sim ?? 0)).toFixed(3)} | ${Math.max(...c.fused.map((r) => r.idf_coverage)).toFixed(3)} | ${c.query} |`);
}
