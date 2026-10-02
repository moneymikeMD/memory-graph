/**
 * LAB-426 end-to-end check: runs the backend's own recallMemories over the
 * negative queries and the gold set, with the default floors and with both
 * floors at 0, hybrid and full-text only.
 *
 * recallMemories has no cutoff argument, so the gold items are replayed in
 * descending cutoff order against a throwaway copy of the corpus database,
 * deleting the memories newer than each cutoff as it goes. The copy is
 * created from the source database as a template and dropped at the end.
 *
 * Usage (from ts/):
 *   RECALL_FLOOR_POSTGRES_URL=postgres://memorygraph@127.0.0.1:55426/memorygraph \
 *   MEMORY_POSTGRES_PASSWORD=... MEMORY_EMBED_URL=http://127.0.0.1:11434 \
 *   bun run tests/recall-floor/evaluate.ts <homelab docs/bench/memory dir> <negative-queries.jsonl> [extra-set.jsonl ...]
 *
 * Extra sets (ordinary-prompts.jsonl, keyword-queries.jsonl) run against the
 * whole corpus like the negatives and are counted per set.
 *
 * Refuses any Postgres host that is not loopback.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import postgres from "postgres";
import { PostgresBackend } from "../../src/backends/postgres.ts";
import { OllamaEmbedder } from "../../src/backends/postgres-embedder.ts";
import { assertLoopback } from "./measure.ts";

const LIMIT = 5;

function jsonl(path: string): Record<string, unknown>[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

interface Arm {
  name: string;
  backend: PostgresBackend;
  negativeCounts: number[];
  extraCounts: Map<string, number[]>;
  recalls: { hook: number[]; model: number[] };
  abstainZero: number;
  abstainTotal: number;
}

async function main(): Promise<void> {
  const [benchDir, negativesPath, ...extraPaths] = process.argv.slice(2);
  const sourceUrl = process.env.RECALL_FLOOR_POSTGRES_URL;
  if (!benchDir || !negativesPath || !sourceUrl) {
    console.error("usage: RECALL_FLOOR_POSTGRES_URL=... bun run tests/recall-floor/evaluate.ts <bench-dir> <negatives.jsonl> [extra-set.jsonl ...]");
    process.exit(2);
  }
  assertLoopback(sourceUrl);
  const password = process.env.MEMORY_POSTGRES_PASSWORD;
  const source = new URL(sourceUrl);
  const copyName = `recall_floor_eval_${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  const copy = new URL(sourceUrl);
  copy.pathname = `/${copyName}`;
  assertLoopback(copy.toString());

  const admin = postgres(sourceUrl, { password, onnotice: () => {}, max: 1 });
  await admin.unsafe(`CREATE DATABASE ${copyName} TEMPLATE ${source.pathname.slice(1)}`);
  const sql = postgres(copy.toString(), { password, onnotice: () => {}, max: 1 });

  const embedder = () => new OllamaEmbedder({ url: process.env.MEMORY_EMBED_URL, timeoutMs: 30_000 });
  const noEmbedder = () => new OllamaEmbedder({ url: undefined });
  const arm = (name: string, emb: OllamaEmbedder, floors: { recallSimilarityFloor?: number; recallFulltextFloor?: number }): Arm => ({
    name,
    backend: new PostgresBackend({ url: copy.toString(), password, embedder: emb, ...floors }),
    negativeCounts: [],
    extraCounts: new Map(),
    recalls: { hook: [], model: [] },
    abstainZero: 0,
    abstainTotal: 0,
  });
  const zero = { recallSimilarityFloor: 0, recallFulltextFloor: 0 };
  const arms = [
    arm("hybrid, default floors", embedder(), {}),
    arm("hybrid, floors 0", embedder(), zero),
    arm("full-text only, default floors", noEmbedder(), {}),
    arm("full-text only, floors 0", noEmbedder(), zero),
  ];

  try {
    for (const a of arms) await a.backend.connect();

    for (const n of jsonl(negativesPath)) {
      for (const a of arms) {
        a.negativeCounts.push((await a.backend.recallMemories(n["query"] as string, { limit: LIMIT })).length);
      }
    }

    for (const e of extraPaths.flatMap(jsonl)) {
      for (const a of arms) {
        const counts = a.extraCounts.get(e["set"] as string) ?? [];
        counts.push((await a.backend.recallMemories(e["query"] as string, { limit: LIMIT })).length);
        a.extraCounts.set(e["set"] as string, counts);
      }
    }

    const gold = jsonl(join(benchDir, "goldset.jsonl"));
    const models = jsonl(join(benchDir, "queries_model.jsonl"));
    gold.sort((x, y) => Date.parse(y["cutoff"] as string) - Date.parse(x["cutoff"] as string));
    for (const g of gold) {
      await sql`DELETE FROM memories WHERE created_at > ${g["cutoff"] as string}`;
      await sql.unsafe("VACUUM memories");
      const goldIds = g["gold_ids"] as string[];
      const queries: Array<["hook" | "model", string]> = [["hook", g["query_hook"] as string]];
      for (const m of models) if (m["item_id"] === g["id"]) queries.push(["model", m["query"] as string]);
      for (const [kind, query] of queries) {
        for (const a of arms) {
          const ids = (await a.backend.recallMemories(query, { limit: LIMIT })).map((m) => m.id!);
          if (goldIds.length === 0) {
            a.abstainTotal++;
            if (ids.length === 0) a.abstainZero++;
          } else {
            a.recalls[kind].push(goldIds.filter((id) => ids.includes(id)).length / goldIds.length);
          }
        }
      }
    }
  } finally {
    for (const a of arms) await a.backend.disconnect().catch(() => {});
    await sql.end({ timeout: 5 });
    await admin.unsafe(`DROP DATABASE IF EXISTS ${copyName} WITH (FORCE)`);
    await admin.end({ timeout: 5 });
  }

  const avg = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
  const summary = arms.map((a) => ({
    arm: a.name,
    negatives: a.negativeCounts.length,
    negatives_zero: a.negativeCounts.filter((n) => n === 0).length,
    negatives_zero_rate: a.negativeCounts.filter((n) => n === 0).length / a.negativeCounts.length,
    negatives_full_page: a.negativeCounts.filter((n) => n >= LIMIT).length,
    recall_at_5_hook: avg(a.recalls.hook),
    recall_at_5_model: avg(a.recalls.model),
    recall_at_5_all: avg([...a.recalls.hook, ...a.recalls.model]),
    scored_hook_queries: a.recalls.hook.length,
    scored_model_queries: a.recalls.model.length,
    abstain_zero: a.abstainZero,
    abstain_queries: a.abstainTotal,
  }));
  console.log("| arm | negatives returning 0 | negatives returning a full page | recall@5 hooks | recall@5 model queries | recall@5 all | abstain queries returning 0 |");
  console.log("|---|---|---|---|---|---|---|");
  for (const s of summary) {
    console.log(
      `| ${s.arm} | ${s.negatives_zero}/${s.negatives} (${(100 * s.negatives_zero_rate).toFixed(1)}%) | ${s.negatives_full_page}/${s.negatives} | ` +
        `${s.recall_at_5_hook.toFixed(4)} | ${s.recall_at_5_model.toFixed(4)} | ${s.recall_at_5_all.toFixed(4)} | ${s.abstain_zero}/${s.abstain_queries} |`
    );
  }
  for (const set of arms[0]!.extraCounts.keys()) {
    console.log(`
| arm | ${set} queries returning 0 | ${set} queries returning a full page |`);
    console.log("|---|---|---|");
    for (const a of arms) {
      const counts = a.extraCounts.get(set)!;
      const zero = counts.filter((n) => n === 0).length;
      console.log(`| ${a.name} | ${zero}/${counts.length} (${((100 * zero) / counts.length).toFixed(1)}%) | ${counts.filter((n) => n >= LIMIT).length}/${counts.length} |`);
    }
  }
}

if (import.meta.main) await main();
