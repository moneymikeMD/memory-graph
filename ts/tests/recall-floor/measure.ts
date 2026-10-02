/**
 * LAB-426 measurement: raw recall signals on the frozen bench corpus.
 *
 * For every gold-set hook, model query, negative query and extra-set query it replays the
 * backend's hybrid ranking (same SQL, plus the gold item's created_at cutoff)
 * and records cosine similarity and the full-text signals of every fused
 * candidate and of every gold memory. Output is one JSON file that
 * analyze.ts turns into the tables in docs/recall-floor.md.
 *
 * Usage (from ts/):
 *   RECALL_FLOOR_POSTGRES_URL=postgres://memorygraph@127.0.0.1:55426/memorygraph \
 *   MEMORY_POSTGRES_PASSWORD=... MEMORY_EMBED_URL=http://127.0.0.1:11434 \
 *   bun run tests/recall-floor/measure.ts <homelab docs/bench/memory dir> <negative-queries.jsonl> <out.json> [extra-set.jsonl ...]
 *
 * Each extra file is a query set with no gold memories (ordinary-prompts.jsonl,
 * keyword-queries.jsonl); its rows name their own set.
 *
 * Refuses any Postgres host that is not loopback.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import postgres from "postgres";
import { buildTsquery, rrf } from "../../src/backends/postgres.ts";
import { OllamaEmbedder, queryEmbedText, vectorLiteral } from "../../src/backends/postgres-embedder.ts";

const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);
const LIMIT = 5;
const OVER_FETCH = Math.max(LIMIT * 4, 50);
const FAR_FUTURE = "9999-01-01T00:00:00Z";

export function assertLoopback(url: string): void {
  const host = new URL(url).hostname;
  if (!LOOPBACK.has(host)) {
    throw new Error(`refusing to run: Postgres host ${host} is not loopback`);
  }
}

interface QueryCase {
  set: "gold" | "model" | "negative" | "ordinary" | "keyword";
  id: string;
  query: string;
  cutoff: string;
  gold_ids: string[];
  kind?: string;
}

function jsonl(path: string): Record<string, unknown>[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l));
}

function loadCases(benchDir: string, negativesPath: string, extraPaths: string[]): QueryCase[] {
  const gold = jsonl(join(benchDir, "goldset.jsonl"));
  const byId = new Map(gold.map((g) => [g["id"] as string, g]));
  const cases: QueryCase[] = gold.map((g) => ({
    set: "gold",
    id: g["id"] as string,
    query: g["query_hook"] as string,
    cutoff: g["cutoff"] as string,
    gold_ids: g["gold_ids"] as string[],
    kind: g["category"] as string,
  }));
  for (const m of jsonl(join(benchDir, "queries_model.jsonl"))) {
    const g = byId.get(m["item_id"] as string);
    if (!g) throw new Error(`model query for unknown gold item ${m["item_id"]}`);
    cases.push({
      set: "model",
      id: `${m["item_id"]}:${m["model"]}`,
      query: m["query"] as string,
      cutoff: g["cutoff"] as string,
      gold_ids: g["gold_ids"] as string[],
      kind: g["category"] as string,
    });
  }
  for (const n of jsonl(negativesPath)) {
    cases.push({
      set: "negative",
      id: n["id"] as string,
      query: n["query"] as string,
      cutoff: FAR_FUTURE,
      gold_ids: [],
      kind: n["kind"] as string,
    });
  }
  for (const o of extraPaths.flatMap(jsonl)) {
    cases.push({
      set: o["set"] as QueryCase["set"],
      id: o["id"] as string,
      query: o["query"] as string,
      cutoff: FAR_FUTURE,
      gold_ids: [],
      kind: o["kind"] as string,
    });
  }
  return cases;
}

async function main(): Promise<void> {
  const [benchDir, negativesPath, outPath, ...extraPaths] = process.argv.slice(2);
  const url = process.env.RECALL_FLOOR_POSTGRES_URL;
  if (!benchDir || !negativesPath || !outPath || !url) {
    console.error("usage: RECALL_FLOOR_POSTGRES_URL=... bun run tests/recall-floor/measure.ts <bench-dir> <negatives.jsonl> <out.json> [extra-set.jsonl ...]");
    process.exit(2);
  }
  assertLoopback(url);
  const sql = postgres(url, { password: process.env.MEMORY_POSTGRES_PASSWORD, onnotice: () => {}, max: 1 });
  const embedder = new OllamaEmbedder({ url: process.env.MEMORY_EMBED_URL, timeoutMs: 30_000 });
  const [col] = await sql`
    SELECT format_type(atttypid, atttypmod) AS t FROM pg_attribute
    WHERE attrelid = 'memories'::regclass AND attname = 'embedding'`;
  const columnType = String(col!["t"]).replace(/\(.*/, "");
  const [count] = await sql`SELECT count(*)::int AS n, count(embedding)::int AS e FROM memories`;

  const out: unknown[] = [];
  for (const c of loadCases(benchDir, negativesPath, extraPaths)) {
    const [vector] = await embedder.embed([queryEmbedText(c.query)]);
    const vec = vectorLiteral(vector!);
    const tsq = buildTsquery(c.query, " | ");
    const lexemeText = tsq ? tsq.split(" | ").join(" ") : "";

    const fulltext = tsq
      ? await sql.unsafe(
          `SELECT id, ts_rank(search_vector, to_tsquery('english', $1)) AS score
           FROM memories WHERE search_vector @@ to_tsquery('english', $1) AND created_at <= $2
           ORDER BY score DESC, id LIMIT $3`,
          [tsq, c.cutoff, OVER_FETCH] as never[]
        )
      : [];
    // Exact scan: the cutoff filter would starve an HNSW scan of candidates.
    const vectorRows = await sql.begin(async (tx) => {
      await tx.unsafe("SET LOCAL enable_indexscan = off");
      return tx.unsafe(
        `SELECT id, 1 - (embedding <=> $1::${columnType}) AS score
         FROM memories WHERE embedding IS NOT NULL AND created_at <= $2
         ORDER BY embedding <=> $1::${columnType}, id LIMIT $3`,
        [vec, c.cutoff, OVER_FETCH] as never[]
      );
    });
    const ftRank = fulltext.map((r) => ({ id: r["id"] as string, score: Number(r["score"]) }));
    const vecRank = (vectorRows as Record<string, unknown>[]).map((r) => ({ id: r["id"] as string, score: Number(r["score"]) }));
    const fused = rrf([ftRank, vecRank]);
    const ids = [...new Set([...fused.map((r) => r.id), ...c.gold_ids])];

    const signals = await sql.unsafe(
      `WITH q AS (
         SELECT tsvector_to_array(to_tsvector('english', $2)) AS lex,
                CASE WHEN $3::text IS NULL THEN NULL ELSE to_tsquery('english', $3) END AS tsq
       ), df AS (
         SELECT l AS lexeme, ln(1 + (n.n - d.df + 0.5) / (d.df + 0.5)) AS idf
         FROM q, unnest(q.lex) AS l,
              LATERAL (SELECT count(*)::float AS n FROM memories) n,
              LATERAL (SELECT count(*)::float AS df FROM memories m WHERE m.search_vector @@ quote_literal(l)::tsquery) d
       )
       SELECT m.id,
              m.created_at <= $5::timestamptz AS before_cutoff,
              1 - (m.embedding <=> $1::${columnType}) AS sim,
              CASE WHEN q.tsq IS NULL THEN 0 ELSE ts_rank(m.search_vector, q.tsq) END AS ts_rank,
              CASE WHEN q.tsq IS NULL THEN 0 ELSE ts_rank_cd(m.search_vector, q.tsq) END AS ts_rank_cd,
              cardinality(q.lex) AS query_lexemes,
              (SELECT count(*)::int FROM df WHERE m.search_vector @@ quote_literal(df.lexeme)::tsquery) AS matched,
              coalesce((SELECT sum(df.idf) FROM df WHERE m.search_vector @@ quote_literal(df.lexeme)::tsquery), 0) AS idf_matched,
              coalesce((SELECT sum(df.idf) FROM df), 0) AS idf_total
       FROM memories m, q WHERE m.id = ANY($4::text[])`,
      [vec, lexemeText, tsq, ids, c.cutoff] as never[]
    );
    const byId = new Map(signals.map((r) => [r["id"] as string, r]));
    const ftPos = new Map(ftRank.map((r, i) => [r.id, i + 1]));
    const vecPos = new Map(vecRank.map((r, i) => [r.id, i + 1]));
    const row = (id: string, rrfScore: number | null) => {
      const s = byId.get(id);
      const lex = Number(s?.["query_lexemes"] ?? 0);
      const matched = Number(s?.["matched"] ?? 0);
      return {
        id,
        rrf: rrfScore,
        ft_pos: ftPos.get(id) ?? null,
        vec_pos: vecPos.get(id) ?? null,
        sim: s ? Number(s["sim"]) : null,
        ts_rank: Number(s?.["ts_rank"] ?? 0),
        ts_rank_cd: Number(s?.["ts_rank_cd"] ?? 0),
        query_lexemes: lex,
        matched,
        coverage: lex > 0 ? matched / lex : 0,
        idf_matched: Number(s?.["idf_matched"] ?? 0),
        idf_total: Number(s?.["idf_total"] ?? 0),
        idf_coverage: Number(s?.["idf_total"] ?? 0) > 0 ? Number(s?.["idf_matched"] ?? 0) / Number(s!["idf_total"]) : 0,
        before_cutoff: s ? Boolean(s["before_cutoff"]) : false,
      };
    };
    out.push({
      ...c,
      fused: fused.map((r) => row(r.id, r.score)),
      gold: c.gold_ids.map((id) => row(id, fused.find((f) => f.id === id)?.score ?? null)),
    });
  }
  writeFileSync(
    outPath,
    JSON.stringify({ limit: LIMIT, over_fetch: OVER_FETCH, memories: count!["n"], embedded: count!["e"], model: embedder.model, cases: out }, null, 1)
  );
  console.error(`wrote ${out.length} cases to ${outPath}`);
  await sql.end({ timeout: 5 });
}

if (import.meta.main) await main();
