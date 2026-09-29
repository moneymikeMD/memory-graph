/**
 * Postgres backend integration suite. Runs against a loopback pgvector
 * container named by MEMORYGRAPH_TEST_POSTGRES_URL (password from
 * MEMORY_POSTGRES_PASSWORD) and skips when that is unset. Each run creates and
 * drops its own databases; a non-loopback URL fails the suite. Embeddings come
 * from a case-sensitive stub Ollama server on 127.0.0.1.
 * MEMORYGRAPH_REQUIRE_POSTGRES_TESTS=1 (`bun run test:postgres`) turns the skip into a failure.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from "bun:test";
import { createServer, type Server } from "node:http";
import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import postgres from "postgres";

import { PostgresBackend, buildTsquery, formatEmbeddingPlan, rrf } from "../src/backends/postgres.ts";
import { OllamaEmbedder, DEFAULT_EMBED_DIMENSION } from "../src/backends/postgres-embedder.ts";
import { SQLiteBackend } from "../src/backends/sqlite.ts";
import { MemoryDatabase } from "../src/database.ts";
import { createMemory, type Memory, type SearchQuery } from "../src/models.ts";
import { exportToJson, importFromJson } from "../src/utils/export-import.ts";
import { MigrationManager, createMigrationOptions } from "../src/migration/index.ts";

const TEST_URL = process.env.MEMORYGRAPH_TEST_POSTGRES_URL;
const REQUIRE_SUITE = process.env.MEMORYGRAPH_REQUIRE_POSTGRES_TESTS === "1";
const PASSWORD = process.env.MEMORY_POSTGRES_PASSWORD;
const CLI = join(import.meta.dir, "..", "src", "cli.ts");
const STUB_DIGEST = "stub0000digest";
const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);
const SMALL_MODEL = "stub-embed:64";
const SMALL_DIMENSION = 64;

function stubVector(text: string, dimension = DEFAULT_EMBED_DIMENSION): number[] {
  const vec = new Array<number>(dimension).fill(0);
  for (const token of text.split(/[^A-Za-z0-9]+/).filter((t) => t.length > 1)) {
    const h = createHash("sha1").update(token).digest();
    vec[h.readUInt16BE(0) % dimension] += 1;
  }
  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
  return vec.map((x) => x / norm);
}

function startStubEmbedder(): Promise<{ server: Server; url: string; calls: () => number; inputs: string[] }> {
  let calls = 0;
  const inputs: string[] = [];
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/api/tags") {
        res.end(
          JSON.stringify({
            models: [
              { name: "qwen3-embedding:0.6b", digest: STUB_DIGEST },
              { name: SMALL_MODEL, digest: STUB_DIGEST },
            ],
          })
        );
      } else if (req.url === "/api/embed") {
        calls++;
        const { input, model } = JSON.parse(body) as { input: string[]; model: string };
        inputs.push(...input);
        const dimension = model === SMALL_MODEL ? SMALL_DIMENSION : DEFAULT_EMBED_DIMENSION;
        res.end(JSON.stringify({ embeddings: input.map((t) => stubVector(t, dimension)) }));
      } else {
        res.statusCode = 404;
        res.end("{}");
      }
    });
  });
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({ server, url: `http://127.0.0.1:${addr.port}`, calls: () => calls, inputs });
    })
  );
}

function dbUrl(name: string): string {
  const u = new URL(TEST_URL!);
  u.pathname = `/${name}`;
  return u.toString();
}

function mem(title: string, content: string, extra: Record<string, unknown> = {}): Memory {
  return createMemory({ type: "solution", title, content, ...extra } as Parameters<typeof createMemory>[0]);
}

function query(overrides: Partial<SearchQuery> = {}): SearchQuery {
  return {
    query: undefined,
    terms: [],
    memory_types: [],
    tags: [],
    project_path: undefined,
    languages: [],
    frameworks: [],
    min_importance: undefined,
    min_confidence: undefined,
    min_effectiveness: undefined,
    created_after: undefined,
    created_before: undefined,
    limit: 50,
    offset: 0,
    include_relationships: false,
    search_tolerance: "normal",
    match_mode: "any",
    relationship_filter: undefined,
    ...overrides,
  } as SearchQuery;
}

function captureStderr<T>(fn: () => Promise<T>): Promise<{ result: T; stderr: string }> {
  const original = console.error;
  const lines: string[] = [];
  console.error = (...args: unknown[]) => lines.push(args.map(String).join(" "));
  return fn()
    .then((result) => ({ result, stderr: lines.join("\n") }))
    .finally(() => {
      console.error = original;
    });
}

function runCli(args: string[], env: Record<string, string>): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const child = spawn("bun", ["run", CLI, ...args], {
      env: {
        PATH: process.env.PATH ?? "",
        HOME: process.env.HOME ?? "",
        MEMORY_LOG_LEVEL: "ERROR",
        MEMORY_FALKORDB_HOST: "127.0.0.1",
        MEMORY_FALKORDB_PORT: "1",
        ...env,
      },
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d));
    child.stderr.on("data", (d) => (stderr += d));
    child.on("close", (code) => resolve({ code: code ?? -1, stdout, stderr }));
  });
}

if (!TEST_URL) {
  console.warn(
    "\n*** SKIPPING the Postgres integration suite: MEMORYGRAPH_TEST_POSTGRES_URL is unset. " +
      "Set it to a loopback pgvector container, or run `bun run test:postgres` to make this a failure. ***\n"
  );
}

describe("postgres suite gate", () => {
  test.if(REQUIRE_SUITE)("MEMORYGRAPH_TEST_POSTGRES_URL is set when the suite is required", () => {
    expect(TEST_URL, "MEMORYGRAPH_REQUIRE_POSTGRES_TESTS=1 but MEMORYGRAPH_TEST_POSTGRES_URL is unset").toBeTruthy();
  });
});

describe.skipIf(!TEST_URL)("postgres backend (loopback pgvector)", () => {
  let admin: ReturnType<typeof postgres>;
  let stub: Awaited<ReturnType<typeof startStubEmbedder>>;
  const created: string[] = [];
  let backend: PostgresBackend;

  async function freshDatabase(): Promise<string> {
    const name = `mg_test_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
    await admin.unsafe(`CREATE DATABASE ${name}`);
    created.push(name);
    return dbUrl(name);
  }

  function embedder(url = stub.url, digest = STUB_DIGEST): OllamaEmbedder {
    return new OllamaEmbedder({ url, digest, timeoutMs: 2000 });
  }

  function smallEmbedder(): OllamaEmbedder {
    return new OllamaEmbedder({ url: stub.url, digest: STUB_DIGEST, model: SMALL_MODEL, dimension: SMALL_DIMENSION, timeoutMs: 2000 });
  }

  async function withClient<T>(url: string, fn: (c: ReturnType<typeof postgres>) => Promise<T>): Promise<T> {
    const c = postgres(url, { password: PASSWORD, onnotice: () => {}, max: 1 });
    try {
      return await fn(c);
    } finally {
      await c.end({ timeout: 5 });
    }
  }

  async function embeddingState(url: string): Promise<{ column: string; index: string; embedded: number; models: string[] }> {
    return withClient(url, async (c) => {
      const [col] = await c`
        SELECT format_type(atttypid, atttypmod) AS column FROM pg_attribute
        WHERE attrelid = 'memories'::regclass AND attname = 'embedding'`;
      const [idx] = await c`SELECT indexdef FROM pg_indexes WHERE indexname = 'memories_embedding_hnsw_idx'`;
      const [cnt] = await c`SELECT count(embedding)::int AS embedded FROM memories`;
      const models = await c`SELECT DISTINCT coalesce(embedding_model, '') AS m FROM memories ORDER BY m`;
      return {
        column: col["column"] as string,
        index: (idx?.["indexdef"] as string) ?? "",
        embedded: cnt["embedded"] as number,
        models: models.map((r) => r["m"] as string),
      };
    });
  }

  async function vectorsById(url: string): Promise<Map<string, number[]>> {
    return withClient(url, async (c) => {
      const rows = await c`SELECT id, embedding::text AS v FROM memories WHERE embedding IS NOT NULL ORDER BY id`;
      return new Map(rows.map((r) => [r["id"] as string, JSON.parse(r["v"] as string) as number[]]));
    });
  }

  /** Turn a fresh store back into the pre-LAB-388 layout: vector(1024) with a vector_cosine_ops index. */
  async function makeLegacyVectorStore(url: string): Promise<void> {
    await withClient(url, (c) =>
      c.unsafe(`
        DROP INDEX memories_embedding_hnsw_idx;
        ALTER TABLE memories ALTER COLUMN embedding TYPE vector(1024) USING embedding::vector(1024);
        CREATE INDEX memories_embedding_hnsw_idx ON memories USING hnsw (embedding vector_cosine_ops);`)
    );
  }

  const CORPUS: Array<[string, string]> = [
    ["Gluetun port forwarding", "gluetun forwards the vpn port to qbittorrent"],
    ["FalkorDB eviction policy", "allkeys-lru evicts memories silently under memory pressure"],
    ["Caddy reverse proxy", "caddy terminates tls for every homelab service"],
    ["Postgres halfvec index", "pgvector halfvec supports hnsw up to four thousand dimensions"],
    ["Ollama embedder digest", "the embedder digest is verified against api tags before embedding"],
  ];
  const QUERIES = ["vpn port forwarding", "memory eviction", "hnsw dimensions", "tls proxy"];

  async function recallIds(b: PostgresBackend): Promise<string[][]> {
    const out: string[][] = [];
    for (const q of QUERIES) out.push((await b.recallMemories(q, { limit: 3 })).map((m) => m.id!));
    return out;
  }

  async function openBackend(url: string, emb = embedder()): Promise<PostgresBackend> {
    const b = new PostgresBackend({ url, password: PASSWORD, embedder: emb });
    await b.connect();
    await b.initializeSchema();
    return b;
  }

  beforeAll(async () => {
    const host = new URL(TEST_URL!).hostname;
    if (!LOOPBACK.has(host)) {
      throw new Error(`MEMORYGRAPH_TEST_POSTGRES_URL must point at a loopback host, not ${host}`);
    }
    admin = postgres(TEST_URL!, { password: PASSWORD, onnotice: () => {}, max: 1 });
    stub = await startStubEmbedder();
  }, 30_000);

  afterAll(async () => {
    await backend?.disconnect();
    for (const name of created) {
      await admin.unsafe(`DROP DATABASE IF EXISTS ${name} WITH (FORCE)`).catch(() => {});
    }
    await admin?.end({ timeout: 5 });
    stub?.server.close();
  }, 60_000);

  beforeEach(async () => {
    await backend?.disconnect();
    backend = await openBackend(await freshDatabase());
  }, 30_000);

  test("schema init is idempotent and safe to run concurrently", async () => {
    const url = await freshDatabase();
    const backends = await Promise.all([1, 2, 3].map(() => openBackend(url)));
    await Promise.all(backends.map((b) => b.initializeSchema()));
    for (const b of backends) await b.disconnect();
  });

  test("store and get round-trip every field", async () => {
    const m = mem("Round trip", "content body", {
      tags: ["Alpha", "beta"],
      importance: 0.9,
      confidence: 0.7,
      effectiveness: 0.4,
      summary: "short",
      created_at: "2026-01-02T03:04:05.678Z",
      context: { project_path: "/p/one", languages: ["ts"] } as any,
    });
    const id = await backend.storeMemory(m);
    const got = await backend.getMemory(id);
    expect(got).not.toBeNull();
    expect(got!.title).toBe("Round trip");
    expect(got!.tags).toEqual(["alpha", "beta"]);
    expect(got!.importance).toBe(0.9);
    expect(got!.effectiveness).toBe(0.4);
    expect(got!.summary).toBe("short");
    expect(got!.created_at).toBe("2026-01-02T03:04:05.678Z");
    expect(got!.context?.project_path).toBe("/p/one");
    const stats = await backend.getMemoryStatistics();
    expect((stats["embeddings"] as any).embedded).toBe(1);
  });

  test("re-storing an unchanged memory keeps its vector without re-embedding", async () => {
    const m = mem("Stable", "unchanged text");
    await backend.storeMemory(m);
    const before = stub.calls();
    await backend.storeMemory({ ...m, importance: 0.2 });
    expect(stub.calls()).toBe(before);
    expect((await backend.getMemory(m.id!))!.importance).toBe(0.2);
  });

  test("update re-embeds only when title or content changes", async () => {
    const id = await backend.storeMemory(mem("Update me", "old content"));
    const m = (await backend.getMemory(id))!;
    let before = stub.calls();
    m.importance = 0.1;
    expect(await backend.updateMemory(m)).toBe(true);
    expect(stub.calls()).toBe(before);
    m.content = "new content about kubernetes";
    before = stub.calls();
    expect(await backend.updateMemory(m)).toBe(true);
    expect(stub.calls()).toBe(before + 1);
    const hits = await backend.recallMemories("kubernetes", { limit: 5 });
    expect(hits[0]?.id).toBe(id);
    expect(await backend.updateMemory({ ...m, id: "missing" })).toBe(false);
  });

  test("links: create, related at depth, type filter, validation, cascade on delete", async () => {
    const a = await backend.storeMemory(mem("A", "node a"));
    const b = await backend.storeMemory(mem("B", "node b"));
    const c = await backend.storeMemory(mem("C", "node c"));
    await backend.createRelationship(b, a, "SOLVES", { strength: 0.9 } as any);
    await backend.createRelationship(c, b, "RELATED_TO", { strength: 0.3 } as any);

    const depth1 = await backend.getRelatedMemories(a, { maxDepth: 1 });
    expect(depth1.map(([m]) => m.id)).toEqual([b]);
    expect(depth1[0][1].type).toBe("SOLVES");
    expect(depth1[0][1].properties.strength).toBe(0.9);

    const depth2 = await backend.getRelatedMemories(a, { maxDepth: 2 });
    expect(depth2.map(([m]) => m.id)).toEqual([b, c]);

    const filtered = await backend.getRelatedMemories(a, { maxDepth: 2, relationshipTypes: ["SOLVES"] });
    expect(filtered.map(([m]) => m.id)).toEqual([b]);

    await expect(backend.createRelationship(a, b, "NOT_A_TYPE")).rejects.toThrow(/Invalid relationship type/);
    await expect(backend.createRelationship(a, "missing", "SOLVES")).rejects.toThrow(/not found/);

    const since = await backend.getRelationshipsSince(new Date(Date.now() - 60_000));
    expect(since.length).toBe(2);

    expect(await backend.deleteMemory(b)).toBe(true);
    expect(await backend.getRelatedMemories(a, { maxDepth: 2 })).toEqual([]);
    expect(await backend.deleteMemory(b)).toBe(false);
  });

  test("related returns every edge to a neighbour: two types on one pair, and a pair linked both ways", async () => {
    const a = await backend.storeMemory(mem("Multi A", "node a"));
    const b = await backend.storeMemory(mem("Multi B", "node b"));
    const c = await backend.storeMemory(mem("Multi C", "node c"));
    await backend.createRelationship(a, b, "CAUSES", { strength: 0.9 } as any);
    await backend.createRelationship(a, b, "CONTRADICTS", { strength: 0.8 } as any);
    await backend.createRelationship(a, c, "CAUSES", { strength: 0.7 } as any);
    await backend.createRelationship(c, a, "CAUSES", { strength: 0.6 } as any);

    const edges = (await backend.getRelatedMemories(a, { maxDepth: 1 })).map(
      ([m, r]) => `${m.id}:${r.from_memory_id}>${r.to_memory_id}:${r.type}`
    );
    expect(edges).toEqual([
      `${b}:${a}>${b}:CAUSES`,
      `${b}:${a}>${b}:CONTRADICTS`,
      `${c}:${a}>${c}:CAUSES`,
      `${c}:${c}>${a}:CAUSES`,
    ]);
    expect((await backend.getRelatedMemories(b, { maxDepth: 1 })).length).toBe(2);
    expect((await backend.getRelatedMemories(a, { maxDepth: 2 })).length).toBe(4);
  });

  test("recall is hybrid, case-insensitive, and survives tsquery operator characters", async () => {
    const target = await backend.storeMemory(
      mem("FalkorDB eviction under memory pressure", "The falkordb container evicted keys when maxmemory was hit.")
    );
    await backend.storeMemory(mem("Unrelated", "postgres vacuum settings"));
    const upper = await backend.recallMemories("FALKORDB EVICTION", { limit: 5 });
    const lower = await backend.recallMemories("falkordb eviction", { limit: 5 });
    expect(upper.map((m) => m.id)).toEqual(lower.map((m) => m.id));
    expect(upper[0].id).toBe(target);
    expect(upper[0].match_info?.["match_quality"]).toBe("hybrid");

    const sent = stub.inputs.slice(-2);
    expect(sent).toEqual(["falkordb eviction", "falkordb eviction"]);

    for (const nasty of ["lint-cmd && test", "a|b", "foo:bar", "it's", "!(x)", "<->", "\\", "***", "'"]) {
      await backend.recallMemories(nasty, { limit: 3 });
      await backend.searchMemories(query({ query: nasty }));
    }
  });

  test("recall filters by type and project", async () => {
    await backend.storeMemory(mem("Redis timeout fix", "redis timeout", { context: { project_path: "/a" } as any }));
    await backend.storeMemory(
      createMemory({ type: "problem", title: "Redis timeout problem", content: "redis timeout", context: { project_path: "/b" } })
    );
    const problems = await backend.recallMemories("redis timeout", { memoryTypes: ["problem"] });
    expect(problems.map((m) => m.type)).toEqual(["problem"]);
    const projA = await backend.recallMemories("redis timeout", { projectPath: "/a" });
    expect(projA.map((m) => m.title)).toEqual(["Redis timeout fix"]);
  });

  test("recall falls back to full-text only and says so when the embedder is down", async () => {
    const url = await freshDatabase();
    const seeded = await openBackend(url);
    await seeded.storeMemory(mem("Gluetun port forwarding", "gluetun forwards the vpn port"));
    await seeded.disconnect();

    const down = await openBackend(url, embedder("http://127.0.0.1:1"));
    const { result, stderr } = await captureStderr(() => down.recallMemories("gluetun", { limit: 5 }));
    await down.disconnect();
    expect(result.map((m) => m.title)).toEqual(["Gluetun port forwarding"]);
    expect(stderr).toContain("full-text only");
  });

  test("store without an embedder records the gap; reindex fills it", async () => {
    const url = await freshDatabase();
    const down = await openBackend(url, embedder("http://127.0.0.1:1"));
    const { stderr } = await captureStderr(() => down.storeMemory(mem("No vector yet", "stored while embedder down")));
    expect(stderr).toContain("without an embedding");
    const stats = await down.getMemoryStatistics();
    expect((stats["embeddings"] as any).missing).toBe(1);
    const failed = await down.reindex();
    expect(failed.error).not.toBeNull();
    expect(failed.remaining).toBe(1);
    await down.disconnect();

    const up = await openBackend(url);
    const done = await up.reindex({ batchSize: 2 });
    expect(done).toEqual({ embedded: 1, remaining: 0, error: null });
    await up.disconnect();
  });

  test("a digest mismatch refuses to embed", async () => {
    const url = await freshDatabase();
    const wrong = await openBackend(url, embedder(stub.url, "some-other-digest"));
    const { stderr } = await captureStderr(() => wrong.storeMemory(mem("Digest", "mismatch")));
    expect(stderr).toContain("digest mismatch");
    await wrong.disconnect();
  });

  test("a fresh store creates halfvec(N) with a halfvec_cosine_ops HNSW index", async () => {
    const url = await freshDatabase();
    const b = await openBackend(url);
    await b.storeMemory(mem("Fresh halfvec", "stored into a halfvec column"));
    await b.disconnect();
    const state = await embeddingState(url);
    expect(state.column).toBe("halfvec(1024)");
    expect(state.index).toContain("halfvec_cosine_ops");
    expect(state.embedded).toBe(1);

    const smallUrl = await freshDatabase();
    const small = await openBackend(smallUrl, smallEmbedder());
    await small.storeMemory(mem("Small halfvec", "a 64 dimension store"));
    expect((await small.planEmbeddingMigration()).action).toBe("none");
    await small.disconnect();
    expect((await embeddingState(smallUrl)).column).toBe(`halfvec(${SMALL_DIMENSION})`);
  });

  test("recall goes through the halfvec HNSW index", async () => {
    const url = await freshDatabase();
    const seeded = await openBackend(url);
    for (const [title, content] of CORPUS) await seeded.storeMemory(mem(title, content));
    const expected = await recallIds(seeded);
    await seeded.disconnect();

    const dbName = new URL(url).pathname.slice(1);
    await admin.unsafe(`ALTER DATABASE ${dbName} SET enable_seqscan = off`);
    const indexed = await openBackend(url);
    expect(await recallIds(indexed)).toEqual(expected);
    expect((await indexed.recallMemories("vpn port forwarding", { limit: 1 }))[0].title).toBe("Gluetun port forwarding");
    await indexed.disconnect();

    const plan = await withClient(url, (c) =>
      c.unsafe(
        `EXPLAIN SELECT id FROM memories WHERE embedding IS NOT NULL
         ORDER BY embedding <=> $1::halfvec, id LIMIT 50`,
        [`[${stubVector("vpn port").join(",")}]`]
      )
    );
    expect(plan.map((r) => r["QUERY PLAN"]).join("\n")).toContain("memories_embedding_hnsw_idx");
  });

  test("a dimension mismatch at startup fails loudly", async () => {
    const url = await freshDatabase();
    const seeded = await openBackend(url);
    await seeded.storeMemory(mem("Seeded at 1024", "the column is halfvec 1024"));
    await seeded.disconnect();

    const mismatched = new PostgresBackend({ url, password: PASSWORD, embedder: smallEmbedder() });
    await mismatched.connect();
    await expect(mismatched.initializeSchema()).rejects.toThrow(
      /memories\.embedding is halfvec\(1024\) but MEMORY_EMBED_DIMENSION is 64; run 'memorygraph migrate embedding --dry-run'/
    );
    await mismatched.disconnect();

    const cli = await runCli(["stats"], {
      MEMORY_BACKEND: "postgres",
      MEMORY_POSTGRES_URL: url,
      MEMORY_POSTGRES_PASSWORD: PASSWORD ?? "",
      MEMORY_EMBED_URL: stub.url,
      MEMORY_EMBED_MODEL: SMALL_MODEL,
      MEMORY_EMBED_DIGEST: STUB_DIGEST,
      MEMORY_EMBED_DIMENSION: String(SMALL_DIMENSION),
    });
    expect(cli.code).not.toBe(0);
    expect(cli.stderr).toContain("MEMORY_EMBED_DIMENSION is 64");
    expect((await embeddingState(url)).column).toBe("halfvec(1024)");
  }, 30_000);

  test("migrate with the same dimension casts in place, keeping every vector and the recall results", async () => {
    const url = await freshDatabase();
    await (await openBackend(url)).disconnect();
    await makeLegacyVectorStore(url);

    const legacy = await openBackend(url);
    for (const [title, content] of CORPUS) await legacy.storeMemory(mem(title, content));
    const before = await recallIds(legacy);
    const dryRun = await legacy.planEmbeddingMigration();
    await legacy.disconnect();
    expect(dryRun.action).toBe("cast");
    expect(dryRun.current?.formatted).toBe("vector(1024)");
    expect(dryRun.statements.join(";")).toContain("USING embedding::halfvec(1024)");
    expect((await embeddingState(url)).column).toBe("vector(1024)");
    const vectorsBefore = await vectorsById(url);
    expect(vectorsBefore.size).toBe(CORPUS.length);

    const migrator = new PostgresBackend({ url, password: PASSWORD, embedder: embedder() });
    await migrator.connect();
    expect((await migrator.migrateEmbedding()).action).toBe("cast");
    expect((await migrator.migrateEmbedding()).action).toBe("none");
    await migrator.disconnect();

    const state = await embeddingState(url);
    expect(state.column).toBe("halfvec(1024)");
    expect(state.index).toContain("halfvec_cosine_ops");
    expect(state.embedded).toBe(CORPUS.length);
    const vectorsAfter = await vectorsById(url);
    expect([...vectorsAfter.keys()]).toEqual([...vectorsBefore.keys()]);
    for (const [id, v] of vectorsBefore) {
      const w = vectorsAfter.get(id)!;
      expect(w.length).toBe(v.length);
      expect(Math.max(...v.map((x, i) => Math.abs(x - w[i])))).toBeLessThan(1e-3);
    }

    const migrated = await openBackend(url);
    expect(await recallIds(migrated)).toEqual(before);
    await migrated.disconnect();
  }, 30_000);

  test("migrate with a new dimension NULLs embeddings, and reindex refills them", async () => {
    const url = await freshDatabase();
    const seeded = await openBackend(url);
    for (const [title, content] of CORPUS) await seeded.storeMemory(mem(title, content));
    await seeded.disconnect();

    const migrator = new PostgresBackend({ url, password: PASSWORD, embedder: smallEmbedder() });
    await migrator.connect();
    const plan = await migrator.planEmbeddingMigration();
    expect(plan.action).toBe("retype");
    expect(plan.embedded).toBe(CORPUS.length);
    expect((await embeddingState(url)).embedded).toBe(CORPUS.length);
    expect((await migrator.migrateEmbedding()).action).toBe("retype");
    expect((await migrator.migrateEmbedding()).action).toBe("none");
    await migrator.disconnect();

    const nulled = await embeddingState(url);
    expect(nulled.column).toBe(`halfvec(${SMALL_DIMENSION})`);
    expect(nulled.index).toContain("halfvec_cosine_ops");
    expect(nulled.embedded).toBe(0);
    expect(nulled.models).toEqual([""]);

    const small = await openBackend(url, smallEmbedder());
    expect(await small.reindex({ batchSize: 2 })).toEqual({ embedded: CORPUS.length, remaining: 0, error: null });
    expect((await small.recallMemories("vpn port forwarding", { limit: 1 }))[0].title).toBe("Gluetun port forwarding");
    await small.disconnect();
    const refilled = await embeddingState(url);
    expect(refilled.embedded).toBe(CORPUS.length);
    expect(refilled.models).toEqual([SMALL_MODEL]);
  }, 30_000);

  test("CLI: migrate embedding --dry-run prints the plan and changes nothing; without it, migrates", async () => {
    const url = await freshDatabase();
    const seeded = await openBackend(url);
    for (const [title, content] of CORPUS) await seeded.storeMemory(mem(title, content));
    await seeded.disconnect();
    await makeLegacyVectorStore(url);
    const env = {
      MEMORY_BACKEND: "postgres",
      MEMORY_POSTGRES_URL: url,
      MEMORY_POSTGRES_PASSWORD: PASSWORD ?? "",
      MEMORY_EMBED_URL: "http://127.0.0.1:1",
    };
    const snapshot = () =>
      withClient(url, async (c) => ({
        state: await embeddingState(url),
        rows: await c`SELECT id, updated_at, embedding::text AS v, embedding_model FROM memories ORDER BY id`,
      }));

    const before = await snapshot();
    const dry = await runCli(["migrate", "embedding", "--dry-run"], env);
    expect(dry.code).toBe(0);
    expect(dry.stdout).toContain("Embedding column: vector(1024) -> halfvec(1024)");
    expect(dry.stdout).toContain(`Memories: ${CORPUS.length} (${CORPUS.length} embedded)`);
    expect(dry.stdout).toContain("Action: cast");
    expect(dry.stdout).toContain("ALTER TABLE memories ALTER COLUMN embedding TYPE halfvec(1024) USING embedding::halfvec(1024);");
    expect(dry.stdout).toContain("Dry run: nothing was changed.");
    expect(dry.stdout + dry.stderr).not.toContain(PASSWORD ?? "\u0000");
    expect(await snapshot()).toEqual(before);

    const run = await runCli(["migrate", "embedding"], env);
    expect(run.code).toBe(0);
    expect(run.stdout).toContain("Migrated.");
    expect((await embeddingState(url)).column).toBe("halfvec(1024)");
    const again = await runCli(["migrate", "embedding", "--dry-run"], env);
    expect(again.stdout).toContain("Action: none");
  }, 60_000);

  test("duplicate check warns with the nearest match, logs it, and still stores", async () => {
    const original = await backend.storeMemory(
      mem("pct exec drops multiline arguments", "pct exec drops multiline arguments passed to the container")
    );
    backend.duplicateCheck = true;
    const dupId = await backend.storeMemory(
      mem("pct exec drops multiline arguments", "pct exec drops multiline arguments passed to the container")
    );
    expect(backend.lastDuplicate?.id).toBe(original);
    expect(backend.lastDuplicate!.similarity).toBeGreaterThan(0.99);
    expect(await backend.getMemory(dupId)).not.toBeNull();

    await backend.storeMemory(mem("Grafana dashboards", "provisioned from json files"));
    expect(backend.lastDuplicate).toBeNull();

    const stats = await backend.getMemoryStatistics();
    expect((stats["duplicate_events"] as any).count).toBe(1);
  });

  test("search applies filters, match modes and pagination", async () => {
    for (let i = 0; i < 5; i++) {
      await backend.storeMemory(
        mem(`Item ${i}`, `docker compose item ${i}`, { tags: i % 2 ? ["odd"] : ["even"], importance: i / 10 })
      );
    }
    expect((await backend.searchMemories(query({ tags: ["odd"] }))).length).toBe(2);
    expect((await backend.searchMemories(query({ min_importance: 0.3 }))).length).toBe(2);
    expect((await backend.searchMemories(query({ query: "compose" }))).length).toBe(5);
    expect((await backend.searchMemories(query({ query: "compose zzzz", match_mode: "all" }))).length).toBe(0);
    const page1 = await backend.searchMemories(query({ limit: 3 }));
    const page2 = await backend.searchMemories(query({ limit: 3, offset: 3 }));
    expect(new Set([...page1, ...page2].map((m) => m.id)).size).toBe(5);
  });

  test("activity reports recent memories and unresolved problems", async () => {
    const p = await backend.storeMemory(createMemory({ type: "problem", title: "Open problem", content: "x" }));
    const activity = await backend.getRecentActivity(7);
    expect(activity["recent_memories_total"]).toBe(1);
    expect((activity["unresolved_problems"] as Memory[]).map((m) => m.id)).toEqual([p]);
  });

  test("export then import round-trips memories and links into a fresh database", async () => {
    const a = await backend.storeMemory(mem("Export A", "alpha"));
    const b = await backend.storeMemory(mem("Export B", "beta"));
    await backend.createRelationship(b, a, "SOLVES");
    const dir = mkdtempSync(join(tmpdir(), "mg-pg-export-"));
    try {
      const file = join(dir, "export.json");
      const out = await exportToJson(new MemoryDatabase(backend), file);
      expect(out["memory_count"]).toBe(2);
      expect(out["relationship_count"]).toBe(1);
      expect(out["backend_type"]).toBe("postgres");

      const target = await openBackend(await freshDatabase());
      const result = await importFromJson(new MemoryDatabase(target), file);
      expect(result).toMatchObject({ imported_memories: 2, imported_relationships: 1 });
      expect((await target.getRelatedMemories(a, { maxDepth: 1 })).map(([m]) => m.id)).toEqual([b]);
      await target.disconnect();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("export then import keeps a second edge on the same pair and both directions of a pair", async () => {
    const a = await backend.storeMemory(mem("Pair A", "alpha"));
    const b = await backend.storeMemory(mem("Pair B", "beta"));
    const c = await backend.storeMemory(mem("Pair C", "gamma"));
    await backend.createRelationship(a, b, "CAUSES");
    await backend.createRelationship(a, b, "CONTRADICTS");
    await backend.createRelationship(a, c, "CAUSES");
    await backend.createRelationship(c, a, "CAUSES");
    const triples = async (db: PostgresBackend): Promise<string[]> => {
      const seen = new Set<string>();
      for (const id of [a, b, c]) {
        for (const [, r] of await db.getRelatedMemories(id, { maxDepth: 1 })) {
          seen.add(`${r.from_memory_id}>${r.to_memory_id}:${r.type}`);
        }
      }
      return [...seen].sort();
    };
    const expected = [`${a}>${b}:CAUSES`, `${a}>${b}:CONTRADICTS`, `${a}>${c}:CAUSES`, `${c}>${a}:CAUSES`].sort();
    expect(await triples(backend)).toEqual(expected);

    const dir = mkdtempSync(join(tmpdir(), "mg-pg-export-pairs-"));
    try {
      const file = join(dir, "export.json");
      const out = await exportToJson(new MemoryDatabase(backend), file);
      expect(out["relationship_count"]).toBe(4);

      const target = await openBackend(await freshDatabase());
      const result = await importFromJson(new MemoryDatabase(target), file);
      expect(result).toMatchObject({ imported_memories: 3, imported_relationships: 4, skipped_relationships: 0 });
      expect(await triples(target)).toEqual(expected);
      await target.disconnect();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("MigrationManager migrates a sqlite store into postgres and verifies it", async () => {
    const dir = mkdtempSync(join(tmpdir(), "mg-pg-migrate-"));
    try {
      const sqlitePath = join(dir, "memory.db");
      const source = new SQLiteBackend(sqlitePath);
      await source.connect();
      await source.initializeSchema();
      const a = await source.storeMemory(mem("Migrated A", "from sqlite"));
      const b = await source.storeMemory(mem("Migrated B", "also from sqlite"));
      await source.createRelationship(a, b, "CAUSES");
      await source.disconnect();

      const targetUrl = await freshDatabase();
      const saved = { url: process.env.MEMORY_EMBED_URL, digest: process.env.MEMORY_EMBED_DIGEST };
      process.env.MEMORY_EMBED_URL = stub.url;
      process.env.MEMORY_EMBED_DIGEST = STUB_DIGEST;
      try {
        const result = await new MigrationManager().migrate(
          { backend_type: "sqlite", path: sqlitePath },
          { backend_type: "postgres", uri: targetUrl, password: PASSWORD },
          createMigrationOptions({ verify: true })
        );
        expect(result.errors).toEqual([]);
        expect(result.success).toBe(true);
        expect(result.imported_memories).toBe(2);
        expect(result.imported_relationships).toBe(1);
      } finally {
        if (saved.url === undefined) delete process.env.MEMORY_EMBED_URL;
        else process.env.MEMORY_EMBED_URL = saved.url;
        if (saved.digest === undefined) delete process.env.MEMORY_EMBED_DIGEST;
        else process.env.MEMORY_EMBED_DIGEST = saved.digest;
      }
      const check = await openBackend(targetUrl);
      expect((await check.getMemoryStatistics())["embeddings"]).toMatchObject({ embedded: 2, missing: 0 });
      await check.disconnect();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("CLI: stats names the backend and host; store warns on a duplicate; migrate --from sqlite", async () => {
    const url = await freshDatabase();
    const env = {
      MEMORY_BACKEND: "postgres",
      MEMORY_POSTGRES_URL: url,
      MEMORY_POSTGRES_PASSWORD: PASSWORD ?? "",
      MEMORY_EMBED_URL: stub.url,
      MEMORY_EMBED_DIGEST: STUB_DIGEST,
    };
    const host = new URL(url).hostname;

    const stats = await runCli(["stats"], env);
    expect(stats.code).toBe(0);
    expect(stats.stdout).toContain("Explicit backend selection: Postgres");
    expect(stats.stdout).toContain(`Backend: postgres (${host}:`);
    expect(stats.stdout + stats.stderr).not.toContain(PASSWORD ?? "\u0000");

    const args = ["store", "--type", "solution", "--title", "cli duplicate probe", "--content", "same words here"];
    expect((await runCli(args, env)).code).toBe(0);
    const second = await runCli(args, env);
    expect(second.code).toBe(0);
    expect(second.stderr).toContain("possible duplicate of");

    const dir = mkdtempSync(join(tmpdir(), "mg-pg-cli-migrate-"));
    try {
      const sqlitePath = join(dir, "memory.db");
      const source = new SQLiteBackend(sqlitePath);
      await source.connect();
      await source.initializeSchema();
      await source.storeMemory(mem("From sqlite via CLI", "migrated"));
      await source.disconnect();
      const migrated = await runCli(["migrate", "--from", "sqlite"], {
        ...env,
        MEMORY_POSTGRES_URL: await freshDatabase(),
        MEMORY_SQLITE_PATH: sqlitePath,
      });
      expect(migrated.stderr).toContain("Migrating: sqlite -> postgres");
      expect(migrated.stderr).toContain("Migration completed successfully!");
      expect(migrated.code).toBe(0);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }

    const reindex = await runCli(["reindex"], env);
    expect(reindex.code).toBe(0);
    expect(reindex.stdout).toContain("0 still without an embedding");
  }, 60_000);
});

describe("postgres helpers", () => {
  test("buildTsquery strips operator characters and joins terms", () => {
    expect(buildTsquery("lint-cmd && test")).toBe("lint-cmd | test");
    expect(buildTsquery("foo:bar it's (x)")).toBe("foo | bar | it | s | x");
    expect(buildTsquery("&& | !")).toBeNull();
    expect(buildTsquery("a b", " & ")).toBe("a & b");
  });

  test("rrf fuses rankings and breaks ties by id", () => {
    const fused = rrf([
      [{ id: "b", score: 1 }, { id: "a", score: 0.5 }],
      [{ id: "a", score: 0.9 }, { id: "b", score: 0.1 }],
    ]);
    expect(fused.map((r) => r.id)).toEqual(["a", "b"]);
  });

  test("the embedder rejects a dimension pgvector cannot index as halfvec", () => {
    expect(new OllamaEmbedder({ dimension: 2560 }).dimension).toBe(2560);
    expect(() => new OllamaEmbedder({ dimension: 4001 })).toThrow(/MEMORY_EMBED_DIMENSION must be an integer from 1 to 4000/);
    expect(() => new OllamaEmbedder({ dimension: Number("abc") })).toThrow(/MEMORY_EMBED_DIMENSION/);
    expect(() => new OllamaEmbedder({ dimension: 0 })).toThrow(/MEMORY_EMBED_DIMENSION/);
  });

  test("formatEmbeddingPlan names a store with no memories table", () => {
    const text = formatEmbeddingPlan(
      { current: null, target: "halfvec(1024)", action: "none", memories: 0, embedded: 0, otherModel: 0, model: "m", statements: [] },
      true
    );
    expect(text).toContain("No memories table yet");
  });

  test("embedder is unavailable without MEMORY_EMBED_URL", async () => {
    const e = new OllamaEmbedder({});
    expect(e.unavailableReason()).toContain("MEMORY_EMBED_URL");
    await expect(e.embed(["x"])).rejects.toThrow(/MEMORY_EMBED_URL/);
  });
});
