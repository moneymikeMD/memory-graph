/**
 * Postgres backend integration suite. Runs against a loopback pgvector
 * container named by MEMORYGRAPH_TEST_POSTGRES_URL (password from
 * MEMORY_POSTGRES_PASSWORD) and skips when that is unset. Each run creates and
 * drops its own databases; a non-loopback URL fails the suite. Embeddings come
 * from a stub Ollama server on 127.0.0.1.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from "bun:test";
import { createServer, type Server } from "node:http";
import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import postgres from "postgres";

import { PostgresBackend, buildTsquery, rrf } from "../src/backends/postgres.ts";
import { OllamaEmbedder, EMBEDDING_DIMENSION } from "../src/backends/postgres-embedder.ts";
import { SQLiteBackend } from "../src/backends/sqlite.ts";
import { MemoryDatabase } from "../src/database.ts";
import { createMemory, type Memory, type SearchQuery } from "../src/models.ts";
import { exportToJson, importFromJson } from "../src/utils/export-import.ts";
import { MigrationManager, createMigrationOptions } from "../src/migration/index.ts";

const TEST_URL = process.env.MEMORYGRAPH_TEST_POSTGRES_URL;
const PASSWORD = process.env.MEMORY_POSTGRES_PASSWORD;
const CLI = join(import.meta.dir, "..", "src", "cli.ts");
const STUB_DIGEST = "stub0000digest";
const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1", "[::1]"]);

function stubVector(text: string): number[] {
  const vec = new Array<number>(EMBEDDING_DIMENSION).fill(0);
  for (const token of text.toLowerCase().split(/[^a-z0-9]+/).filter((t) => t.length > 1)) {
    const h = createHash("sha1").update(token).digest();
    vec[h.readUInt16BE(0) % EMBEDDING_DIMENSION] += 1;
  }
  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
  return vec.map((x) => x / norm);
}

function startStubEmbedder(): Promise<{ server: Server; url: string; calls: () => number }> {
  let calls = 0;
  const server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/api/tags") {
        res.end(JSON.stringify({ models: [{ name: "qwen3-embedding:0.6b", digest: STUB_DIGEST }] }));
      } else if (req.url === "/api/embed") {
        calls++;
        const input = JSON.parse(body).input as string[];
        res.end(JSON.stringify({ embeddings: input.map(stubVector) }));
      } else {
        res.statusCode = 404;
        res.end("{}");
      }
    });
  });
  return new Promise((resolve) =>
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({ server, url: `http://127.0.0.1:${addr.port}`, calls: () => calls });
    })
  );
}

function dbUrl(name: string): string {
  const u = new URL(TEST_URL!);
  u.pathname = `/${name}`;
  return u.toString();
}

function mem(title: string, content: string, extra: Partial<Memory> = {}): Memory {
  return createMemory({ type: "solution", title, content, ...extra });
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
  });

  afterAll(async () => {
    await backend?.disconnect();
    for (const name of created) {
      await admin.unsafe(`DROP DATABASE IF EXISTS ${name} WITH (FORCE)`).catch(() => {});
    }
    await admin?.end({ timeout: 5 });
    stub?.server.close();
  });

  beforeEach(async () => {
    await backend?.disconnect();
    backend = await openBackend(await freshDatabase());
  });

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

  test("embedder is unavailable without MEMORY_EMBED_URL", async () => {
    const e = new OllamaEmbedder({});
    expect(e.unavailableReason()).toContain("MEMORY_EMBED_URL");
    await expect(e.embed(["x"])).rejects.toThrow(/MEMORY_EMBED_URL/);
  });
});
