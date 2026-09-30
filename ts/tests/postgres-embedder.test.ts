/**
 * OllamaEmbedder URL failover and breaker (LAB-397). Every embedder here is a
 * stub HTTP server or a closed port on 127.0.0.1; no test reaches another host.
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { createServer, type Server } from "node:http";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  OllamaEmbedder,
  EmbedderUnavailableError,
  parseEmbedUrls,
  type EmbedderOptions,
} from "../src/backends/postgres-embedder.ts";
import { Config } from "../src/config.ts";

const MODEL = "stub-embed:4";
const DIGEST = "stub-digest-good";
const DIMENSION = 4;

interface Stub {
  url: string;
  tags: number;
  embeds: number;
  close: () => Promise<void>;
}

type StubMode = "ok" | "wrong-digest" | "http-500" | "hang-tags" | "embed-500";

async function startStub(mode: StubMode = "ok"): Promise<Stub> {
  const stub = { url: "", tags: 0, embeds: 0, close: async () => {} };
  const server: Server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/api/tags") {
        stub.tags++;
        if (mode === "hang-tags") return;
        if (mode === "http-500") {
          res.statusCode = 500;
          res.end("{}");
          return;
        }
        const digest = mode === "wrong-digest" ? "stub-digest-other" : DIGEST;
        res.end(JSON.stringify({ models: [{ name: MODEL, digest }] }));
      } else if (req.url === "/api/embed") {
        stub.embeds++;
        if (mode === "embed-500") {
          res.statusCode = 500;
          res.end("{}");
          return;
        }
        const input = (JSON.parse(body).input as string[]) ?? [];
        res.end(JSON.stringify({ embeddings: input.map(() => [1, 0, 0, 0]) }));
      } else {
        res.statusCode = 404;
        res.end("{}");
      }
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const addr = server.address() as { port: number };
  stub.url = `http://127.0.0.1:${addr.port}`;
  stub.close = () =>
    new Promise<void>((resolve) => {
      server.closeAllConnections();
      server.close(() => resolve());
    });
  return stub;
}

/** A loopback URL whose port was just released, so connecting to it is refused. */
async function closedPortUrl(): Promise<string> {
  const s = await startStub();
  await s.close();
  return s.url;
}

function captureStderr<T>(fn: () => Promise<T>): Promise<{ result: T | Error; stderr: string[] }> {
  const original = console.error;
  const lines: string[] = [];
  console.error = (...args: unknown[]) => lines.push(args.map(String).join(" "));
  return fn()
    .then((result) => ({ result: result as T | Error, stderr: lines }))
    .catch((err: Error) => ({ result: err, stderr: lines }))
    .finally(() => {
      console.error = original;
    });
}

describe("OllamaEmbedder URL list", () => {
  let dir: string;
  let breakerPath: string;
  const stubs: Stub[] = [];

  function make(urls: string[], extra: EmbedderOptions = {}): OllamaEmbedder {
    return new OllamaEmbedder({
      url: urls.join(","),
      model: MODEL,
      digest: DIGEST,
      dimension: DIMENSION,
      timeoutMs: 1000,
      connectTimeoutMs: 250,
      breakerPath,
      ...extra,
    });
  }

  async function stub(mode: StubMode = "ok"): Promise<Stub> {
    const s = await startStub(mode);
    stubs.push(s);
    return s;
  }

  function breakerState(): Record<string, { until: number; reason: string }> {
    return JSON.parse(readFileSync(breakerPath, "utf8")).urls;
  }

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "mg-embed-breaker-"));
    breakerPath = join(dir, "breaker.json");
  });

  afterEach(async () => {
    for (const s of stubs.splice(0)) await s.close();
    rmSync(dir, { recursive: true, force: true });
  });

  test("parses a comma-separated list in order, trimming whitespace, slashes and duplicates", () => {
    expect(parseEmbedUrls(" http://a:1/ , ,http://b:2//,http://a:1")).toEqual(["http://a:1", "http://b:2"]);
    expect(parseEmbedUrls("http://only:11434")).toEqual(["http://only:11434"]);
    expect(parseEmbedUrls(undefined)).toEqual([]);
    expect(new OllamaEmbedder({ url: " , " }).unavailableReason()).toBe("MEMORY_EMBED_URL is not set");
  });

  test("a single URL embeds exactly as before", async () => {
    const a = await stub();
    const emb = make([a.url]);
    expect(await emb.embed(["x", "y"])).toEqual([
      [1, 0, 0, 0],
      [1, 0, 0, 0],
    ]);
    await emb.embed(["z"]);
    expect(emb.activeUrl()).toBe(a.url);
    expect(a.tags).toBe(1);
    expect(a.embeds).toBe(2);
  });

  test("first URL up: the second URL is never contacted", async () => {
    const a = await stub();
    const b = await stub();
    const emb = make([a.url, b.url]);
    await emb.embed(["x"]);
    expect(emb.activeUrl()).toBe(a.url);
    expect(b.tags + b.embeds).toBe(0);
  });

  test("first URL refusing connections: falls back to the second and opens the breaker", async () => {
    const down = await closedPortUrl();
    const b = await stub();
    const emb = make([down, b.url]);
    expect(await emb.embed(["x"])).toHaveLength(1);
    expect(emb.activeUrl()).toBe(b.url);
    const state = breakerState();
    expect(state[down].reason).toContain("unreachable");
    expect(state[down].until).toBeGreaterThan(Date.now());
    expect(state[b.url]).toBeUndefined();
  });

  test("first URL down (HTTP 500 on /api/tags): a later process skips it while the breaker holds", async () => {
    const a = await stub("http-500");
    const b = await stub();
    await make([a.url, b.url]).embed(["x"]);
    expect(a.tags).toBe(1);

    const nextProcess = make([a.url, b.url]);
    await nextProcess.embed(["y"]);
    expect(nextProcess.activeUrl()).toBe(b.url);
    expect(a.tags).toBe(1);
  });

  test("first URL hanging on /api/tags times out and falls back", async () => {
    const a = await stub("hang-tags");
    const b = await stub();
    const emb = make([a.url, b.url], { timeoutMs: 200 });
    await emb.embed(["x"]);
    expect(emb.activeUrl()).toBe(b.url);
    expect(breakerState()[a.url].reason).toContain("timed out after 200 ms");
  });

  test("first URL serving the wrong digest is skipped, not used, and the skip is logged once", async () => {
    const a = await stub("wrong-digest");
    const b = await stub();
    const first = await captureStderr(() => make([a.url, b.url]).embed(["x"]));
    expect(first.result).toHaveLength(1);
    expect(a.embeds).toBe(0);
    expect(first.stderr.filter((l) => l.includes("skipping embedder"))).toHaveLength(1);
    expect(first.stderr[0]).toContain("digest mismatch");

    const whileOpen = await captureStderr(() => make([a.url, b.url]).embed(["y"]));
    expect(whileOpen.stderr).toEqual([]);
    expect(a.tags).toBe(1);

    const state = breakerState();
    state[a.url].until = Date.now() - 1;
    writeFileSync(breakerPath, JSON.stringify({ version: 1, urls: state }));
    const afterExpiry = await captureStderr(() => make([a.url, b.url]).embed(["z"]));
    expect(a.tags).toBe(2);
    expect(a.embeds).toBe(0);
    expect(afterExpiry.stderr).toEqual([]);
  });

  test("all URLs down: throws naming each URL, then latches for the process", async () => {
    const down = await closedPortUrl();
    const bad = await stub("wrong-digest");
    const emb = make([down, bad.url]);
    const { result } = await captureStderr(() => emb.embed(["x"]));
    expect(result).toBeInstanceOf(EmbedderUnavailableError);
    const message = (result as Error).message;
    expect(message).toContain(down);
    expect(message).toContain(bad.url);
    expect(emb.unavailableReason()).toBe(message);

    await expect(emb.embed(["y"])).rejects.toThrow(EmbedderUnavailableError);
    expect(bad.tags).toBe(1);
  });

  test("an embed failure on the active URL fails over to the next one", async () => {
    const a = await stub("embed-500");
    const b = await stub();
    const emb = make([a.url, b.url]);
    expect(await emb.embed(["x"])).toHaveLength(1);
    expect(emb.activeUrl()).toBe(b.url);
    expect(breakerState()[a.url].reason).toContain("HTTP 500");
  });

  test("a recovered URL is used again once its breaker expires, and its entry is cleared", async () => {
    const a = await stub();
    const b = await stub();
    writeFileSync(
      breakerPath,
      JSON.stringify({ version: 1, urls: { [a.url]: { until: Date.now() - 1, reason: "was down" } } })
    );
    const emb = make([a.url, b.url]);
    await emb.embed(["x"]);
    expect(emb.activeUrl()).toBe(a.url);
    expect(breakerState()[a.url]).toBeUndefined();
  });

  test("breakerMs 0 never skips a URL", async () => {
    const a = await stub("http-500");
    const b = await stub();
    await make([a.url, b.url], { breakerMs: 0 }).embed(["x"]);
    await make([a.url, b.url], { breakerMs: 0 }).embed(["y"]);
    expect(a.tags).toBe(2);
  });

  test("an unwritable or corrupt breaker file does not stop embedding", async () => {
    const down = await closedPortUrl();
    const b = await stub();
    const unwritable = make([down, b.url], { breakerPath: join(dir, "missing", "\0bad", "breaker.json") });
    expect(await unwritable.embed(["x"])).toHaveLength(1);

    writeFileSync(breakerPath, "not json");
    const corrupt = make([down, b.url]);
    expect(await corrupt.embed(["x"])).toHaveLength(1);
    expect(breakerState()[down]).toBeDefined();
  });
});

describe("embedder config", () => {
  const keys = ["MEMORY_EMBED_CONNECT_TIMEOUT_MS", "MEMORY_EMBED_BREAKER_MS", "MEMORY_EMBED_BREAKER_PATH"];
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = Object.fromEntries(keys.map((k) => [k, process.env[k]]));
    for (const k of keys) delete process.env[k];
  });

  afterEach(() => {
    for (const k of keys) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  test("defaults: 250 ms connect timeout, 60 s breaker, a per-user file in the temp directory", () => {
    expect(Config.EMBED_CONNECT_TIMEOUT_MS).toBe(250);
    expect(Config.EMBED_BREAKER_MS).toBe(60000);
    expect(Config.EMBED_BREAKER_PATH).toBe(
      join(tmpdir(), `memorygraph-embed-breaker-${process.getuid?.() ?? "user"}.json`)
    );
  });

  test("each variable overrides its default", () => {
    process.env.MEMORY_EMBED_CONNECT_TIMEOUT_MS = "100";
    process.env.MEMORY_EMBED_BREAKER_MS = "0";
    process.env.MEMORY_EMBED_BREAKER_PATH = "/somewhere/breaker.json";
    expect(Config.EMBED_CONNECT_TIMEOUT_MS).toBe(100);
    expect(Config.EMBED_BREAKER_MS).toBe(0);
    expect(Config.EMBED_BREAKER_PATH).toBe("/somewhere/breaker.json");
  });
});
