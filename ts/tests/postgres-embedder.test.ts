/**
 * OllamaEmbedder URL failover and breaker (LAB-397). Every embedder here is a
 * stub HTTP server or a closed port on 127.0.0.1; no test reaches another host.
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { createServer, type Server } from "node:http";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createConnection } from "node:net";
import { EventEmitter } from "node:events";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  OllamaEmbedder,
  EmbedderUnavailableError,
  parseEmbedUrls,
  type EmbedderOptions,
  type SocketFactory,
} from "../src/backends/postgres-embedder.ts";
import { Config, embedBreakerDir } from "../src/config.ts";

const MODEL = "stub-embed:4";
const DIGEST = "stub-digest-good";
const DIMENSION = 4;

interface Stub {
  url: string;
  tags: number;
  embeds: number;
  close: () => Promise<void>;
}

type StubMode = "ok" | "wrong-digest" | "wrong-model" | "http-500" | "slow-500" | "hang-tags" | "embed-500";

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
        if (mode === "slow-500") {
          setTimeout(() => {
            res.statusCode = 500;
            res.end("{}");
          }, 250);
          return;
        }
        if (mode === "http-500") {
          res.statusCode = 500;
          res.end("{}");
          return;
        }
        const digest = mode === "wrong-digest" ? "stub-digest-other" : DIGEST;
        const name = mode === "wrong-model" ? "other-model:1" : MODEL;
        res.end(JSON.stringify({ models: [{ name, digest }] }));
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

/** A socket that never connects and never errors, like a SYN to a sleeping host. */
function silentSocket(): EventEmitter & { destroy: () => void } {
  return Object.assign(new EventEmitter(), { destroy: () => {} });
}

/** Connect normally, except to the given URLs, whose sockets never connect. */
function blackhole(...urls: string[]): SocketFactory {
  const ports = new Set(urls.map((u) => Number(new URL(u).port)));
  return (target) => (ports.has(target.port) ? silentSocket() : createConnection(target));
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

  test("a single URL keeps 8744d66's error text, logs nothing, probes no socket and persists no breaker", async () => {
    const noSocket: SocketFactory = () => {
      throw new Error("a single URL must not open a probe socket");
    };
    const cases: Array<[StubMode | "refused", (url: string) => string | RegExp]> = [
      ["wrong-digest", () => `embedder digest mismatch for ${MODEL}: expected ${DIGEST}, served stub-digest-other`],
      ["wrong-model", (url) => `embedder at ${url} does not serve ${MODEL}`],
      ["http-500", () => "embedder /api/tags returned HTTP 500"],
      ["hang-tags", () => "embedder /api/tags timed out after 200 ms"],
      ["embed-500", () => "embedder /api/embed returned HTTP 500"],
      ["refused", (url) => new RegExp(`^embedder at ${url.replace(/[.]/g, "[.]")} unreachable: TypeError: Unable to connect`)],
    ];
    for (const [mode, expected] of cases) {
      const url = mode === "refused" ? await closedPortUrl() : (await stub(mode)).url;
      const emb = make([url], { timeoutMs: 200, connect: noSocket });
      const { result, stderr } = await captureStderr(() => emb.embed(["x"]));
      expect(result).toBeInstanceOf(EmbedderUnavailableError);
      const want = expected(url);
      if (typeof want === "string") expect((result as Error).message).toBe(want);
      else expect((result as Error).message).toMatch(want);
      expect(emb.unavailableReason()).toBe((result as Error).message);
      expect(stderr).toEqual([]);
      expect(existsSync(breakerPath)).toBe(false);
    }
    const flaky = await stub("http-500");
    await make([flaky.url]).embed(["x"]).catch(() => {});
    await make([flaky.url]).embed(["x"]).catch(() => {});
    expect(flaky.tags).toBe(2);
  });

  test("a never-connecting first URL fails at connectTimeoutMs and the chain fails over", async () => {
    const asleep = await stub();
    const b = await stub();
    const emb = make([asleep.url, b.url], { connectTimeoutMs: 80, connect: blackhole(asleep.url) });
    const started = Date.now();
    expect(await emb.embed(["x"])).toHaveLength(1);
    const elapsed = Date.now() - started;
    expect(elapsed).toBeGreaterThanOrEqual(75);
    expect(elapsed).toBeLessThan(1000);
    expect(emb.activeUrl()).toBe(b.url);
    expect(asleep.tags).toBe(0);
    expect(breakerState()[asleep.url].reason).toContain("did not accept a connection within 80 ms");
  });

  test("concurrent embed() calls share one selection and both return", async () => {
    for (const urls of [1, 2]) {
      const a = await stub();
      const list = urls === 1 ? [a.url] : [a.url, (await stub()).url];
      const emb = make(list);
      const results = await Promise.all([emb.embed(["x"]), emb.embed(["y"]), emb.embed(["z"])]);
      expect(results.map((r) => r.length)).toEqual([1, 1, 1]);
      expect(a.tags).toBe(1);
      expect(emb.unavailableReason()).toBeNull();
    }
  });

  test("a breaker entry far in the future is clamped to one breaker window", async () => {
    const a = await stub();
    const b = await stub();
    writeFileSync(breakerPath, JSON.stringify({ version: 1, urls: { [a.url]: { until: 9e15, reason: "planted" } } }));
    const first = make([a.url, b.url], { breakerMs: 100 });
    await first.embed(["x"]);
    expect(first.activeUrl()).toBe(b.url);
    expect(breakerState()[a.url].until).toBeLessThanOrEqual(Date.now() + 100);
    await Bun.sleep(150);
    const later = make([a.url, b.url], { breakerMs: 100 });
    await later.embed(["y"]);
    expect(later.activeUrl()).toBe(a.url);
  });

  test("MEMORY_EMBED_TIMEOUT_MS bounds the whole chain in one embed() call", async () => {
    const a = await stub("hang-tags");
    const b = await stub("hang-tags");
    const emb = make([a.url, b.url], { timeoutMs: 300 });
    const started = Date.now();
    const { result } = await captureStderr(() => emb.embed(["x"]));
    const elapsed = Date.now() - started;
    expect(result).toBeInstanceOf(EmbedderUnavailableError);
    expect(elapsed).toBeLessThan(550);
    expect((result as Error).message).toContain("embed deadline of 300 ms spent");
    expect(b.tags).toBe(0);
    expect(emb.unavailableReason()).toBeNull();
    expect(Object.keys(breakerState())).toEqual([a.url]);
    expect(breakerState()[a.url].reason).toMatch(/timed out after \d+ ms/);
  });

  test("a timeout cut short by the deadline opens no breaker for that URL", async () => {
    const slow = await stub("slow-500");
    const hang = await stub("hang-tags");
    const emb = make([slow.url, hang.url], { timeoutMs: 400, connectTimeoutMs: 100 });
    const { result } = await captureStderr(() => emb.embed(["x"]));
    expect(result).toBeInstanceOf(EmbedderUnavailableError);
    expect((result as Error).message).toContain(`${hang.url}: embedder /api/tags timed out`);
    expect(hang.tags).toBe(1);
    expect(Object.keys(breakerState())).toEqual([slow.url]);
  });

  test("the temporary breaker file is removed when the rename fails", async () => {
    const down = await closedPortUrl();
    const b = await stub();
    const target = join(dir, "state");
    mkdirSync(join(target, "occupied"), { recursive: true });
    const emb = make([down, b.url], { breakerPath: target });
    expect(await emb.embed(["x"])).toHaveLength(1);
    expect(readdirSync(dir).filter((f) => f.endsWith(".tmp"))).toEqual([]);
  });

  test("the remaining URLs get what is left of the deadline", async () => {
    const asleep = await stub();
    const b = await stub();
    const emb = make([asleep.url, b.url], { timeoutMs: 400, connectTimeoutMs: 250, connect: blackhole(asleep.url) });
    const started = Date.now();
    expect(await emb.embed(["x"])).toHaveLength(1);
    expect(Date.now() - started).toBeLessThan(400);
    expect(emb.activeUrl()).toBe(b.url);
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

  test("defaults: 250 ms connect timeout, 60 s breaker, a per-user breaker file", () => {
    expect(Config.EMBED_CONNECT_TIMEOUT_MS).toBe(250);
    expect(Config.EMBED_BREAKER_MS).toBe(60000);
    const dir = process.platform === "linux" && process.env.XDG_RUNTIME_DIR ? process.env.XDG_RUNTIME_DIR : tmpdir();
    expect(Config.EMBED_BREAKER_PATH).toBe(join(dir, `memorygraph-embed-breaker-${process.getuid?.() ?? "user"}.json`));
  });

  test("the breaker directory is a non-empty $XDG_RUNTIME_DIR on Linux, else the OS temp dir", () => {
    expect(embedBreakerDir("linux", "/run/user/1000")).toBe("/run/user/1000");
    expect(embedBreakerDir("linux", "")).toBe(tmpdir());
    expect(embedBreakerDir("linux", undefined)).toBe(tmpdir());
    expect(embedBreakerDir("darwin", "/run/user/1000")).toBe(tmpdir());
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
