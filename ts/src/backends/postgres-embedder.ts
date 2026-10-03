/**
 * Ollama embedding client for the Postgres backend.
 *
 * One URL behaves as it always has: the digest is verified before the first
 * embed and the first failure latches unavailability for the process. A
 * comma-separated list adds failover: each URL must accept a TCP connection
 * within the connect timeout and serve the pinned digest, one deadline covers
 * the whole chain in each embed() call, and a failed URL is skipped by a
 * breaker persisted to a small JSON file shared by short-lived hook processes.
 */

import { createConnection } from "node:net";
import { mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

export const DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b";
export const DEFAULT_EMBED_DIGEST =
  "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d";
export const DEFAULT_EMBED_DIMENSION = 1024;
/** pgvector's HNSW limit for halfvec columns. */
export const MAX_EMBED_DIMENSION = 4000;
export const DEFAULT_EMBED_CONNECT_TIMEOUT_MS = 250;
export const DEFAULT_EMBED_BREAKER_MS = 60_000;

export class EmbedderUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmbedderUnavailableError";
  }
}

/** The subset of a net.Socket that the connect probe uses. */
export interface ProbeSocket {
  once(event: "connect", listener: () => void): unknown;
  once(event: "error", listener: (err: Error) => void): unknown;
  destroy(): unknown;
}

export type SocketFactory = (target: { host: string; port: number }) => ProbeSocket;

export interface EmbedderOptions {
  /** One URL, or a comma-separated list tried in order. */
  url?: string;
  model?: string;
  digest?: string;
  dimension?: number;
  /** One URL: the limit per HTTP request. A list: the total for one embed() call across the chain. */
  timeoutMs?: number;
  /** A list only: the TCP connect timeout per URL. */
  connectTimeoutMs?: number;
  /** A list only: breaker state file shared across processes; unset keeps the breaker in this process. */
  breakerPath?: string;
  /** A list only: how long a failed URL is skipped; 0 disables the breaker. */
  breakerMs?: number;
  /** Ollama keep_alive for /api/embed: a duration such as "30m" or seconds, -1 to pin; unset sends none. */
  keepAlive?: string;
  /** Opens the probe's TCP connection; replaceable so tests need no network. */
  connect?: SocketFactory;
}

interface BreakerEntry {
  until: number;
  reason: string;
}

class DeadlineTimeoutError extends EmbedderUnavailableError {
  readonly shortened: boolean;
  constructor(message: string, shortened: boolean) {
    super(message);
    this.shortened = shortened;
  }
}

class WrongModelError extends EmbedderUnavailableError {}

/** Split MEMORY_EMBED_URL into its ordered URLs, trimmed, without trailing slashes or empty entries. */
export function parseEmbedUrls(raw: string | undefined): string[] {
  if (!raw) return [];
  const out: string[] = [];
  for (const part of raw.split(",")) {
    const url = part.trim().replace(/\/+$/, "");
    if (url && !out.includes(url)) out.push(url);
  }
  return out;
}

/** Ollama takes keep_alive as a number of seconds (-1 pins) or a duration string; blank means none. */
export function parseKeepAlive(raw: string | undefined): string | number | undefined {
  const v = raw?.trim();
  if (!v) return undefined;
  return /^-?\d+$/.test(v) ? Number(v) : v;
}

export class OllamaEmbedder {
  readonly urls: string[];
  readonly model: string;
  readonly digest: string;
  readonly dimension: number;
  readonly timeoutMs: number;
  readonly connectTimeoutMs: number;
  readonly breakerMs: number;
  readonly breakerPath: string | undefined;
  readonly keepAlive: string | number | undefined;
  private readonly multi: boolean;
  private readonly connect: SocketFactory;
  private active: string | null = null;
  private selecting: Promise<string | null> | null = null;
  private readonly tried = new Set<string>();
  private readonly failures: string[] = [];
  private readonly localBreaker: Record<string, BreakerEntry> = {};
  private unavailable: string | null = null;

  constructor(opts: EmbedderOptions = {}) {
    this.urls = parseEmbedUrls(opts.url);
    this.multi = this.urls.length > 1;
    this.model = opts.model ?? DEFAULT_EMBED_MODEL;
    this.digest = opts.digest ?? DEFAULT_EMBED_DIGEST;
    this.dimension = validateDimension(opts.dimension ?? DEFAULT_EMBED_DIMENSION);
    this.timeoutMs = opts.timeoutMs ?? 30000;
    this.connectTimeoutMs = opts.connectTimeoutMs ?? DEFAULT_EMBED_CONNECT_TIMEOUT_MS;
    this.breakerMs = Math.max(0, opts.breakerMs ?? DEFAULT_EMBED_BREAKER_MS);
    this.breakerPath = opts.breakerPath || undefined;
    this.keepAlive = parseKeepAlive(opts.keepAlive);
    this.connect = opts.connect ?? ((target) => createConnection(target));
    if (this.urls.length === 0) this.unavailable = "MEMORY_EMBED_URL is not set";
  }

  /** Reason the embedder cannot be used, or null while it is still usable. */
  unavailableReason(): string | null {
    return this.unavailable;
  }

  /** The URL this process is embedding with, or null before the first successful probe. */
  activeUrl(): string | null {
    return this.active;
  }

  /**
   * Embed each text with the pinned model, failing over down the URL list.
   * Throws EmbedderUnavailableError when no URL is set, or every URL is
   * unreachable, serving a different digest, or returning the wrong dimension.
   */
  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    if (this.unavailable) throw new EmbedderUnavailableError(this.unavailable);
    const deadline = this.multi ? Date.now() + this.timeoutMs : Number.POSITIVE_INFINITY;
    for (;;) {
      const url = this.active ?? (await this.selectOnce(deadline));
      if (!url) {
        if (this.unavailable) throw new EmbedderUnavailableError(this.unavailable);
        if (this.urls.every((u) => this.tried.has(u))) {
          this.unavailable = this.failures.join("; ") || "no embedder URL is usable";
          throw new EmbedderUnavailableError(this.unavailable);
        }
        throw new EmbedderUnavailableError(
          `embed deadline of ${this.timeoutMs} ms spent before every URL was tried: ${this.failures.join("; ")}`
        );
      }
      try {
        return await this.embedAt(url, texts, deadline);
      } catch (err) {
        if (this.active !== url) continue;
        this.active = null;
        this.fail(url, err);
      }
    }
  }

  private selectOnce(deadline: number): Promise<string | null> {
    if (!this.selecting) {
      this.selecting = this.select(deadline).finally(() => {
        this.selecting = null;
      });
    }
    return this.selecting;
  }

  private async select(deadline: number): Promise<string | null> {
    const breaker = this.readBreaker();
    const now = Date.now();
    for (const url of this.urls) {
      if (this.tried.has(url)) continue;
      const open = breaker[url];
      if (open && open.until > now) {
        this.tried.add(url);
        this.failures.push(`${url}: skipped until ${new Date(open.until).toISOString()} (${open.reason})`);
        continue;
      }
      if (deadline - Date.now() <= 0) return null;
      this.tried.add(url);
      try {
        await this.probe(url, deadline);
      } catch (err) {
        this.fail(url, err, open);
        continue;
      }
      if (open) this.clearBreaker(url);
      this.active = url;
      return url;
    }
    return null;
  }

  private async probe(url: string, deadline: number): Promise<void> {
    if (this.multi) {
      const connectMs = Math.min(this.connectTimeoutMs, remaining(deadline));
      try {
        await connectProbe(url, connectMs, this.connect);
      } catch (err) {
        if (err instanceof DeadlineTimeoutError && connectMs < this.connectTimeoutMs) {
          throw new DeadlineTimeoutError(err.message, true);
        }
        throw err;
      }
    }
    const tags = (await this.request(url, "/api/tags", deadline)) as {
      models?: Array<{ name?: string; model?: string; digest?: string }>;
    };
    const entry = (tags.models ?? []).find((m) => m.name === this.model || m.model === this.model);
    if (!entry) {
      throw new WrongModelError(`embedder at ${url} does not serve ${this.model}`);
    }
    if (entry.digest !== this.digest) {
      const where = this.multi ? ` at ${url}` : "";
      throw new WrongModelError(
        `embedder digest mismatch for ${this.model}${where}: expected ${this.digest}, served ${entry.digest}`
      );
    }
  }

  private async embedAt(url: string, texts: string[], deadline: number): Promise<number[][]> {
    const body = (await this.request(url, "/api/embed", deadline, {
      model: this.model,
      input: texts,
      ...(this.keepAlive === undefined ? {} : { keep_alive: this.keepAlive }),
    })) as {
      embeddings?: unknown;
    };
    const vectors = body.embeddings;
    if (!Array.isArray(vectors) || vectors.length !== texts.length) {
      throw new EmbedderUnavailableError("embedder returned no embeddings");
    }
    for (const v of vectors) {
      if (!Array.isArray(v) || v.length !== this.dimension) {
        throw new EmbedderUnavailableError(
          `embedder returned a ${Array.isArray(v) ? v.length : "non-array"}-dimension vector, expected ${this.dimension}`
        );
      }
    }
    return vectors as number[][];
  }

  /** Record a URL's failure; with a list, open its breaker and log a wrong-model skip once per reason. */
  private fail(url: string, err: unknown, previous?: BreakerEntry): void {
    const reason = errorMessage(err);
    this.failures.push(this.multi ? `${url}: ${reason}` : reason);
    if (!this.multi) return;
    if (err instanceof WrongModelError && previous?.reason !== reason) {
      console.error(`memorygraph: skipping embedder ${url}: ${reason}`);
    }
    if (this.breakerMs === 0) return;
    if (err instanceof DeadlineTimeoutError && err.shortened) return;
    const entry = { until: Date.now() + this.breakerMs, reason };
    this.localBreaker[url] = entry;
    this.updateBreakerFile((state) => {
      state[url] = entry;
    });
  }

  private clearBreaker(url: string): void {
    delete this.localBreaker[url];
    this.updateBreakerFile((state) => {
      delete state[url];
    });
  }

  private readBreaker(): Record<string, BreakerEntry> {
    if (!this.multi || this.breakerMs === 0) return {};
    const state = { ...readBreakerFile(this.breakerPath), ...this.localBreaker };
    const ceiling = Date.now() + this.breakerMs;
    const clamped = Object.keys(state).filter((url) => state[url].until > ceiling);
    if (clamped.length === 0) return state;
    for (const url of clamped) state[url].until = ceiling;
    this.updateBreakerFile((file) => {
      for (const url of clamped) if (file[url] && file[url].until > ceiling) file[url].until = ceiling;
    });
    return state;
  }

  private updateBreakerFile(mutate: (state: Record<string, BreakerEntry>) => void): void {
    if (!this.breakerPath) return;
    const state = readBreakerFile(this.breakerPath);
    mutate(state);
    const tmp = `${this.breakerPath}.${process.pid}.tmp`;
    try {
      mkdirSync(dirname(this.breakerPath), { recursive: true });
      writeFileSync(tmp, JSON.stringify({ version: 1, urls: state }), { mode: 0o600 });
      renameSync(tmp, this.breakerPath);
    } catch {
      // The breaker only saves time; an unwritable state file must not stop embedding.
      try {
        rmSync(tmp, { force: true });
      } catch {
        // Nothing left to clean up.
      }
    }
  }

  private async request(url: string, path: string, deadline: number, body?: unknown): Promise<unknown> {
    const allotted = Math.min(this.timeoutMs, remaining(deadline));
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), allotted);
    try {
      const res = await fetch(`${url}${path}`, {
        method: body === undefined ? "GET" : "POST",
        headers: body === undefined ? undefined : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      if (!res.ok) {
        throw new EmbedderUnavailableError(`embedder ${path} returned HTTP ${res.status}`);
      }
      return await res.json();
    } catch (err) {
      if (err instanceof EmbedderUnavailableError) throw err;
      const name = err instanceof Error ? err.name : "";
      if (name === "AbortError") {
        throw new DeadlineTimeoutError(
          `embedder ${path} timed out after ${allotted} ms`,
          allotted < this.timeoutMs - this.connectTimeoutMs
        );
      }
      throw new EmbedderUnavailableError(`embedder at ${url} unreachable: ${err}`);
    } finally {
      clearTimeout(timer);
    }
  }
}

function remaining(deadline: number): number {
  return Math.max(0, deadline - Date.now());
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function readBreakerFile(path: string | undefined): Record<string, BreakerEntry> {
  if (!path) return {};
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as { urls?: Record<string, BreakerEntry> };
    const out: Record<string, BreakerEntry> = {};
    for (const [url, e] of Object.entries(parsed.urls ?? {})) {
      if (e && Number.isFinite(e.until) && typeof e.reason === "string") out[url] = { until: e.until, reason: e.reason };
    }
    return out;
  } catch {
    return {};
  }
}

/** Resolve when a TCP connection to the URL's host and port opens within timeoutMs, else throw. */
export function connectProbe(
  url: string,
  timeoutMs: number,
  connect: SocketFactory = (target) => createConnection(target)
): Promise<void> {
  let target: URL;
  try {
    target = new URL(url);
  } catch {
    return Promise.reject(new EmbedderUnavailableError(`embedder URL ${url} is not a valid URL`));
  }
  const host = target.hostname.replace(/^\[|\]$/g, "");
  const port = Number(target.port) || (target.protocol === "https:" ? 443 : 80);
  return new Promise((resolve, reject) => {
    const socket = connect({ host, port });
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new DeadlineTimeoutError(`embedder at ${url} did not accept a connection within ${timeoutMs} ms`, false));
    }, timeoutMs);
    socket.once("connect", () => {
      clearTimeout(timer);
      socket.destroy();
      resolve();
    });
    socket.once("error", (err) => {
      clearTimeout(timer);
      socket.destroy();
      reject(new EmbedderUnavailableError(`embedder at ${url} unreachable: ${err.message}`));
    });
  });
}

/** Return the dimension if it is an integer pgvector can index as halfvec, else throw. */
export function validateDimension(dimension: number): number {
  if (!Number.isInteger(dimension) || dimension < 1 || dimension > MAX_EMBED_DIMENSION) {
    throw new Error(`MEMORY_EMBED_DIMENSION must be an integer from 1 to ${MAX_EMBED_DIMENSION}, got ${dimension}`);
  }
  return dimension;
}

/** The text a memory is embedded from, matching the LAB-348 benchmark. */
export function memoryEmbedText(title: string, content: string): string {
  return `${title}\n\n${content}`;
}

/** Recall embeds the lower-cased query so results do not depend on query case. */
export function queryEmbedText(query: string): string {
  return query.toLowerCase();
}

export function vectorLiteral(vec: number[]): string {
  return `[${vec.join(",")}]`;
}
