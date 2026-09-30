/**
 * Ollama embedding client for the Postgres backend.
 *
 * Takes an ordered list of embedder URLs and uses the first one that accepts
 * a TCP connection within the connect timeout and serves the pinned model
 * digest. A URL that fails is skipped by a breaker, persisted to a small JSON
 * file so each short-lived hook process does not pay the same timeout again.
 * When every URL fails, unavailability latches for the rest of the process so
 * a down embedder costs one round of probes, not one per memory.
 */

import { createConnection } from "node:net";
import { mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
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

export interface EmbedderOptions {
  /** One URL, or a comma-separated list tried in order. */
  url?: string;
  model?: string;
  digest?: string;
  dimension?: number;
  timeoutMs?: number;
  connectTimeoutMs?: number;
  /** Breaker state file shared across processes; unset keeps the breaker in this process only. */
  breakerPath?: string;
  /** How long a failed URL is skipped; 0 disables the breaker. */
  breakerMs?: number;
}

interface BreakerEntry {
  until: number;
  reason: string;
}

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

export class OllamaEmbedder {
  readonly urls: string[];
  readonly model: string;
  readonly digest: string;
  readonly dimension: number;
  readonly timeoutMs: number;
  readonly connectTimeoutMs: number;
  readonly breakerMs: number;
  readonly breakerPath: string | undefined;
  private active: string | null = null;
  private readonly tried = new Set<string>();
  private readonly localBreaker: Record<string, BreakerEntry> = {};
  private unavailable: string | null = null;

  constructor(opts: EmbedderOptions = {}) {
    this.urls = parseEmbedUrls(opts.url);
    this.model = opts.model ?? DEFAULT_EMBED_MODEL;
    this.digest = opts.digest ?? DEFAULT_EMBED_DIGEST;
    this.dimension = validateDimension(opts.dimension ?? DEFAULT_EMBED_DIMENSION);
    this.timeoutMs = opts.timeoutMs ?? 30000;
    this.connectTimeoutMs = opts.connectTimeoutMs ?? DEFAULT_EMBED_CONNECT_TIMEOUT_MS;
    this.breakerMs = Math.max(0, opts.breakerMs ?? DEFAULT_EMBED_BREAKER_MS);
    this.breakerPath = opts.breakerPath || undefined;
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
    const failures: string[] = [];
    for (;;) {
      const url = this.active ?? (await this.select(failures));
      if (!url) {
        this.unavailable = failures.length > 0 ? failures.join("; ") : "no embedder URL is usable";
        throw new EmbedderUnavailableError(this.unavailable);
      }
      try {
        return await this.embedAt(url, texts);
      } catch (err) {
        const reason = errorMessage(err);
        failures.push(`${url}: ${reason}`);
        this.trip(url, reason, false);
        this.active = null;
      }
    }
  }

  private async select(failures: string[]): Promise<string | null> {
    const breaker = this.readBreaker();
    const now = Date.now();
    for (const url of this.urls) {
      if (this.tried.has(url)) continue;
      this.tried.add(url);
      const open = breaker[url];
      if (open && open.until > now) {
        failures.push(`${url}: skipped until ${new Date(open.until).toISOString()} (${open.reason})`);
        continue;
      }
      try {
        await this.probe(url);
      } catch (err) {
        const reason = errorMessage(err);
        failures.push(`${url}: ${reason}`);
        this.trip(url, reason, err instanceof WrongModelError, open);
        continue;
      }
      if (open) this.clearBreaker(url);
      this.active = url;
      return url;
    }
    return null;
  }

  private async probe(url: string): Promise<void> {
    await connectProbe(url, this.connectTimeoutMs);
    const tags = (await this.request(url, "/api/tags")) as {
      models?: Array<{ name?: string; model?: string; digest?: string }>;
    };
    const entry = (tags.models ?? []).find((m) => m.name === this.model || m.model === this.model);
    if (!entry) {
      throw new WrongModelError(`embedder at ${url} does not serve ${this.model}`);
    }
    if (entry.digest !== this.digest) {
      throw new WrongModelError(
        `embedder digest mismatch for ${this.model} at ${url}: expected ${this.digest}, served ${entry.digest}`
      );
    }
  }

  private async embedAt(url: string, texts: string[]): Promise<number[][]> {
    const body = (await this.request(url, "/api/embed", { model: this.model, input: texts })) as {
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

  /** Open the breaker for a URL; a wrong-model skip is logged once per distinct reason. */
  private trip(url: string, reason: string, logSkip: boolean, previous?: BreakerEntry): void {
    if (logSkip && previous?.reason !== reason) {
      console.error(`memorygraph: skipping embedder ${url}: ${reason}`);
    }
    if (this.breakerMs === 0) return;
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
    if (this.breakerMs === 0) return {};
    return { ...readBreakerFile(this.breakerPath), ...this.localBreaker };
  }

  private updateBreakerFile(mutate: (state: Record<string, BreakerEntry>) => void): void {
    if (!this.breakerPath) return;
    const state = readBreakerFile(this.breakerPath);
    mutate(state);
    try {
      mkdirSync(dirname(this.breakerPath), { recursive: true });
      const tmp = `${this.breakerPath}.${process.pid}.tmp`;
      writeFileSync(tmp, JSON.stringify({ version: 1, urls: state }), { mode: 0o600 });
      renameSync(tmp, this.breakerPath);
    } catch {
      // The breaker only saves time; an unwritable state file must not stop embedding.
    }
  }

  private async request(url: string, path: string, body?: unknown): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
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
        throw new EmbedderUnavailableError(`embedder ${path} timed out after ${this.timeoutMs} ms`);
      }
      throw new EmbedderUnavailableError(`embedder at ${url} unreachable: ${err}`);
    } finally {
      clearTimeout(timer);
    }
  }
}

class WrongModelError extends EmbedderUnavailableError {}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function readBreakerFile(path: string | undefined): Record<string, BreakerEntry> {
  if (!path) return {};
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8")) as { urls?: Record<string, BreakerEntry> };
    const out: Record<string, BreakerEntry> = {};
    for (const [url, e] of Object.entries(parsed.urls ?? {})) {
      if (e && typeof e.until === "number" && typeof e.reason === "string") out[url] = e;
    }
    return out;
  } catch {
    return {};
  }
}

/** Resolve when a TCP connection to the URL's host and port opens within timeoutMs, else throw. */
export function connectProbe(url: string, timeoutMs: number): Promise<void> {
  let target: URL;
  try {
    target = new URL(url);
  } catch {
    return Promise.reject(new EmbedderUnavailableError(`embedder URL ${url} is not a valid URL`));
  }
  const host = target.hostname.replace(/^\[|\]$/g, "");
  const port = Number(target.port) || (target.protocol === "https:" ? 443 : 80);
  return new Promise((resolve, reject) => {
    const socket = createConnection({ host, port });
    const timer = setTimeout(() => {
      socket.destroy();
      reject(new EmbedderUnavailableError(`embedder at ${url} did not accept a connection within ${timeoutMs} ms`));
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
