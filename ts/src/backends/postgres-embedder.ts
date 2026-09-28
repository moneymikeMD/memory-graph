/**
 * Ollama embedding client for the Postgres backend.
 *
 * Verifies the served model's digest against the pinned one before the first
 * embed, and latches unavailability for the rest of the process so a down
 * embedder costs one timeout, not one per memory.
 */

export const DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b";
export const DEFAULT_EMBED_DIGEST =
  "ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d";
export const EMBEDDING_DIMENSION = 1024;

export class EmbedderUnavailableError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmbedderUnavailableError";
  }
}

export interface EmbedderOptions {
  url?: string;
  model?: string;
  digest?: string;
  timeoutMs?: number;
}

export class OllamaEmbedder {
  readonly url: string | undefined;
  readonly model: string;
  readonly digest: string;
  readonly timeoutMs: number;
  private verified = false;
  private unavailable: string | null = null;

  constructor(opts: EmbedderOptions = {}) {
    this.url = opts.url ? opts.url.replace(/\/+$/, "") : undefined;
    this.model = opts.model ?? DEFAULT_EMBED_MODEL;
    this.digest = opts.digest ?? DEFAULT_EMBED_DIGEST;
    this.timeoutMs = opts.timeoutMs ?? 30000;
    if (!this.url) this.unavailable = "MEMORY_EMBED_URL is not set";
  }

  /** Reason the embedder cannot be used, or null while it is still usable. */
  unavailableReason(): string | null {
    return this.unavailable;
  }

  /**
   * Embed each text with the pinned model. Throws EmbedderUnavailableError
   * when the embedder is unset, unreachable, serving a different digest, or
   * returns vectors of the wrong dimension.
   */
  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];
    if (this.unavailable) throw new EmbedderUnavailableError(this.unavailable);
    try {
      if (!this.verified) await this.verifyDigest();
      const body = (await this.request("/api/embed", {
        model: this.model,
        input: texts,
      })) as { embeddings?: unknown };
      const vectors = body.embeddings;
      if (!Array.isArray(vectors) || vectors.length !== texts.length) {
        throw new EmbedderUnavailableError("embedder returned no embeddings");
      }
      for (const v of vectors) {
        if (!Array.isArray(v) || v.length !== EMBEDDING_DIMENSION) {
          throw new EmbedderUnavailableError(
            `embedder returned a ${Array.isArray(v) ? v.length : "non-array"}-dimension vector, expected ${EMBEDDING_DIMENSION}`
          );
        }
      }
      return vectors as number[][];
    } catch (err) {
      const reason = err instanceof Error ? err.message : String(err);
      this.unavailable = reason;
      throw err instanceof EmbedderUnavailableError ? err : new EmbedderUnavailableError(reason);
    }
  }

  private async verifyDigest(): Promise<void> {
    const tags = (await this.request("/api/tags")) as {
      models?: Array<{ name?: string; model?: string; digest?: string }>;
    };
    const entry = (tags.models ?? []).find((m) => m.name === this.model || m.model === this.model);
    if (!entry) {
      throw new EmbedderUnavailableError(`embedder at ${this.url} does not serve ${this.model}`);
    }
    if (entry.digest !== this.digest) {
      throw new EmbedderUnavailableError(
        `embedder digest mismatch for ${this.model}: expected ${this.digest}, served ${entry.digest}`
      );
    }
    this.verified = true;
  }

  private async request(path: string, body?: unknown): Promise<unknown> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(`${this.url}${path}`, {
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
      throw new EmbedderUnavailableError(`embedder at ${this.url} unreachable: ${err}`);
    } finally {
      clearTimeout(timer);
    }
  }
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
