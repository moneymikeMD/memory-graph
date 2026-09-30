/**
 * Postgres backend for MemoryGraph: memories plus a links table, with hybrid
 * recall (tsvector full-text and pgvector cosine fused by reciprocal-rank
 * fusion, as benchmarked in homelab LAB-348 postgres_hybrid).
 *
 * Embeddings come from an Ollama embedder (MEMORY_EMBED_URL). A memory stored
 * while the embedder is down keeps a NULL vector and the reason; `reindex`
 * fills it later. Recall degrades to full-text only and says so on stderr.
 */

import postgres from "postgres";
import { randomUUID } from "node:crypto";

import { Config } from "../config.ts";
import {
  type Memory,
  type Relationship,
  type RelationshipProperties,
  type SearchQuery,
  createMemory,
  createRelationshipProperties,
  isRelationshipType,
  ALL_RELATIONSHIP_TYPES,
} from "../models.ts";
import type { GraphBackend, HealthCheckResult } from "./base.ts";
import { DatabaseConnectionError, RelationshipError, ValidationError } from "../errors.ts";
import {
  EmbedderUnavailableError,
  OllamaEmbedder,
  memoryEmbedText,
  queryEmbedText,
  vectorLiteral,
} from "./postgres-embedder.ts";

/**
 * Nearest-match cosine at or above which `store` warns of a possible
 * duplicate. Calibrated on the LAB-354 audit corpus; see
 * docs/postgres/README.md for the method and numbers.
 */
export const DEFAULT_DUPLICATE_THRESHOLD = 0.86;

const RRF_K = 60;
const MAX_QUERY_TERMS = 32;
const TSQUERY_UNSAFE = /[&|!()<>\\:*'"@~^{}[\]]/g;

export interface PostgresBackendOptions {
  url?: string;
  host?: string;
  port?: number;
  database?: string;
  user?: string;
  password?: string;
  embedder?: OllamaEmbedder;
  duplicateThreshold?: number;
}

export interface DuplicateMatch {
  id: string;
  title: string;
  similarity: number;
}

interface RankedId {
  id: string;
  score: number;
}

export interface EmbeddingColumn {
  type: string;
  dimension: number | null;
  formatted: string;
}

/**
 * What `migrate embedding` does to move memories.embedding to halfvec(N).
 * `current` is null when the store has no memories table yet.
 */
export interface EmbeddingMigrationPlan {
  current: EmbeddingColumn | null;
  target: string;
  action: "none" | "create-index" | "cast" | "retype";
  memories: number;
  embedded: number;
  otherModel: number;
  model: string;
  statements: string[];
}

type Sql = ReturnType<typeof postgres>;
type Tx = postgres.TransactionSql;

export class PostgresBackend implements GraphBackend {
  readonly embedder: OllamaEmbedder;
  readonly duplicateThreshold: number;
  /** Set by the `store` command only, so import and migrate do not log duplicate events. */
  duplicateCheck = false;
  lastDuplicate: DuplicateMatch | null = null;

  private readonly opts: PostgresBackendOptions;
  private sql: Sql | null = null;
  private connected = false;
  private warnedFulltextOnly = false;
  private columnType = "halfvec";

  constructor(opts: PostgresBackendOptions = {}) {
    this.opts = {
      url: opts.url ?? Config.POSTGRES_URL,
      host: opts.host ?? Config.POSTGRES_HOST,
      port: opts.port ?? Config.POSTGRES_PORT,
      database: opts.database ?? Config.POSTGRES_DB,
      user: opts.user ?? Config.POSTGRES_USER,
      password: opts.password ?? Config.POSTGRES_PASSWORD,
    };
    this.embedder =
      opts.embedder ??
      new OllamaEmbedder({
        url: Config.EMBED_URL,
        model: Config.EMBED_MODEL,
        digest: Config.EMBED_DIGEST,
        dimension: Config.EMBED_DIMENSION,
        timeoutMs: Config.EMBED_TIMEOUT_MS,
        connectTimeoutMs: Config.EMBED_CONNECT_TIMEOUT_MS,
        breakerMs: Config.EMBED_BREAKER_MS,
        breakerPath: Config.EMBED_BREAKER_PATH,
      });
    this.duplicateThreshold =
      opts.duplicateThreshold ?? Config.DUPLICATE_THRESHOLD ?? DEFAULT_DUPLICATE_THRESHOLD;
  }

  /** host:port/database of the server, never including credentials. */
  target(): string {
    if (this.opts.url) {
      try {
        const u = new URL(this.opts.url);
        return `${u.hostname}:${u.port || "5432"}${u.pathname || ""}`;
      } catch {
        return "(unparseable MEMORY_POSTGRES_URL)";
      }
    }
    return `${this.opts.host}:${this.opts.port}/${this.opts.database}`;
  }

  private db(): Sql {
    if (!this.sql) throw new DatabaseConnectionError("Not connected");
    return this.sql;
  }

  async connect(): Promise<boolean> {
    const common = {
      max: 4,
      idle_timeout: 5,
      connect_timeout: 10,
      onnotice: () => {},
    };
    try {
      this.sql = this.opts.url
        ? postgres(this.opts.url, { ...common, ...(this.opts.password ? { password: this.opts.password } : {}) })
        : postgres({
            ...common,
            host: this.opts.host,
            port: this.opts.port,
            database: this.opts.database,
            username: this.opts.user,
            password: this.opts.password,
          });
      await this.sql`SELECT 1`;
      this.connected = true;
      console.log(`Successfully connected to Postgres at ${this.target()}`);
      return true;
    } catch (err) {
      await this.sql?.end({ timeout: 1 }).catch(() => {});
      this.sql = null;
      throw new DatabaseConnectionError(`Failed to connect to Postgres at ${this.target()}: ${redact(err)}`);
    }
  }

  async disconnect(): Promise<void> {
    if (this.sql) {
      await this.sql.end({ timeout: 5 });
      this.sql = null;
    }
    this.connected = false;
  }

  /**
   * Create the schema if absent and check memories.embedding against the
   * configured dimension. Throws on a mismatch; `migrate embedding` fixes it.
   */
  async initializeSchema(): Promise<void> {
    const sql = this.db();
    const dimension = this.embedder.dimension;
    this.columnType = await sql.begin(async (tx) => {
      await tx`SELECT pg_advisory_xact_lock(hashtext('memorygraph_schema'))`;
      await tx.unsafe(schemaSql(dimension));
      const column = await readEmbeddingColumn(tx);
      if (!column || column.dimension !== dimension || (column.type !== "vector" && column.type !== "halfvec")) {
        throw new DatabaseConnectionError(
          `memories.embedding is ${column?.formatted ?? "missing"} but MEMORY_EMBED_DIMENSION is ${dimension}; ` +
            "run 'memorygraph migrate embedding --dry-run' to see the migration plan"
        );
      }
      await tx.unsafe(
        `CREATE INDEX IF NOT EXISTS ${EMBEDDING_INDEX} ON memories USING hnsw (embedding ${column.type}_cosine_ops)`
      );
      return column.type;
    });
  }

  /** The migration `migrate embedding` would run, computed in a read-only transaction. */
  async planEmbeddingMigration(): Promise<EmbeddingMigrationPlan> {
    return this.db().begin("read only", (tx) => buildEmbeddingPlan(tx, this.embedder.dimension, this.embedder.model));
  }

  /**
   * Move memories.embedding to halfvec(N) under the schema lock: cast in
   * place when N is unchanged, else NULL every vector for `reindex`.
   * Idempotent; returns the plan it ran.
   */
  async migrateEmbedding(): Promise<EmbeddingMigrationPlan> {
    return this.db().begin(async (tx) => {
      await tx`SELECT pg_advisory_xact_lock(hashtext('memorygraph_schema'))`;
      const plan = await buildEmbeddingPlan(tx, this.embedder.dimension, this.embedder.model);
      for (const statement of plan.statements) await tx.unsafe(statement);
      return plan;
    });
  }

  async executeQuery(): Promise<Record<string, unknown>[]> {
    throw new Error(
      "Postgres backend does not support Cypher queries. Use storeMemory(), searchMemories(), etc."
    );
  }

  async healthCheck(): Promise<HealthCheckResult> {
    const info: HealthCheckResult = {
      connected: this.connected,
      backend_type: "postgres",
      host: this.target(),
    };
    if (this.connected) {
      try {
        const [row] = await this.db()`SELECT count(*)::int AS count FROM memories`;
        info["statistics"] = { memory_count: row["count"] };
        const [ver] = await this.db()`SELECT extversion FROM pg_extension WHERE extname = 'vector'`;
        if (ver) info["pgvector_version"] = ver["extversion"];
      } catch (err) {
        info["warning"] = redact(err);
      }
    }
    return info;
  }

  backendName(): string {
    return "postgres";
  }
  supportsFulltextSearch(): boolean {
    return true;
  }
  supportsTransactions(): boolean {
    return true;
  }
  isCypherCapable(): boolean {
    return false;
  }

  // -- Embedding helpers --

  private async tryEmbed(texts: string[]): Promise<{ vectors: number[][] | null; error: string | null }> {
    try {
      return { vectors: await this.embedder.embed(texts), error: null };
    } catch (err) {
      if (err instanceof EmbedderUnavailableError) return { vectors: null, error: err.message };
      throw err;
    }
  }

  // -- Memory CRUD --

  async storeMemory(memory: Memory): Promise<string> {
    const sql = this.db();
    if (!memory.id) memory.id = randomUUID();
    const now = new Date().toISOString();
    const createdAt = memory.created_at ? toIso(memory.created_at) : now;
    const updatedAt = memory.updated_at ? toIso(memory.updated_at) : now;
    memory.updated_at = updatedAt;
    this.lastDuplicate = null;

    const [existing] = await sql`
      SELECT title, content, embedding IS NOT NULL AS has_embedding
      FROM memories WHERE id = ${memory.id}`;
    const textUnchanged =
      existing && existing["title"] === memory.title && existing["content"] === memory.content;
    const keepVector = Boolean(textUnchanged && existing["has_embedding"]);

    let vector: number[] | null = null;
    let embedError: string | null = null;
    if (!keepVector) {
      const { vectors, error } = await this.tryEmbed([memoryEmbedText(memory.title, memory.content)]);
      vector = vectors?.[0] ?? null;
      embedError = error;
      if (error) {
        console.error(`memorygraph: stored ${memory.id} without an embedding (${error}); run 'memorygraph reindex' later`);
      }
    }

    if (this.duplicateCheck && !existing) {
      if (vector) {
        this.lastDuplicate = await this.findDuplicate(memory, vector);
      } else {
        console.error("memorygraph: duplicate check skipped: the new memory has no embedding");
      }
    }

    const vectorParam = vector ? vectorLiteral(vector) : null;
    const row = {
      id: memory.id,
      type: memory.type,
      title: memory.title,
      content: memory.content,
      summary: memory.summary ?? null,
      tags: Array.isArray(memory.tags) ? memory.tags : [],
      importance: typeof memory.importance === "number" ? memory.importance : 0.5,
      confidence: typeof memory.confidence === "number" ? memory.confidence : 0.8,
      effectiveness: memory.effectiveness ?? null,
      usage_count: typeof memory.usage_count === "number" ? memory.usage_count : 0,
      created_at: createdAt,
      updated_at: updatedAt,
      last_accessed: memory.last_accessed ? toIso(memory.last_accessed) : null,
      version: typeof memory.version === "number" ? memory.version : 1,
      updated_by: memory.updated_by ?? null,
      context: memory.context ? sql.json(memory.context as never) : null,
    };

    try {
      await sql`
        INSERT INTO memories (
          id, type, title, content, summary, tags, importance, confidence, effectiveness,
          usage_count, created_at, updated_at, last_accessed, version, updated_by, context,
          embedding, embedding_model, embedding_error
        ) VALUES (
          ${row.id}, ${row.type}, ${row.title}, ${row.content}, ${row.summary}, ${row.tags}::text[],
          ${row.importance}, ${row.confidence}, ${row.effectiveness}, ${row.usage_count},
          ${row.created_at}, ${row.updated_at}, ${row.last_accessed}, ${row.version},
          ${row.updated_by}, ${row.context},
          ${vectorParam}::${sql.unsafe(this.columnType)}, ${vector ? this.embedder.model : null}, ${embedError}
        )
        ON CONFLICT (id) DO UPDATE SET
          type = EXCLUDED.type,
          title = EXCLUDED.title,
          content = EXCLUDED.content,
          summary = EXCLUDED.summary,
          tags = EXCLUDED.tags,
          importance = EXCLUDED.importance,
          confidence = EXCLUDED.confidence,
          effectiveness = EXCLUDED.effectiveness,
          usage_count = EXCLUDED.usage_count,
          updated_at = EXCLUDED.updated_at,
          last_accessed = EXCLUDED.last_accessed,
          version = EXCLUDED.version,
          updated_by = EXCLUDED.updated_by,
          context = EXCLUDED.context,
          embedding = CASE WHEN ${keepVector} THEN memories.embedding ELSE EXCLUDED.embedding END,
          embedding_model = CASE WHEN ${keepVector} THEN memories.embedding_model ELSE EXCLUDED.embedding_model END,
          embedding_error = CASE WHEN ${keepVector} THEN memories.embedding_error ELSE EXCLUDED.embedding_error END`;
    } catch (err) {
      throw new DatabaseConnectionError(`Failed to store memory: ${redact(err)}`);
    }

    if (this.lastDuplicate) {
      await sql`
        INSERT INTO duplicate_events (memory_id, match_id, similarity, threshold)
        VALUES (${memory.id}, ${this.lastDuplicate.id}, ${this.lastDuplicate.similarity}, ${this.duplicateThreshold})`;
    }
    return memory.id;
  }

  /**
   * Run the hybrid search for a memory about to be stored and return its
   * nearest match by cosine when that clears the duplicate threshold.
   */
  async findDuplicate(memory: Memory, vector: number[]): Promise<DuplicateMatch | null> {
    const fused = await this.hybridRank(
      `${memory.title} ${memory.content}`,
      vector,
      { excludeId: memory.id ?? undefined },
      5
    );
    if (fused.length === 0) return null;
    const ids = fused.map((r) => r.id);
    const rows = await this.db()`
      SELECT id, title, 1 - (embedding <=> ${vectorLiteral(vector)}::${this.db().unsafe(this.columnType)}) AS similarity
      FROM memories
      WHERE id = ANY(${ids}::text[]) AND embedding IS NOT NULL
      ORDER BY similarity DESC, id
      LIMIT 1`;
    const best = rows[0];
    if (!best) return null;
    const similarity = Number(best["similarity"]);
    if (similarity < this.duplicateThreshold) return null;
    return { id: best["id"] as string, title: best["title"] as string, similarity };
  }

  async getMemory(memoryId: string, _includeRelationships = true): Promise<Memory | null> {
    const [row] = await this.db()`SELECT ${this.db().unsafe(MEMORY_COLUMNS)} FROM memories WHERE id = ${memoryId}`;
    return row ? rowToMemory(row) : null;
  }

  async searchMemories(searchQuery: SearchQuery): Promise<Memory[]> {
    const params: unknown[] = [];
    const p = (v: unknown): string => {
      params.push(v);
      return `$${params.length}`;
    };
    const where = filterClauses(searchQuery, p);

    const explicitTerms = (searchQuery.terms ?? []).map((t) => t.trim()).filter((t) => t.length > 0);
    const text = explicitTerms.length > 0 ? explicitTerms.join(" ") : searchQuery.query ?? "";
    const joiner = searchQuery.match_mode === "all" ? " & " : " | ";
    const tsq = buildTsquery(text, joiner);

    let rank = "0";
    if (tsq) {
      const ref = p(tsq);
      where.push(`search_vector @@ to_tsquery('english', ${ref})`);
      rank = `ts_rank(search_vector, to_tsquery('english', ${ref}))`;
    } else if (text.trim()) {
      const pattern = p(`%${escapeLike(text.trim())}%`);
      where.push(`(title ILIKE ${pattern} OR content ILIKE ${pattern} OR coalesce(summary, '') ILIKE ${pattern})`);
    }

    const limit = p(Math.max(0, Math.trunc(searchQuery.limit ?? 20)));
    const offset = p(Math.max(0, Math.trunc(searchQuery.offset ?? 0)));
    const rows = await this.db().unsafe(
      `SELECT ${MEMORY_COLUMNS}, ${rank} AS rank FROM memories
       ${where.length ? `WHERE ${where.join(" AND ")}` : ""}
       ORDER BY rank DESC, importance DESC, created_at DESC, id
       LIMIT ${limit} OFFSET ${offset}`,
      params as never[]
    );
    return rows.map(rowToMemory).filter((m): m is Memory => m !== null);
  }

  /**
   * Hybrid recall: full-text and vector rankings fused by RRF. Falls back to
   * full-text only, with a note on stderr, when the query cannot be embedded.
   */
  async recallMemories(
    query: string,
    opts?: { memoryTypes?: string[]; projectPath?: string; limit?: number }
  ): Promise<Memory[]> {
    const limit = Math.max(1, Math.trunc(opts?.limit ?? 20));
    const { vectors, error } = await this.tryEmbed([queryEmbedText(query)]);
    if (error && !this.warnedFulltextOnly) {
      console.error(`memorygraph: recall is full-text only: the query could not be embedded (${error})`);
      this.warnedFulltextOnly = true;
    }
    const ranked = await this.hybridRank(
      query,
      vectors?.[0] ?? null,
      { memoryTypes: opts?.memoryTypes, projectPath: opts?.projectPath },
      limit
    );
    if (ranked.length === 0) return [];
    const rows = await this.db()`
      SELECT ${this.db().unsafe(MEMORY_COLUMNS)} FROM memories WHERE id = ANY(${ranked.map((r) => r.id)}::text[])`;
    const byId = new Map(rows.map((r) => [r["id"] as string, r]));
    const out: Memory[] = [];
    for (const r of ranked) {
      const row = byId.get(r.id);
      const mem = row ? rowToMemory(row) : null;
      if (mem) {
        mem.match_info = { match_quality: vectors ? "hybrid" : "fulltext", rrf_score: r.score };
        out.push(mem);
      }
    }
    return out;
  }

  private async hybridRank(
    text: string,
    vector: number[] | null,
    filters: { memoryTypes?: string[]; projectPath?: string; excludeId?: string },
    limit: number
  ): Promise<RankedId[]> {
    const overFetch = Math.max(limit * 4, 50);
    const fulltext = await this.fulltextRank(text, filters, overFetch);
    if (!vector) return fulltext.slice(0, limit);
    const vec = await this.vectorRank(vector, filters, overFetch);
    return rrf([fulltext, vec]).slice(0, limit);
  }

  private async fulltextRank(
    text: string,
    filters: { memoryTypes?: string[]; projectPath?: string; excludeId?: string },
    limit: number
  ): Promise<RankedId[]> {
    const tsq = buildTsquery(text, " | ");
    if (!tsq) return [];
    const params: unknown[] = [tsq];
    const where = ["search_vector @@ to_tsquery('english', $1)", ...recallFilters(filters, params)];
    params.push(limit);
    const rows = await this.db().unsafe(
      `SELECT id, ts_rank(search_vector, to_tsquery('english', $1)) AS score
       FROM memories WHERE ${where.join(" AND ")}
       ORDER BY score DESC, id LIMIT $${params.length}`,
      params as never[]
    );
    return rows.map((r) => ({ id: r["id"] as string, score: Number(r["score"]) }));
  }

  private async vectorRank(
    vector: number[],
    filters: { memoryTypes?: string[]; projectPath?: string; excludeId?: string },
    limit: number
  ): Promise<RankedId[]> {
    const params: unknown[] = [vectorLiteral(vector)];
    const where = ["embedding IS NOT NULL", ...recallFilters(filters, params)];
    params.push(limit);
    const efSearch = Math.min(Math.max(limit, 40), 1000);
    const rows = await this.db().begin(async (tx) => {
      await tx.unsafe(`SET LOCAL hnsw.ef_search = ${efSearch}`);
      return tx.unsafe(
        `SELECT id, 1 - (embedding <=> $1::${this.columnType}) AS score
         FROM memories WHERE ${where.join(" AND ")}
         ORDER BY embedding <=> $1::${this.columnType}, id LIMIT $${params.length}`,
        params as never[]
      );
    });
    return (rows as Record<string, unknown>[]).map((r) => ({ id: r["id"] as string, score: Number(r["score"]) }));
  }

  async updateMemory(memory: Memory): Promise<boolean> {
    if (!memory.id) throw new ValidationError("Memory must have an ID to update");
    const sql = this.db();
    memory.updated_at = new Date().toISOString();
    const [existing] = await sql`SELECT title, content FROM memories WHERE id = ${memory.id}`;
    if (!existing) return false;

    const textChanged = existing["title"] !== memory.title || existing["content"] !== memory.content;
    let vectorSql = sql`embedding`;
    let modelSql = sql`embedding_model`;
    let errorSql = sql`embedding_error`;
    if (textChanged) {
      const { vectors, error } = await this.tryEmbed([memoryEmbedText(memory.title, memory.content)]);
      const vec = vectors?.[0] ? vectorLiteral(vectors[0]) : null;
      if (error) console.error(`memorygraph: updated ${memory.id} without an embedding (${error}); run 'memorygraph reindex' later`);
      vectorSql = sql`${vec}::${sql.unsafe(this.columnType)}`;
      modelSql = sql`${vec ? this.embedder.model : null}`;
      errorSql = sql`${error}`;
    }

    const result = await sql`
      UPDATE memories SET
        type = ${memory.type}, title = ${memory.title}, content = ${memory.content},
        summary = ${memory.summary ?? null}, tags = ${memory.tags ?? []}::text[],
        importance = ${memory.importance}, confidence = ${memory.confidence},
        effectiveness = ${memory.effectiveness ?? null}, usage_count = ${memory.usage_count ?? 0},
        updated_at = ${toIso(memory.updated_at)},
        last_accessed = ${memory.last_accessed ? toIso(memory.last_accessed) : null},
        version = ${memory.version ?? 1}, updated_by = ${memory.updated_by ?? null},
        context = ${memory.context ? sql.json(memory.context as never) : null},
        embedding = ${vectorSql}, embedding_model = ${modelSql}, embedding_error = ${errorSql}
      WHERE id = ${memory.id}`;
    return result.count > 0;
  }

  async deleteMemory(memoryId: string): Promise<boolean> {
    const result = await this.db()`DELETE FROM memories WHERE id = ${memoryId}`;
    return result.count > 0;
  }

  // -- Relationships --

  async createRelationship(
    fromMemoryId: string,
    toMemoryId: string,
    relationshipType: string,
    properties?: RelationshipProperties
  ): Promise<string> {
    if (!isRelationshipType(relationshipType)) {
      throw new RelationshipError(
        `Invalid relationship type: '${relationshipType}'. Valid types are: ${ALL_RELATIONSHIP_TYPES.join(", ")}`
      );
    }
    const sql = this.db();
    const found = await sql`SELECT id FROM memories WHERE id IN (${fromMemoryId}, ${toMemoryId})`;
    const foundIds = new Set(found.map((r) => r["id"]));
    if (!foundIds.has(fromMemoryId) || !foundIds.has(toMemoryId)) {
      throw new RelationshipError("One or both memories not found", {
        from_id: fromMemoryId,
        to_id: toMemoryId,
      });
    }
    const id = randomUUID();
    const props = createRelationshipProperties(properties);
    await sql`
      INSERT INTO links (
        id, from_id, to_id, rel_type, strength, confidence, context, evidence_count, success_rate,
        created_at, last_validated, validation_count, counter_evidence_count,
        valid_from, valid_until, recorded_at, invalidated_by
      ) VALUES (
        ${id}, ${fromMemoryId}, ${toMemoryId}, ${relationshipType}, ${props.strength}, ${props.confidence},
        ${props.context ?? null}, ${props.evidence_count}, ${props.success_rate ?? null},
        ${toIso(props.created_at)}, ${toIso(props.last_validated)}, ${props.validation_count},
        ${props.counter_evidence_count}, ${toIso(props.valid_from)},
        ${props.valid_until ? toIso(props.valid_until) : null}, ${toIso(props.recorded_at)},
        ${props.invalidated_by ?? null}
      )`;
    return id;
  }

  async getRelatedMemories(
    memoryId: string,
    opts?: { relationshipTypes?: string[]; maxDepth?: number; limit?: number }
  ): Promise<[Memory, Relationship][]> {
    const sql = this.db();
    const maxDepth = Math.max(1, Math.min(Number(opts?.maxDepth ?? 2) || 2, 10));
    const relTypes = opts?.relationshipTypes?.length ? opts.relationshipTypes : null;
    const visited = new Set<string>([memoryId]);
    const reached: Array<{ id: string; link: Record<string, unknown> }> = [];
    let frontier = [memoryId];

    for (let depth = 0; depth < maxDepth && frontier.length > 0; depth++) {
      const links = relTypes
        ? await sql`SELECT * FROM links
            WHERE (from_id = ANY(${frontier}::text[]) OR to_id = ANY(${frontier}::text[]))
              AND rel_type = ANY(${relTypes}::text[])
            ORDER BY strength DESC, id`
        : await sql`SELECT * FROM links
            WHERE from_id = ANY(${frontier}::text[]) OR to_id = ANY(${frontier}::text[])
            ORDER BY strength DESC, id`;
      const frontierSet = new Set(frontier);
      const layer = new Set<string>();
      for (const link of links) {
        const from = link["from_id"] as string;
        const to = link["to_id"] as string;
        const other = frontierSet.has(from) && !visited.has(to) ? to : frontierSet.has(to) && !visited.has(from) ? from : null;
        if (!other) continue;
        layer.add(other);
        reached.push({ id: other, link });
      }
      for (const id of layer) visited.add(id);
      frontier = [...layer];
    }

    if (reached.length === 0) return [];
    const rows = await sql`
      SELECT ${sql.unsafe(MEMORY_COLUMNS)} FROM memories WHERE id = ANY(${reached.map((r) => r.id)}::text[])`;
    const byId = new Map(rows.map((r) => [r["id"] as string, r]));
    const results: [Memory, Relationship][] = [];
    for (const { id, link } of reached) {
      const row = byId.get(id);
      const mem = row ? rowToMemory(row) : null;
      if (mem) results.push([mem, rowToRelationship(link)]);
    }
    return opts?.limit !== undefined ? results.slice(0, Math.max(0, Math.trunc(opts.limit))) : results;
  }

  // -- Statistics and activity --

  async getMemoryStatistics(): Promise<Record<string, unknown>> {
    const sql = this.db();
    const [totals] = await sql`
      SELECT count(*)::int AS count,
             count(embedding)::int AS embedded,
             avg(importance) AS avg_importance,
             avg(confidence) AS avg_confidence
      FROM memories`;
    const [rels] = await sql`SELECT count(*)::int AS count FROM links`;
    const byType = await sql`SELECT type, count(*)::int AS count FROM memories GROUP BY type ORDER BY count DESC, type`;
    const [dups] = await sql`SELECT count(*)::int AS count FROM duplicate_events`;
    const memoriesByType: Record<string, number> = {};
    for (const row of byType) memoriesByType[row["type"] as string] = row["count"] as number;
    return {
      backend: { name: "postgres", host: this.target() },
      total_memories: { count: totals["count"] },
      total_relationships: { count: rels["count"] },
      memories_by_type: memoriesByType,
      avg_importance: { avg_importance: numOrNull(totals["avg_importance"]) },
      avg_confidence: { avg_confidence: numOrNull(totals["avg_confidence"]) },
      embeddings: {
        embedded: totals["embedded"],
        missing: (totals["count"] as number) - (totals["embedded"] as number),
        model: this.embedder.model,
      },
      duplicate_events: { count: dups["count"] },
    };
  }

  async getRecentActivity(days = 7, project?: string | null): Promise<Record<string, unknown>> {
    const sql = this.db();
    const cutoff = new Date(Date.now() - days * 86400_000).toISOString();
    const RECENT_CAP = 50;
    const UNRESOLVED_CAP = 20;
    const recentRows = await sql`
      SELECT ${sql.unsafe(MEMORY_COLUMNS)} FROM memories
      WHERE created_at >= ${cutoff} ORDER BY created_at DESC, id LIMIT ${RECENT_CAP}`;
    const [recentTotal] = await sql`SELECT count(*)::int AS count FROM memories WHERE created_at >= ${cutoff}`;
    const problemRows = await sql`
      SELECT ${sql.unsafe(MEMORY_COLUMNS)} FROM memories m
      WHERE m.type = 'problem'
        AND NOT EXISTS (SELECT 1 FROM links l WHERE l.to_id = m.id AND l.rel_type = 'SOLVES')
      ORDER BY m.importance DESC, m.id LIMIT ${UNRESOLVED_CAP}`;
    const [unresolvedTotal] = await sql`
      SELECT count(*)::int AS count FROM memories m
      WHERE m.type = 'problem'
        AND NOT EXISTS (SELECT 1 FROM links l WHERE l.to_id = m.id AND l.rel_type = 'SOLVES')`;

    const recent = recentRows.map(rowToMemory).filter((m): m is Memory => m !== null);
    const unresolved = problemRows.map(rowToMemory).filter((m): m is Memory => m !== null);
    const byType: Record<string, number> = {};
    for (const mem of recent) byType[mem.type] = (byType[mem.type] ?? 0) + 1;
    const recentCapped = (recentTotal["count"] as number) > recent.length;
    const unresolvedCapped = (unresolvedTotal["count"] as number) > unresolved.length;
    const capParts: string[] = [];
    if (recentCapped) {
      capParts.push(`Recent memories capped at ${RECENT_CAP} (${recentTotal["count"]} total in the last ${days} days)`);
    }
    if (unresolvedCapped) {
      capParts.push(`Unresolved problems capped at ${UNRESOLVED_CAP} (${unresolvedTotal["count"]} total)`);
    }
    return {
      total_count: recent.length,
      memories_by_type: byType,
      recent_memories: recent,
      recent_memories_total: recentTotal["count"],
      recent_memories_capped: recentCapped,
      unresolved_problems: unresolved,
      unresolved_problems_total: unresolvedTotal["count"],
      unresolved_problems_capped: unresolvedCapped,
      cap_message: capParts.length > 0 ? capParts.join("; ") : null,
      days,
      project,
    };
  }

  async getRelationshipsSince(since: Date): Promise<Relationship[]> {
    const sinceIso = since instanceof Date ? since.toISOString() : String(since);
    const rows = await this.db()`
      SELECT * FROM links
      WHERE recorded_at >= ${sinceIso} OR (valid_until IS NOT NULL AND valid_until >= ${sinceIso})
      ORDER BY recorded_at ASC, id`;
    return rows.map(rowToRelationship);
  }

  // -- Reindex --

  /**
   * Embed memories that have no vector (or every memory with `all`), in
   * batches. Stops at the first embedder failure and reports what it did.
   */
  async reindex(opts: { all?: boolean; batchSize?: number } = {}): Promise<{
    embedded: number;
    remaining: number;
    error: string | null;
  }> {
    const sql = this.db();
    const batchSize = Math.max(1, opts.batchSize ?? 16);
    let embedded = 0;
    let error: string | null = null;
    let afterId = "";
    for (;;) {
      const rows = opts.all
        ? await sql`SELECT id, title, content FROM memories WHERE id > ${afterId} ORDER BY id LIMIT ${batchSize}`
        : await sql`SELECT id, title, content FROM memories
            WHERE embedding IS NULL AND id > ${afterId} ORDER BY id LIMIT ${batchSize}`;
      if (rows.length === 0) break;
      afterId = rows[rows.length - 1]["id"] as string;
      const result = await this.tryEmbed(rows.map((r) => memoryEmbedText(r["title"] as string, r["content"] as string)));
      if (!result.vectors) {
        error = result.error;
        break;
      }
      await sql.begin(async (tx) => {
        for (let i = 0; i < rows.length; i++) {
          await tx`
            UPDATE memories SET embedding = ${vectorLiteral(result.vectors![i])}::${tx.unsafe(this.columnType)},
              embedding_model = ${this.embedder.model}, embedding_error = NULL
            WHERE id = ${rows[i]["id"] as string}`;
        }
      });
      embedded += rows.length;
    }
    const [left] = await sql`SELECT count(*)::int AS count FROM memories WHERE embedding IS NULL`;
    return { embedded, remaining: left["count"] as number, error };
  }
}

// ---------------------------------------------------------------------------
// SQL
// ---------------------------------------------------------------------------

const EMBEDDING_INDEX = "memories_embedding_hnsw_idx";

const schemaSql = (dimension: number): string => `
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
  id text PRIMARY KEY,
  type text NOT NULL,
  title text NOT NULL,
  content text NOT NULL,
  summary text,
  tags text[] NOT NULL DEFAULT '{}',
  importance double precision NOT NULL DEFAULT 0.5,
  confidence double precision NOT NULL DEFAULT 0.8,
  effectiveness double precision,
  usage_count integer NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  last_accessed timestamptz,
  version integer NOT NULL DEFAULT 1,
  updated_by text,
  context jsonb,
  embedding halfvec(${dimension}),
  embedding_model text,
  embedding_error text,
  search_vector tsvector GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, '') || ' ' || coalesce(summary, ''))
  ) STORED
);
CREATE INDEX IF NOT EXISTS memories_search_idx ON memories USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS memories_created_at_idx ON memories (created_at);
CREATE INDEX IF NOT EXISTS memories_type_idx ON memories (type);

CREATE TABLE IF NOT EXISTS links (
  id text PRIMARY KEY,
  from_id text NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  to_id text NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  rel_type text NOT NULL,
  strength double precision NOT NULL DEFAULT 0.5,
  confidence double precision NOT NULL DEFAULT 0.8,
  context text,
  evidence_count integer NOT NULL DEFAULT 1,
  success_rate double precision,
  created_at timestamptz NOT NULL,
  last_validated timestamptz NOT NULL,
  validation_count integer NOT NULL DEFAULT 0,
  counter_evidence_count integer NOT NULL DEFAULT 0,
  valid_from timestamptz NOT NULL,
  valid_until timestamptz,
  recorded_at timestamptz NOT NULL,
  invalidated_by text
);
CREATE INDEX IF NOT EXISTS links_from_idx ON links (from_id);
CREATE INDEX IF NOT EXISTS links_to_idx ON links (to_id);
CREATE INDEX IF NOT EXISTS links_type_idx ON links (rel_type);

CREATE TABLE IF NOT EXISTS duplicate_events (
  id bigserial PRIMARY KEY,
  memory_id text NOT NULL,
  match_id text NOT NULL,
  similarity double precision NOT NULL,
  threshold double precision NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
`;

const MEMORY_COLUMNS = `id, type, title, content, summary, tags, importance, confidence, effectiveness,
  usage_count, created_at, updated_at, last_accessed, version, updated_by, context`;

// ---------------------------------------------------------------------------
// Embedding column migration
// ---------------------------------------------------------------------------

async function readEmbeddingColumn(tx: Tx): Promise<EmbeddingColumn | null> {
  const [row] = await tx`
    SELECT t.typname AS type, a.atttypmod AS typmod, format_type(a.atttypid, a.atttypmod) AS formatted
    FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid
    WHERE a.attrelid = to_regclass('memories') AND a.attname = 'embedding' AND NOT a.attisdropped`;
  if (!row) return null;
  const typmod = Number(row["typmod"]);
  return {
    type: row["type"] as string,
    dimension: typmod > 0 ? typmod : null,
    formatted: row["formatted"] as string,
  };
}

async function buildEmbeddingPlan(tx: Tx, dimension: number, model: string): Promise<EmbeddingMigrationPlan> {
  const target = `halfvec(${dimension})`;
  const plan: EmbeddingMigrationPlan = {
    current: await readEmbeddingColumn(tx),
    target,
    action: "none",
    memories: 0,
    embedded: 0,
    otherModel: 0,
    model,
    statements: [],
  };
  const current = plan.current;
  if (!current) return plan;
  if (current.type !== "vector" && current.type !== "halfvec") {
    throw new ValidationError(`memories.embedding is ${current.formatted}; migrate embedding handles vector and halfvec only`);
  }

  const [counts] = await tx`
    SELECT count(*)::int AS memories, count(embedding)::int AS embedded,
           (count(*) FILTER (WHERE embedding IS NOT NULL AND embedding_model IS DISTINCT FROM ${model}))::int AS other_model
    FROM memories`;
  plan.memories = counts["memories"] as number;
  plan.embedded = counts["embedded"] as number;
  plan.otherModel = counts["other_model"] as number;

  const [index] = await tx`
    SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = ${EMBEDDING_INDEX}`;
  const indexReady = typeof index?.["indexdef"] === "string" && index["indexdef"].includes("halfvec_cosine_ops");
  const dropIndex = `DROP INDEX IF EXISTS ${EMBEDDING_INDEX}`;
  const createIndex = `CREATE INDEX ${EMBEDDING_INDEX} ON memories USING hnsw (embedding halfvec_cosine_ops)`;

  if (current.type === "halfvec" && current.dimension === dimension) {
    if (!indexReady) {
      plan.action = "create-index";
      plan.statements = [dropIndex, createIndex];
    }
  } else if (current.dimension === dimension) {
    plan.action = "cast";
    plan.statements = [
      dropIndex,
      `ALTER TABLE memories ALTER COLUMN embedding TYPE ${target} USING embedding::${target}`,
      createIndex,
    ];
  } else {
    plan.action = "retype";
    plan.statements = [
      dropIndex,
      `ALTER TABLE memories ALTER COLUMN embedding TYPE ${target} USING NULL::${target}`,
      "UPDATE memories SET embedding_model = NULL, embedding_error = NULL",
      createIndex,
    ];
  }
  return plan;
}

/** Human-readable plan for `migrate embedding`, stating whether it ran. */
export function formatEmbeddingPlan(plan: EmbeddingMigrationPlan, dryRun: boolean): string {
  if (!plan.current) {
    return `No memories table yet; the first command creates it with embedding ${plan.target}.`;
  }
  const lines = [
    `Embedding column: ${plan.current.formatted} -> ${plan.target}`,
    `Memories: ${plan.memories} (${plan.embedded} embedded)`,
  ];
  const describe: Record<EmbeddingMigrationPlan["action"], string> = {
    none: `nothing to do; the column is already ${plan.target} with a halfvec HNSW index`,
    "create-index": "rebuild the HNSW index with halfvec_cosine_ops",
    cast: "cast every vector in place and rebuild the HNSW index; vectors are kept",
    retype: `the dimension changes, so every embedding becomes NULL; run 'memorygraph reindex' afterwards to embed ${plan.memories} memories`,
  };
  lines.push(`Action: ${plan.action}: ${describe[plan.action]}`);
  for (const statement of plan.statements) lines.push(`  ${statement};`);
  if (plan.otherModel > 0 && plan.action !== "retype") {
    lines.push(
      `Note: ${plan.otherModel} memories were embedded with a model other than ${plan.model}; run 'memorygraph reindex --all' to re-embed them.`
    );
  }
  lines.push(dryRun ? "Dry run: nothing was changed." : plan.action === "none" ? "Nothing changed." : "Migrated.");
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Turn free text into an OR (or AND) of to_tsquery terms. Operator
 * characters are stripped first: to_tsquery rejects them inside a term
 * ("&&" crashed the LAB-351 bench).
 */
export function buildTsquery(text: string, joiner = " | "): string | null {
  const terms = text
    .replace(TSQUERY_UNSAFE, " ")
    .split(/\s+/)
    .map((t) => t.replace(/^[-.,;/]+|[-.,;/]+$/g, ""))
    .filter((t) => t.length > 0)
    .slice(0, MAX_QUERY_TERMS);
  return terms.length > 0 ? terms.join(joiner) : null;
}

/** Reciprocal-rank fusion over best-first rankings, ties broken by id. */
export function rrf(rankings: RankedId[][], k = RRF_K): RankedId[] {
  const scores = new Map<string, number>();
  for (const ranking of rankings) {
    ranking.forEach((r, rank) => scores.set(r.id, (scores.get(r.id) ?? 0) + 1 / (k + rank + 1)));
  }
  return [...scores.entries()]
    .map(([id, score]) => ({ id, score }))
    .sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

function recallFilters(
  filters: { memoryTypes?: string[]; projectPath?: string; excludeId?: string },
  params: unknown[]
): string[] {
  const out: string[] = [];
  if (filters.memoryTypes && filters.memoryTypes.length > 0) {
    params.push(filters.memoryTypes);
    out.push(`type = ANY($${params.length}::text[])`);
  }
  if (filters.projectPath) {
    params.push(filters.projectPath);
    out.push(`context->>'project_path' = $${params.length}`);
  }
  if (filters.excludeId) {
    params.push(filters.excludeId);
    out.push(`id <> $${params.length}`);
  }
  return out;
}

function filterClauses(q: SearchQuery, p: (v: unknown) => string): string[] {
  const where: string[] = [];
  if (q.memory_types?.length) where.push(`type = ANY(${p(q.memory_types)}::text[])`);
  if (q.tags?.length) where.push(`tags && ${p(q.tags.map((t) => t.toLowerCase().trim()))}::text[]`);
  if (q.project_path) where.push(`context->>'project_path' = ${p(q.project_path)}`);
  if (q.min_importance != null) where.push(`importance >= ${p(q.min_importance)}`);
  if (q.min_confidence != null) where.push(`confidence >= ${p(q.min_confidence)}`);
  if (q.min_effectiveness != null) where.push(`effectiveness >= ${p(q.min_effectiveness)}`);
  if (q.created_after) where.push(`created_at >= ${p(toIso(q.created_after))}`);
  if (q.created_before) where.push(`created_at <= ${p(toIso(q.created_before))}`);
  return where;
}

function escapeLike(value: string): string {
  return value.replace(/[\\%_]/g, (c) => `\\${c}`);
}

function toIso(value: string | Date): string {
  return value instanceof Date ? value.toISOString() : value;
}

function isoOrUndefined(value: unknown): string | undefined {
  if (value === null || value === undefined) return undefined;
  return value instanceof Date ? value.toISOString() : String(value);
}

function numOrNull(value: unknown): number | null {
  return value === null || value === undefined ? null : Number(value);
}

function redact(err: unknown): string {
  return String(err instanceof Error ? err.message : err).replace(/postgres(ql)?:\/\/[^@\s]*@/g, "postgres://***@");
}

function rowToMemory(row: Record<string, unknown>): Memory | null {
  try {
    return createMemory({
      id: row["id"] as string,
      type: row["type"] as string,
      title: row["title"] as string,
      content: row["content"] as string,
      summary: (row["summary"] as string) ?? undefined,
      tags: (row["tags"] as string[]) ?? [],
      importance: Number(row["importance"] ?? 0.5),
      confidence: Number(row["confidence"] ?? 0.8),
      effectiveness: row["effectiveness"] === null || row["effectiveness"] === undefined ? null : Number(row["effectiveness"]),
      usage_count: Number(row["usage_count"] ?? 0),
      created_at: isoOrUndefined(row["created_at"]),
      updated_at: isoOrUndefined(row["updated_at"]),
      last_accessed: isoOrUndefined(row["last_accessed"]),
      version: Number(row["version"] ?? 1),
      updated_by: (row["updated_by"] as string) ?? undefined,
      context: (row["context"] as Record<string, unknown>) ?? undefined,
    });
  } catch (err) {
    console.error(`Failed to parse memory row: ${err}`);
    return null;
  }
}

function rowToRelationship(row: Record<string, unknown>): Relationship {
  return {
    id: row["id"] as string,
    from_memory_id: row["from_id"] as string,
    to_memory_id: row["to_id"] as string,
    type: row["rel_type"] as string,
    properties: createRelationshipProperties({
      strength: Number(row["strength"] ?? 0.5),
      confidence: Number(row["confidence"] ?? 0.8),
      context: (row["context"] as string) ?? undefined,
      evidence_count: Number(row["evidence_count"] ?? 1),
      success_rate: row["success_rate"] === null ? undefined : Number(row["success_rate"]),
      created_at: isoOrUndefined(row["created_at"]),
      last_validated: isoOrUndefined(row["last_validated"]),
      validation_count: Number(row["validation_count"] ?? 0),
      counter_evidence_count: Number(row["counter_evidence_count"] ?? 0),
      valid_from: isoOrUndefined(row["valid_from"]),
      valid_until: isoOrUndefined(row["valid_until"]),
      recorded_at: isoOrUndefined(row["recorded_at"]),
      invalidated_by: (row["invalidated_by"] as string) ?? undefined,
    }),
    description: undefined,
    bidirectional: false,
  };
}
