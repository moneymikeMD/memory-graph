# Postgres backend

`MEMORY_BACKEND=postgres` stores memories in Postgres with pgvector. It is a
full backend: store, get, update, delete, link, related, search, recall,
activity, stats, export, import and migrate all work. Like `sqlite` it is not
Cypher-capable, so the intelligence, analytics, proactive and temporal
commands (including `briefing`) print their unsupported message instead of
running.

Recall is the `postgres_hybrid` design benchmarked in homelab LAB-345/LAB-348
(`scripts/bench/memory/adapters/postgres_hybrid`): a tsvector full-text ranking
and a pgvector cosine ranking fused by reciprocal-rank fusion (k = 60).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MEMORY_BACKEND` | | `postgres` |
| `MEMORY_POSTGRES_URL` | | Connection URL. If unset, the four variables below are used. |
| `MEMORY_POSTGRES_HOST` | `localhost` | |
| `MEMORY_POSTGRES_PORT` | `5432` | |
| `MEMORY_POSTGRES_DB` | `memorygraph` | |
| `MEMORY_POSTGRES_USER` | `memorygraph` | |
| `MEMORY_POSTGRES_PASSWORD` | | Read from the environment only. There is no flag for it. |
| `MEMORY_EMBED_URL` | | Ollama base URL, for example `http://embedder:11434`. Unset means no vectors and full-text-only recall. |
| `MEMORY_EMBED_MODEL` | `qwen3-embedding:0.6b` | |
| `MEMORY_EMBED_DIGEST` | `ac6da0df…15621d` | The model digest the benchmark used. It is checked against `/api/tags` before the first embed in each process. |
| `MEMORY_EMBED_TIMEOUT_MS` | `30000` | |
| `MEMORY_DUPLICATE_THRESHOLD` | `0.86` | Cosine at which `store` warns about a possible duplicate. |

Connection messages and `stats` name the server as `host:port/database`. They
never print the password.

## Schema

`initializeSchema` runs on every CLI invocation. It is idempotent and holds a
transaction-scoped advisory lock, so concurrent CLI calls (for example from
hooks) do not race each other's `CREATE ... IF NOT EXISTS`.

- `memories` holds every `Memory` field plus `embedding vector(1024)`,
  `embedding_model` and `embedding_error`, and a stored generated
  `search_vector` (`to_tsvector('english', title || content || summary)`). It
  is indexed with GIN (full-text) and HNSW (`vector_cosine_ops`). This matches
  the bench's `sql/bench-memory/schema.sql`.
- `links` holds relationships: type, strength, confidence and the bi-temporal
  fields. Deleting a memory cascades to its links.
- `duplicate_events` has one row per duplicate warning:
  `(memory_id, match_id, similarity, threshold, created_at)`.

## Embedding

The embedded text is `title + "\n\n" + content`, the same text the benchmark's
`backfill_embeddings.py` embedded. When the embedder is unset, unreachable,
serving a different digest, or returning the wrong dimension:

- `store` and `update` still write the memory, with `embedding` NULL and the
  reason in `embedding_error`. A line on stderr says to run `reindex`.
- The first failure latches for the rest of the process, so an import against
  a down embedder costs one timeout, not one per memory.
- `memorygraph reindex` embeds every memory whose vector is NULL, in batches
  (`--batch-size`, default 16). `--all` re-embeds everything, for example
  after a model change. `stats` reports how many memories are still missing a
  vector.

## Recall

1. The full-text arm splits the query into terms. It strips the tsquery
   operator characters, because to_tsquery rejects them inside a term and the
   `&&` in `lint-cmd` crashed the LAB-351 bench. It ORs the terms into
   `to_tsquery('english', …)` and ranks the matches by `ts_rank`. The `english`
   configuration lower-cases terms, so this arm ignores case.
2. The vector arm embeds the query and ranks by cosine. It runs with
   `SET LOCAL hnsw.ef_search` raised to the over-fetch size, because the
   default of 40 would cap the arm below its 50-row over-fetch.
3. Both arms over-fetch `max(4 × limit, 50)` rows and are fused by RRF. Ties
   are broken by id, so results are deterministic.

The query is lower-cased before it is embedded. That makes the whole ranking
independent of query case: `recall --query 'FALKORDB EVICTION'` returns the
same ids, in the same order, as the lower-case query. The benchmark embedded
the raw query and measured recall as unchanged by case. Lower-casing turns
"recall unchanged" into "identical results".

When the query cannot be embedded, recall uses the full-text arm alone and
says so once on stderr: `recall is full-text only: the query could not be
embedded (<reason>)`.

## Duplicate warning

`memorygraph store` (only the CLI `store` command, not import or migrate)
embeds the new memory, then runs the hybrid search with the memory's title
and content as the query. It excludes the memory itself and takes the top 5
fused results. If the highest cosine among those results reaches the
threshold, it prints the following and logs a `duplicate_events` row:

```
memorygraph: possible duplicate of <id> "<title>" (cosine 0.912); stored anyway
```

The memory is stored either way. When the new memory has no embedding, the
check is skipped, and stderr says so.

### Calibration

The threshold is calibrated against the LAB-354 audit
(homelab `docs/bench/memory/baseline.md`). The audit uses TF-IDF cosine
≥ 0.65 on the frozen 1,469-memory corpus. It flagged 44 memories (3.0%) as
having an earlier near-duplicate, from 47 candidate pairs.
`calibrate_duplicate_threshold.py` re-derives those pairs with the audit's own
`find_candidates` and measures their embedding cosine. It then counts how many
corpus memories each threshold would flag against their nearest earlier
memory:

Measured 2026-09-28 against a loopback copy of the corpus, embedded with
`qwen3-embedding:0.6b` at the pinned digest:

- The 47 audit pairs have embedding cosines of min 0.761, p10 0.800, p25
  0.865, median 0.892 and max 0.997.
- `qwen3-embedding` scores this single-domain corpus high across the board.
  At 0.80, 22.7% of memories have an earlier memory above the line.

| Threshold | Memories flagged | Rate | Audit-flagged memories recovered |
|---|---|---|---|
| 0.80 | 334 | 22.7% | 43/44 |
| 0.83 | 194 | 13.2% | 39/44 |
| **0.86** | **98** | **6.7%** | **37/44** |
| 0.88 | 60 | 4.1% | 28/44 |
| 0.89 | 39 | 2.7% | 24/44 |
| 0.90 | 28 | 1.9% | 19/44 |

**0.86 is the default.** The choice rests on samples of nearest pairs that the
audit did not flag, judged by eye:

- From 0.80 to 0.83, most pairs cover the same area but state different facts,
  for example two Atlassian GraphQL notes or two skill comparisons.
- From 0.83 to 0.86, the pairs are mixed.
- At 0.86 and above, most pairs restate or correct the same fact, for example
  the two `land-branch.sh` `origin/main` notes, the two `decisions.sh`
  cwd-resolution notes, and "Owner reversed the WO-053 Open binding" against
  "WO-053 CORRECTED".

At 0.86 the warning catches 37 of the audit's 44 memories. It flags about
twice the audit's rate, and the sampled extras are paraphrases that TF-IDF
misses. Matching the audit's 3.0% rate exactly would need 0.89, which drops
recovery to 24/44. Override the threshold with `MEMORY_DUPLICATE_THRESHOLD`.

Reproduce the measurement with:

```bash
python3 docs/postgres/calibrate_duplicate_threshold.py \
  --corpus ~/code/home_workspace/homelab/docs/bench/memory/corpus/memories.json \
  --bench-dir ~/code/home_workspace/homelab/scripts/bench/memory \
  --psql "docker exec -i <loopback-container> psql -U memorygraph -d <db>"
```

## Migration

`memorygraph migrate --from falkordb` copies the FalkorDB store named by
`MEMORY_FALKORDB_*` into the configured backend (`MEMORY_BACKEND=postgres`). It
uses the existing export, import and verify path: it counts both sides and
compares a sample of 10 memories' content, and it rolls back on a mismatch.
Each imported memory is embedded as it is written. The verify step requires
equal counts, so the target must start empty. `--dry-run` validates both ends
and exports without writing. `--to <backend>` still works as before; with
`--to`, the target's connection settings now come from the environment unless
`--to-path` or `--to-uri` is given.

## Testing

`ts/tests/postgres-backend.test.ts` runs only when
`MEMORYGRAPH_TEST_POSTGRES_URL` is set, and it refuses any host that is not
loopback. Each test creates its own database and drops it afterwards.
Embeddings come from a stub Ollama server on 127.0.0.1. The suite needs a
role that can create databases.

```bash
docker run -d --name mg-pg --env-file pg.env -p 127.0.0.1:55432:5432 \
  pgvector/pgvector:pg17@sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f
# pg.env: POSTGRES_USER=memorygraph, POSTGRES_DB=memorygraph, POSTGRES_PASSWORD=<generated>
export MEMORY_POSTGRES_PASSWORD=<same>
MEMORYGRAPH_TEST_POSTGRES_URL=postgres://memorygraph@127.0.0.1:55432/memorygraph bun test
```

Run the suite with any production `MEMORY_BACKEND` / `MEMORY_FALKORDB_*`
variables unset. Several existing tests spawn the CLI with the parent
environment, and `never-throw-sweep`'s `migrate --to sqlite --dry-run` connects
to whatever FalkorDB `MEMORY_FALKORDB_HOST` names.
