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
| `MEMORY_EMBED_DIMENSION` | `1024` | The model's output dimension, 1 to 4000 (pgvector's HNSW limit for `halfvec`). It must match the `embedding` column; see Changing the embedding model. |
| `MEMORY_EMBED_TIMEOUT_MS` | `30000` | |
| `MEMORY_DUPLICATE_THRESHOLD` | `0.86` | Cosine at which `store` warns about a possible duplicate. |

Connection messages and `stats` name the server as `host:port/database`. They
never print the password.

## Schema

`initializeSchema` runs on every CLI invocation. It is idempotent and holds a
transaction-scoped advisory lock, so concurrent CLI calls (for example from
hooks) do not race each other's `CREATE ... IF NOT EXISTS`.

- `memories` holds every `Memory` field plus `embedding halfvec(N)`, where N
  is `MEMORY_EMBED_DIMENSION`, `embedding_model` and `embedding_error`, and a
  stored generated `search_vector`
  (`to_tsvector('english', title || content || summary)`). It is indexed with
  GIN (full-text) and HNSW (`halfvec_cosine_ops`). The bench's
  `sql/bench-memory/schema.sql` used `vector(1024)`. `halfvec` stores 16-bit
  floats, so models over `vector`'s 2,000-dimension index limit can be
  indexed, for example `qwen3-embedding:4b` at 2,560.
- At every start, `initializeSchema` checks the `embedding` column's
  dimension against `MEMORY_EMBED_DIMENSION` and fails on a mismatch:
  `memories.embedding is halfvec(1024) but MEMORY_EMBED_DIMENSION is 2560;
  run 'memorygraph migrate embedding --dry-run' to see the migration plan`.
  A store created before `halfvec` (a `vector(1024)` column) keeps working at
  the same dimension until it is migrated.
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
   operator characters, ORs the terms into `to_tsquery('english', …)` and
   ranks the matches by `ts_rank`. The `english` configuration lower-cases
   terms, so this arm ignores case on its own.
2. The vector arm embeds the lower-cased query and ranks by cosine.
3. Both arms over-fetch `max(4 × limit, 50)` rows and are fused by RRF.

When the query cannot be embedded, recall uses the full-text arm alone and
says so once on stderr: `recall is full-text only: the query could not be
embedded (<reason>)`.

### Deviations

The first five items are differences from the benchmark adapter (homelab
`scripts/bench/memory/adapters/postgres_hybrid`). The last is a difference
from upstream memorygraph's `migrate`. Items marked *owner-acknowledged* were
reviewed and kept by the owner on 2026-09-28.

- **The query is lower-cased before it is embedded** (*owner-acknowledged*). The benchmark embedded
  the raw query and measured recall as unchanged by case. Full-text alone
  already returns the same set for either case, because `to_tsquery`
  lower-cases, but the vector arm does not, so the fused order could differ.
  Lower-casing makes `recall --query 'FALKORDB EVICTION'` return the same ids
  in the same order as the lower-case query. `ts/tests/postgres-backend.test.ts`
  pins the string sent to `/api/embed`, and its stub embedder is
  case-sensitive.
- **`hnsw.ef_search` is raised to the over-fetch size** (`SET LOCAL`, capped
  at 1000). At the corpus's 1,469 rows the planner does not use the HNSW
  index: `EXPLAIN` shows Seq Scan plus Sort, which is exact. So the benchmark
  did not run under the default `ef_search` cap of 40. The raise matters
  once the index is used on a larger store. With `enable_seqscan = off`, the
  default returned 40 rows for a `LIMIT 50` (measured 2026-09-28). In the spec
  review, the top 5 changed on 13 of 400 proxy queries.
- **The full-text query is capped at 32 terms** (`MAX_QUERY_TERMS`). In the
  spec review, the measured queries had at most 17 terms, so the cap does not
  bind on them. It bounds `store`'s duplicate search, which uses the whole
  title and content as its query.
- **The tsquery sanitiser is stricter.** The benchmark stripped
  `& | ! ( ) < > \`, because `to_tsquery` rejects them inside a term and the
  `&&` in `lint-cmd` crashed the LAB-351 bench. This backend also strips
  `: * ' " @ ~ ^ { } [ ]` and trims `- . , ; /` from term ends, so no
  user-supplied text is parsed as tsquery syntax (weights, prefixes, quoting).
- **RRF ties are broken by id.** The benchmark kept Python's stable sort,
  which is full-text-first insertion order. The id tie-break makes the order
  independent of which arm returned a row first. In the spec review, on 400
  proxy queries, the top-5 set differed from the benchmark's on 5 and the
  order on 32.
- **`migrate --to <backend>` falls back to the environment**
  (*owner-acknowledged*). Upstream builds the target from the flags alone.
  Here, when neither `--to-path` nor `--to-uri` is given, the target's
  settings come from the environment, the same way the source's do. For
  example, `--to sqlite` uses `MEMORY_SQLITE_PATH` or the default store path.
  A postgres target always takes its password from
  `MEMORY_POSTGRES_PASSWORD`, even when `--to-uri` is given, so the password
  never goes on the command line. The `migrate` case in `never-throw-sweep`
  is sealed to loopback, because this fallback lets a dry run get as far as
  exporting from the source (see Testing).

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
and exports without writing. With `--to <backend>` and no `--to-path` or
`--to-uri`, the target's settings come from the environment (see Deviations).

## Changing the embedding model

`memorygraph migrate embedding` moves an existing store's `embedding` column
to `halfvec(N)`, where N is `MEMORY_EMBED_DIMENSION`. `--dry-run` prints the
plan and changes nothing: it runs only reads, inside a `READ ONLY`
transaction, and skips `initializeSchema`. Without `--dry-run`, it runs the
plan in one transaction under the schema advisory lock.

| Current column | Action | What happens |
|---|---|---|
| `vector(N)`, same N | `cast` | Drops the HNSW index, runs `ALTER COLUMN embedding TYPE halfvec(N) USING embedding::halfvec(N)`, and rebuilds the index. Every vector is kept, rounded to 16-bit floats. |
| any other dimension | `retype` | Drops the index, retypes the column with every embedding NULL, clears `embedding_model`, and rebuilds the index. Run `memorygraph reindex` afterwards to embed every memory with the new model. |
| `halfvec(N)` without a `halfvec_cosine_ops` index | `create-index` | Rebuilds the index. |
| `halfvec(N)` with the index | `none` | Nothing. The command is idempotent. |

If the model changes but the dimension does not, `cast` keeps the old
model's vectors. The plan counts the memories embedded with a model other
than `MEMORY_EMBED_MODEL` and says to run `memorygraph reindex --all`.

To move a store to a new model:

```bash
MEMORY_EMBED_MODEL=<model> MEMORY_EMBED_DIGEST=<digest> MEMORY_EMBED_DIMENSION=<N> \
  memorygraph migrate embedding --dry-run     # read the plan
# take a backup, then run the same command without --dry-run
MEMORY_EMBED_MODEL=<model> MEMORY_EMBED_DIGEST=<digest> MEMORY_EMBED_DIMENSION=<N> \
  memorygraph reindex                          # after a retype; --all after a same-dimension model change
```

Every client of the store needs the same three variables. A client left on
the old dimension fails at startup with the mismatch error.

## Testing

`ts/tests/postgres-backend.test.ts` runs only when
`MEMORYGRAPH_TEST_POSTGRES_URL` is set, and it refuses any host that is not
loopback. When the variable is unset, the suite prints a `SKIPPING the
Postgres integration suite` banner. `bun run test:postgres` sets
`MEMORYGRAPH_REQUIRE_POSTGRES_TESTS=1`, which turns a missing URL into a
failing test, so the exit status shows whether the suite ran. Each test
creates its own database and drops it afterwards. Embeddings come from a
case-sensitive stub Ollama server on 127.0.0.1. The suite needs a role that
can create databases.

```bash
docker run -d --name mg-pg --env-file pg.env -p 127.0.0.1:55432:5432 \
  pgvector/pgvector:pg17@sha256:cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f
# pg.env: POSTGRES_USER=memorygraph, POSTGRES_DB=memorygraph, POSTGRES_PASSWORD=<generated>
env $(env | awk -F= '/^MEMORY_/{printf "-u %s ", $1}') \
  MEMORY_POSTGRES_PASSWORD=<same> \
  MEMORYGRAPH_TEST_POSTGRES_URL=postgres://memorygraph@127.0.0.1:55432/memorygraph \
  bun run test:postgres
```

Run the suite with every inherited `MEMORY_*` variable dropped, not a named
list. Several tests spawn the CLI with the parent environment and set only
`MEMORY_BACKEND` themselves. An inherited `MEMORY_FALKORDBLITE_PATH` alone
fails four `store-path.test.ts` cases (VAL-LOCAL-007, 008 and 009), because
it overrides `--store` and the cwd-relative default. The `env $(...)` form
needs a shell that word-splits, such as bash or zsh with the substitution
unquoted; a zsh `$VAR` holding the list does not split. `never-throw-sweep` drops
every inherited `MEMORY_*` variable and points FalkorDB at 127.0.0.1:1,
because its `migrate` case builds the migration source from the environment.
