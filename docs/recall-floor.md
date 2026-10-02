# Recall relevance floor (LAB-426)

Postgres hybrid recall used to fill `--limit` for every query. It fused the
nearest vectors and every memory matching any one query term by reciprocal
rank, and a reciprocal-rank score says nothing about relevance. A query about
a topic that was never stored returned a full page of unrelated memories.

Recall now applies a floor to the raw signals. A candidate survives when
either of these holds:

- its cosine similarity to the query is at least `MEMORY_RECALL_SIMILARITY_FLOOR` (default **0.58**), or
- its full-text coverage is at least `MEMORY_RECALL_FULLTEXT_FLOOR` (default **0.5**).

Full-text coverage is the share of the query's lexemes that the memory
contains, each lexeme weighted by its BM25 inverse document frequency
`ln(1 + (N - df + 0.5) / (df + 0.5))`. It runs from 0 to 1. A match on a rare
term counts for more than a match on a common one.

Both defaults were measured, on 2026-10-02, as described below. All numbers
in this document are measured unless a sentence says otherwise.

## Result

Run through the backend's own `recallMemories` (`evaluate.ts`), limit 5:

| arm | negatives returning 0 | negatives returning a full page | recall@5 hooks | recall@5 model queries | recall@5 all | abstain queries returning 0 |
|---|---|---|---|---|---|---|
| hybrid, default floors | 45/48 (93.8%) | 0/48 | 0.9774 | 0.9337 | 0.9446 | 10/16 |
| hybrid, floors 0 | 0/48 (0.0%) | 48/48 | 0.9613 | 0.9480 | 0.9513 | 0/16 |
| full-text only, default floors | 48/48 (100.0%) | 0/48 | 0.8161 | 0.8262 | 0.8237 | 13/16 |
| full-text only, floors 0 | 10/48 (20.8%) | 36/48 | 0.8269 | 0.8674 | 0.8573 | 0/16 |

Against the two LAB-426 targets, in hybrid mode:

- **At least 90% of negative queries return nothing.** 45 of 48 (93.8%) return
  nothing and none returns a full page. With both floors at 0, all 48 return a
  full page.
- **Gold-set recall@5 drops by no more than 2 percentage points.** Over the 31
  scored `goldset.jsonl` hooks it rises from 0.9613 to 0.9774 (+1.6 points),
  because dropping irrelevant candidates lets a gold memory move into the top
  5. Over the 93 scored model-written queries it falls from 0.9480 to 0.9337
  (-1.4 points). Over all 124 scored queries it falls from 0.9513 to 0.9446
  (-0.7 points).

The floor adds one SQL statement per recall. On the scratch store, recall p50
went from 40.8 ms to 42.7 ms and p95 from 44.4 ms to 47.9 ms (105 recalls per
arm, embedder on the same machine).

## Method

- **Corpus.** homelab `docs/bench/memory/corpus/memories.json` (sha256
  `1c6e6a10…`, 1,469 memories, homelab commit `2f40bf5`), imported with this
  fork's `memorygraph import` into a scratch `pgvector/pgvector:0.8.6-pg17`
  container bound to 127.0.0.1 (PostgreSQL 17.11, pgvector 0.8.6). The
  embedding column is `halfvec(1024)`.
- **Embedder.** `qwen3-embedding:0.6b`, digest `ac6da0df…15621d`, on the local
  Ollama. All 1,469 memories embedded.
- **Positive queries.** The 35 `goldset.jsonl` hooks and the 105
  `queries_model.jsonl` queries. Each runs with its gold item's `cutoff`, as
  the homelab harness does: only memories with `created_at <= cutoff` are
  candidates. Four gold items are `abstain` items with no gold memory; their
  16 queries score no recall and are reported separately. Two model queries
  are empty strings and score 0 with or without a floor. That leaves 31
  scored hooks and 93 scored model queries. The harness also sends each hook
  upper-cased and lower-cased; recall is case-insensitive in both arms, so
  those variants are identical and are not repeated here.
- **Negative queries.** The 48 queries in
  [`recall-floor/negative-queries.jsonl`](recall-floor/negative-queries.jsonl),
  written for this ticket: 32 plain questions on topics the corpus cannot
  contain (cooking, zoology, history, sport), 6 two-word queries, and 10
  homonym queries that reuse a technical word in a non-technical sense
  ("python snake feeding schedule", "container ship cargo capacity"). They
  run against the whole corpus with no cutoff.
- **Signals.** `ts/tests/recall-floor/measure.ts` replays the backend's ranking
  SQL for each query, with the cutoff added and an exact vector scan, and
  records the signals of every fused candidate (up to 100) and of every gold
  memory. `analyze.ts` applies candidate floors to those records and
  produces the tables below. It simulates the floor; `evaluate.ts` then checks
  the chosen floors against the real code path.
- **End-to-end check.** `evaluate.ts` calls `recallMemories` itself. Since
  `recallMemories` has no cutoff argument, it copies the database, walks the
  gold items in descending cutoff order and deletes the memories newer than
  each cutoff before running that item's queries. The hybrid rows of its table
  equal the simulation exactly. The full-text-only rows differ from the
  simulation by about one point, because the copy's document frequencies
  shrink as memories are deleted.

Both scripts refuse to run when the Postgres host is not loopback.

## Distributions

Cosine similarity, three full-text signals. "Unweighted coverage" is the plain
share of query lexemes matched; `ts_rank` is what the full-text arm sorts by.

| signal | population | n | min | p5 | p10 | p25 | p50 | p75 | p90 | p95 | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cosine similarity | gold memories, hook queries | 49 | 0.565 | 0.590 | 0.608 | 0.667 | 0.713 | 0.774 | 0.820 | 0.835 | 0.873 |
| cosine similarity | gold memories, model queries | 147 | 0.208 | 0.536 | 0.591 | 0.653 | 0.704 | 0.747 | 0.796 | 0.843 | 0.915 |
| cosine similarity | non-gold in the unfloored top 5, hook and model queries | 518 | 0.390 | 0.459 | 0.484 | 0.535 | 0.593 | 0.668 | 0.720 | 0.754 | 0.885 |
| cosine similarity | unfloored top 5, negative queries | 240 | 0.096 | 0.188 | 0.229 | 0.340 | 0.404 | 0.464 | 0.512 | 0.546 | 0.622 |
| cosine similarity | best candidate per negative query | 48 | 0.343 | 0.366 | 0.394 | 0.423 | 0.472 | 0.512 | 0.560 | 0.600 | 0.622 |
| IDF-weighted coverage | gold memories, hook queries | 49 | 0.339 | 0.428 | 0.533 | 0.669 | 0.809 | 1.000 | 1.000 | 1.000 | 1.000 |
| IDF-weighted coverage | gold memories, model queries | 147 | 0.000 | 0.340 | 0.528 | 0.715 | 0.816 | 0.959 | 1.000 | 1.000 | 1.000 |
| IDF-weighted coverage | non-gold in the unfloored top 5, hook and model queries | 518 | 0.000 | 0.104 | 0.135 | 0.318 | 0.500 | 0.723 | 0.885 | 1.000 | 1.000 |
| IDF-weighted coverage | unfloored top 5, negative queries | 240 | 0.000 | 0.000 | 0.000 | 0.000 | 0.084 | 0.159 | 0.222 | 0.253 | 0.386 |
| IDF-weighted coverage | best candidate per negative query | 48 | 0.000 | 0.000 | 0.000 | 0.128 | 0.205 | 0.257 | 0.313 | 0.327 | 0.386 |
| unweighted coverage | gold memories, hook queries | 49 | 0.286 | 0.429 | 0.556 | 0.667 | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 |
| unweighted coverage | gold memories, model queries | 147 | 0.000 | 0.429 | 0.600 | 0.714 | 0.833 | 0.889 | 1.000 | 1.000 | 1.000 |
| unweighted coverage | non-gold in the unfloored top 5, hook and model queries | 518 | 0.000 | 0.200 | 0.250 | 0.400 | 0.571 | 0.750 | 0.857 | 1.000 | 1.000 |
| unweighted coverage | unfloored top 5, negative queries | 240 | 0.000 | 0.000 | 0.000 | 0.000 | 0.167 | 0.250 | 0.333 | 0.400 | 0.500 |
| unweighted coverage | best candidate per negative query | 48 | 0.000 | 0.000 | 0.000 | 0.200 | 0.333 | 0.400 | 0.500 | 0.500 | 0.500 |
| ts_rank | gold memories, hook queries | 49 | 0.022 | 0.030 | 0.041 | 0.054 | 0.063 | 0.074 | 0.079 | 0.084 | 0.087 |
| ts_rank | gold memories, model queries | 147 | 0.000 | 0.033 | 0.043 | 0.054 | 0.064 | 0.074 | 0.083 | 0.087 | 0.092 |
| ts_rank | non-gold in the unfloored top 5, hook and model queries | 518 | 0.000 | 0.014 | 0.021 | 0.030 | 0.042 | 0.058 | 0.069 | 0.076 | 0.088 |
| ts_rank | unfloored top 5, negative queries | 240 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 | 0.018 | 0.024 | 0.030 | 0.034 |
| ts_rank | best candidate per negative query | 48 | 0.000 | 0.000 | 0.000 | 0.012 | 0.021 | 0.027 | 0.033 | 0.034 | 0.037 |

Reading the table:

- Similarity alone does not separate the populations. The best candidate for a
  negative query reaches 0.622, while 10% of gold memories sit below 0.61 on
  hook queries and 0.59 on model queries. A similarity-only floor at 0.58 costs
  7.9 points of recall on model queries (third table below).
- IDF-weighted coverage separates better. No negative query has a candidate
  above 0.386, and 90% of gold memories are above 0.53.
- The two signals fail on different memories. Of the 196 gold rows, 170 clear
  both default floors, 10 clear only the similarity floor, 10 clear only the
  coverage floor and 6 clear neither. That is why the rule is an OR.

## Trade-off curve

Simulated from the recorded signals, limit 5. "Meets both targets" means at
least 90% of negatives return nothing, none returns a full page, and recall@5
is within 2 points of the unfloored value on hooks and on model queries
separately.

| similarity / coverage floor | recall@5 hooks (pp vs no floor) | recall@5 model queries (pp vs no floor) | negatives returning 0 | negatives returning a full page | abstain queries returning 0 | meets both targets |
|---|---|---|---|---|---|---|
| 0 / 0 (no floor) | 0.9613 (+0.0) | 0.9480 (+0.0) | 0/48 (0.0%) | 48 | 0/16 | no |
| 0.50 / 0.40 | 0.9613 (+0.0) | 0.9480 (+0.0) | 33/48 (68.8%) | 9 | 3/16 | no |
| 0.50 / 0.50 | 0.9613 (+0.0) | 0.9480 (+0.0) | 33/48 (68.8%) | 9 | 3/16 | no |
| 0.50 / 0.60 | 0.9613 (+0.0) | 0.9480 (+0.0) | 33/48 (68.8%) | 9 | 3/16 | no |
| 0.50 / 0.70 | 0.9613 (+0.0) | 0.9480 (+0.0) | 33/48 (68.8%) | 9 | 3/16 | no |
| 0.52 / 0.40 | 0.9613 (+0.0) | 0.9480 (+0.0) | 38/48 (79.2%) | 5 | 5/16 | no |
| 0.52 / 0.50 | 0.9774 (+1.6) | 0.9480 (+0.0) | 38/48 (79.2%) | 5 | 5/16 | no |
| 0.52 / 0.60 | 0.9774 (+1.6) | 0.9373 (-1.1) | 38/48 (79.2%) | 5 | 5/16 | no |
| 0.52 / 0.70 | 0.9774 (+1.6) | 0.9373 (-1.1) | 38/48 (79.2%) | 5 | 5/16 | no |
| 0.54 / 0.40 | 0.9613 (+0.0) | 0.9444 (-0.4) | 39/48 (81.3%) | 3 | 8/16 | no |
| 0.54 / 0.50 | 0.9774 (+1.6) | 0.9444 (-0.4) | 39/48 (81.3%) | 3 | 8/16 | no |
| 0.54 / 0.60 | 0.9774 (+1.6) | 0.9337 (-1.4) | 39/48 (81.3%) | 3 | 8/16 | no |
| 0.54 / 0.70 | 0.9774 (+1.6) | 0.9337 (-1.4) | 39/48 (81.3%) | 3 | 8/16 | no |
| 0.56 / 0.40 | 0.9613 (+0.0) | 0.9337 (-1.4) | 41/48 (85.4%) | 2 | 10/16 | no |
| 0.56 / 0.50 | 0.9774 (+1.6) | 0.9337 (-1.4) | 41/48 (85.4%) | 2 | 10/16 | no |
| 0.56 / 0.60 | 0.9774 (+1.6) | 0.9229 (-2.5) | 41/48 (85.4%) | 2 | 10/16 | no |
| 0.56 / 0.70 | 0.9774 (+1.6) | 0.9229 (-2.5) | 41/48 (85.4%) | 2 | 10/16 | no |
| 0.58 / 0.40 | 0.9613 (+0.0) | 0.9337 (-1.4) | 45/48 (93.8%) | 0 | 10/16 | yes |
| 0.58 / 0.50 | 0.9774 (+1.6) | 0.9337 (-1.4) | 45/48 (93.8%) | 0 | 10/16 | yes |
| 0.58 / 0.60 | 0.9774 (+1.6) | 0.9229 (-2.5) | 45/48 (93.8%) | 0 | 11/16 | no |
| 0.58 / 0.70 | 0.9839 (+2.3) | 0.9229 (-2.5) | 45/48 (93.8%) | 0 | 11/16 | no |
| 0.60 / 0.40 | 0.9290 (-3.2) | 0.9337 (-1.4) | 46/48 (95.8%) | 0 | 10/16 | no |
| 0.60 / 0.50 | 0.9452 (-1.6) | 0.9337 (-1.4) | 46/48 (95.8%) | 0 | 13/16 | yes |
| 0.60 / 0.60 | 0.9452 (-1.6) | 0.9122 (-3.6) | 46/48 (95.8%) | 0 | 15/16 | no |
| 0.60 / 0.70 | 0.9516 (-1.0) | 0.9086 (-3.9) | 46/48 (95.8%) | 0 | 15/16 | no |
| 0.62 / 0.40 | 0.9290 (-3.2) | 0.9337 (-1.4) | 46/48 (95.8%) | 0 | 10/16 | no |
| 0.62 / 0.50 | 0.9452 (-1.6) | 0.9337 (-1.4) | 46/48 (95.8%) | 0 | 13/16 | yes |
| 0.62 / 0.60 | 0.9129 (-4.8) | 0.9122 (-3.6) | 46/48 (95.8%) | 0 | 16/16 | no |
| 0.62 / 0.70 | 0.9194 (-4.2) | 0.9086 (-3.9) | 46/48 (95.8%) | 0 | 16/16 | no |
| 0.64 / 0.40 | 0.9290 (-3.2) | 0.9194 (-2.9) | 48/48 (100.0%) | 0 | 10/16 | no |
| 0.64 / 0.50 | 0.9452 (-1.6) | 0.9194 (-2.9) | 48/48 (100.0%) | 0 | 13/16 | no |
| 0.64 / 0.60 | 0.9022 (-5.9) | 0.8978 (-5.0) | 48/48 (100.0%) | 0 | 16/16 | no |
| 0.64 / 0.70 | 0.9086 (-5.3) | 0.8943 (-5.4) | 48/48 (100.0%) | 0 | 16/16 | no |
| 0.66 / 0.40 | 0.9290 (-3.2) | 0.9194 (-2.9) | 48/48 (100.0%) | 0 | 10/16 | no |
| 0.66 / 0.50 | 0.9452 (-1.6) | 0.9194 (-2.9) | 48/48 (100.0%) | 0 | 13/16 | no |
| 0.66 / 0.60 | 0.8914 (-7.0) | 0.8817 (-6.6) | 48/48 (100.0%) | 0 | 16/16 | no |
| 0.66 / 0.70 | 0.8978 (-6.3) | 0.8728 (-7.5) | 48/48 (100.0%) | 0 | 16/16 | no |

One-step resolution around the chosen similarity floor:

| similarity floor | recall@5 hooks (pp vs no floor) | recall@5 model queries (pp vs no floor) | negatives returning 0 | negatives returning a full page | abstain queries returning 0 | meets both targets |
|---|---|---|---|---|---|---|
| 0.55 | 0.9774 (+1.6) | 0.9444 (-0.4) | 41/48 (85.4%) | 3 | 8/16 | no |
| 0.56 | 0.9774 (+1.6) | 0.9337 (-1.4) | 41/48 (85.4%) | 2 | 10/16 | no |
| 0.57 | 0.9774 (+1.6) | 0.9337 (-1.4) | 44/48 (91.7%) | 0 | 10/16 | yes |
| 0.58 | 0.9774 (+1.6) | 0.9337 (-1.4) | 45/48 (93.8%) | 0 | 10/16 | yes |
| 0.59 | 0.9452 (-1.6) | 0.9337 (-1.4) | 45/48 (93.8%) | 0 | 12/16 | yes |
| 0.60 | 0.9452 (-1.6) | 0.9337 (-1.4) | 46/48 (95.8%) | 0 | 13/16 | yes |
| 0.61 | 0.9452 (-1.6) | 0.9337 (-1.4) | 46/48 (95.8%) | 0 | 13/16 | yes |
| 0.62 | 0.9452 (-1.6) | 0.9337 (-1.4) | 46/48 (95.8%) | 0 | 13/16 | yes |
| 0.63 | 0.9452 (-1.6) | 0.9194 (-2.9) | 48/48 (100.0%) | 0 | 13/16 | no |

## Choosing the full-text criterion

| full-text criterion | recall@5 hooks (pp vs no floor) | recall@5 model queries (pp vs no floor) | negatives returning 0 | negatives returning a full page | abstain queries returning 0 | meets both targets |
|---|---|---|---|---|---|---|
| none (similarity only) | 0.9516 (-1.0) | 0.8692 (-7.9) | 45/48 (93.8%) | 0 | 11/16 | no |
| unweighted coverage >= 0.50 | 0.9774 (+1.6) | 0.9373 (-1.1) | 38/48 (79.2%) | 1 | 4/16 | no |
| unweighted coverage >= 0.67 | 0.9774 (+1.6) | 0.9122 (-3.6) | 45/48 (93.8%) | 0 | 11/16 | no |
| unweighted coverage >= 0.75 | 0.9774 (+1.6) | 0.9014 (-4.7) | 45/48 (93.8%) | 0 | 11/16 | no |
| ts_rank >= 0.04 | 0.9774 (+1.6) | 0.9373 (-1.1) | 45/48 (93.8%) | 0 | 10/16 | yes |
| ts_rank >= 0.05 | 0.9839 (+2.3) | 0.9229 (-2.5) | 45/48 (93.8%) | 0 | 11/16 | no |
| IDF-weighted coverage >= 0.40 | 0.9613 (+0.0) | 0.9337 (-1.4) | 45/48 (93.8%) | 0 | 10/16 | yes |
| IDF-weighted coverage >= 0.50 | 0.9774 (+1.6) | 0.9337 (-1.4) | 45/48 (93.8%) | 0 | 10/16 | yes |
| IDF-weighted coverage >= 0.60 | 0.9774 (+1.6) | 0.9229 (-2.5) | 45/48 (93.8%) | 0 | 11/16 | no |

Unweighted coverage cannot meet both targets: at 0.50 a two-word negative
query passes on one common word, and at 0.67 model queries lose 3.6 points.
`ts_rank >= 0.04` does meet both, but `ts_rank` has no fixed scale and the
best negative candidate scores 0.037, which leaves almost no margin.
IDF-weighted coverage is bounded to 0 to 1, and the chosen 0.5 sits between
the best negative candidate (0.386) and the 10th percentile of gold memories
(0.53).

## Chosen defaults

**Similarity 0.58, coverage 0.5.**

The pairs that meet both targets form a small region: similarity 0.57 to 0.62
with coverage 0.45 to 0.50 (a step wider on either side at some similarity
values). Inside it there are two plateaus.

- Similarity 0.57 to 0.58 keeps every gold memory the hooks find, and hook
  recall improves by 1.6 points. Three negatives still return something.
- Similarity 0.59 to 0.62 silences one more negative (46/48) but drops gold
  memories with cosine between 0.58 and 0.59, and hook recall goes to -1.6
  points, 0.4 points from the limit.

A missed real memory costs more than an occasional loose result, because
agents are told to recall before saying something is absent. So the default
is on the first plateau, at its upper end, 0.58. Coverage 0.5 is the middle of
the passing range; 0.40 gives up the hook improvement and 0.60 fails on model
queries.

## What still gets through, and what is lost

Three negative queries return results at the defaults, all homonym queries
whose best candidates clear the similarity floor with no full-text match:

| id | results | cosines | query |
|---|---|---|---|
| neg-41 | 2 | 0.600, 0.591 | apache helicopter rotor blade maintenance |
| neg-46 | 1 | 0.622 | container ship cargo capacity of the Suez canal |
| neg-47 | 3 | 0.621, 0.604, 0.595 | bridge loan interest rates for home buyers |

Gold memories that clear neither floor, all on model-written queries:

- `node-nvm` (claude-opus-5-5 query): the gold memory was ranked 4th, cosine 0.556, coverage 0.29.
- `mg-supersedes` (claude-haiku-4-5 query): three gold memories, one ranked 1st with cosine 0.526 and coverage 0.39; the other two were already outside the top 5.
- `lab-ssh-leak` and `lab79` (claude-opus-5-5): the cached query is an empty string, so nothing is recalled with or without a floor.

Of the 124 scored positive queries, 109 still return a full page of 5, 13
return 1 to 4 results and 2 return nothing (the two empty queries).

## Limits of this measurement

- The sample is small: 48 negatives and 31 scored hooks. One negative more or
  less moves the zero-result rate by 2.1 points; one hook moves recall by up
  to 3.2 points.
- The similarity floor has little margin on the negative side. `neg-08`
  ("migration route of monarch butterflies to Mexico") has a best cosine of
  0.577, 0.003 under the floor. At 0.57 the rate is 44/48 (91.7%), still above
  the target.
- The floors are specific to `qwen3-embedding:0.6b` embedding the bare
  lower-cased query. A different embedding model or a query instruction prefix
  changes the cosine scale and needs a new measurement.
- The corpus is one domain (homelab and developer tooling). The negatives are
  off-domain by construction. A query on an adjacent technical topic that was
  never stored is harder, and this measurement does not cover it.
- In the full-text-only fallback (embedder unreachable) the coverage floor
  costs recall: hooks 0.8269 to 0.8161 (-1.1 points), model queries 0.8674 to
  0.8262 (-4.1 points), end to end. The LAB-426 targets are stated for normal
  recall, and the ticket requires the floor in the fallback, so this is
  reported here and not tuned separately. No coverage floor in the simulated
  fallback table keeps model queries within 2 points and also silences 90% of
  negatives:

| coverage floor | recall@5 hooks (pp vs no floor) | recall@5 model queries (pp vs no floor) | negatives returning 0 | negatives returning a full page | abstain queries returning 0 | meets both targets |
|---|---|---|---|---|---|---|
| 0 (no floor) | 0.8269 (+0.0) | 0.8674 (+0.0) | 10/48 (20.8%) | 36 | 0/16 | no |
| 0.30 | 0.8591 (+3.2) | 0.8566 (-1.1) | 41/48 (85.4%) | 2 | 3/16 | no |
| 0.40 | 0.8269 (+0.0) | 0.8369 (-3.0) | 48/48 (100.0%) | 0 | 10/16 | no |
| 0.50 | 0.8269 (+0.0) | 0.8369 (-3.0) | 48/48 (100.0%) | 0 | 13/16 | no |
| 0.60 | 0.7516 (-7.5) | 0.7882 (-7.9) | 48/48 (100.0%) | 0 | 16/16 | no |

## Behaviour

- Recall may return fewer than `--limit` results, including none.
- Each result's `match_info` carries `similarity` (null in the fallback) and
  `fulltext_coverage` next to `rrf_score`, and the CLI prints both on the
  `Match:` line.
- When candidates existed but none cleared the floor, the CLI prints
  `No memories cleared the relevance floor: …` instead of `Found N relevant memories`.
- A floor of 0 on either variable passes every candidate, which restores the
  previous ranking exactly. To gate on one signal alone, set the other floor
  above 1.
- In the full-text-only fallback only the coverage floor applies.
- The duplicate warning on `store` does not use the floor.

## Reproducing

```bash
docker run -d --name recall-floor-pg --env-file pg.env -p 127.0.0.1:55426:5432 pgvector/pgvector:0.8.6-pg17
# pg.env: POSTGRES_USER=memorygraph, POSTGRES_DB=memorygraph, POSTGRES_PASSWORD=<generated>
cd ts
for v in $(env | awk -F= '/^MEMORY_/{print $1}'); do unset "$v"; done
export MEMORY_BACKEND=postgres MEMORY_POSTGRES_HOST=127.0.0.1 MEMORY_POSTGRES_PORT=55426 \
  MEMORY_POSTGRES_PASSWORD=<same> MEMORY_EMBED_URL=http://127.0.0.1:11434
BENCH=<homelab>/docs/bench/memory
bun run src/cli.ts import --input $BENCH/corpus/memories.json
export RECALL_FLOOR_POSTGRES_URL=postgres://memorygraph@127.0.0.1:55426/memorygraph
bun run tests/recall-floor/measure.ts $BENCH ../docs/recall-floor/negative-queries.jsonl /tmp/signals.json
bun run tests/recall-floor/analyze.ts /tmp/signals.json
bun run tests/recall-floor/evaluate.ts $BENCH ../docs/recall-floor/negative-queries.jsonl
docker rm -f recall-floor-pg
```

## Negative queries: best candidate per query

| id | kind | best cosine | best IDF-weighted coverage | query |
|---|---|---|---|---|
| neg-01 | plain | 0.429 | 0.000 | penguin breeding season on the Antarctic ice shelf |
| neg-02 | plain | 0.391 | 0.299 | how long to proof sourdough bread dough overnight |
| neg-03 | plain | 0.408 | 0.128 | best fertilizer for tomato plants in clay soil |
| neg-04 | plain | 0.356 | 0.222 | symptoms of vitamin D deficiency in adults |
| neg-05 | plain | 0.457 | 0.253 | how to tune a violin by ear |
| neg-06 | plain | 0.427 | 0.327 | rules for offside in association football |
| neg-07 | plain | 0.411 | 0.205 | history of the Byzantine empire after Justinian |
| neg-08 | plain | 0.577 | 0.195 | migration route of monarch butterflies to Mexico |
| neg-09 | plain | 0.401 | 0.000 | how to knit a cable stitch sweater |
| neg-10 | plain | 0.445 | 0.085 | causes of the French Revolution of 1789 |
| neg-11 | plain | 0.474 | 0.158 | recipe for lamb tagine with apricots |
| neg-12 | plain | 0.560 | 0.324 | training plan for a first marathon in 16 weeks |
| neg-13 | plain | 0.418 | 0.212 | how do volcanoes form at subduction zones |
| neg-14 | plain | 0.472 | 0.171 | watercolor wet on wet technique for skies |
| neg-15 | plain | 0.343 | 0.244 | what do hedgehogs eat in winter |
| neg-16 | plain | 0.488 | 0.000 | baroque counterpoint in Bach fugues |
| neg-17 | plain | 0.393 | 0.278 | how to prune apple trees in late winter |
| neg-18 | plain | 0.472 | 0.224 | origin of the Silk Road trade in the Han dynasty |
| neg-19 | plain | 0.423 | 0.181 | treatment for a sprained ankle swelling |
| neg-20 | plain | 0.506 | 0.277 | wedding seating chart etiquette for divorced parents |
| neg-21 | plain | 0.518 | 0.186 | chess opening Sicilian defense Najdorf variation |
| neg-22 | plain | 0.432 | 0.192 | how glaciers carve U-shaped valleys |
| neg-23 | plain | 0.366 | 0.189 | beginner yoga poses for lower back pain |
| neg-24 | plain | 0.464 | 0.308 | how to brew kombucha scoby at home |
| neg-25 | plain | 0.442 | 0.268 | life cycle of the Atlantic salmon |
| neg-26 | plain | 0.399 | 0.000 | impressionist painters Monet and Renoir |
| neg-27 | plain | 0.398 | 0.247 | how to change a bicycle tire inner tube |
| neg-28 | plain | 0.438 | 0.233 | photosynthesis light reactions in chloroplasts |
| neg-29 | plain | 0.528 | 0.320 | mortgage refinance closing costs explained |
| neg-30 | plain | 0.493 | 0.089 | sea turtle nesting beaches in Costa Rica |
| neg-31 | plain | 0.475 | 0.000 | how to housebreak a labrador puppy |
| neg-32 | plain | 0.568 | 0.313 | origami crane folding instructions |
| neg-33 | short | 0.512 | 0.000 | sourdough starter |
| neg-34 | short | 0.546 | 0.000 | giraffe gestation |
| neg-35 | short | 0.472 | 0.378 | crochet blanket |
| neg-36 | short | 0.436 | 0.000 | Renaissance fresco |
| neg-37 | short | 0.512 | 0.000 | trout fishing lures |
| neg-38 | short | 0.488 | 0.000 | ballet pirouette |
| neg-39 | homonym | 0.501 | 0.276 | python snake feeding schedule for a ball python |
| neg-40 | homonym | 0.486 | 0.239 | rust removal from a cast iron skillet |
| neg-41 | homonym | 0.600 | 0.159 | apache helicopter rotor blade maintenance |
| neg-42 | homonym | 0.547 | 0.194 | java island coffee harvest season |
| neg-43 | homonym | 0.471 | 0.251 | shell collecting on the beach at low tide |
| neg-44 | homonym | 0.560 | 0.179 | docker strike at the port of Rotterdam in 1970 |
| neg-45 | homonym | 0.394 | 0.159 | how to bake a raspberry pie with a lattice crust |
| neg-46 | homonym | 0.622 | 0.233 | container ship cargo capacity of the Suez canal |
| neg-47 | homonym | 0.621 | 0.257 | bridge loan interest rates for home buyers |
| neg-48 | homonym | 0.497 | 0.386 | kernel of corn popping temperature |
