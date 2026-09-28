#!/usr/bin/env python3
"""Calibrate the Postgres backend's duplicate-warning cosine threshold.

Compares embedding cosine against the LAB-354 audit's TF-IDF near-duplicate
pairs on the same corpus. Needs the corpus imported into a Postgres
memorygraph database (with embeddings) and homelab's scripts/bench/memory
on disk for the audit's own find_candidates. SQL runs through --psql, a
command that reads SQL on stdin (for example a docker exec into a loopback
container), so no credential is passed here.
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def psql(cmd, sql):
    out = subprocess.run(
        shlex.split(cmd) + ["-A", "-t", "-F", "|", "-v", "ON_ERROR_STOP=1"],
        input=sql, capture_output=True, text=True, check=True,
    ).stdout
    return [line.split("|") for line in out.splitlines() if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="homelab docs/bench/memory/corpus/memories.json")
    p.add_argument("--bench-dir", required=True, help="homelab scripts/bench/memory")
    p.add_argument("--psql", required=True, help="command that runs psql against the imported database")
    p.add_argument("--audit-min-score", type=float, default=0.65)
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.bench_dir).resolve()))
    from mine_recurrences import find_candidates, load_memories

    memories = load_memories(args.corpus)
    pairs = find_candidates(memories, args.audit_min_score)
    audit_flagged = {c["b_id"] for c in pairs}

    values = ",".join(f"('{c['a_id']}','{c['b_id']}')" for c in pairs)
    pair_cos = {
        (a, b): float(s)
        for a, b, s in psql(args.psql, f"""
            SELECT p.a, p.b, 1 - (ma.embedding <=> mb.embedding)
            FROM (VALUES {values}) AS p(a, b)
            JOIN memories ma ON ma.id = p.a JOIN memories mb ON mb.id = p.b
            WHERE ma.embedding IS NOT NULL AND mb.embedding IS NOT NULL;""")
    }
    nearest = {
        mid: float(s)
        for mid, s in psql(args.psql, """
            SELECT m.id, max(1 - (m.embedding <=> e.embedding))
            FROM memories m JOIN memories e
              ON e.created_at < m.created_at AND e.embedding IS NOT NULL
            WHERE m.embedding IS NOT NULL
            GROUP BY m.id;""")
    }
    total = len(memories)
    cos = sorted(pair_cos.values())

    def pct(q):
        return cos[min(len(cos) - 1, int(q * len(cos)))]

    print(f"corpus memories: {total}; with an earlier memory and an embedding: {len(nearest)}")
    print(f"audit pairs (TF-IDF >= {args.audit_min_score}): {len(pairs)}; flagged memories: "
          f"{len(audit_flagged)} ({100 * len(audit_flagged) / total:.1f}%)")
    print(f"embedding cosine of audit pairs: min {cos[0]:.3f} p10 {pct(0.10):.3f} "
          f"p25 {pct(0.25):.3f} median {pct(0.5):.3f} max {cos[-1]:.3f}")
    print("threshold  flagged  rate   audit_flagged_recovered")
    for t in [x / 100 for x in range(70, 97)]:
        flagged = {m for m, s in nearest.items() if s >= t}
        print(f"{t:.2f}      {len(flagged):5d}  {100 * len(flagged) / total:4.1f}%  "
              f"{len(flagged & audit_flagged)}/{len(audit_flagged)}")


if __name__ == "__main__":
    main()
