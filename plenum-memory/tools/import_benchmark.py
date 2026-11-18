"""Turns one split of Memorilla's published data into the plain files `plenum-memory evaluate` reads.

Memorilla publishes its benchmarks as zstd-compressed parquet (memorilla/Memorilla-Data,
data/<config>/<split>-*.parquet). This reads them once with pyarrow and writes:

    <out>/rows.jsonl          one row per question: row, collection_id, question, answer,
                              and choices / hard where the split has them
    <out>/collections.jsonl   one line per collection: collection_id, documents

The embeddings stay where they are (embeddings/qwen3-embedding-4b/<config>/<split>/);
the evaluation reads their safetensors files directly.

    python plenum-memory/tools/import_benchmark.py <data dir> pmv2 test data/plenum/pmv2-test
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

data, config, split, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
files = sorted((data / "data" / config).glob(f"{split}-*.parquet"))
if not files:
    sys.exit(f"no {split}-*.parquet under {data / 'data' / config}")
out.mkdir(parents=True, exist_ok=True)
seen, n = set(), 0
with open(out / "rows.jsonl", "w", encoding="utf-8") as rows, open(out / "collections.jsonl", "w", encoding="utf-8") as cols:
    for f in files:
        for r in pq.read_table(f).to_pylist():
            row = {"row": n, "collection_id": r["collection_id"], "question": r["question"], "answer": r["answer"]}
            for k in ("choices", "hard"):
                if r.get(k) is not None:
                    row[k] = r[k]
            rows.write(json.dumps(row, ensure_ascii=False) + "\n")
            if r["collection_id"] not in seen:
                seen.add(r["collection_id"])
                cols.write(json.dumps({"collection_id": r["collection_id"], "documents": r["documents"]}, ensure_ascii=False) + "\n")
            n += 1
print(f"{n} rows, {len(seen)} collections -> {out}")
