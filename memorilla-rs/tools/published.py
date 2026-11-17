"""Checks the Rust port against what Memorilla publishes (memorilla/Memorilla-Data).

    python tools/published.py export  <data dir> <config> <split> <row> <store dir>
        writes the row's collection, with the PUBLISHED document embeddings, as the
        store user "<config>_<row>", plus <store dir>/<config>_<row>.sample.jsonl (the
        question and a few documents, to embed with the Rust encoder) and prints the question.

    python tools/published.py compare <data dir> <config> <split> <row> <store dir>
        compares the Rust embeddings of that sample (store user "<config>_<row>_rust",
        made with `memorilla-rs add --no-chunk`) with the published ones.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file

mode, data, config, split, row, store = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), Path(sys.argv[6])
SAMPLE_DOCS = 6

emb_dir = data / "embeddings" / "qwen3-embedding-4b" / config / split
table = pq.read_table(next((data / "data" / config).glob(f"{split}-*.parquet")))
example = table.slice(row, 1).to_pylist()[0]
index = load_file(emb_dir / "index.safetensors")
at = int(np.nonzero(index["collection_ids"] == example["collection_id"])[0][0])
shard, start, length = int(index["shard"][at]), int(index["start"][at]), int(index["length"][at])
docs = load_file(emb_dir / f"documents-{shard:05d}.safetensors")["embeddings"][start : start + length].astype("<f4")
question = load_file(emb_dir / "questions.safetensors")["embeddings"][row].astype("<f4")
assert len(example["documents"]) == length, (len(example["documents"]), length)
# the sample: the question, then the shortest documents (the encoder is slow on a CPU)
order = sorted(range(length), key=lambda i: len(example["documents"][i]))[:SAMPLE_DOCS]
name = f"{config}_{row}"

if mode == "export":
    user = store / name
    user.mkdir(parents=True, exist_ok=True)
    with open(user / "docs.jsonl", "w", encoding="utf-8") as handle:
        for text in example["documents"]:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    docs.tofile(user / "embeddings.f32")
    question.tofile(store / f"{name}.question.f32")
    with open(store / f"{name}.sample.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"text": example["question"].strip()}, ensure_ascii=False) + "\n")
        for i in order:
            handle.write(json.dumps({"text": example["documents"][i]}, ensure_ascii=False) + "\n")
    print(f"user '{name}': {length} documents with the published embeddings")
    print("question:", example["question"])
    print("reference answer:", example["answer"])
else:
    rust = np.fromfile(store / f"{name}_rust" / "embeddings.f32", dtype="<f4").reshape(-1, docs.shape[1])
    published = np.concatenate([question[None], docs[order]])[: len(rust)]
    labels = ["question"] + [f"document {i} ({len(example['documents'][i])} chars)" for i in order]
    for label, a, b in zip(labels, rust, published):
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"cosine {cosine:.5f}   norms rust {np.linalg.norm(a):.4f} published {np.linalg.norm(b):.4f}   {label}")
