# plenum-memory

Plenum Tech's per-user memory for a local LLM, in Rust: latent semantic memory on the
CPU, with no Python and no GPU at run time. It implements the
[Memorilla](https://github.com/snap-stanford/memorilla) method and runs its released
memory modules.

A user's documents are embedded once and kept. For every question, a small memory
module compresses the user's whole collection into **16 memory tokens**, conditioned on
the question, and a frozen decoder reads them as 16 input positions in its system prompt.

This is an inference-only port of the parts of Memorilla's `memorilla/` package that
answer questions. It runs on the [plenum-engine](../plenum-engine) engine.

## The three models

| Role | File | Size |
|---|---|---|
| Encoder | `Qwen3-Embedding-4B-Q8_0.gguf` (Qwen/Qwen3-Embedding-4B-GGUF) | 3.99 GB |
| Memory module | `memorilla/stage3/memory.pt` or `memorilla/pmv2/memory.pt` (memorilla/Memorilla-Qwen3-8B) | 388 MB |
| Decoder | `Qwen3-8B-Q4_K_M.gguf` (Qwen/Qwen3-8B-GGUF) | 4.68 GB |

The memory module was trained for exactly this encoder and this decoder; other models
of other widths are refused. By default the files are looked for in `./models`
(`--models DIR` or `PLENUM_MODELS` changes that). Commands are run from the
repository root.

## Build

```bash
cargo build --release -p plenum-memory
```

## Use

```bash
plenum-memory add   --user priya plenum-memory/examples/priya.jsonl      # embed and keep her documents
plenum-memory ask   --user priya -q "What is my dog's name?"
plenum-memory chat  --user priya
plenum-memory search --user priya -q "allergy"             # the closest stored passages
plenum-memory users
plenum-memory serve --port 8000                            # the server an agent talks to
```

`plenum-memory help` lists every option. Files given to `add` are `.txt`/`.md` (split
into passages of at most 256 tokens) or `.jsonl` (one `{"text": ...}` per line, each
kept whole). Collections live in `./memory-store/<user>/` (`--store DIR`).

### Memory tokens, and passages

16 vectors hold the gist of a collection, not its wording. On the test collection here
they knew that the dog is four and that the daughter has an allergy, and got the dog's
name and the allergen wrong. So by default the decoder is also given, as text, the
**3 stored passages closest to the question** (`--retrieve K`), found with the same
embeddings at no extra cost. With them the same questions are answered exactly.

`--retrieve 0` gives the decoder the memory tokens alone, which is what Memorilla
itself does.

The memory tokens make the decoder answer with confidence, including about things the
memory does not hold. `--grounded` (or the same sentence in your own system prompt)
tells it to say that it does not know.

### The memory learns from each conversation

`plenum-memory serve` saves every question asked through it, with its answer, to the
asking user's memory once the answer has been sent, so a later question can recall it:

1. **Add** documents for a user (`add`, the page, or `POST /v1/memory/documents`).
2. **Ask.** The memory tokens and the closest passages go to the decoder.
3. **Save.** In the background, the exchange is embedded and stored as one entry:
   "On 2026-11-17, the user asked: … The assistant answered: …". Answers with nothing
   to remember (empty, an error, "I don't know") are not saved.
4. **Recall.** The next question can draw on it like any other entry.
5. **Review and forget.** `POST /v1/memory/list` shows the newest entries, marked
   `document` or `chat`; `POST /v1/memory/forget` removes one by its index. The chat
   page shows the same list under "What this user's memory holds", with a forget button.

A saved answer is stored as what the assistant said, not as a fact, but it can still
be wrong, and once saved it is recalled like anything else. Review the entries and
forget the wrong ones. To turn saving off: untick "save Q&A to memory" on the page,
send `"remember": false` in a request, or start the server with `--no-remember`. From
the command line, `ask` and `chat` save only with `--remember`.

### Server, for an agent

`plenum-memory serve` is the OpenAI-compatible server of plenum-engine with the memory
behind it. Its page at `http://127.0.0.1:8000/` is a chat interface to try it: choose
a user, paste text into their memory, ask questions, and switch the memory, the
passages and the "I don't know" prompt on and off to see what each one changes.

| Route | |
|---|---|
| `POST /v1/chat/completions` | `"user": "U"` selects whose memory is used; the last user message is the question. `"memory": false` turns it off, `"retrieve": K` overrides `--retrieve`. |
| `POST /v1/memory/documents` | `{"user": "U", "documents": ["...", "..."]}` or `{"user": "U", "text": "..."}`: embeds and stores. `"chunk": false` keeps texts whole. |
| `POST /v1/memory/tokens` | `{"user": "U", "question": "..."}`: the 16 memory vectors and the closest passages, for an agent that runs its own decoder. |
| `POST /v1/memory/clear` | `{"user": "U"}` |
| `GET /v1/memory/users` | the collections and their sizes |

```bash
curl http://127.0.0.1:8000/v1/memory/documents -d '{"user":"ravi","documents":["Ravi runs a bakery called Crumb and Co. It is closed on Mondays."]}'
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"user":"ravi","messages":[{"role":"user","content":"What day is my shop closed?"}]}'
```

Any OpenAI client works: base URL `http://127.0.0.1:8000/v1`, any API key, and the
`user` field of the request set to the user's id.

## Speed, on a 16 GB laptop without a GPU

Measured on an Intel Core 7 150U:

- adding a document: about 8 s for a passage of 60 tokens (once per document);
- a question with memory tokens alone: about 45 s;
- a question with 3 passages as well: about 80 s, most of it the decoder reading the
  longer prompt at about 3 tokens/s;
- the memory module itself: 0.2 s for 37 documents, 0.5 s for 500.

These times need the encoder and the decoder in RAM together (about 9 GB with the
Q8_0 encoder). When less is available, Windows pages the models in and out and a
question takes minutes: through the server, with about 4 GB available when it started,
the same question took 220 s with the Q8_0 encoder and 140 to 170 s with the Q4_K_M
one (`--encoder .../Qwen3-Embedding-4B-Q4_K_M.gguf`, 2.33 GB). The smaller encoder is
less faithful: its vectors have cosine 0.97 to 0.985 with the published ones, against
0.9993 to 0.9997 for Q8_0. It does not embed faster. Closing other applications helps
more than changing the encoder.

This is a personal-scale memory. A collection of thousands of documents is cheap to
query (the module's cost) but slow to write on a CPU (the encoder's).

## What has been verified

- **Memory module against PyTorch.** `plenum-memory/tools/reference.py` runs Memorilla's own
  `MemoryModule` with a real checkpoint on fixed inputs; `plenum-memory verify` compares.
  Both released checkpoints tried (stage3 with 37 documents, pmv2 with 500): largest
  difference 4e-5 on values up to 34, cosine 1.000000000. This also checks the
  `memory.pt` reader (`src/torchpt.rs`), which needs no PyTorch.
- **Encoder against the published embeddings.** `plenum-memory/tools/published.py` takes a row of
  `memorilla/Memorilla-Data`; the Rust encoder's vectors for its question and six of
  its documents have cosine 0.9993 to 0.9997 with the published ones (the gap is the
  Q8_0 quantization; the published vectors are unit-norm, as these are).
- **The example of Memorilla's README, end to end.** PersonaMem-v2 test row 73: 107
  past conversations, the `pmv2` checkpoint, memory tokens alone (`--retrieve 0`). The
  answer begins "Since you've got a family history of type 2 diabetes, you might enjoy
  breakfasts that balance protein and fiber without spiking blood sugar", as the
  original's does; without the memory the decoder gives a generic list of breakfasts.
- **Per-user memory.** Two users in one store get their own answers, over the server.

Not verified: no benchmark of the paper has been run, so there is no accuracy number
for this port (a Q8_0 encoder and a Q4_K_M decoder instead of the bfloat16 models the
module was trained with). One example reproduced is not a benchmark.

### Which checkpoint

`pmv2` is trained on a user's chat history and answers in a way that fits the user
(with the system prompt "You are a personalized AI assistant. Answer the question about
the user based on your understanding of the user."). `stage3` (the default) is the
multitask module.

On a collection of 12 short chats (`plenum-memory/examples/priya.jsonl`), memory tokens alone were
not reliable with either checkpoint. `stage3` kept the gist and lost names. `pmv2` got
four questions out of four wrong: the dog's name, the daughter's allergy (twice), and
where the user is moving. With 3 passages as well, the same questions are answered
correctly. For exact facts about a user, keep `--retrieve` on; see "Memory tokens,
and passages" above.

To repeat the two checks (after downloading the pmv2 test split of `memorilla/Memorilla-Data` into `./data`):

```bash
python plenum-memory/tools/published.py export data pmv2 test 73 memory-store
plenum-memory add --user pmv2_73_rust --no-chunk memory-store/pmv2_73.sample.jsonl
python plenum-memory/tools/published.py compare data pmv2 test 73 memory-store
plenum-memory ask --user pmv2_73 --memory models/memorilla/pmv2 --retrieve 0 \
  --system "You are a personalized AI assistant. Answer the question about the user based on your understanding of the user." \
  -q "What are some easy breakfast ideas that will keep me full all morning?"
```

## Evaluation

`plenum-memory evaluate` is a port of Memorilla's `evaluate.py` and `metrics.py`: for each
row of a test set the memory module compresses that row's collection, the decoder
answers greedily with the memory tokens alone (as Memorilla does; `--retrieve K` adds
passages), and the answers are scored as the paper scores them.

| Benchmark | Scoring |
|---|---|
| `pmv2`, `pv4`, `factkg` | answer-choice accuracy: the answer and every choice are embedded with the encoder, and the nearest choice must be the reference (`pv4` also reports the hard subset) |
| `triviaqa` | exact match (SQuAD normalisation), plus BLEU, ROUGE and F1 |
| `pubmedqa` | the first word, yes / no / maybe, must match |
| `narrativeqa`, `lamp4`, `lamp7` | corpus BLEU-1/4 (NLTK, method-1 smoothing), ROUGE-1/L F-measure (rouge_score with the Porter stemmer), token F1 |

The prompts, system prompts and generation budgets are those of `memorilla/benchmarks.py`.
The rows come from Memorilla's published parquet, turned into plain files once by
`tools/import_benchmark.py` (pyarrow; Rust cannot read the zstd parquet here without a C
compiler), and the document and question embeddings are read from Memorilla's published
safetensors, so a run uses the same inputs as the original:

```bash
python plenum-memory/tools/import_benchmark.py data pmv2 test data/plenum/pmv2-test
plenum-memory evaluate --benchmark pmv2 --memory models/memorilla/pmv2   --data data/plenum/pmv2-test --embeddings data/embeddings/qwen3-embedding-4b/pmv2/test --limit 20
```

Your own questions are scored the same way, answered from a user's memory:

```bash
# qa.jsonl: {"question": "...", "answer": "...", "user": "compliance"} per line ("choices" for choice scoring)
plenum-memory evaluate --data qa.jsonl --scoring generation --retrieve 3
```

Each run writes `results/<name>/predictions.jsonl` as it goes (a rerun continues where it
stopped; `--fresh` starts over) and `results/<name>/metrics.json` at the end, in percent.
On a laptop CPU it is about a minute or two per row, so `--limit` a slice.

Checked against the Python packages Memorilla scores with (`tools/check_metrics.py`, with
`examples/metrics.rs`): BLEU-1/4 and ROUGE-1/L equal NLTK and rouge_score to 1e-16 on 300
random pairs, and the Porter stemmer gives NLTK's stem for all 12,082 distinct words of 120
PersonaMem-v2 rows. METEOR is not computed: it needs WordNet's synonym data.

`--no-memory` runs the same decoder and prompts with no memory tokens and no passages, as a
baseline; its results go to `results/<name>-no-memory/`.

### First results (PersonaMem-v2 test, rows 0–39, Q4_K_M decoder)

| Run | Rows 0–19 | Rows 20–39 | All 40 |
|---|---|---|---|
| `pmv2` memory module (16 memory tokens, no passages) | 9/20 | 9/20 | **45.0%** (18/40) |
| `--no-memory` baseline | 9/20 | 3/20 | **30.0%** (12/40) |
| Gain | 0 | +6 | **+15.0 points** |

Memory turned 14 questions right that the baseline missed and 8 wrong that it got right.
With 40 rows each score is uncertain by about ±8 points, and a sign test on those 22
disagreements gives p ≈ 0.29, so the gain is encouraging but not yet significant. Neither
score is comparable with the paper's figure over all 5,000 rows, which used a
full-precision decoder.

## Not ported

Training (`train.py`, the recipes, the RL code), METEOR, the baselines of
`evaluate_baselines.py`, modules with `retrieval_init` or document self-attention layers (none of the released
checkpoints has them), and batching several questions at once.

## Layout

| File | Role | Python counterpart |
|---|---|---|
| `src/torchpt.rs` | reads `memory.pt` (zip + pickle) without PyTorch | `torch.load` |
| `src/memory.rs` | the memory module: `write` a collection once, `read` it per question | `memorilla/memory.py` |
| `src/store.rs` | per-user documents and embeddings on disk | `memorilla/embeddings.py` |
| `src/stack.rs` | encoder -> store -> module -> prompt; the server's memory hook and routes | `memorilla/vllm.py`, `data.py` |
| `src/main.rs` | the command line | |
| `tools/reference.py` | PyTorch reference outputs for `verify` | |

## License

Copyright 2026 Plenum Tech. Apache-2.0; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
Written from Memorilla (https://github.com/snap-stanford/memorilla), by its authors at Stanford University, also Apache-2.0.
