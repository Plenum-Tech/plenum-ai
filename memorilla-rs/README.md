# memorilla-rs

[Memorilla](https://github.com/snap-stanford/memorilla) in Rust, for a local PC: latent
semantic memory for an LLM, on the CPU, with no Python and no GPU at run time.

A user's documents are embedded once and kept. For every question, a small memory
module compresses the user's whole collection into **16 memory tokens**, conditioned on
the question, and a frozen decoder reads them as 16 input positions in its system prompt.

This is an inference-only port of the parts of Memorilla's `memorilla/` package that
answer questions. It runs on the [colibri-rs](../colibri-rs) engine.

## The three models

| Role | File | Size |
|---|---|---|
| Encoder | `Qwen3-Embedding-4B-Q8_0.gguf` (Qwen/Qwen3-Embedding-4B-GGUF) | 3.99 GB |
| Memory module | `memorilla/stage3/memory.pt` or `memorilla/pmv2/memory.pt` (memorilla/Memorilla-Qwen3-8B) | 388 MB |
| Decoder | `Qwen3-8B-Q4_K_M.gguf` (Qwen/Qwen3-8B-GGUF) | 4.68 GB |

The memory module was trained for exactly this encoder and this decoder; other models
of other widths are refused. By default the files are looked for in `./models`
(`--models DIR` or `MEMORILLA_MODELS` changes that). Commands are run from the
repository root.

## Build

```bash
cargo build --release -p memorilla-rs
```

## Use

```bash
memorilla-rs add   --user priya memorilla-rs/examples/priya.jsonl      # embed and keep her documents
memorilla-rs ask   --user priya -q "What is my dog's name?"
memorilla-rs chat  --user priya
memorilla-rs search --user priya -q "allergy"             # the closest stored passages
memorilla-rs users
memorilla-rs serve --port 8000                            # the server an agent talks to
```

`memorilla-rs help` lists every option. Files given to `add` are `.txt`/`.md` (split
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

### Server, for an agent

`memorilla-rs serve` is the OpenAI-compatible server of colibri-rs with the memory
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

- **Memory module against PyTorch.** `memorilla-rs/tools/reference.py` runs Memorilla's own
  `MemoryModule` with a real checkpoint on fixed inputs; `memorilla-rs verify` compares.
  Both released checkpoints tried (stage3 with 37 documents, pmv2 with 500): largest
  difference 4e-5 on values up to 34, cosine 1.000000000. This also checks the
  `memory.pt` reader (`src/torchpt.rs`), which needs no PyTorch.
- **Encoder against the published embeddings.** `memorilla-rs/tools/published.py` takes a row of
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

On a collection of 12 short chats (`memorilla-rs/examples/priya.jsonl`), memory tokens alone were
not reliable with either checkpoint. `stage3` kept the gist and lost names. `pmv2` got
four questions out of four wrong: the dog's name, the daughter's allergy (twice), and
where the user is moving. With 3 passages as well, the same questions are answered
correctly. For exact facts about a user, keep `--retrieve` on; see "Memory tokens,
and passages" above.

To repeat the two checks (after downloading the pmv2 test split of `memorilla/Memorilla-Data` into `./data`):

```bash
python memorilla-rs/tools/published.py export data pmv2 test 73 memory-store
memorilla-rs add --user pmv2_73_rust --no-chunk memory-store/pmv2_73.sample.jsonl
python memorilla-rs/tools/published.py compare data pmv2 test 73 memory-store
memorilla-rs ask --user pmv2_73 --memory models/memorilla/pmv2 --retrieve 0 \
  --system "You are a personalized AI assistant. Answer the question about the user based on your understanding of the user." \
  -q "What are some easy breakfast ideas that will keep me full all morning?"
```

## Not ported

Training (`train.py`, the recipes, the RL code), evaluation and the benchmarks,
modules with `retrieval_init` or document self-attention layers (none of the released
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
