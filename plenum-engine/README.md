# plenum-engine

Plenum Tech's local LLM engine, in Rust: run a GGUF language model **in place** on the CPU, keep the
dense part of the model resident, and read the routed **experts of a MoE model from
the disk** when the router asks for them.

This is a Rust engine written from the C sources of [colibri](https://github.com/JustVugg/colibri) (its `c/` folder), not a line-for-line
translation of the whole project. What it covers and what it does not is listed
under [Scope](#scope).

## Build

```bash
cargo build --release -p plenum-engine
```

Run it from the repository root. The binary is `target/release/plenum-engine` (`plenum-engine.exe` on Windows). On Windows without
Visual Studio's C++ tools, use the GNU toolchain: `rustup default stable-x86_64-pc-windows-gnu`.

## Use

```bash
plenum-engine chat  model.gguf                    # talk in the terminal
plenum-engine run   model.gguf -p "your prompt"   # one answer
plenum-engine serve model.gguf --port 8000        # OpenAI-compatible server + chat page
plenum-engine info  model.gguf                    # what is in the file, and whether it runs
plenum-engine selftest                            # check the engine on tiny built-in models
```

`plenum-engine help` lists the options (`--temp`, `--top-k`, `--top-p`, `--max-tokens`,
`--ctx`, `--threads`, `--think`, `--expert-cache-mb`, `--prof`, ...). A small dense
model is bound by memory bandwidth, so fewer threads than cores is often faster there
(`--threads 6`); a MoE model is faster with all of them (the default).

### Expert streaming (MoE models)

The routed experts are most of a MoE model (16.35 GB of the 17.28 GB of
Qwen3-30B-A3B Q4_K_M) and are never loaded as a whole. Two ways to stream them:

- **Mapped** (the default): an expert is a range of the memory-mapped file, used
  where it is with no copy. The operating system keeps the experts in use in RAM and
  drops the others when memory is short.
- **Private cache** (`--expert-cache-mb N`): an expert is read from the disk into a
  cache of at most N MB, least recently used out first. This is a hard limit on the
  experts' RAM, at the price of a copy and of sharing RAM with the OS file cache.

Measured on a 16 GB laptop (Intel Core 7 150U, NVMe SSD, no GPU), Qwen3-30B-A3B
Q4_K_M: 951 MB resident, about 2 tokens/s generating in mapped mode and 1.6 with a
4 GB private cache. Qwen3-0.6B Q8_0: about 20 tokens/s.

### Server

| Route | |
|---|---|
| `GET /` | a chat page |
| `POST /v1/chat/completions` | OpenAI chat, `stream: true` or not |
| `POST /v1/completions` | raw text completion |
| `GET /v1/models`, `GET /health` | |

Request fields: `messages`, `stream`, `temperature`, `top_p`, `top_k`, `max_tokens`,
`seed`, `stop`, `repetition_penalty`, `enable_thinking` (Qwen3 reasoning, off by
default) and `memory_embeddings` (below). Requests run one at a time; a request whose
messages begin like the previous one reuses its KV cache. Tool calling is not implemented.

### Latent memory tokens

A prompt can carry vectors instead of tokens: each `<|memory|>` in the text is
replaced, in order, by one vector of the model's width, which enters the model as an
input embedding at that position. This is the interface a memory module such as
[Memorilla](https://github.com/snap-stanford/memorilla) needs from its frozen decoder
(Memorilla gives vLLM `prompt_embeds` with its memory tokens in the `<|memory|>` positions).

- Library: `Engine::generate_with_memory(sess, prompt, &vectors, ...)`, or
  `Model::forward_embd` for full control. See [`examples/memory.rs`](examples/memory.rs).
- Server: `"memory_embeddings": [[...], ...]` in the request, with as many
  `<|memory|>` in the messages as there are vectors.

The engine only places the vectors. Producing them (Memorilla's encoder and memory
module) is a separate model that this crate does not run.

## Scope

Runs:

- GGUF architectures `qwen3` (dense), `qwen3moe` (MoE, experts streamed from disk) and `qwen2`.
- Tensor types F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
  decoded per block exactly as `c/ggml_quants.h` does.
- The byte-level BPE tokenizer in the GGUF (`tokenizer.ggml.model = gpt2`), ChatML prompts.

Does not (yet): the other colibri engines (GLM, DeepSeek V4, Kimi, Inkling, MiMo,
OLMoE, Qwen3.8, image and decision models), GPU backends (CUDA, Vulkan, Metal), IQ
and MXFP4 blocks, split GGUF files, the model's own Jinja chat template, tool calls,
speculative decoding, the Python tooling (`coli` installer, downloader, MCP server).

## What has been verified

- `cargo test` and `plenum-engine selftest`: block decoders against hand-built blocks; the
  tokenizer; a prompt run at once equals token by token; KV prefix reuse; input
  vectors equal token ids; a MoE whose experts all hold the dense FFN equals the dense
  model; expert streaming with a 1-byte cache equals everything cached; Q8_0 against f32.
- The fused kernels (Q8_0, Q4_K, Q6_K codes multiplied in place) equal decode-then-dot
  on blocks of arbitrary bytes.
- End to end on real models:
  - Qwen3-0.6B Q8_0, dense (`chat`, `run`, `serve`, memory vectors).
  - Qwen3-30B-A3B Q4_K_M, MoE with 128 experts per layer and 8 routed, Q4_K and Q6_K
    blocks: correct answers in both streaming modes, with the same text from each.

Not verified against real weights: `qwen2`, and the Q4_0/Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q5_K
decoders (ported from `c/ggml_quants.h`, not yet run on a file that uses them).
Outputs have not been compared token for token with the C engine or llama.cpp.

## Layout

| File | Role | C counterpart |
|---|---|---|
| `src/gguf.rs` | GGUF reader (mmap) and a small writer | `c/gguf.h` |
| `src/quant.rs` | block decoders, dot, matmul | `c/ggml_quants.h`, `c/matmul_f32.h` |
| `src/expert.rs` | expert store: mapped ranges, or positioned reads into a byte-bounded LRU | `c/expert_store.h` |
| `src/model.rs` | the transformer: attention, dense FFN, MoE routing | `c/qwen36.c`, `c/olmoe.c` |
| `src/tokenizer.rs` | BPE tokenizer, streaming UTF-8 decoder | `c/tok.h` |
| `src/generate.rs` | sampler, ChatML, generation loop, prefix reuse, memory vectors | `c/sample.h`, `c/kv_prefix.h` |
| `src/server.rs`, `src/chat.html` | HTTP server and chat page | `c/openai_server.py` |
| `src/tiny.rs` | synthetic models and the self-check | `c/*_tiny` |

## License

Copyright 2026 Plenum Tech. Apache-2.0; see [LICENSE](../LICENSE) and [NOTICE](../NOTICE).
Written from colibri (https://github.com/JustVugg/colibri), Copyright 2026 Vincenzo Fornaro, also Apache-2.0.
