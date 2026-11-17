# Plenum AI

Two Rust crates for running a language model with per-user memory on an ordinary PC:
CPU only, no GPU, no Python at run time.

| Crate | What it is |
|---|---|
| [`plenum-engine`](plenum-engine) | An engine that runs GGUF language models in place on the CPU, and streams the experts of mixture-of-experts models from the disk instead of holding them in RAM. Chat CLI and an OpenAI-compatible server. Written from the C sources of [colibri](https://github.com/JustVugg/colibri). |
| [`plenum-memory`](plenum-memory) | Latent semantic memory on top of it: a user's documents are embedded once, and for every question a memory module compresses the whole collection into 16 memory tokens for the decoder. Inference-only port of [Memorilla](https://github.com/snap-stanford/memorilla). |

Each crate's README says what it covers, what it does not, and what has been verified.

## Status

This is a working slice, not a full port of either project.

- `plenum-engine` runs the Qwen3 dense, Qwen3 MoE and Qwen2 architectures, and the
  hybrid Qwen3.5 / 3.8 family (Gated DeltaNet + gated attention: Qwen3.8-27B dense,
  Qwen3.6 MoE). Colibri's other engines (GLM, DeepSeek, Kimi, Inkling, MiMo, OLMoE,
  Qwen3.8-Flash-Next, the image and decision models), its GPU backends and its Python
  tooling are not ported.
- `plenum-memory` answers questions with the released memory modules. Training,
  evaluation and the benchmarks are not ported.
- Verified: a 30B MoE model (Qwen3-30B-A3B Q4_K_M) runs on a 16 GB laptop with under
  1 GB resident; Qwen3.8-27B answers correctly, and its hybrid layers match Hugging
  Face's implementation to 4e-6 on a tiny model; the memory module matches Memorilla's PyTorch code to 4e-5; the
  encoder's embeddings match the published ones (cosine 0.9993 to 0.9997 with the
  Q8_0 encoder); the example of Memorilla's README is reproduced.
- Not verified: no benchmark of either paper has been run, and outputs have not been
  compared token for token with the C engine or llama.cpp.
- It is slow on a laptop CPU: about 2 tokens/s for the 30B MoE model, about 0.1 for
  the 27B dense one (its 15 GB do not fit in 16 GB of RAM), and from under
  one to several minutes per question with memory, depending on prompt length and free RAM.

## Build

Rust 1.82 or later. On Windows without the Visual Studio C++ tools, use the GNU
toolchain (`rustup default stable-x86_64-pc-windows-gnu`).

```bash
cargo build --release
cargo test --release
```

The binaries are `target/release/plenum-engine` and `target/release/plenum-memory`. The build
targets the CPU it runs on (`.cargo/config.toml` sets `target-cpu=native`), so a binary
built on one machine may not run on an older one.

## Models

Nothing is downloaded for you. Put the files in `./models` (it is git-ignored):

| File | From (huggingface.co) | Size | Used by |
|---|---|---|---|
| `Qwen3-0.6B-Q8_0.gguf` | Qwen/Qwen3-0.6B-GGUF | 0.6 GB | `plenum-engine`, quick tests |
| `Qwen3-30B-A3B-Q4_K_M.gguf` | Qwen/Qwen3-30B-A3B-GGUF | 17.3 GB | `plenum-engine`, MoE with expert streaming |
| `Qwen3-8B-Q4_K_M.gguf` | Qwen/Qwen3-8B-GGUF | 4.7 GB | `plenum-memory`, decoder |
| `Qwen3-Embedding-4B-Q8_0.gguf` | Qwen/Qwen3-Embedding-4B-GGUF | 4.0 GB | `plenum-memory`, encoder |
| `memorilla/stage3/memory.pt`, `memorilla/pmv2/memory.pt` (with their `config.json`) | memorilla/Memorilla-Qwen3-8B | 0.4 GB each | `plenum-memory`, memory module |

## Quick start

```bash
# chat with a model
target/release/plenum-engine chat models/Qwen3-0.6B-Q8_0.gguf

# a user's memory: add documents, ask, or serve it to an agent
target/release/plenum-memory add --user priya plenum-memory/examples/priya.jsonl
target/release/plenum-memory ask --user priya -q "What is my dog's name?"
target/release/plenum-memory serve --port 8000     # chat page at http://127.0.0.1:8000/
```

The server speaks the OpenAI chat API; the `user` field of a request selects whose
memory is used. See [`plenum-memory/README.md`](plenum-memory/README.md#server-for-an-agent).

## License

Copyright 2026 Plenum Tech. Licensed under Apache-2.0, as both upstream projects are.
colibri is by Vincenzo Fornaro and Memorilla by its authors at Stanford University;
[NOTICE](NOTICE) says which parts of this code were written from which of their
sources. See [LICENSE](LICENSE).
