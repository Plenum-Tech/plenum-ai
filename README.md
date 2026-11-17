# colibri-memorilla-rs

Two Rust crates for running a language model with per-user memory on an ordinary PC:
CPU only, no GPU, no Python at run time.

| Crate | What it is |
|---|---|
| [`colibri-rs`](colibri-rs) | An engine that runs GGUF language models in place on the CPU, and streams the experts of mixture-of-experts models from the disk instead of holding them in RAM. Chat CLI and an OpenAI-compatible server. Written from the C sources of [colibri](https://github.com/JustVugg/colibri). |
| [`memorilla-rs`](memorilla-rs) | Latent semantic memory on top of it: a user's documents are embedded once, and for every question a memory module compresses the whole collection into 16 memory tokens for the decoder. Inference-only port of [Memorilla](https://github.com/snap-stanford/memorilla). |

Each crate's README says what it covers, what it does not, and what has been verified.

## Status

This is a working slice, not a full port of either project.

- `colibri-rs` runs the Qwen3 dense, Qwen3 MoE and Qwen2 architectures. The other
  twelve colibri engines, the GPU backends and the Python tooling are not ported.
- `memorilla-rs` answers questions with the released memory modules. Training,
  evaluation and the benchmarks are not ported.
- Verified: a 30B MoE model (Qwen3-30B-A3B Q4_K_M) runs on a 16 GB laptop with under
  1 GB resident; the memory module matches Memorilla's PyTorch code to 4e-5; the
  encoder's embeddings match the published ones (cosine 0.9993 to 0.9997 with the
  Q8_0 encoder); the example of Memorilla's README is reproduced.
- Not verified: no benchmark of either paper has been run, and outputs have not been
  compared token for token with the C engine or llama.cpp.
- It is slow on a laptop CPU: about 2 tokens/s for the 30B MoE model, and from under
  one to several minutes per question with memory, depending on prompt length and free RAM.

## Build

Rust 1.82 or later. On Windows without the Visual Studio C++ tools, use the GNU
toolchain (`rustup default stable-x86_64-pc-windows-gnu`).

```bash
cargo build --release
cargo test --release
```

The binaries are `target/release/coli-rs` and `target/release/memorilla-rs`. The build
targets the CPU it runs on (`.cargo/config.toml` sets `target-cpu=native`), so a binary
built on one machine may not run on an older one.

## Models

Nothing is downloaded for you. Put the files in `./models` (it is git-ignored):

| File | From (huggingface.co) | Size | Used by |
|---|---|---|---|
| `Qwen3-0.6B-Q8_0.gguf` | Qwen/Qwen3-0.6B-GGUF | 0.6 GB | `coli-rs`, quick tests |
| `Qwen3-30B-A3B-Q4_K_M.gguf` | Qwen/Qwen3-30B-A3B-GGUF | 17.3 GB | `coli-rs`, MoE with expert streaming |
| `Qwen3-8B-Q4_K_M.gguf` | Qwen/Qwen3-8B-GGUF | 4.7 GB | `memorilla-rs`, decoder |
| `Qwen3-Embedding-4B-Q8_0.gguf` | Qwen/Qwen3-Embedding-4B-GGUF | 4.0 GB | `memorilla-rs`, encoder |
| `memorilla/stage3/memory.pt`, `memorilla/pmv2/memory.pt` (with their `config.json`) | memorilla/Memorilla-Qwen3-8B | 0.4 GB each | `memorilla-rs`, memory module |

## Quick start

```bash
# chat with a model
target/release/coli-rs chat models/Qwen3-0.6B-Q8_0.gguf

# a user's memory: add documents, ask, or serve it to an agent
target/release/memorilla-rs add --user priya memorilla-rs/examples/priya.jsonl
target/release/memorilla-rs ask --user priya -q "What is my dog's name?"
target/release/memorilla-rs serve --port 8000     # chat page at http://127.0.0.1:8000/
```

The server speaks the OpenAI chat API; the `user` field of a request selects whose
memory is used. See [`memorilla-rs/README.md`](memorilla-rs/README.md#server-for-an-agent).

## License

Copyright 2026 Plenum Tech. Licensed under Apache-2.0, as both upstream projects are.
colibri is by Vincenzo Fornaro and Memorilla by its authors at Stanford University;
[NOTICE](NOTICE) says which parts of this code were written from which of their
sources. See [LICENSE](LICENSE).
