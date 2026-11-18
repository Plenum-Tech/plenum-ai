//! Plenum Memory: per-user latent semantic memory for a local LLM.
//!
//! A user's documents are embedded once by an encoder and kept. For each question, a
//! small memory module compresses the whole collection into 16 vectors, conditioned on
//! the question, and a frozen decoder reads them as 16 input positions in its system
//! prompt. Inference-only port of <https://github.com/snap-stanford/memorilla>, on the
//! plenum-engine engine (CPU, GGUF models).

pub mod eval;
pub mod memory;
pub mod metrics;
pub mod safetensors;
pub mod stack;
pub mod store;
pub mod torchpt;
