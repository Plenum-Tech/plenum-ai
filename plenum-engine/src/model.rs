//! The Qwen transformers, CPU only, read from GGUF in place:
//!
//! - `qwen3`, `qwen2`: dense; every layer attention.
//! - `qwen3moe`: Qwen3 MoE (Qwen3-30B-A3B, Qwen3-Coder); routed experts.
//! - `qwen35`: the dense Qwen3.5 / Qwen3.8 models (Qwen3.8-27B). Hybrid: three of every
//!   four layers are Gated DeltaNet (linear attention with a recurrent state), the
//!   fourth is gated attention with partial rotary.
//! - `qwen35moe`: the same layers with routed experts and a gated shared expert
//!   (Qwen3.5 / Qwen3.6 MoE).
//!
//! The hybrid layers are a port of `c/qwen36.c` (`deltanet`, `attention`), including
//! the three things llama.cpp's GGUF conversion changes and the engine reads as stored:
//! RMSNorm weights carry the +1 already, `ssm_a` holds A = -exp(A_log), and value head
//! h of a DeltaNet layer reads key head h % key_heads ("tiled"), not h / repeat.
//!
//! Dense matrices are views over the mapped file and are computed in their own blocks.
//! The routed experts of a MoE model come from the ExpertStore, read when routed to.

use crate::expert::{ExpertStore, ExpertTensor};
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{self, axpy, dequant_row, dot, matmul, GType};
use memmap2::Mmap;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// The DeltaNet layers' shape (GGUF `ssm.*`).
#[derive(Debug, Clone, Copy)]
pub struct DeltaCfg {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
}

impl DeltaCfg {
    pub fn conv_dim(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }
    pub fn value_dim(&self) -> usize {
        self.v_heads * self.v_dim
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub arch: String,
    pub name: String,
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_ctx_train: usize,
    pub eps: f32,
    pub rope_base: f32,
    /// Dims of each head that are rotated (the first `rope_dim`); the rest pass as they are.
    pub rope_dim: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Hybrid models: the DeltaNet shape, and how many layers are attention.
    pub delta: Option<DeltaCfg>,
    pub n_attn_layers: usize,
}

/// A matrix [rows, cols] in the mapped file.
#[derive(Clone)]
pub struct Mat {
    map: Arc<Mmap>,
    off: usize,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    fn bytes(&self) -> &[u8] {
        &self.map[self.off..self.off + self.ty.bytes_for(self.cols) * self.rows]
    }
    /// y[s, rows] = x[s, cols] @ W^T
    pub fn mul(&self, x: &[f32], s: usize, y: &mut [f32]) {
        matmul(self.bytes(), self.ty, self.rows, self.cols, x, s, y);
    }
    pub fn row(&self, r: usize, out: &mut [f32]) {
        let rb = self.ty.bytes_for(self.cols);
        dequant_row(self.ty, &self.bytes()[r * rb..(r + 1) * rb], out);
    }
}

/// The always-on expert of a qwen35moe layer, scaled by a sigmoid gate of the input.
struct Shared {
    gate: Mat,
    up: Mat,
    down: Mat,
    gate_inp: Option<Vec<f32>>,
}

enum Ffn {
    Dense { gate: Mat, up: Mat, down: Mat },
    Moe { router: Mat, gate: ExpertTensor, up: ExpertTensor, down: ExpertTensor, shared: Option<Shared> },
}

struct Attn {
    /// [heads * head_dim] rows, or [heads * 2 * head_dim] when `gated`: per head the query
    /// then the output gate
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    bq: Option<Vec<f32>>,
    bk: Option<Vec<f32>>,
    bv: Option<Vec<f32>>,
    bo: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    gated: bool,
}

/// A Gated DeltaNet layer (HF `linear_attn`).
struct Delta {
    qkv: Mat,
    z: Mat,
    beta: Mat,
    alpha: Mat,
    /// [conv_dim][kernel]: per channel its taps, the last one for the current token
    conv: Vec<f32>,
    dt_bias: Vec<f32>,
    /// A = -exp(A_log), per value head
    a: Vec<f32>,
    /// the gated RMSNorm's weight, [v_dim], shared by every head
    norm: Vec<f32>,
    out: Mat,
}

enum Mixer {
    Attn(Attn),
    Delta(Delta),
}

struct Layer {
    attn_norm: Vec<f32>,
    mixer: Mixer,
    ffn_norm: Vec<f32>,
    ffn: Ffn,
}

pub struct Model {
    pub cfg: Config,
    embed: Mat,
    layers: Vec<Layer>,
    out_norm: Vec<f32>,
    lm_head: Mat,
    inv_freq: Vec<f32>,
    pub experts: Option<ExpertStore>,
    /// Bytes of the file that are the resident (mapped) part and the streamed part.
    pub dense_bytes: usize,
    pub expert_bytes: usize,
    pub quant_summary: String,
    pub prof: Prof,
}

/// Where the time of the forward pass goes (nanoseconds), printed by `plenum-engine run --prof`.
#[derive(Default)]
pub struct Prof {
    pub attn_mm: AtomicU64,
    pub attn: AtomicU64,
    pub delta: AtomicU64,
    pub ffn_dense: AtomicU64,
    pub router: AtomicU64,
    pub expert_mm: AtomicU64,
    pub head: AtomicU64,
}

impl Prof {
    pub fn line(&self) -> String {
        let f = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
        format!(
            "time: attention matmul {:.1}s · attention {:.1}s · deltanet {:.1}s · dense ffn {:.1}s · router {:.1}s · experts (fetch + matmul) {:.1}s · output layer {:.1}s",
            f(&self.attn_mm), f(&self.attn), f(&self.delta), f(&self.ffn_dense), f(&self.router), f(&self.expert_mm), f(&self.head)
        )
    }
}

fn tick(a: &AtomicU64, t: Instant) -> Instant {
    a.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Instant::now()
}

/// The state of one conversation: the KV cache of the attention layers, and the conv
/// window and recurrent state of the DeltaNet layers.
pub struct State {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    /// per DeltaNet layer: the last kernel-1 inputs of every channel, [conv_dim][kernel-1]
    conv: Vec<Vec<f32>>,
    /// per DeltaNet layer: per value head a [k_dim][v_dim] matrix
    rec: Vec<Vec<f32>>,
    pub n_past: usize,
}

pub struct LoadOptions {
    /// MoE models. 0 (the default): experts are used from the mapped file and the
    /// operating system decides which stay in RAM. Otherwise: experts are read into a
    /// private cache of at most this many bytes.
    pub expert_cache_bytes: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions { expert_cache_bytes: 0 }
    }
}

fn gtype(t: &TensorInfo) -> Result<GType, String> {
    GType::from_id(t.type_id).ok_or_else(|| {
        format!(
            "tensor '{}' is {}: this engine computes F32, F16, BF16, Q8_0, Q4_0/1, Q5_0/1 and Q2_K..Q6_K blocks",
            t.name,
            quant::type_id_name(t.type_id)
        )
    })
}

struct Loader<'a> {
    g: &'a Gguf,
    dense_bytes: usize,
    types: std::collections::BTreeMap<&'static str, usize>,
}

impl<'a> Loader<'a> {
    fn info(&self, name: &str) -> Result<&'a TensorInfo, String> {
        self.g.tensor(name).ok_or_else(|| format!("GGUF has no tensor '{name}'"))
    }
    fn has(&self, name: &str) -> bool {
        self.g.tensor(name).is_some()
    }
    fn check_fit(&self, t: &TensorInfo, ty: GType) -> Result<(), String> {
        let (be, _) = ty.block();
        if t.dims.is_empty() || t.dims[0] % be != 0 {
            return Err(format!("tensor '{}': row length {:?} is not a whole number of {} blocks", t.name, t.dims.first(), ty.name()));
        }
        let need = ty.bytes_for(t.n_elements());
        if t.offset + need > self.g.map.len() {
            return Err(format!("tensor '{}' runs past the end of the file (is the download complete?)", t.name));
        }
        Ok(())
    }
    fn mat(&mut self, name: &str) -> Result<Mat, String> {
        let t = self.info(name)?;
        let ty = gtype(t)?;
        if t.dims.len() != 2 {
            return Err(format!("tensor '{name}' has {} dimensions, a matrix has 2", t.dims.len()));
        }
        self.check_fit(t, ty)?;
        let bytes = ty.bytes_for(t.n_elements());
        self.dense_bytes += bytes;
        *self.types.entry(ty.name()).or_insert(0) += bytes;
        Ok(Mat { map: self.g.map.clone(), off: t.offset, ty, rows: t.dims[1], cols: t.dims[0] })
    }
    /// A matrix of the given shape, or an error that names both.
    fn mat_shape(&mut self, name: &str, rows: usize, cols: usize) -> Result<Mat, String> {
        let m = self.mat(name)?;
        if m.rows != rows || m.cols != cols {
            return Err(format!("tensor '{name}' is [{} x {}], expected [{rows} x {cols}]", m.rows, m.cols));
        }
        Ok(m)
    }
    fn vec(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.info(name)?;
        let ty = gtype(t)?;
        self.check_fit(t, ty)?;
        let n = t.n_elements();
        let mut out = vec![0f32; n];
        dequant_row(ty, &self.g.map[t.offset..t.offset + ty.bytes_for(n)], &mut out);
        self.dense_bytes += n * 4;
        Ok(out)
    }
    fn vec_len(&mut self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let v = self.vec(name)?;
        if v.len() != len {
            return Err(format!("tensor '{name}' has {} values, expected {len}", v.len()));
        }
        Ok(v)
    }
    fn opt_vec(&mut self, name: &str) -> Result<Option<Vec<f32>>, String> {
        if self.has(name) {
            self.vec(name).map(Some)
        } else {
            Ok(None)
        }
    }
    fn experts(&mut self, name: &str, n_expert: usize) -> Result<(ExpertTensor, usize), String> {
        let t = self.info(name)?;
        let ty = gtype(t)?;
        if t.dims.len() != 3 || t.dims[2] != n_expert {
            return Err(format!("tensor '{name}' has shape {:?}, expected [cols, rows, {n_expert}]", t.dims));
        }
        self.check_fit(t, ty)?;
        let bytes = ty.bytes_for(t.n_elements());
        *self.types.entry(ty.name()).or_insert(0) += bytes;
        Ok((ExpertTensor { offset: t.offset, ty, rows: t.dims[1], cols: t.dims[0] }, bytes))
    }
}

fn rmsnorm(x: &mut [f32], w: &[f32], eps: f32) {
    let n = w.len();
    for row in x.chunks_exact_mut(n) {
        let ms = dot(row, row) / n as f32;
        let r = 1.0 / (ms + eps).sqrt();
        for (v, g) in row.iter_mut().zip(w) {
            *v = *v * r * g;
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// log(1 + e^x), without overflow for large x.
#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// Scales `v` to unit length (with 1e-6 inside the root, as HF's l2norm), times `scale`.
#[inline]
fn l2norm(v: &mut [f32], scale: f32) {
    let n = (dot(v, v) + 1e-6).sqrt();
    let f = scale / n;
    for x in v.iter_mut() {
        *x *= f;
    }
}

impl Model {
    pub fn load(path: &Path, opts: &LoadOptions) -> Result<Model, String> {
        let g = Gguf::open(path)?;
        Self::from_gguf(&g, opts)
    }

    pub fn from_gguf(g: &Gguf, opts: &LoadOptions) -> Result<Model, String> {
        let arch = g.str("general.architecture").unwrap_or("").to_string();
        let hybrid = matches!(arch.as_str(), "qwen35" | "qwen35moe");
        if !hybrid && !matches!(arch.as_str(), "qwen3" | "qwen3moe" | "qwen2") {
            return Err(format!(
                "GGUF architecture '{arch}' is not supported by the Rust engine yet (qwen3, qwen3moe, qwen2, qwen35 and qwen35moe are)"
            ));
        }
        let key = |k: &str| format!("{arch}.{k}");
        let need = |k: &str| g.int(&key(k)).map(|v| v as usize).ok_or_else(|| format!("GGUF has no {}", key(k)));
        // the multi-token-prediction layers at the end are not part of the forward pass
        let n_mtp = g.int(&key("nextn_predict_layers")).unwrap_or(0) as usize;
        let n_layer = need("block_count")?.checked_sub(n_mtp).filter(|n| *n > 0).ok_or("GGUF block_count leaves no layers")?;
        let n_embd = need("embedding_length")?;
        let n_head = need("attention.head_count")?;
        let n_kv_head = g.int(&key("attention.head_count_kv")).map(|v| v as usize).unwrap_or(n_head);
        let head_dim = g.int(&key("attention.key_length")).map(|v| v as usize).unwrap_or(n_embd / n_head.max(1));
        let n_expert = g.int(&key("expert_count")).unwrap_or(0) as usize;
        let n_expert_used = g.int(&key("expert_used_count")).unwrap_or(0) as usize;
        let rope_dim = g.int(&key("rope.dimension_count")).map(|v| v as usize).unwrap_or(head_dim);
        if n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 || head_dim % 2 != 0 || rope_dim % 2 != 0 || rope_dim > head_dim {
            return Err(format!("bad attention shape: {n_head} heads, {n_kv_head} kv heads, head dim {head_dim}, rotary {rope_dim}"));
        }
        let delta = if hybrid {
            let k_heads = need("ssm.group_count")?;
            let v_heads = need("ssm.time_step_rank")?;
            let k_dim = need("ssm.state_size")?;
            let inner = need("ssm.inner_size")?;
            let conv_kernel = need("ssm.conv_kernel")?;
            if k_heads == 0 || v_heads == 0 || inner % v_heads != 0 || conv_kernel < 2 {
                return Err(format!("bad DeltaNet shape: {k_heads} key heads, {v_heads} value heads, inner {inner}, kernel {conv_kernel}"));
            }
            Some(DeltaCfg { k_heads, v_heads, k_dim, v_dim: inner / v_heads, conv_kernel })
        } else {
            None
        };

        let mut ld = Loader { g, dense_bytes: 0, types: Default::default() };
        let embed = ld.mat("token_embd.weight")?;
        if embed.cols != n_embd {
            return Err(format!("token_embd.weight has rows of {}, the model is {n_embd} wide", embed.cols));
        }
        let lm_head = if ld.has("output.weight") { ld.mat("output.weight")? } else { embed.clone() };
        let out_norm = ld.vec_len("output_norm.weight", n_embd)?;
        let moe = n_expert > 0 && ld.has("blk.0.ffn_gate_inp.weight");
        let mut expert_bytes = 0usize;
        let mut layers = Vec::with_capacity(n_layer);
        let mut n_attn_layers = 0;
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let ffn = if moe {
                let router = ld.mat_shape(&p("ffn_gate_inp.weight"), n_expert, n_embd)?;
                let (gate, b1) = ld.experts(&p("ffn_gate_exps.weight"), n_expert)?;
                let (up, b2) = ld.experts(&p("ffn_up_exps.weight"), n_expert)?;
                let (down, b3) = ld.experts(&p("ffn_down_exps.weight"), n_expert)?;
                if gate.cols != n_embd || up.cols != n_embd || down.rows != n_embd || down.cols != gate.rows || up.rows != gate.rows {
                    return Err(format!("layer {i}: expert shapes do not fit the model"));
                }
                expert_bytes += b1 + b2 + b3;
                let shared = if ld.has(&p("ffn_up_shexp.weight")) {
                    let up_s = ld.mat(&p("ffn_up_shexp.weight"))?;
                    Some(Shared {
                        gate: ld.mat_shape(&p("ffn_gate_shexp.weight"), up_s.rows, n_embd)?,
                        down: ld.mat_shape(&p("ffn_down_shexp.weight"), n_embd, up_s.rows)?,
                        up: up_s,
                        gate_inp: match ld.opt_vec(&p("ffn_gate_inp_shexp.weight"))? {
                            Some(v) if v.len() != n_embd => return Err(format!("layer {i}: ffn_gate_inp_shexp has {} values", v.len())),
                            v => v,
                        },
                    })
                } else {
                    None
                };
                Ffn::Moe { router, gate, up, down, shared }
            } else {
                let up = ld.mat(&p("ffn_up.weight"))?;
                Ffn::Dense {
                    gate: ld.mat_shape(&p("ffn_gate.weight"), up.rows, n_embd)?,
                    down: ld.mat_shape(&p("ffn_down.weight"), n_embd, up.rows)?,
                    up,
                }
            };
            // a hybrid layer is attention when it has the attention tensors, DeltaNet otherwise
            let mixer = if ld.has(&p("attn_q.weight")) {
                n_attn_layers += 1;
                let gated = hybrid;
                let qrows = n_head * head_dim * if gated { 2 } else { 1 };
                Mixer::Attn(Attn {
                    wq: ld.mat_shape(&p("attn_q.weight"), qrows, n_embd)?,
                    wk: ld.mat_shape(&p("attn_k.weight"), n_kv_head * head_dim, n_embd)?,
                    wv: ld.mat_shape(&p("attn_v.weight"), n_kv_head * head_dim, n_embd)?,
                    wo: ld.mat_shape(&p("attn_output.weight"), n_embd, n_head * head_dim)?,
                    bq: ld.opt_vec(&p("attn_q.bias"))?,
                    bk: ld.opt_vec(&p("attn_k.bias"))?,
                    bv: ld.opt_vec(&p("attn_v.bias"))?,
                    bo: ld.opt_vec(&p("attn_output.bias"))?,
                    q_norm: ld.opt_vec(&p("attn_q_norm.weight"))?,
                    k_norm: ld.opt_vec(&p("attn_k_norm.weight"))?,
                    gated,
                })
            } else if let Some(dc) = delta {
                let (cd, vd) = (dc.conv_dim(), dc.value_dim());
                let conv = ld.vec_len(&p("ssm_conv1d.weight"), cd * dc.conv_kernel)?;
                // Ollama's converter calls the dt bias ssm_dt (the same values)
                let dt = if ld.has(&p("ssm_dt.bias")) { p("ssm_dt.bias") } else { p("ssm_dt") };
                Mixer::Delta(Delta {
                    qkv: ld.mat_shape(&p("attn_qkv.weight"), cd, n_embd)?,
                    z: ld.mat_shape(&p("attn_gate.weight"), vd, n_embd)?,
                    beta: ld.mat_shape(&p("ssm_beta.weight"), dc.v_heads, n_embd)?,
                    alpha: ld.mat_shape(&p("ssm_alpha.weight"), dc.v_heads, n_embd)?,
                    conv,
                    dt_bias: ld.vec_len(&dt, dc.v_heads)?,
                    a: ld.vec_len(&p("ssm_a"), dc.v_heads)?,
                    norm: ld.vec_len(&p("ssm_norm.weight"), dc.v_dim)?,
                    out: ld.mat_shape(&p("ssm_out.weight"), n_embd, vd)?,
                })
            } else {
                return Err(format!("layer {i} has no attention tensors"));
            };
            // the norm before the feed-forward: ffn_norm in llama.cpp's Qwen3, post_attention_norm in Qwen3.5
            let fnorm = if ld.has(&p("ffn_norm.weight")) { p("ffn_norm.weight") } else { p("post_attention_norm.weight") };
            layers.push(Layer { attn_norm: ld.vec_len(&p("attn_norm.weight"), n_embd)?, mixer, ffn_norm: ld.vec_len(&fnorm, n_embd)?, ffn });
        }
        if n_attn_layers == 0 && delta.is_none() {
            return Err("the model has no attention layers".into());
        }
        let n_ff = match &layers[0].ffn {
            Ffn::Dense { gate, .. } => gate.rows,
            Ffn::Moe { gate, .. } => gate.rows,
        };
        let cfg = Config {
            name: g.str("general.name").unwrap_or("model").to_string(),
            n_layer,
            n_embd,
            n_head,
            n_kv_head,
            head_dim,
            n_ff,
            n_vocab: embed.rows,
            n_ctx_train: g.int(&key("context_length")).unwrap_or(4096) as usize,
            eps: g.float(&key("attention.layer_norm_rms_epsilon")).unwrap_or(1e-6) as f32,
            rope_base: g.float(&key("rope.freq_base")).unwrap_or(1_000_000.0) as f32,
            rope_dim,
            n_expert: if moe { n_expert } else { 0 },
            n_expert_used: if moe { n_expert_used.clamp(1, n_expert) } else { 0 },
            delta,
            n_attn_layers,
            arch,
        };
        // rotary frequencies over the rotated dims: theta^(-2i/rope_dim)
        let half = rope_dim / 2;
        let inv_freq = (0..half).map(|i| cfg.rope_base.powf(-(i as f32) / half as f32)).collect();
        let experts = if moe {
            let map = if opts.expert_cache_bytes == 0 { Some(g.map.clone()) } else { None };
            Some(ExpertStore::open(&g.path, opts.expert_cache_bytes, map).map_err(|e| format!("{}: {e}", g.path.display()))?)
        } else {
            None
        };
        let total: usize = ld.types.values().sum();
        let quant_summary = ld
            .types
            .iter()
            .map(|(k, v)| format!("{k} {:.0}%", 100.0 * *v as f64 / total.max(1) as f64))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(Model { cfg, embed, layers, out_norm, lm_head, inv_freq, experts, dense_bytes: ld.dense_bytes, expert_bytes, quant_summary, prof: Prof::default() })
    }

    pub fn new_state(&self) -> State {
        let n = self.cfg.n_layer;
        let mut st = State { k: vec![Vec::new(); n], v: vec![Vec::new(); n], conv: vec![Vec::new(); n], rec: vec![Vec::new(); n], n_past: 0 };
        st.reset_recurrent(self);
        st
    }

    /// Whether the state can be cut back to an earlier position. A DeltaNet layer's
    /// state is a running sum over every token so far: it can only start over.
    pub fn rewindable(&self) -> bool {
        self.cfg.delta.is_none()
    }

    /// Rotates the first rope_dim dims of each head: the pairs (i, i + rope_dim/2).
    fn rope(&self, x: &mut [f32], n_heads: usize, pos: usize) {
        let (hd, half) = (self.cfg.head_dim, self.cfg.rope_dim / 2);
        for hh in 0..n_heads {
            let v = &mut x[hh * hd..(hh + 1) * hd];
            for i in 0..half {
                let (s, c) = (pos as f32 * self.inv_freq[i]).sin_cos();
                let (a, b) = (v[i], v[i + half]);
                v[i] = a * c - b * s;
                v[i + half] = a * s + b * c;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn moe(&self, li: usize, h: &[f32], s: usize, out: &mut [f32], router: &Mat, gate: &ExpertTensor, up: &ExpertTensor, down: &ExpertTensor, shared: Option<&Shared>) -> Result<(), String> {
        let (ne, k, d, ff) = (self.cfg.n_expert, self.cfg.n_expert_used, self.cfg.n_embd, gate.rows);
        let store = self.experts.as_ref().expect("a MoE model has an expert store");
        let mut logits = vec![0f32; s * ne];
        let t = Instant::now();
        router.mul(h, s, &mut logits);
        let t = tick(&self.prof.router, t);
        // (token, expert, weight) for every routed expert of every token
        let mut jobs: Vec<(usize, usize, f32)> = Vec::with_capacity(s * k);
        for si in 0..s {
            // softmax over every expert, the top k, their weights renormalized to sum to 1
            let lg = &mut logits[si * ne..(si + 1) * ne];
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for v in lg.iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            let mut order: Vec<usize> = (0..ne).collect();
            order.sort_by(|&a, &b| lg[b].partial_cmp(&lg[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            order.truncate(k);
            let wsum: f32 = order.iter().map(|&e| lg[e] / sum).sum();
            order.sort_unstable(); // by expert id: reads go forward in the file
            jobs.extend(order.iter().map(|&e| (si, e, (lg[e] / sum) / wsum)));
        }
        out[..s * d].fill(0.0);
        // One parallel region for a batch of experts, each expert computed on one thread:
        // an expert is three small products, too small to be worth splitting further.
        for batch in jobs.chunks(512) {
            // each distinct expert of the batch is fetched once, however many tokens routed to it
            let mut ids: Vec<usize> = batch.iter().map(|j| j.1).collect();
            ids.sort_unstable();
            ids.dedup();
            let fetched: Vec<_> = ids
                .par_iter()
                .map(|&e| store.get(li, e, gate, up, down).map_err(|e| format!("reading an expert: {e}")))
                .collect::<Result<_, _>>()?;
            let ys: Vec<Result<Vec<f32>, String>> = batch
                .par_iter()
                .map(|&(si, e, w)| {
                    let ex = &fetched[ids.binary_search(&e).expect("fetched above")];
                    let x = &h[si * d..(si + 1) * d];
                    let (mut g, mut u, mut y) = (vec![0f32; ff], vec![0f32; ff], vec![0f32; d]);
                    quant::matmul1_serial(&ex.gate, gate.ty, ff, d, x, &mut g);
                    quant::matmul1_serial(&ex.up, up.ty, ff, d, x, &mut u);
                    for (a, b) in g.iter_mut().zip(&u) {
                        *a = silu(*a) * b;
                    }
                    quant::matmul1_serial(&ex.down, down.ty, d, ff, &g, &mut y);
                    for v in y.iter_mut() {
                        *v *= w;
                    }
                    Ok(y)
                })
                .collect();
            for (&(si, _, _), y) in batch.iter().zip(ys) {
                axpy(1.0, &y?, &mut out[si * d..(si + 1) * d]);
            }
        }
        // the shared expert runs for every token, scaled by sigmoid(gate_inp . x)
        if let Some(sh) = shared {
            let sf = sh.up.rows;
            let (mut g, mut u, mut y) = (vec![0f32; s * sf], vec![0f32; s * sf], vec![0f32; s * d]);
            sh.gate.mul(h, s, &mut g);
            sh.up.mul(h, s, &mut u);
            for (a, b) in g.iter_mut().zip(&u) {
                *a = silu(*a) * b;
            }
            sh.down.mul(&g, s, &mut y);
            for si in 0..s {
                let x = &h[si * d..(si + 1) * d];
                let w = sh.gate_inp.as_ref().map(|gi| sigmoid(dot(gi, x))).unwrap_or(1.0);
                axpy(w, &y[si * d..(si + 1) * d], &mut out[si * d..(si + 1) * d]);
            }
        }
        tick(&self.prof.expert_mm, t);
        Ok(())
    }

    /// Softmax attention over the cache, for `s` new rows starting at position `p0`.
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, l: &Attn, li: usize, st: &mut State, h: &[f32], s: usize, p0: usize, out: &mut [f32]) {
        let c = &self.cfg;
        let (d, hd, nh, nkv) = (c.n_embd, c.head_dim, c.n_head, c.n_kv_head);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let mut t = Instant::now();
        let mut q = vec![0f32; s * qd];
        let mut gate = if l.gated { vec![0f32; s * qd] } else { Vec::new() };
        if l.gated {
            // per head the query then its output gate
            let mut qg = vec![0f32; s * 2 * qd];
            l.wq.mul(h, s, &mut qg);
            for (r, row) in qg.chunks_exact(2 * hd).enumerate() {
                q[r * hd..(r + 1) * hd].copy_from_slice(&row[..hd]);
                gate[r * hd..(r + 1) * hd].copy_from_slice(&row[hd..]);
            }
        } else {
            l.wq.mul(h, s, &mut q);
        }
        let mut kk = vec![0f32; s * kvd];
        let mut vv = vec![0f32; s * kvd];
        l.wk.mul(h, s, &mut kk);
        l.wv.mul(h, s, &mut vv);
        for (buf, bias, w) in [(&mut q, &l.bq, qd), (&mut kk, &l.bk, kvd), (&mut vv, &l.bv, kvd)] {
            if let Some(b) = bias {
                for row in buf.chunks_exact_mut(w) {
                    axpy(1.0, b, row);
                }
            }
        }
        if let Some(w) = &l.q_norm {
            rmsnorm(&mut q, w, c.eps);
        }
        if let Some(w) = &l.k_norm {
            rmsnorm(&mut kk, w, c.eps);
        }
        for i in 0..s {
            self.rope(&mut q[i * qd..(i + 1) * qd], nh, p0 + i);
            self.rope(&mut kk[i * kvd..(i + 1) * kvd], nkv, p0 + i);
        }
        t = tick(&self.prof.attn_mm, t);
        st.k[li].truncate(p0 * kvd);
        st.v[li].truncate(p0 * kvd);
        st.k[li].extend_from_slice(&kk);
        st.v[li].extend_from_slice(&vv);
        let (ck, cv) = (&st.k[li], &st.v[li]);
        let mut att = vec![0f32; s * qd];
        att.par_chunks_mut(hd).enumerate().for_each_init(
            Vec::<f32>::new,
            |sc, (idx, o)| {
                let (i, hh) = (idx / nh, idx % nh);
                let kh = hh / group;
                let n = p0 + i + 1; // causal: this token and everything before it
                let qv = &q[i * qd + hh * hd..i * qd + (hh + 1) * hd];
                sc.clear();
                let mut mx = f32::NEG_INFINITY;
                for t in 0..n {
                    let v = dot(qv, &ck[t * kvd + kh * hd..t * kvd + (kh + 1) * hd]) * scale;
                    mx = mx.max(v);
                    sc.push(v);
                }
                let mut sum = 0f32;
                for v in sc.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                o.fill(0.0);
                for t in 0..n {
                    axpy(sc[t] / sum, &cv[t * kvd + kh * hd..t * kvd + (kh + 1) * hd], o);
                }
            },
        );
        if l.gated {
            for (a, g) in att.iter_mut().zip(&gate) {
                *a *= sigmoid(*g);
            }
        }
        t = tick(&self.prof.attn, t);
        l.wo.mul(&att, s, out);
        if let Some(b) = &l.bo {
            for row in out[..s * d].chunks_exact_mut(d) {
                axpy(1.0, b, row);
            }
        }
        tick(&self.prof.attn_mm, t);
    }

    /// A Gated DeltaNet layer over `s` new rows (port of c/qwen36.c `deltanet`):
    /// projections, a causal depthwise conv with SiLU, then per value head the gated
    /// delta rule on a [k_dim x v_dim] state, a gated RMSNorm, and the out projection.
    fn deltanet(&self, l: &Delta, li: usize, st: &mut State, h: &[f32], s: usize, out: &mut [f32]) {
        let dc = self.cfg.delta.expect("a DeltaNet layer has its shape");
        let (d, eps) = (self.cfg.n_embd, self.cfg.eps);
        let (kh, vh, kd, vdim, kn) = (dc.k_heads, dc.v_heads, dc.k_dim, dc.v_dim, dc.conv_kernel);
        let (cd, vd) = (dc.conv_dim(), dc.value_dim());
        let key_tot = kh * kd;
        let scale = 1.0 / (kd as f32).sqrt();
        let t = Instant::now();
        let mut qkv = vec![0f32; s * cd];
        let mut z = vec![0f32; s * vd];
        let mut bproj = vec![0f32; s * vh];
        let mut aproj = vec![0f32; s * vh];
        l.qkv.mul(h, s, &mut qkv);
        l.z.mul(h, s, &mut z);
        l.beta.mul(h, s, &mut bproj);
        l.alpha.mul(h, s, &mut aproj);

        // the conv, channel by channel through every row, carrying the window
        let mut conv_out = vec![0f32; s * cd];
        {
            let ring = &mut st.conv[li];
            let w = &l.conv;
            // channels are independent: split them across threads
            let chunk = 256;
            let mut cols: Vec<(usize, &mut [f32])> = ring.chunks_mut(chunk * (kn - 1)).enumerate().collect();
            let parts: Vec<Vec<f32>> = cols
                .par_iter_mut()
                .map(|(ci, rg)| {
                    let c0 = *ci * chunk;
                    let nc = rg.len() / (kn - 1);
                    let mut part = vec![0f32; s * nc];
                    for j in 0..nc {
                        let cc = c0 + j;
                        let wc = &w[cc * kn..(cc + 1) * kn];
                        let r = &mut rg[j * (kn - 1)..(j + 1) * (kn - 1)];
                        for row in 0..s {
                            let xin = qkv[row * cd + cc];
                            let mut acc = wc[kn - 1] * xin;
                            for k in 0..kn - 1 {
                                acc += wc[k] * r[k];
                            }
                            part[row * nc + j] = silu(acc);
                            r.copy_within(1.., 0);
                            r[kn - 2] = xin;
                        }
                    }
                    part
                })
                .collect();
            let mut c0 = 0;
            for part in parts {
                let nc = part.len() / s;
                for row in 0..s {
                    conv_out[row * cd + c0..row * cd + c0 + nc].copy_from_slice(&part[row * nc..(row + 1) * nc]);
                }
                c0 += nc;
            }
        }

        // the gated delta rule: value heads are independent, each walks the rows in order
        let mut normed = vec![0f32; s * vd];
        let per_head: Vec<Vec<f32>> = st.rec[li]
            .par_chunks_mut(kd * vdim)
            .enumerate()
            .map(|(hh, sm)| {
                let ki = hh % kh; // GGUF: value heads tiled over the key heads
                let mut o = vec![0f32; s * vdim];
                let (mut qv, mut kv) = (vec![0f32; kd], vec![0f32; kd]);
                let (mut mem, mut delta) = (vec![0f32; vdim], vec![0f32; vdim]);
                for row in 0..s {
                    let cv = &conv_out[row * cd..(row + 1) * cd];
                    qv.copy_from_slice(&cv[ki * kd..(ki + 1) * kd]);
                    kv.copy_from_slice(&cv[key_tot + ki * kd..key_tot + (ki + 1) * kd]);
                    l2norm(&mut qv, scale);
                    l2norm(&mut kv, 1.0);
                    let v = &cv[2 * key_tot + hh * vdim..2 * key_tot + (hh + 1) * vdim];
                    let beta = sigmoid(bproj[row * vh + hh]);
                    let decay = (l.a[hh] * softplus(aproj[row * vh + hh] + l.dt_bias[hh])).exp();
                    // decay the state, then read what it holds for k
                    mem.fill(0.0);
                    for (k, sr) in sm.chunks_exact_mut(vdim).enumerate() {
                        for x in sr.iter_mut() {
                            *x *= decay;
                        }
                        axpy(kv[k], sr, &mut mem);
                    }
                    for j in 0..vdim {
                        delta[j] = (v[j] - mem[j]) * beta;
                    }
                    // write k (x) delta, and read the updated state for q
                    let orow = &mut o[row * vdim..(row + 1) * vdim];
                    for (k, sr) in sm.chunks_exact_mut(vdim).enumerate() {
                        axpy(kv[k], &delta, sr);
                        axpy(qv[k], sr, orow);
                    }
                    // gated RMSNorm: norm(o) * w * silu(z)
                    let r = 1.0 / (dot(orow, orow) / vdim as f32 + eps).sqrt();
                    let zr = &z[row * vd + hh * vdim..row * vd + (hh + 1) * vdim];
                    for j in 0..vdim {
                        orow[j] = orow[j] * r * l.norm[j] * silu(zr[j]);
                    }
                }
                o
            })
            .collect();
        for (hh, o) in per_head.iter().enumerate() {
            for row in 0..s {
                normed[row * vd + hh * vdim..row * vd + (hh + 1) * vdim].copy_from_slice(&o[row * vdim..(row + 1) * vdim]);
            }
        }
        l.out.mul(&normed, s, &mut out[..s * d]);
        tick(&self.prof.delta, t);
    }

    /// The embedding rows of `tokens`, [tokens.len(), n_embd].
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        let (d, nv) = (self.cfg.n_embd, self.cfg.n_vocab);
        let mut x = vec![0f32; tokens.len() * d];
        for (i, &t) in tokens.iter().enumerate() {
            if t as usize >= nv {
                return Err(format!("token id {t} is outside the vocabulary ({nv})"));
            }
            self.embed.row(t as usize, &mut x[i * d..(i + 1) * d]);
        }
        Ok(x)
    }

    /// Runs `tokens` after what `st` already holds and returns the logits of the last one.
    pub fn forward(&self, st: &mut State, tokens: &[u32], n_ctx: usize) -> Result<Vec<f32>, String> {
        self.forward_embd(st, self.embed_tokens(tokens)?, n_ctx)
    }

    /// The same, from input vectors [s, n_embd] instead of token ids: the way latent
    /// "soft" tokens (a memory module's output, a vision projector's) enter the model.
    pub fn forward_embd(&self, st: &mut State, x: Vec<f32>, n_ctx: usize) -> Result<Vec<f32>, String> {
        let last = self.forward_hidden(st, x, n_ctx)?;
        let mut logits = vec![0f32; self.lm_head.rows];
        let t = Instant::now();
        self.lm_head.mul(&last, 1, &mut logits);
        tick(&self.prof.head, t);
        Ok(logits)
    }

    /// A text embedding, as the Qwen3-Embedding models define it: the hidden state of
    /// the last token (which the caller makes the end-of-text token), L2-normalized.
    pub fn embed_text(&self, tokens: &[u32], n_ctx: usize) -> Result<Vec<f32>, String> {
        let mut st = self.new_state();
        let mut v = self.forward_hidden(&mut st, self.embed_tokens(tokens)?, n_ctx)?;
        let norm = dot(&v, &v).sqrt().max(1e-12);
        for x in v.iter_mut() {
            *x /= norm;
        }
        Ok(v)
    }

    /// Runs input vectors [s, n_embd] and returns the last position's hidden state after
    /// the final norm (what the output layer, or an embedding pooler, reads).
    pub fn forward_hidden(&self, st: &mut State, mut x: Vec<f32>, n_ctx: usize) -> Result<Vec<f32>, String> {
        let c = &self.cfg;
        let d = c.n_embd;
        let s = x.len() / d;
        if s == 0 || x.len() != s * d {
            return Err(format!("forward: {} input values are not whole rows of {d}", x.len()));
        }
        if st.n_past + s > n_ctx {
            return Err(format!("the context is full ({} + {s} tokens, limit {n_ctx}); raise --ctx or start a new conversation", st.n_past));
        }
        let p0 = st.n_past;
        let mut h = vec![0f32; s * d];
        let mut o = vec![0f32; s * d];
        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention or DeltaNet ----
            h.copy_from_slice(&x);
            rmsnorm(&mut h, &l.attn_norm, c.eps);
            match &l.mixer {
                Mixer::Attn(a) => self.attention(a, li, st, &h, s, p0, &mut o),
                Mixer::Delta(dn) => self.deltanet(dn, li, st, &h, s, &mut o),
            }
            axpy(1.0, &o, &mut x);

            // ---- feed-forward ----
            let t = Instant::now();
            h.copy_from_slice(&x);
            rmsnorm(&mut h, &l.ffn_norm, c.eps);
            match &l.ffn {
                Ffn::Dense { gate, up, down } => {
                    let ff = gate.rows;
                    let mut g = vec![0f32; s * ff];
                    let mut u = vec![0f32; s * ff];
                    gate.mul(&h, s, &mut g);
                    up.mul(&h, s, &mut u);
                    for (a, b) in g.iter_mut().zip(&u) {
                        *a = silu(*a) * b;
                    }
                    down.mul(&g, s, &mut o);
                    tick(&self.prof.ffn_dense, t);
                }
                Ffn::Moe { router, gate, up, down, shared } => self.moe(li, &h, s, &mut o, router, gate, up, down, shared.as_ref())?,
            }
            axpy(1.0, &o, &mut x);
        }
        st.n_past = p0 + s;

        let mut last = x[(s - 1) * d..].to_vec();
        rmsnorm(&mut last, &self.out_norm, c.eps);
        Ok(last)
    }
}

impl State {
    /// Zeroes the DeltaNet layers' conv windows and recurrent states.
    fn reset_recurrent(&mut self, m: &Model) {
        let Some(dc) = m.cfg.delta else { return };
        for (li, l) in m.layers.iter().enumerate() {
            if let Mixer::Delta(_) = l.mixer {
                self.conv[li] = vec![0f32; dc.conv_dim() * (dc.conv_kernel - 1)];
                self.rec[li] = vec![0f32; dc.v_heads * dc.k_dim * dc.v_dim];
            }
        }
    }

    /// Keeps the first `n` positions, so a conversation whose beginning is unchanged
    /// reuses it (c/kv_prefix.h), and returns how many positions are kept. A model with
    /// DeltaNet layers cannot cut its state back: unless `n` is where it already is, it
    /// starts over and keeps none.
    pub fn truncate(&mut self, m: &Model, n: usize) -> usize {
        if n >= self.n_past {
            return self.n_past;
        }
        if !m.rewindable() && n > 0 {
            return self.truncate(m, 0);
        }
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            let per = if self.n_past > 0 { k.len() / self.n_past } else { 0 };
            k.truncate(n * per);
            v.truncate(n * per);
        }
        if n == 0 {
            self.reset_recurrent(m);
        }
        self.n_past = n;
        n
    }

    pub fn bytes(&self) -> usize {
        self.k.iter().chain(self.v.iter()).chain(self.conv.iter()).chain(self.rec.iter()).map(|v| v.len() * 4).sum()
    }
}
