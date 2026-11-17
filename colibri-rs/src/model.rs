//! The Qwen transformer (GGUF architectures `qwen3`, `qwen3moe`, `qwen2`), CPU only.
//!
//! Dense matrices (attention, router, dense FFN, embeddings) are views over the mapped
//! file and are computed in their own blocks. The routed experts of a MoE model are
//! not mapped: they come from the ExpertStore, read from the disk when routed to.

use crate::expert::{ExpertStore, ExpertTensor};
use crate::gguf::{Gguf, TensorInfo};
use crate::quant::{self, axpy, dequant_row, dot, matmul, GType};
use memmap2::Mmap;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

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
    pub n_expert: usize,
    pub n_expert_used: usize,
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

enum Ffn {
    Dense { gate: Mat, up: Mat, down: Mat },
    Moe { router: Mat, gate: ExpertTensor, up: ExpertTensor, down: ExpertTensor },
}

struct Layer {
    attn_norm: Vec<f32>,
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

/// Where the time of the forward pass goes (nanoseconds), printed by `coli-rs run --prof`.
#[derive(Default)]
pub struct Prof {
    pub attn_mm: AtomicU64,
    pub attn: AtomicU64,
    pub ffn_dense: AtomicU64,
    pub router: AtomicU64,
    pub expert_mm: AtomicU64,
    pub head: AtomicU64,
}

impl Prof {
    pub fn line(&self) -> String {
        let f = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1e9;
        format!(
            "time: attention matmul {:.1}s · attention {:.1}s · dense ffn {:.1}s · router {:.1}s · experts (fetch + matmul) {:.1}s · output layer {:.1}s",
            f(&self.attn_mm), f(&self.attn), f(&self.ffn_dense), f(&self.router), f(&self.expert_mm), f(&self.head)
        )
    }
}

fn tick(a: &AtomicU64, t: Instant) -> Instant {
    a.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    Instant::now()
}

/// The KV cache of one conversation.
pub struct State {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
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
    fn opt_vec(&mut self, name: &str) -> Result<Option<Vec<f32>>, String> {
        if self.g.tensor(name).is_some() {
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

impl Model {
    pub fn load(path: &Path, opts: &LoadOptions) -> Result<Model, String> {
        let g = Gguf::open(path)?;
        Self::from_gguf(&g, opts)
    }

    pub fn from_gguf(g: &Gguf, opts: &LoadOptions) -> Result<Model, String> {
        let arch = g.str("general.architecture").unwrap_or("").to_string();
        if !matches!(arch.as_str(), "qwen3" | "qwen3moe" | "qwen2") {
            return Err(format!(
                "GGUF architecture '{arch}' is not supported by the Rust engine yet (qwen3, qwen3moe and qwen2 are)"
            ));
        }
        let key = |k: &str| format!("{arch}.{k}");
        let need = |k: &str| g.int(&key(k)).map(|v| v as usize).ok_or_else(|| format!("GGUF has no {}", key(k)));
        let n_layer = need("block_count")?;
        let n_embd = need("embedding_length")?;
        let n_head = need("attention.head_count")?;
        let n_kv_head = g.int(&key("attention.head_count_kv")).map(|v| v as usize).unwrap_or(n_head);
        let head_dim = g.int(&key("attention.key_length")).map(|v| v as usize).unwrap_or(n_embd / n_head.max(1));
        let n_expert = g.int(&key("expert_count")).unwrap_or(0) as usize;
        let n_expert_used = g.int(&key("expert_used_count")).unwrap_or(0) as usize;
        if n_head == 0 || n_kv_head == 0 || n_head % n_kv_head != 0 || head_dim % 2 != 0 {
            return Err(format!("bad attention shape: {n_head} heads, {n_kv_head} kv heads, head dim {head_dim}"));
        }

        let mut ld = Loader { g, dense_bytes: 0, types: Default::default() };
        let embed = ld.mat("token_embd.weight")?;
        if embed.cols != n_embd {
            return Err(format!("token_embd.weight has rows of {}, the model is {n_embd} wide", embed.cols));
        }
        let lm_head = if g.tensor("output.weight").is_some() { ld.mat("output.weight")? } else { embed.clone() };
        let out_norm = ld.vec("output_norm.weight")?;
        let moe = n_expert > 0 && g.tensor("blk.0.ffn_gate_inp.weight").is_some();
        let mut expert_bytes = 0usize;
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            let ffn = if moe {
                let router = ld.mat(&p("ffn_gate_inp.weight"))?;
                let (gate, b1) = ld.experts(&p("ffn_gate_exps.weight"), n_expert)?;
                let (up, b2) = ld.experts(&p("ffn_up_exps.weight"), n_expert)?;
                let (down, b3) = ld.experts(&p("ffn_down_exps.weight"), n_expert)?;
                if gate.cols != n_embd || up.cols != n_embd || down.rows != n_embd || down.cols != gate.rows || up.rows != gate.rows {
                    return Err(format!("layer {i}: expert shapes do not fit the model"));
                }
                expert_bytes += b1 + b2 + b3;
                Ffn::Moe { router, gate, up, down }
            } else {
                Ffn::Dense { gate: ld.mat(&p("ffn_gate.weight"))?, up: ld.mat(&p("ffn_up.weight"))?, down: ld.mat(&p("ffn_down.weight"))? }
            };
            let layer = Layer {
                attn_norm: ld.vec(&p("attn_norm.weight"))?,
                wq: ld.mat(&p("attn_q.weight"))?,
                wk: ld.mat(&p("attn_k.weight"))?,
                wv: ld.mat(&p("attn_v.weight"))?,
                wo: ld.mat(&p("attn_output.weight"))?,
                bq: ld.opt_vec(&p("attn_q.bias"))?,
                bk: ld.opt_vec(&p("attn_k.bias"))?,
                bv: ld.opt_vec(&p("attn_v.bias"))?,
                bo: ld.opt_vec(&p("attn_output.bias"))?,
                q_norm: ld.opt_vec(&p("attn_q_norm.weight"))?,
                k_norm: ld.opt_vec(&p("attn_k_norm.weight"))?,
                ffn_norm: ld.vec(&p("ffn_norm.weight"))?,
                ffn,
            };
            if layer.wq.rows != n_head * head_dim || layer.wk.rows != n_kv_head * head_dim || layer.wv.rows != n_kv_head * head_dim {
                return Err(format!("layer {i}: attention matrices do not fit {n_head}x{head_dim} / {n_kv_head}x{head_dim}"));
            }
            layers.push(layer);
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
            n_expert: if moe { n_expert } else { 0 },
            n_expert_used: if moe { n_expert_used.clamp(1, n_expert) } else { 0 },
            arch,
        };
        let half = head_dim / 2;
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
        State { k: vec![Vec::new(); self.cfg.n_layer], v: vec![Vec::new(); self.cfg.n_layer], n_past: 0 }
    }

    fn rope(&self, x: &mut [f32], n_heads: usize, pos: usize) {
        let hd = self.cfg.head_dim;
        let half = hd / 2;
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

    fn moe(&self, li: usize, h: &[f32], s: usize, out: &mut [f32], router: &Mat, gate: &ExpertTensor, up: &ExpertTensor, down: &ExpertTensor) -> Result<(), String> {
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
        tick(&self.prof.expert_mm, t);
        Ok(())
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
        let s = x.len() / c.n_embd;
        if s == 0 || x.len() != s * c.n_embd {
            return Err(format!("forward: {} input values are not whole rows of {}", x.len(), c.n_embd));
        }
        if st.n_past + s > n_ctx {
            return Err(format!("the context is full ({} + {s} tokens, limit {n_ctx}); raise --ctx or start a new conversation", st.n_past));
        }
        let (d, hd, nh, nkv) = (c.n_embd, c.head_dim, c.n_head, c.n_kv_head);
        let (qd, kvd) = (nh * hd, nkv * hd);
        let group = nh / nkv;
        let scale = 1.0 / (hd as f32).sqrt();
        let p0 = st.n_past;

        let mut h = vec![0f32; s * d];
        let mut q = vec![0f32; s * qd];
        let mut kk = vec![0f32; s * kvd];
        let mut vv = vec![0f32; s * kvd];
        let mut att = vec![0f32; s * qd];
        let mut o = vec![0f32; s * d];

        for (li, l) in self.layers.iter().enumerate() {
            // ---- attention ----
            let mut t = Instant::now();
            h.copy_from_slice(&x);
            rmsnorm(&mut h, &l.attn_norm, c.eps);
            l.wq.mul(&h, s, &mut q);
            l.wk.mul(&h, s, &mut kk);
            l.wv.mul(&h, s, &mut vv);
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
            att.par_chunks_mut(hd).enumerate().for_each_init(
                || Vec::<f32>::new(),
                |sc, (idx, out)| {
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
                    out.fill(0.0);
                    for t in 0..n {
                        axpy(sc[t] / sum, &cv[t * kvd + kh * hd..t * kvd + (kh + 1) * hd], out);
                    }
                },
            );
            t = tick(&self.prof.attn, t);
            l.wo.mul(&att, s, &mut o);
            if let Some(b) = &l.bo {
                for row in o.chunks_exact_mut(d) {
                    axpy(1.0, b, row);
                }
            }
            axpy(1.0, &o, &mut x);

            t = tick(&self.prof.attn_mm, t);

            // ---- feed-forward ----
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
                Ffn::Moe { router, gate, up, down } => self.moe(li, &h, s, &mut o, router, gate, up, down)?,
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
    /// Keep the first `n` positions (a conversation whose beginning is unchanged
    /// reuses it: c/kv_prefix.h).
    pub fn truncate(&mut self, n: usize) {
        if n >= self.n_past {
            return;
        }
        for (k, v) in self.k.iter_mut().zip(self.v.iter_mut()) {
            let per = if self.n_past > 0 { k.len() / self.n_past } else { 0 };
            k.truncate(n * per);
            v.truncate(n * per);
        }
        self.n_past = n;
    }

    pub fn bytes(&self) -> usize {
        self.k.iter().chain(self.v.iter()).map(|v| v.len() * 4).sum()
    }
}
