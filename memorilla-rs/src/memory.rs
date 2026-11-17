//! The Memorilla memory module (port of memorilla/memory.py, inference only).
//!
//! It compresses any number of document embeddings into `num_memories` vectors in the
//! decoder's input space, conditioned on the question:
//!
//! 1. the slots start as learned vectors;
//! 2. they cross-attend to the question embedding (residual + LayerNorm);
//! 3. `num_cross_attn_layers` blocks: pre-norm cross-attention from the slots to the
//!    normalized documents, then a pre-norm feed-forward network, both residual;
//! 4. a final LayerNorm and a linear projection to the decoder's width.
//!
//! The documents' keys and values do not depend on the question, so they are computed
//! once per collection (`write`) and every question only reads them (`read`).

use crate::torchpt::{load_state_dict, Tensor};
use colibri::quant::{axpy, dot};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

const LN_EPS: f32 = 1e-5;

struct Norm {
    w: Vec<f32>,
    b: Vec<f32>,
}

impl Norm {
    /// LayerNorm over rows of `w.len()`.
    fn apply(&self, x: &mut [f32]) {
        let d = self.w.len();
        for row in x.chunks_exact_mut(d) {
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let r = 1.0 / (var + LN_EPS).sqrt();
            for ((v, w), b) in row.iter_mut().zip(&self.w).zip(&self.b) {
                *v = (*v - mean) * r * w + b;
            }
        }
    }
}

/// torch.nn.MultiheadAttention's parameters: in_proj holds the query, key and value
/// projections stacked ([3d, d]).
struct Attn {
    in_w: Vec<f32>,
    in_b: Vec<f32>,
    out_w: Vec<f32>,
    out_b: Vec<f32>,
}

struct Block {
    ca_norm: Norm,
    ca: Attn,
    ffn_norm: Norm,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

pub struct MemoryModule {
    pub dim: usize,
    pub out_dim: usize,
    pub num_memories: usize,
    pub heads: usize,
    queries: Vec<f32>,
    qc: Attn,
    qc_norm: Norm,
    doc_norm: Norm,
    blocks: Vec<Block>,
    final_norm: Norm,
    proj_w: Vec<f32>,
    proj_b: Vec<f32>,
}

/// A document collection as the memory module holds it: per cross-attention block,
/// the documents' keys and values ([n, dim] each).
pub struct Collection {
    pub n: usize,
    layers: Vec<(Vec<f32>, Vec<f32>)>,
}

/// y[n, o] = x[n, i] @ W^T + b, W [o, i].
fn linear(w: &[f32], b: &[f32], o: usize, i: usize, x: &[f32]) -> Vec<f32> {
    let n = x.len() / i;
    let mut y = vec![0f32; n * o];
    if n == 1 {
        y.par_iter_mut().with_min_len(64).enumerate().for_each(|(k, v)| *v = dot(&w[k * i..(k + 1) * i], x) + b[k]);
    } else {
        y.par_chunks_mut(o).enumerate().for_each(|(r, yr)| {
            let xr = &x[r * i..(r + 1) * i];
            for (k, v) in yr.iter_mut().enumerate() {
                *v = dot(&w[k * i..(k + 1) * i], xr) + b[k];
            }
        });
    }
    y
}

/// erf to about 1e-7 (Abramowitz & Stegun 7.1.26), enough for f32 GELU.
fn erf(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs() as f64;
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t * (-x * x).exp();
    s * y as f32
}

/// torch.nn.GELU (the exact form).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

struct Weights(HashMap<String, Tensor>);

impl Weights {
    fn take(&mut self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let t = self.0.remove(name).ok_or_else(|| format!("the checkpoint has no '{name}'"))?;
        if t.data.len() != len {
            return Err(format!("'{name}' has {} values (shape {:?}), expected {len}", t.data.len(), t.shape));
        }
        Ok(t.data)
    }
    fn norm(&mut self, name: &str, d: usize) -> Result<Norm, String> {
        Ok(Norm { w: self.take(&format!("{name}.weight"), d)?, b: self.take(&format!("{name}.bias"), d)? })
    }
    fn attn(&mut self, name: &str, d: usize) -> Result<Attn, String> {
        Ok(Attn {
            in_w: self.take(&format!("{name}.in_proj_weight"), 3 * d * d)?,
            in_b: self.take(&format!("{name}.in_proj_bias"), 3 * d)?,
            out_w: self.take(&format!("{name}.out_proj.weight"), d * d)?,
            out_b: self.take(&format!("{name}.out_proj.bias"), d)?,
        })
    }
}

impl MemoryModule {
    /// Loads a checkpoint directory (`memory.pt` + `config.json`) or a `memory.pt` file.
    pub fn load(path: &Path) -> Result<MemoryModule, String> {
        let file = if path.is_dir() { path.join("memory.pt") } else { path.to_path_buf() };
        let mut w = Weights(load_state_dict(&file)?);
        let shape = |w: &Weights, n: &str| w.0.get(n).map(|t| t.shape.clone()).ok_or_else(|| format!("the checkpoint has no '{n}'"));
        let q = shape(&w, "memory_queries").map_err(|e| format!("{e} (modules with retrieval_init are not supported)"))?;
        let p = shape(&w, "memory_projection.weight")?;
        if q.len() != 3 || p.len() != 2 || p[1] != q[2] {
            return Err(format!("unexpected shapes: memory_queries {q:?}, memory_projection.weight {p:?}"));
        }
        let (num_memories, dim, out_dim) = (q[1], q[2], p[0]);
        if w.0.keys().any(|k| k.starts_with("encoder.")) {
            return Err("this checkpoint has document self-attention layers, which this port does not run".into());
        }
        // the number of heads leaves no trace in the weights: config.json holds it (default 8)
        let mut heads = 8usize;
        if let Ok(text) = std::fs::read_to_string(file.with_file_name("config.json")) {
            if let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&text) {
                heads = cfg.get("num_heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
            }
        }
        if heads == 0 || dim % heads != 0 {
            return Err(format!("{heads} heads do not divide the width {dim}"));
        }
        let n_blocks = (0..).take_while(|i| w.0.contains_key(&format!("cross_attn_layers.{i}.in_proj_weight"))).count();
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            blocks.push(Block {
                ca_norm: w.norm(&format!("cross_attn_norms.{i}"), dim)?,
                ca: w.attn(&format!("cross_attn_layers.{i}"), dim)?,
                ffn_norm: w.norm(&format!("ffn_norms.{i}"), dim)?,
                w1: w.take(&format!("ffn_layers.{i}.0.weight"), 4 * dim * dim)?,
                b1: w.take(&format!("ffn_layers.{i}.0.bias"), 4 * dim)?,
                w2: w.take(&format!("ffn_layers.{i}.3.weight"), 4 * dim * dim)?,
                b2: w.take(&format!("ffn_layers.{i}.3.bias"), dim)?,
            });
        }
        Ok(MemoryModule {
            dim,
            out_dim,
            num_memories,
            heads,
            queries: w.take("memory_queries", num_memories * dim)?,
            qc: w.attn("query_cond_attn", dim)?,
            qc_norm: w.norm("query_cond_norm", dim)?,
            doc_norm: w.norm("doc_norm", dim)?,
            blocks,
            final_norm: w.norm("layer_norm", dim)?,
            proj_w: w.take("memory_projection.weight", out_dim * dim)?,
            proj_b: w.take("memory_projection.bias", out_dim)?,
        })
    }

    /// The question-independent part: the documents' keys and values for every block.
    /// `docs` is [n, dim].
    pub fn write(&self, docs: &[f32]) -> Result<Collection, String> {
        let d = self.dim;
        if docs.len() % d != 0 {
            return Err(format!("{} document values are not whole embeddings of {d}", docs.len()));
        }
        let mut normed = docs.to_vec();
        self.doc_norm.apply(&mut normed);
        let layers = self
            .blocks
            .iter()
            .map(|b| (linear(&b.ca.in_w[d * d..2 * d * d], &b.ca.in_b[d..2 * d], d, d, &normed), linear(&b.ca.in_w[2 * d * d..], &b.ca.in_b[2 * d..], d, d, &normed)))
            .collect();
        Ok(Collection { n: docs.len() / d, layers })
    }

    /// The memory tokens of a question over a collection: `num_memories` vectors of `out_dim`.
    pub fn read(&self, coll: &Collection, question: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        let (d, m, nh) = (self.dim, self.num_memories, self.heads);
        let hd = d / nh;
        if question.len() != d {
            return Err(format!("the question embedding has {} values, the memory module is {d} wide", question.len()));
        }
        if coll.n == 0 {
            return Err("the collection is empty".into());
        }
        // question conditioning: one key, so every slot receives out_proj(value(question))
        let v = linear(&self.qc.in_w[2 * d * d..], &self.qc.in_b[2 * d..], d, d, question);
        let cond = linear(&self.qc.out_w, &self.qc.out_b, d, d, &v);
        let mut slots = self.queries.clone();
        for row in slots.chunks_exact_mut(d) {
            axpy(1.0, &cond, row);
        }
        self.qc_norm.apply(&mut slots);

        let scale = 1.0 / (hd as f32).sqrt();
        for (b, (keys, values)) in self.blocks.iter().zip(&coll.layers) {
            let mut x = slots.clone();
            b.ca_norm.apply(&mut x);
            let q = linear(&b.ca.in_w[..d * d], &b.ca.in_b[..d], d, d, &x);
            let mut att = vec![0f32; m * d];
            att.par_chunks_mut(hd).enumerate().for_each(|(idx, out)| {
                let (s, h) = (idx / nh, idx % nh);
                let qv = &q[s * d + h * hd..s * d + (h + 1) * hd];
                let mut sc: Vec<f32> = (0..coll.n).map(|t| dot(qv, &keys[t * d + h * hd..t * d + (h + 1) * hd]) * scale).collect();
                let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0f32;
                for v in sc.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v;
                }
                for (t, w) in sc.iter().enumerate() {
                    axpy(w / sum, &values[t * d + h * hd..t * d + (h + 1) * hd], out);
                }
            });
            let attended = linear(&b.ca.out_w, &b.ca.out_b, d, d, &att);
            axpy(1.0, &attended, &mut slots);

            let mut x = slots.clone();
            b.ffn_norm.apply(&mut x);
            let mut hid = linear(&b.w1, &b.b1, 4 * d, d, &x);
            for v in hid.iter_mut() {
                *v = gelu(*v);
            }
            let y = linear(&b.w2, &b.b2, d, 4 * d, &hid);
            axpy(1.0, &y, &mut slots);
        }
        self.final_norm.apply(&mut slots);
        let out = linear(&self.proj_w, &self.proj_b, self.out_dim, d, &slots);
        Ok(out.chunks_exact(self.out_dim).map(|r| r.to_vec()).collect())
    }

    /// `read(write(docs), question)`: the module's forward pass.
    pub fn forward(&self, docs: &[f32], question: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        self.read(&self.write(docs)?, question)
    }
}
