//! Tiny synthetic models and the engine's self-check (the role c/*_tiny + ref.json
//! play for the C engines): the properties below hold for any correct engine, so they
//! are checked without downloading a model.

use crate::gguf::Writer;
use crate::model::{LoadOptions, Model};
use crate::quant::quantize_q8_0;
use std::path::{Path, PathBuf};

pub struct TinySpec {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    /// 0 = dense model
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// every expert holds the dense model's FFN weights
    pub identical_experts: bool,
    pub q8: bool,
}

impl Default for TinySpec {
    fn default() -> Self {
        TinySpec { n_layer: 2, n_embd: 64, n_head: 4, n_kv_head: 2, head_dim: 16, n_ff: 96, n_vocab: 50, n_expert: 0, n_expert_used: 0, identical_experts: false, q8: false }
    }
}

/// Values that depend only on the tensor's name, so two models share what they share.
fn values(name: &str, n: usize, scale: f32, bias: f32) -> Vec<f32> {
    let mut s = 0xcbf2_9ce4_8422_2325u64;
    for b in name.bytes() {
        s = (s ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            bias + scale * (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0)
        })
        .collect()
}

pub fn write_tiny(path: &Path, sp: &TinySpec) -> std::io::Result<()> {
    let arch = if sp.n_expert > 0 { "qwen3moe" } else { "qwen3" };
    let mut w = Writer::new();
    w.str("general.architecture", arch);
    w.str("general.name", "tiny");
    let k = |s: &str| format!("{arch}.{s}");
    w.u32(&k("block_count"), sp.n_layer as u32);
    w.u32(&k("context_length"), 256);
    w.u32(&k("embedding_length"), sp.n_embd as u32);
    w.u32(&k("feed_forward_length"), sp.n_ff as u32);
    w.u32(&k("attention.head_count"), sp.n_head as u32);
    w.u32(&k("attention.head_count_kv"), sp.n_kv_head as u32);
    w.u32(&k("attention.key_length"), sp.head_dim as u32);
    w.f32(&k("attention.layer_norm_rms_epsilon"), 1e-6);
    w.f32(&k("rope.freq_base"), 10000.0);
    if sp.n_expert > 0 {
        w.u32(&k("expert_count"), sp.n_expert as u32);
        w.u32(&k("expert_used_count"), sp.n_expert_used as u32);
    }
    let (d, ff, qd, kvd) = (sp.n_embd, sp.n_ff, sp.n_head * sp.head_dim, sp.n_kv_head * sp.head_dim);
    let mat = |w: &mut Writer, name: &str, dims: &[usize], data: Vec<f32>| {
        if sp.q8 {
            w.tensor(name, dims, 8, quantize_q8_0(&data));
        } else {
            w.tensor_f32(name, dims, &data);
        }
    };
    mat(&mut w, "token_embd.weight", &[d, sp.n_vocab], values("token_embd.weight", d * sp.n_vocab, 0.5, 0.0));
    mat(&mut w, "output.weight", &[d, sp.n_vocab], values("output.weight", d * sp.n_vocab, 0.5, 0.0));
    w.tensor_f32("output_norm.weight", &[d], &values("output_norm.weight", d, 0.1, 1.0));
    for i in 0..sp.n_layer {
        let p = |s: &str| format!("blk.{i}.{s}");
        for (n, len) in [("attn_norm.weight", d), ("ffn_norm.weight", d), ("attn_q_norm.weight", sp.head_dim), ("attn_k_norm.weight", sp.head_dim)] {
            w.tensor_f32(&p(n), &[len], &values(&p(n), len, 0.1, 1.0));
        }
        for (n, rows, cols) in [("attn_q.weight", qd, d), ("attn_k.weight", kvd, d), ("attn_v.weight", kvd, d), ("attn_output.weight", d, qd)] {
            mat(&mut w, &p(n), &[cols, rows], values(&p(n), rows * cols, 0.3, 0.0));
        }
        let dense = [("ffn_gate", ff, d), ("ffn_up", ff, d), ("ffn_down", d, ff)];
        if sp.n_expert == 0 {
            for (n, rows, cols) in dense {
                let name = p(&format!("{n}.weight"));
                mat(&mut w, &name, &[cols, rows], values(&name, rows * cols, 0.3, 0.0));
            }
        } else {
            let name = p("ffn_gate_inp.weight");
            mat(&mut w, &name, &[d, sp.n_expert], values(&name, d * sp.n_expert, 0.5, 0.0));
            for (n, rows, cols) in dense {
                let mut all = Vec::with_capacity(rows * cols * sp.n_expert);
                for e in 0..sp.n_expert {
                    let src = if sp.identical_experts { p(&format!("{n}.weight")) } else { p(&format!("{n}_exps.{e}")) };
                    all.extend(values(&src, rows * cols, 0.3, 0.0));
                }
                mat(&mut w, &p(&format!("{n}_exps.weight")), &[cols, rows, sp.n_expert], all);
            }
        }
    }
    w.write(path)
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn check(name: &str, diff: f32, tol: f32, log: &mut Vec<String>) -> Result<(), String> {
    if diff.is_nan() || diff > tol {
        return Err(format!("{name}: max difference {diff:e} is over {tol:e}"));
    }
    log.push(format!("ok  {name} (max difference {diff:.2e})"));
    Ok(())
}

/// Builds tiny models in `dir` and checks the engine against itself. Returns one
/// line per check.
pub fn selfcheck(dir: &Path) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let file = |n: &str| -> PathBuf { dir.join(n) };
    let io = |e: std::io::Error| e.to_string();
    let toks: Vec<u32> = vec![3, 17, 42, 8, 29, 11, 5, 33];
    let mut log = Vec::new();
    let big = LoadOptions { expert_cache_bytes: 1 << 30 };

    // 1. a prompt run at once and one token at a time give the same logits (KV cache, batching)
    write_tiny(&file("dense.gguf"), &TinySpec::default()).map_err(io)?;
    let dense = Model::load(&file("dense.gguf"), &big)?;
    let mut st = dense.new_state();
    let batch = dense.forward(&mut st, &toks, 256)?;
    let mut st1 = dense.new_state();
    let mut inc = Vec::new();
    for t in &toks {
        inc = dense.forward(&mut st1, &[*t], 256)?;
    }
    check("prompt at once = token by token", max_diff(&batch, &inc), 2e-3, &mut log)?;

    // 1b. token ids, and their embedding rows given as input vectors, are the same input
    let viaembd = dense.forward_embd(&mut dense.new_state(), dense.embed_tokens(&toks)?, 256)?;
    check("input vectors = token ids", max_diff(&batch, &viaembd), 0.0, &mut log)?;

    // 2. the prefix is reused: cut the cache back, run the tail again
    st.truncate(4);
    let again = dense.forward(&mut st, &toks[4..], 256)?;
    check("prefix reuse after truncating the KV cache", max_diff(&batch, &again), 2e-3, &mut log)?;

    // 3. a MoE whose experts all hold the dense FFN is the dense model, whatever the router picks
    let moe_same = TinySpec { n_expert: 4, n_expert_used: 2, identical_experts: true, ..Default::default() };
    write_tiny(&file("moe_same.gguf"), &moe_same).map_err(io)?;
    let m = Model::load(&file("moe_same.gguf"), &big)?;
    let out = m.forward(&mut m.new_state(), &toks, 256)?;
    check("MoE with identical experts = dense model", max_diff(&batch, &out), 2e-3, &mut log)?;

    // 4. expert streaming: a cache too small to hold the experts gives the same logits, from the disk
    let moe = TinySpec { n_expert: 8, n_expert_used: 2, ..Default::default() };
    write_tiny(&file("moe.gguf"), &moe).map_err(io)?;
    let roomy = Model::load(&file("moe.gguf"), &big)?;
    let tight = Model::load(&file("moe.gguf"), &LoadOptions { expert_cache_bytes: 1 })?;
    let a = roomy.forward(&mut roomy.new_state(), &toks, 256)?;
    // token by token, so that an expert routed to again must come from the cache or the disk
    let (mut sr, mut stt) = (roomy.new_state(), tight.new_state());
    let (mut a1, mut b) = (Vec::new(), Vec::new());
    for t in &toks {
        a1 = roomy.forward(&mut sr, &[*t], 256)?;
        b = tight.forward(&mut stt, &[*t], 256)?;
    }
    check("MoE prompt at once = token by token", max_diff(&a, &a1), 2e-3, &mut log)?;
    check("expert streaming with a 1-byte cache = everything cached", max_diff(&a1, &b), 0.0, &mut log)?;
    let mapped = Model::load(&file("moe.gguf"), &LoadOptions { expert_cache_bytes: 0 })?;
    let c = mapped.forward(&mut mapped.new_state(), &toks, 256)?;
    check("experts used from the mapped file = experts read into the cache", max_diff(&a, &c), 0.0, &mut log)?;
    let (r, t) = (roomy.experts.as_ref().unwrap(), tight.experts.as_ref().unwrap());
    let load = std::sync::atomic::Ordering::Relaxed;
    if t.misses.load(load) <= r.misses.load(load) || t.resident_bytes() >= r.resident_bytes() {
        return Err("the tight expert cache did not evict".into());
    }
    log.push(format!("ok  eviction: roomy cache {} disk reads, 1-byte cache {}", r.misses.load(load), t.misses.load(load)));
    if max_diff(&a, &batch) < 1e-3 {
        return Err("distinct experts gave the dense model's logits: the experts are not being used".into());
    }

    // 5. the same weights as Q8_0 blocks stay close to f32 (the quantized path computes the same model)
    write_tiny(&file("dense_q8.gguf"), &TinySpec { q8: true, ..Default::default() }).map_err(io)?;
    let q8 = Model::load(&file("dense_q8.gguf"), &big)?;
    let outq = q8.forward(&mut q8.new_state(), &toks, 256)?;
    let scale = batch.iter().fold(0f32, |m, v| m.max(v.abs()));
    check("Q8_0 blocks vs f32 (relative)", max_diff(&batch, &outq) / scale, 0.08, &mut log)?;
    write_tiny(&file("moe_q8.gguf"), &TinySpec { q8: true, ..moe }).map_err(io)?;
    let mq8 = Model::load(&file("moe_q8.gguf"), &big)?;
    let outm = mq8.forward(&mut mq8.new_state(), &toks, 256)?;
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    check("Q8_0 MoE experts vs f32 (relative)", max_diff(&a, &outm) / scale, 0.08, &mut log)?;
    Ok(log)
}

#[cfg(test)]
mod tests {
    #[test]
    fn engine_selfcheck() {
        let dir = std::env::temp_dir().join(format!("colibri-rs-test-{}", std::process::id()));
        let r = super::selfcheck(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        for l in r.expect("selfcheck") {
            println!("{l}");
        }
    }
}
