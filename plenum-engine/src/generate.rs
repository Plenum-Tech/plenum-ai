//! Sampling, the chat prompt, and the generation loop with prefix reuse between turns
//! (Rust counterparts of c/sample.h and c/kv_prefix.h).

use crate::model::{Model, State};
use crate::tokenizer::{StreamDecoder, Tokenizer};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct SampleParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub max_tokens: usize,
    pub seed: u64,
}

impl Default for SampleParams {
    fn default() -> Self {
        SampleParams { temperature: 0.7, top_k: 20, top_p: 0.8, repeat_penalty: 1.05, repeat_last_n: 64, max_tokens: 512, seed: 0 }
    }
}

/// xorshift64*: small, seedable, the same stream on every machine.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let s = if seed == 0 {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
        } else {
            seed
        };
        Rng(s | 1)
    }
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// The first of the largest (ties go to the lowest id, as c/argmax_first.h).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

pub fn sample(logits: &mut [f32], recent: &[u32], p: &SampleParams, rng: &mut Rng) -> u32 {
    if p.repeat_penalty != 1.0 && p.repeat_last_n > 0 {
        let from = recent.len().saturating_sub(p.repeat_last_n);
        let mut seen = std::collections::HashSet::new();
        for &t in &recent[from..] {
            if seen.insert(t) {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = if *l > 0.0 { *l / p.repeat_penalty } else { *l * p.repeat_penalty };
                }
            }
        }
    }
    if p.temperature <= 0.0 {
        return argmax(logits);
    }
    let mut cand: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
    let k = if p.top_k == 0 { cand.len() } else { p.top_k.min(cand.len()) };
    let desc = |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0));
    if k < cand.len() {
        cand.select_nth_unstable_by(k - 1, desc);
        cand.truncate(k);
    }
    cand.sort_by(desc);
    let mx = cand[0].1;
    let mut sum = 0f32;
    for c in cand.iter_mut() {
        c.1 = ((c.1 - mx) / p.temperature).exp();
        sum += c.1;
    }
    if p.top_p < 1.0 {
        let mut acc = 0f32;
        let mut keep = cand.len();
        for (i, c) in cand.iter().enumerate() {
            acc += c.1 / sum;
            if acc >= p.top_p {
                keep = i + 1;
                break;
            }
        }
        cand.truncate(keep);
        sum = cand.iter().map(|c| c.1).sum();
    }
    let mut r = rng.next_f32() * sum;
    for c in &cand {
        r -= c.1;
        if r <= 0.0 {
            return c.0;
        }
    }
    cand[cand.len() - 1].0
}

/// Where a latent vector goes in a prompt given to `Engine::generate_with_memory`.
pub const MEMORY_PLACEHOLDER: &str = "<|memory|>";

#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// ChatML, the format of the Qwen family: each message between <|im_start|>role and
/// <|im_end|>, then the assistant's turn opened. With `think` off, a Qwen3 model is
/// given an empty <think></think> block, which is how its own template disables reasoning.
pub fn chatml_prompt(tok: &Tokenizer, messages: &[Message], think: bool) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str("<|im_start|>");
        s.push_str(&m.role);
        s.push('\n');
        s.push_str(&m.content);
        s.push_str("<|im_end|>\n");
    }
    s.push_str("<|im_start|>assistant\n");
    if !think && tok.token_id("<think>").is_some() && tok.token_id("</think>").is_some() {
        s.push_str("<think>\n\n</think>\n\n");
    }
    s
}

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub reused_tokens: usize,
    pub completion_tokens: usize,
    pub prompt_secs: f64,
    pub gen_secs: f64,
    /// "stop" (the model ended its turn), "length" (max_tokens or the context), "cancel"
    pub finish: &'static str,
}

impl GenStats {
    pub fn line(&self) -> String {
        let evald = self.prompt_tokens - self.reused_tokens;
        format!(
            "prompt {} tok ({} reused) {:.1} tok/s · generated {} tok {:.1} tok/s",
            self.prompt_tokens,
            self.reused_tokens,
            if self.prompt_secs > 0.0 { evald as f64 / self.prompt_secs } else { 0.0 },
            self.completion_tokens,
            if self.gen_secs > 0.0 { self.completion_tokens as f64 / self.gen_secs } else { 0.0 },
        )
    }
}

/// One conversation: its KV cache and the tokens that cache stands for.
pub struct Session {
    pub state: State,
    pub tokens: Vec<u32>,
}

pub struct Engine {
    pub model: Model,
    pub tok: Tokenizer,
    pub n_ctx: usize,
    stop_ids: Vec<u32>,
}

impl Engine {
    pub fn new(model: Model, tok: Tokenizer, n_ctx: usize) -> Engine {
        let mut stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"].iter().filter_map(|s| tok.token_id(s)).collect();
        if let Some(e) = tok.eos {
            if !stop_ids.contains(&e) {
                stop_ids.push(e);
            }
        }
        let n_ctx = if n_ctx == 0 { model.cfg.n_ctx_train.min(8192) } else { n_ctx.min(model.cfg.n_ctx_train) };
        Engine { model, tok, n_ctx, stop_ids }
    }

    pub fn new_session(&self) -> Session {
        Session { state: self.model.new_state(), tokens: Vec::new() }
    }

    /// Generates after `prompt`. `on_text` receives each piece of text and returns
    /// false to stop. The part of `prompt` the session already computed is not run again.
    pub fn generate(
        &self,
        sess: &mut Session,
        prompt: &[u32],
        p: &SampleParams,
        stops: &[String],
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<(String, GenStats), String> {
        if prompt.is_empty() {
            return Err("the prompt is empty".into());
        }
        if prompt.len() >= self.n_ctx {
            return Err(format!("the prompt is {} tokens and the context is {}; raise --ctx or shorten the conversation", prompt.len(), self.n_ctx));
        }
        let mut st = GenStats { prompt_tokens: prompt.len(), finish: "length", ..Default::default() };
        // prefix reuse: keep what the session and the new prompt have in common
        let mut lcp = sess.tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        if lcp == prompt.len() {
            lcp -= 1; // the last token must run again: its logits are what we sample from
        }
        sess.state.truncate(lcp);
        sess.tokens.truncate(lcp);
        st.reused_tokens = lcp;

        let t0 = Instant::now();
        let logits = self.model.forward(&mut sess.state, &prompt[lcp..], self.n_ctx)?;
        sess.tokens.extend_from_slice(&prompt[lcp..]);
        st.prompt_secs = t0.elapsed().as_secs_f64();

        self.decode(sess, logits, st, p, stops, on_text)
    }

    /// Generates after a prompt that carries latent vectors: every `<|memory|>` in
    /// `prompt` is replaced, in order, by one row of `memory` (each `n_embd` long),
    /// which enters the model as an input embedding in that position. This is how a
    /// memory module such as Memorilla hands its memory tokens to a frozen decoder.
    pub fn generate_with_memory(
        &self,
        sess: &mut Session,
        prompt: &str,
        memory: &[Vec<f32>],
        p: &SampleParams,
        stops: &[String],
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<(String, GenStats), String> {
        let d = self.model.cfg.n_embd;
        let parts: Vec<&str> = prompt.split(MEMORY_PLACEHOLDER).collect();
        if parts.len() - 1 != memory.len() {
            return Err(format!(
                "the prompt has {} {MEMORY_PLACEHOLDER} placeholders and {} memory vectors were given",
                parts.len() - 1,
                memory.len()
            ));
        }
        let mut x = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            x.extend(self.model.embed_tokens(&self.tok.encode(part, true))?);
            if let Some(m) = memory.get(i) {
                if m.len() != d {
                    return Err(format!("memory vector {i} has {} values, the model is {d} wide", m.len()));
                }
                x.extend_from_slice(m);
            }
        }
        let n = x.len() / d;
        if n == 0 || n >= self.n_ctx {
            return Err(format!("the prompt is {n} positions and the context is {}", self.n_ctx));
        }
        // the cache of a prompt with vectors in it cannot be matched against token ids: start clean, leave clean
        sess.state.truncate(0);
        sess.tokens.clear();
        let mut st = GenStats { prompt_tokens: n, finish: "length", ..Default::default() };
        let t0 = Instant::now();
        let logits = self.model.forward_embd(&mut sess.state, x, self.n_ctx)?;
        st.prompt_secs = t0.elapsed().as_secs_f64();
        let r = self.decode(sess, logits, st, p, stops, on_text);
        sess.state.truncate(0);
        sess.tokens.clear();
        r
    }

    fn decode(
        &self,
        sess: &mut Session,
        mut logits: Vec<f32>,
        mut st: GenStats,
        p: &SampleParams,
        stops: &[String],
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<(String, GenStats), String> {
        let mut rng = Rng::new(p.seed);
        let mut dec = StreamDecoder::default();
        let mut text = String::new();
        let mut sent = 0usize; // bytes of `text` already given to on_text
        let mut flushed = false; // a stop string ended the text: nothing more to send
        let hold = stops.iter().map(|s| s.len()).max().unwrap_or(0);
        let t1 = Instant::now();
        'gen: while st.completion_tokens < p.max_tokens {
            let id = sample(&mut logits, &sess.tokens, p, &mut rng);
            if self.stop_ids.contains(&id) {
                st.finish = "stop";
                break;
            }
            st.completion_tokens += 1;
            text.push_str(&dec.push(&self.tok, id));
            // a stop string ends the text before it; until one can no longer begin, hold the tail back
            let cut = stops.iter().filter(|s| !s.is_empty()).filter_map(|s| text[sent..].find(s.as_str())).min();
            if let Some(i) = cut {
                let at = sent + i;
                if at > sent {
                    on_text(&text[sent..at]);
                }
                text.truncate(at);
                st.finish = "stop";
                flushed = true;
                break 'gen;
            }
            let mut upto = text.len().saturating_sub(hold.saturating_sub(1));
            while upto > sent && !text.is_char_boundary(upto) {
                upto -= 1;
            }
            if upto > sent {
                if !on_text(&text[sent..upto]) {
                    st.finish = "cancel";
                    sent = upto;
                    break;
                }
                sent = upto;
            }
            if sess.state.n_past + 1 > self.n_ctx {
                break; // the context is full
            }
            logits = self.model.forward(&mut sess.state, &[id], self.n_ctx)?;
            sess.tokens.push(id);
        }
        if st.finish != "cancel" && !flushed {
            text.push_str(&dec.finish());
            if text.len() > sent {
                on_text(&text[sent..]);
            }
        }
        st.gen_secs = t1.elapsed().as_secs_f64();
        Ok((text, st))
    }

    /// A chat turn: the messages become the ChatML prompt and the reply is generated.
    pub fn chat(
        &self,
        sess: &mut Session,
        messages: &[Message],
        think: bool,
        p: &SampleParams,
        stops: &[String],
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<(String, GenStats), String> {
        let prompt = chatml_prompt(&self.tok, messages, think);
        let ids = self.tok.encode(&prompt, true);
        self.generate(sess, &ids, p, stops, on_text)
    }
}

/// Splits a Qwen3 reply into (reasoning, answer) at its <think> block.
pub fn split_thinking(text: &str) -> (Option<String>, String) {
    if let Some(a) = text.find("<think>") {
        if let Some(b) = text[a..].find("</think>") {
            let think = text[a + 7..a + b].trim().to_string();
            let rest = format!("{}{}", &text[..a], text[a + b + 8..].trim_start());
            return (if think.is_empty() { None } else { Some(think) }, rest);
        }
        // cut off while still reasoning: there is no answer yet
        return (Some(text[a + 7..].trim().to_string()), text[..a].to_string());
    }
    (None, text.to_string())
}
