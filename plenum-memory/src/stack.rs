//! The memory pipeline: encoder -> store -> memory module -> memory tokens, and the
//! prompt Memorilla gives its decoder (port of memorilla/vllm.py and data.py).

use crate::memory::{Collection, MemoryModule};
use crate::store::Store;
use plenum_engine::generate::{chatml_prompt, Engine, GenStats, Message, SampleParams, Session, MEMORY_PLACEHOLDER};
use plenum_engine::server::Extension;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
/// For an agent that must not guess: the memory tokens make the decoder answer with
/// confidence, and without this it will also do so about things the memory does not hold.
pub const GROUNDED_SYSTEM_PROMPT: &str = "You are a helpful assistant. Answer only from what you know about the user. If you do not know, say that you do not know.";

/// A user's memory, loaded: the documents, their embeddings, and the collection as
/// the memory module holds it.
pub struct UserMemory {
    pub docs: Vec<String>,
    pub embeddings: Vec<f32>,
    pub collection: Collection,
}

/// What the memory gives for one question.
pub struct Recall {
    /// The memory tokens: the whole collection compressed for this question.
    pub tokens: Vec<Vec<f32>>,
    /// The stored passages closest to the question, best first, with their cosine.
    pub passages: Vec<(f32, String)>,
}

pub struct MemoryStack {
    pub encoder: Engine,
    pub module: MemoryModule,
    pub store: Store,
    /// Most tokens of a document chunk (`add` splits longer texts).
    pub chunk_tokens: usize,
    /// How many of the closest stored passages are also given to the decoder as text
    /// (0 = memory tokens alone, as Memorilla itself does).
    pub retrieve: usize,
    /// A passage is only given when its cosine with the question is at least this.
    pub min_similarity: f32,
    cache: Mutex<HashMap<String, Arc<UserMemory>>>,
}

impl MemoryStack {
    pub fn new(encoder: Engine, module: MemoryModule, store: Store) -> Result<MemoryStack, String> {
        if encoder.model.cfg.n_embd != module.dim {
            return Err(format!(
                "the encoder is {} wide and the memory module expects embeddings of {} (it was trained with Qwen3-Embedding-4B)",
                encoder.model.cfg.n_embd, module.dim
            ));
        }
        Ok(MemoryStack { encoder, module, store, chunk_tokens: 256, retrieve: 3, min_similarity: 0.3, cache: Mutex::new(HashMap::new()) })
    }

    /// The embedding of a text: its tokens and the end-of-text token through the encoder,
    /// the last hidden state normalized (pooled vector, no instruction prefix).
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, String> {
        let tok = &self.encoder.tok;
        let eot = tok.token_id("<|endoftext|>").or(tok.eos).ok_or("the encoder's tokenizer has no end-of-text token")?;
        let mut ids = tok.encode(text, false);
        ids.truncate(self.encoder.n_ctx - 1);
        ids.push(eot);
        self.encoder.model.embed_text(&ids, self.encoder.n_ctx)
    }

    fn n_tokens(&self, text: &str) -> usize {
        self.encoder.tok.encode(text, false).len()
    }

    /// Splits a text into passages of at most `chunk_tokens` tokens, at line ends where
    /// it can, then at sentence ends, then between words. Each passage becomes one vector.
    pub fn chunk(&self, text: &str) -> Vec<String> {
        let max = self.chunk_tokens.max(16);
        let mut units: Vec<String> = Vec::new();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            if self.n_tokens(line) <= max {
                units.push(line.to_string());
                continue;
            }
            for sent in line.split_inclusive(|c| matches!(c, '.' | '!' | '?' | ';')).map(str::trim).filter(|s| !s.is_empty()) {
                if self.n_tokens(sent) <= max {
                    units.push(sent.to_string());
                    continue;
                }
                let mut cur = String::new();
                for word in sent.split_whitespace() {
                    if !cur.is_empty() && self.n_tokens(&cur) + self.n_tokens(word) + 1 > max {
                        units.push(std::mem::take(&mut cur));
                    }
                    if !cur.is_empty() {
                        cur.push(' ');
                    }
                    cur.push_str(word);
                }
                if !cur.is_empty() {
                    units.push(cur);
                }
            }
        }
        let mut chunks = Vec::new();
        let (mut cur, mut cur_n) = (String::new(), 0usize);
        for u in units {
            let n = self.n_tokens(&u) + 1;
            if !cur.is_empty() && cur_n + n > max {
                chunks.push(std::mem::take(&mut cur));
                cur_n = 0;
            }
            if !cur.is_empty() {
                cur.push('\n');
            }
            cur.push_str(&u);
            cur_n += n;
        }
        if !cur.is_empty() {
            chunks.push(cur);
        }
        chunks
    }

    /// Adds documents to a user's collection (each embedded once, now). With `chunk`,
    /// long texts are split into passages first. Returns how many vectors were added.
    pub fn add(&self, user: &str, texts: &[String], chunk: bool, mut progress: impl FnMut(usize, usize)) -> Result<usize, String> {
        let docs: Vec<String> = if chunk { texts.iter().flat_map(|t| self.chunk(t)).collect() } else { texts.iter().filter(|t| !t.trim().is_empty()).cloned().collect() };
        for (i, d) in docs.iter().enumerate() {
            progress(i, docs.len());
            let e = self.embed(d)?;
            self.store.add(user, d, &e)?;
        }
        self.cache.lock().unwrap().remove(&crate::store::user_key(user)?);
        Ok(docs.len())
    }

    pub fn clear(&self, user: &str) -> Result<(), String> {
        self.cache.lock().unwrap().remove(&crate::store::user_key(user)?);
        self.store.clear(user)
    }

    /// The user's memory (None when it is empty). Built from the stored embeddings on
    /// first use, then kept until documents are added.
    pub fn user_memory(&self, user: &str) -> Result<Option<Arc<UserMemory>>, String> {
        let key = crate::store::user_key(user)?;
        if let Some(c) = self.cache.lock().unwrap().get(&key) {
            return Ok(Some(c.clone()));
        }
        let (docs, embeddings) = self.store.load(user)?;
        if docs.is_empty() {
            return Ok(None);
        }
        let collection = self.module.write(&embeddings)?;
        let m = Arc::new(UserMemory { docs, embeddings, collection });
        self.cache.lock().unwrap().insert(key, m.clone());
        Ok(Some(m))
    }

    /// What the user's memory gives for a question: the memory tokens, and the `k`
    /// closest stored passages (None when the collection is empty).
    pub fn recall(&self, user: &str, question: &str, k: usize) -> Result<Option<Recall>, String> {
        let Some(mem) = self.user_memory(user)? else { return Ok(None) };
        let q = self.embed(question.trim())?;
        let d = self.module.dim;
        let mut scored: Vec<(f32, usize)> = (0..mem.docs.len()).map(|i| (plenum_engine::quant::dot(&q, &mem.embeddings[i * d..(i + 1) * d]), i)).collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let passages = scored.into_iter().take(k).filter(|(s, _)| *s >= self.min_similarity).map(|(s, i)| (s, mem.docs[i].clone())).collect();
        Ok(Some(Recall { tokens: self.module.read(&mem.collection, &q)?, passages }))
    }

    /// The memory tokens for a question over a user's collection (None when it is empty).
    pub fn memory_tokens(&self, user: &str, question: &str) -> Result<Option<Vec<Vec<f32>>>, String> {
        Ok(self.recall(user, question, 0)?.map(|r| r.tokens))
    }

    /// The user turn when passages are given with the question.
    pub fn with_passages(question: &str, passages: &[(f32, String)]) -> String {
        if passages.is_empty() {
            return question.trim().to_string();
        }
        let mut s = String::from("Notes from the user's history that may be relevant:\n");
        for (i, (_, p)) in passages.iter().enumerate() {
            s.push_str(&format!("[{}] {}\n", i + 1, p.trim().replace('\n', " ")));
        }
        s.push_str("\nQuestion: ");
        s.push_str(question.trim());
        s
    }

    /// What Memorilla appends to the system prompt to reserve the memory positions.
    pub fn placeholder(&self) -> String {
        format!("\nMemory{}Memory", MEMORY_PLACEHOLDER.repeat(self.module.num_memories))
    }

    /// Answers a question about a user from their collection: the memory tokens go in
    /// the system prompt, the decoder answers. Without a collection (or with
    /// `use_memory` off) the decoder answers on its own. Returns (answer, stats, memory used).
    #[allow(clippy::too_many_arguments)]
    pub fn answer(
        &self,
        decoder: &Engine,
        sess: &mut Session,
        user: &str,
        question: &str,
        system: &str,
        use_memory: bool,
        p: &SampleParams,
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<(String, GenStats, bool), String> {
        if decoder.model.cfg.n_embd != self.module.out_dim {
            return Err(format!(
                "the decoder is {} wide and the memory module writes tokens of {} (it was trained with Qwen3-8B)",
                decoder.model.cfg.n_embd, self.module.out_dim
            ));
        }
        let recall = if use_memory { self.recall(user, question, self.retrieve)? } else { None };
        let (sys, ask) = match &recall {
            Some(r) => (format!("{system}{}", self.placeholder()), Self::with_passages(question, &r.passages)),
            None => (system.to_string(), question.trim().to_string()),
        };
        let msgs = [Message { role: "system".into(), content: sys }, Message { role: "user".into(), content: ask }];
        let prompt = chatml_prompt(&decoder.tok, &msgs, false);
        let used = recall.is_some();
        let memory: &[Vec<f32>] = recall.as_ref().map(|r| r.tokens.as_slice()).unwrap_or(&[]);
        let (text, st) = decoder.generate_with_memory(sess, &prompt, memory, p, &[], on_text)?;
        Ok((text.trim().to_string(), st, used))
    }
}

fn user_of(v: &Value) -> String {
    v.get("user").and_then(|u| u.as_str()).filter(|u| !u.trim().is_empty()).unwrap_or("default").to_string()
}

fn ok(v: Value) -> Option<(String, Value)> {
    Some(("200 OK".into(), v))
}

fn bad(msg: String) -> Option<(String, Value)> {
    Some(("400 Bad Request".into(), json!({"error": {"message": msg, "type": "invalid_request_error"}})))
}

/// The memory behind the OpenAI-compatible server: a chat request's `user` field
/// selects the collection, the last user message is the question, and the memory
/// tokens enter the system prompt. Plus routes to manage the collections.
impl Extension for MemoryStack {
    fn memory(&self, request: &Value, messages: &mut Vec<Message>) -> Result<Option<Vec<Vec<f32>>>, String> {
        if request.get("memory").and_then(|m| m.as_bool()) == Some(false) {
            return Ok(None);
        }
        let Some(at) = messages.iter().rposition(|m| m.role == "user") else { return Ok(None) };
        let question = messages[at].content.clone();
        let k = request.get("retrieve").and_then(|k| k.as_u64()).map(|k| k as usize).unwrap_or(self.retrieve);
        let Some(Recall { tokens, passages }) = self.recall(&user_of(request), &question, k)? else { return Ok(None) };
        messages[at].content = Self::with_passages(&question, &passages);
        match messages.first_mut() {
            Some(m) if m.role == "system" => m.content.push_str(&self.placeholder()),
            _ => messages.insert(0, Message { role: "system".into(), content: format!("{DEFAULT_SYSTEM_PROMPT}{}", self.placeholder()) }),
        }
        Ok(Some(tokens))
    }

    fn route(&self, method: &str, path: &str, body: &[u8]) -> Option<(String, Value)> {
        let parse = || serde_json::from_slice::<Value>(body).map_err(|e| format!("the body is not JSON: {e}"));
        match (method, path) {
            ("GET", "/v1/memory/users") => ok(json!({"users": self.store.users().into_iter().map(|(u, n)| json!({"user": u, "documents": n})).collect::<Vec<_>>()})),
            ("POST", "/v1/memory/documents") => {
                let v = match parse() {
                    Ok(v) => v,
                    Err(e) => return bad(e),
                };
                let user = user_of(&v);
                let mut texts: Vec<String> = v.get("documents").and_then(|d| d.as_array()).map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect()).unwrap_or_default();
                if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                    texts.push(t.to_string());
                }
                if texts.is_empty() {
                    return bad("give \"documents\": [\"...\"] or \"text\": \"...\"".into());
                }
                let chunk = v.get("chunk").and_then(|c| c.as_bool()).unwrap_or(true);
                match self.add(&user, &texts, chunk, |_, _| {}) {
                    Ok(n) => ok(json!({"user": user, "added": n, "documents": self.store.count(&user)})),
                    Err(e) => bad(e),
                }
            }
            ("POST", "/v1/memory/clear") => match parse().and_then(|v| self.clear(&user_of(&v)).map(|_| user_of(&v))) {
                Ok(user) => ok(json!({"user": user, "documents": 0})),
                Err(e) => bad(e),
            },
            ("POST", "/v1/memory/tokens") => {
                let v = match parse() {
                    Ok(v) => v,
                    Err(e) => return bad(e),
                };
                let Some(q) = v.get("question").and_then(|q| q.as_str()) else { return bad("give \"question\": \"...\"".into()) };
                let k = v.get("retrieve").and_then(|k| k.as_u64()).map(|k| k as usize).unwrap_or(self.retrieve);
                match self.recall(&user_of(&v), q, k) {
                    Ok(Some(r)) => ok(json!({
                        "user": user_of(&v),
                        "passages": r.passages.iter().map(|(s, p)| json!({"similarity": s, "text": p})).collect::<Vec<_>>(),
                        "memory_embeddings": r.tokens,
                    })),
                    Ok(None) => ok(json!({"user": user_of(&v), "passages": [], "memory_embeddings": Value::Null})),
                    Err(e) => bad(e),
                }
            }
            _ => None,
        }
    }
}
