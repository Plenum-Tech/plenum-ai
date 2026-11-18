//! Memorilla's evaluation (port of evaluate.py and memorilla/metrics.py): for every row
//! of a test set, the memory module turns the row's document collection into memory
//! tokens, the decoder answers greedily, and the answers are scored as the paper does.
//!
//! Two sources of rows:
//! - a published benchmark (`--benchmark pmv2 ...`): the rows and collections written by
//!   tools/import_benchmark.py, and Memorilla's precomputed embeddings of every document
//!   and question, read from its safetensors files, so the run uses exactly the inputs
//!   the original evaluation does;
//! - your own questions (`--data file.jsonl`): one {"question", "answer", "user"} per
//!   line (and "choices" for choice scoring), answered from that user's memory.
//!
//! Predictions are written as they are made (predictions.jsonl) and a rerun continues
//! where the last one stopped; metrics.json is written at the end, in percent.

use crate::memory::Collection;
use crate::metrics::{self, clean_generation};
use crate::safetensors::SafeTensors;
use crate::stack::{MemoryStack, Recall};
use plenum_engine::generate::{chatml_prompt, Engine, Message, SampleParams};
use plenum_engine::quant::dot;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const PERSONA_PROMPT: &str = "You are a personalized AI assistant. Answer the question about the user based on your understanding of the user.";
pub const TRIVIA_PROMPT: &str = "You are a helpful assistant that answers trivia questions using a compressed memory of retrieved Wikipedia articles. Answer concisely with a short factual phrase or single word.";
pub const HELPFUL_PROMPT: &str = "You are a helpful assistant.";
const TRIVIA_SUFFIX: &str = "\n\nAnswer with the exact short factual phrase that answers the question. For example:\n- \"Mark Twain\"\n- \"1969\"\n- \"Mount Everest\"\n- \"the Pacific Ocean\"\n- \"carbon dioxide\"\n\nYour answer:";

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scoring {
    /// the answer's nearest choice (by embedding) must be the reference
    Choice,
    ExactMatch,
    /// the first word, yes / no / maybe, must be the reference's
    Polarity,
    /// BLEU, ROUGE and token F1 against the reference
    Generation,
}

impl Scoring {
    pub fn parse(s: &str) -> Result<Scoring, String> {
        Ok(match s {
            "choice" => Scoring::Choice,
            "exact_match" => Scoring::ExactMatch,
            "polarity" => Scoring::Polarity,
            "generation" => Scoring::Generation,
            _ => return Err(format!("scoring '{s}': choice, exact_match, polarity or generation")),
        })
    }
}

pub struct Benchmark {
    pub name: String,
    pub system_prompt: String,
    pub max_new_tokens: usize,
    pub scoring: Scoring,
    pub suffix: &'static str,
}

/// The benchmarks of evaluate.py, with their prompts and generation budgets.
pub fn benchmark(name: &str) -> Result<Benchmark, String> {
    let (sys, n, sc, suffix) = match name {
        "pmv2" | "pv4" => (PERSONA_PROMPT, 256, Scoring::Choice, ""),
        "factkg" => (PERSONA_PROMPT, 8, Scoring::Choice, ""),
        "triviaqa" => (TRIVIA_PROMPT, 64, Scoring::ExactMatch, TRIVIA_SUFFIX),
        "narrativeqa" => (PERSONA_PROMPT, 32, Scoring::Generation, ""),
        "pubmedqa" => (PERSONA_PROMPT, 64, Scoring::Polarity, ""),
        "lamp4" | "lamp7" => (PERSONA_PROMPT, 128, Scoring::Generation, ""),
        _ => return Err(format!("unknown benchmark '{name}': pmv2, pv4, factkg, triviaqa, narrativeqa, pubmedqa, lamp4, lamp7")),
    };
    Ok(Benchmark { name: name.to_string(), system_prompt: sys.to_string(), max_new_tokens: n, scoring: sc, suffix })
}

struct Row {
    /// position in the split (the row of questions.safetensors)
    index: usize,
    collection_id: i64,
    user: Option<String>,
    question: String,
    answer: String,
    choices: Vec<String>,
    hard: bool,
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    text.lines().filter(|l| !l.trim().is_empty()).enumerate().map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("{} line {}: {e}", path.display(), i + 1))).collect()
}

fn strs(v: &Value, k: &str) -> Vec<String> {
    v.get(k).and_then(|a| a.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn row_of(i: usize, v: &Value) -> Row {
    Row {
        index: v.get("row").and_then(|r| r.as_u64()).map(|r| r as usize).unwrap_or(i),
        collection_id: v.get("collection_id").and_then(|c| c.as_i64()).unwrap_or(-1),
        user: v.get("user").and_then(|u| u.as_str()).map(String::from),
        question: v.get("question").and_then(|q| q.as_str()).unwrap_or_default().to_string(),
        answer: v.get("answer").and_then(|q| q.as_str()).unwrap_or_default().to_string(),
        choices: strs(v, "choices"),
        hard: v.get("hard").and_then(|h| h.as_bool()).unwrap_or(false),
    }
}

/// Memorilla's published embeddings of one split: every collection's documents and
/// every question, as computed by Qwen3-Embedding-4B.
struct Embeddings {
    dir: PathBuf,
    index: HashMap<i64, (usize, usize, usize)>,
    questions: SafeTensors,
    shards: HashMap<usize, SafeTensors>,
}

impl Embeddings {
    fn open(dir: &Path) -> Result<Embeddings, String> {
        let ix = SafeTensors::open(&dir.join("index.safetensors"))?;
        let (ids, shard, start, len) = (ix.ints("collection_ids")?, ix.ints("shard")?, ix.ints("start")?, ix.ints("length")?);
        let index = (0..ids.len()).map(|i| (ids[i], (shard[i] as usize, start[i] as usize, len[i] as usize))).collect();
        Ok(Embeddings { dir: dir.to_path_buf(), index, questions: SafeTensors::open(&dir.join("questions.safetensors"))?, shards: HashMap::new() })
    }
    fn documents(&mut self, id: i64) -> Result<Vec<f32>, String> {
        let &(sh, start, len) = self.index.get(&id).ok_or_else(|| format!("collection {id} has no embeddings"))?;
        if !self.shards.contains_key(&sh) {
            self.shards.insert(sh, SafeTensors::open(&self.dir.join(format!("documents-{sh:05}.safetensors")))?);
        }
        self.shards[&sh].rows("embeddings", start, start + len)
    }
    fn question(&self, row: usize) -> Result<Vec<f32>, String> {
        self.questions.rows("embeddings", row, row + 1)
    }
}

pub struct EvalOptions {
    pub benchmark: Option<String>,
    /// a benchmark: the folder tools/import_benchmark.py wrote; own data: a .jsonl file
    pub data: PathBuf,
    pub embeddings: Option<PathBuf>,
    pub scoring: Option<Scoring>,
    pub system: Option<String>,
    pub max_new_tokens: Option<usize>,
    pub offset: usize,
    pub limit: Option<usize>,
    /// the closest stored passages given as text too (0: memory tokens alone, as Memorilla)
    pub retrieve: usize,
    pub out: PathBuf,
    pub fresh: bool,
    /// the baseline: the decoder alone, with no memory tokens and no passages
    pub no_memory: bool,
}

fn passages(q: &[f32], docs: &[f32], texts: &[String], dim: usize, k: usize, min_sim: f32) -> Vec<(f32, String)> {
    if k == 0 || texts.is_empty() {
        return Vec::new();
    }
    let qn = dot(q, q).sqrt().max(1e-8);
    let mut sc: Vec<(f32, usize)> = (0..texts.len())
        .map(|i| {
            let d = &docs[i * dim..(i + 1) * dim];
            (dot(q, d) / (qn * dot(d, d).sqrt().max(1e-8)), i)
        })
        .collect();
    sc.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    sc.into_iter().take(k).filter(|(s, _)| *s >= min_sim).map(|(s, i)| (s, texts[i].clone())).collect()
}

/// Which non-empty choice the prediction is closest to (cosine of the encoder's
/// embeddings), as an index into `choices`; and whether that is the reference.
fn pick_choice(stack: &MemoryStack, prediction: &str, choices: &[String], reference: &str) -> Result<(usize, bool), String> {
    let p = stack.embed(prediction.trim())?;
    let mut best = (f32::NEG_INFINITY, 0usize);
    for (i, c) in choices.iter().enumerate() {
        if c.trim().is_empty() {
            continue;
        }
        let e = stack.embed(c)?;
        let s = dot(&p, &e) / (dot(&p, &p).sqrt() * dot(&e, &e).sqrt() + 1e-8);
        if s > best.0 {
            best = (s, i);
        }
    }
    let gold = choices.iter().position(|c| c.trim() == reference.trim()).ok_or_else(|| format!("the reference {reference:?} is not one of the choices"))?;
    Ok((best.1, best.1 == gold))
}

/// Runs the evaluation; returns the metrics written to metrics.json.
pub fn run(stack: &MemoryStack, decoder: &Engine, o: &EvalOptions) -> Result<Value, String> {
    if decoder.model.cfg.n_embd != stack.module.out_dim {
        return Err(format!("the decoder is {} wide and the memory module writes tokens of {}", decoder.model.cfg.n_embd, stack.module.out_dim));
    }
    let bench = match &o.benchmark {
        Some(b) => benchmark(b)?,
        None => Benchmark {
            name: o.data.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "custom".into()),
            system_prompt: HELPFUL_PROMPT.into(),
            max_new_tokens: 128,
            scoring: o.scoring.ok_or("own data needs --scoring choice|exact_match|polarity|generation")?,
            suffix: "",
        },
    };
    let mut bench = bench;
    if o.no_memory {
        bench.name.push_str("-no-memory");
    }
    let system = o.system.clone().unwrap_or(bench.system_prompt.clone());
    let max_new = o.max_new_tokens.unwrap_or(bench.max_new_tokens);

    // ---- the rows ----
    let (rows, collections, mut emb) = if o.benchmark.is_some() {
        let rows: Vec<Row> = read_jsonl(&o.data.join("rows.jsonl"))?.iter().enumerate().map(|(i, v)| row_of(i, v)).collect();
        let mut cols = HashMap::new();
        for v in read_jsonl(&o.data.join("collections.jsonl"))? {
            cols.insert(v.get("collection_id").and_then(|c| c.as_i64()).unwrap_or(-1), strs(&v, "documents"));
        }
        let e = Embeddings::open(o.embeddings.as_ref().ok_or("a benchmark needs --embeddings <its embeddings folder>")?)?;
        (rows, cols, Some(e))
    } else {
        let rows: Vec<Row> = read_jsonl(&o.data)?.iter().enumerate().map(|(i, v)| row_of(i, v)).collect();
        if let Some(r) = rows.iter().find(|r| r.user.is_none()) {
            return Err(format!("row {} has no \"user\": own data is answered from a user's memory", r.index + 1));
        }
        (rows, HashMap::new(), None)
    };
    let end = o.limit.map(|l| (o.offset + l).min(rows.len())).unwrap_or(rows.len());
    let rows = rows.get(o.offset..end).ok_or("--offset is past the end of the data")?;
    if bench.scoring == Scoring::Choice {
        if let Some(r) = rows.iter().find(|r| r.choices.iter().all(|c| c.trim().is_empty())) {
            return Err(format!("row {} has no choices to score against", r.index + 1));
        }
    }

    // ---- predictions so far ----
    let dir = o.out.join(&bench.name);
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let pred_path = dir.join("predictions.jsonl");
    if o.fresh {
        let _ = fs::remove_file(&pred_path);
    }
    let mut done: Vec<Value> = if pred_path.exists() { read_jsonl(&pred_path)? } else { Vec::new() };
    let done_rows: std::collections::HashSet<u64> = done.iter().filter_map(|v| v.get("row").and_then(|r| r.as_u64())).collect();
    let todo: Vec<&Row> = rows.iter().filter(|r| !done_rows.contains(&(r.index as u64))).collect();
    eprintln!("  {}: {} rows, {} already answered, {} to go · scoring {:?} · {} new tokens at most", bench.name, rows.len(), rows.len() - todo.len(), todo.len(), bench.scoring, max_new);

    let p = SampleParams { temperature: 0.0, max_tokens: max_new, repeat_penalty: 1.0, ..Default::default() };
    let mut cache: Option<(i64, Arc<Collection>, Vec<f32>)> = None;
    let t_all = Instant::now();
    for (n, r) in todo.iter().enumerate() {
        let t0 = Instant::now();
        let question = format!("{}{}", r.question.trim(), bench.suffix);
        // the memory tokens and the closest passages
        let (tokens, pass) = if o.no_memory {
            (Vec::new(), Vec::new())
        } else if let Some(e) = emb.as_mut() {
            if cache.as_ref().map(|c| c.0) != Some(r.collection_id) {
                let docs = e.documents(r.collection_id)?;
                cache = Some((r.collection_id, Arc::new(stack.module.write(&docs)?), docs));
            }
            let (_, coll, docs) = cache.as_ref().unwrap();
            let q = e.question(r.index)?;
            let texts = collections.get(&r.collection_id).map(|v| v.as_slice()).unwrap_or(&[]);
            (stack.module.read(coll, &q)?, passages(&q, docs, texts, stack.module.dim, o.retrieve, stack.min_similarity))
        } else {
            let user = r.user.as_deref().unwrap_or_default();
            match stack.recall(user, &r.question, o.retrieve)? {
                Some(Recall { tokens, passages }) => (tokens, passages),
                None => return Err(format!("user '{user}' has no documents in the memory store")),
            }
        };
        let msgs = [
            Message { role: "system".into(), content: if o.no_memory { system.clone() } else { format!("{system}{}", stack.placeholder()) } },
            Message { role: "user".into(), content: MemoryStack::with_passages(&question, &pass) },
        ];
        let prompt = chatml_prompt(&decoder.tok, &msgs, false);
        let mut sess = decoder.new_session();
        let (text, st) = decoder.generate_with_memory(&mut sess, &prompt, &tokens, &p, &[], |_| true)?;
        let prediction = clean_generation(&text);
        let mut rec = json!({"row": r.index, "question": r.question, "answer": r.answer, "prediction": prediction});
        if !r.choices.is_empty() {
            rec["choices"] = json!(r.choices);
        }
        if r.hard {
            rec["hard"] = json!(true);
        }
        if bench.scoring == Scoring::Choice {
            let (pick, ok) = pick_choice(stack, &prediction, &r.choices, &r.answer)?;
            rec["picked"] = json!(pick);
            rec["correct"] = json!(ok);
        }
        let mut f = OpenOptions::new().create(true).append(true).open(&pred_path).map_err(|e| format!("{}: {e}", pred_path.display()))?;
        writeln!(f, "{rec}").map_err(|e| e.to_string())?;
        let mark = match rec.get("correct").and_then(|c| c.as_bool()) {
            Some(true) => " ✓",
            Some(false) => " ✗",
            None => "",
        };
        eprintln!(
            "  [{}/{}] row {}{mark} · {} tokens in {:.0}s · {}",
            n + 1,
            todo.len(),
            r.index,
            st.completion_tokens,
            t0.elapsed().as_secs_f64(),
            prediction.chars().take(90).collect::<String>().replace('\n', " ")
        );
        done.push(rec);
    }
    if !todo.is_empty() {
        eprintln!("  answered {} rows in {:.1} min", todo.len(), t_all.elapsed().as_secs_f64() / 60.0);
    }

    // ---- scores, over the requested rows ----
    let wanted: std::collections::HashSet<u64> = rows.iter().map(|r| r.index as u64).collect();
    let recs: Vec<&Value> = done.iter().filter(|v| v.get("row").and_then(|r| r.as_u64()).is_some_and(|i| wanted.contains(&i))).collect();
    let preds: Vec<String> = recs.iter().map(|v| v.get("prediction").and_then(|s| s.as_str()).unwrap_or_default().to_string()).collect();
    let refs: Vec<String> = recs.iter().map(|v| v.get("answer").and_then(|s| s.as_str()).unwrap_or_default().to_string()).collect();
    let pct = |x: f64| (x * 10000.0).round() / 100.0;
    let mut scores = serde_json::Map::new();
    match bench.scoring {
        Scoring::Choice => {
            let ok: Vec<bool> = recs.iter().map(|v| v.get("correct").and_then(|c| c.as_bool()).unwrap_or(false)).collect();
            scores.insert("accuracy".into(), json!(pct(ok.iter().filter(|b| **b).count() as f64 / ok.len().max(1) as f64)));
            let hard: Vec<bool> = recs.iter().zip(&ok).filter(|(v, _)| v.get("hard").and_then(|h| h.as_bool()).unwrap_or(false)).map(|(_, b)| *b).collect();
            if !hard.is_empty() {
                scores.insert("accuracy_hard".into(), json!(pct(hard.iter().filter(|b| **b).count() as f64 / hard.len() as f64)));
            }
        }
        Scoring::Polarity => {
            scores.insert("accuracy".into(), json!(pct(metrics::polarity_accuracy(&preds, &refs))));
        }
        Scoring::ExactMatch | Scoring::Generation => {
            if bench.scoring == Scoring::ExactMatch {
                scores.insert("exact_match".into(), json!(pct(metrics::exact_match(&preds, &refs))));
            }
            let (b1, b4) = metrics::bleu(&preds, &refs);
            let (r1, rl) = metrics::rouge(&preds, &refs);
            scores.insert("bleu1".into(), json!(pct(b1)));
            scores.insert("bleu4".into(), json!(pct(b4)));
            scores.insert("rouge1".into(), json!(pct(r1)));
            scores.insert("rougeL".into(), json!(pct(rl)));
            scores.insert("f1".into(), json!(pct(metrics::token_f1(&preds, &refs))));
        }
    }
    let out = json!({
        "benchmark": bench.name,
        "num_examples": recs.len(),
        "scores": scores,
        "setup": {
            "system_prompt": system,
            "max_new_tokens": max_new,
            "retrieved_passages": if o.no_memory { 0 } else { o.retrieve },
            "memory": !o.no_memory,
            "memory_module": stack.module.num_memories,
            "decoder": decoder.model.cfg.name,
            "encoder": stack.encoder.model.cfg.name,
            "note": "METEOR is not computed (it needs WordNet).",
        },
    });
    let mpath = dir.join("metrics.json");
    fs::write(&mpath, serde_json::to_string_pretty(&out).unwrap()).map_err(|e| format!("{}: {e}", mpath.display()))?;
    Ok(out)
}
