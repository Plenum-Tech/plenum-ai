//! plenum-memory: the command line of Plenum Memory.

use plenum_engine::generate::{Engine, SampleParams};
use plenum_engine::gguf::Gguf;
use plenum_engine::model::{LoadOptions, Model};
use plenum_engine::server::{serve, ServerConfig};
use plenum_engine::tokenizer::Tokenizer;
use plenum_memory::memory::MemoryModule;
use plenum_memory::stack::{MemoryStack, DEFAULT_SYSTEM_PROMPT, GROUNDED_SYSTEM_PROMPT};
use plenum_memory::store::Store;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "\
plenum-memory: per-user latent semantic memory for a local LLM (the Memorilla method).

A user's documents are embedded once and kept. For every question the memory module
turns the whole collection into 16 memory tokens, and the decoder answers with them.

USAGE
  plenum-memory add   --user U [--text \"...\"] [FILE ...]   add documents to a user's memory
  plenum-memory ask   --user U -q \"...\"                    answer a question from that memory
  plenum-memory chat  --user U                             ask question after question
  plenum-memory serve [--port N]                           OpenAI-compatible server with memory
  plenum-memory search --user U -q \"...\"                   the stored documents closest to a question
  plenum-memory users                                      the collections in the store
  plenum-memory clear --user U                             remove a user's memory
  plenum-memory verify --docs F --question F --expect F    compare the memory module with PyTorch
  plenum-memory evaluate --benchmark B --data DIR --embeddings DIR   score it as Memorilla's paper does
  plenum-memory evaluate --data qa.jsonl --scoring S       score it on your own questions

EVALUATE
  --benchmark B       pmv2, pv4, factkg, triviaqa, narrativeqa, pubmedqa, lamp4 or lamp7: their
                      prompts, budgets and scoring. --data is the folder that
                      tools/import_benchmark.py wrote, --embeddings Memorilla's embeddings of
                      the split (embeddings/qwen3-embedding-4b/<benchmark>/<split>)
  --data qa.jsonl     your own rows: {\"question\", \"answer\", \"user\"} (and \"choices\"), answered
                      from that user's memory; with --scoring choice|exact_match|polarity|generation
  --limit N --offset K   a slice of the rows (it is about a minute per row on a laptop CPU)
  --out DIR           results/<name>/predictions.jsonl and metrics.json (default results)
  --fresh             start over instead of continuing the predictions already written
  --no-memory         the baseline: the same questions to the decoder alone (results/<name>-no-memory)
  Passages default to 0 here (the memory tokens alone, as Memorilla); --retrieve K adds them.

OPTIONS
  --models DIR        where the models are (default: $PLENUM_MODELS, else ./models)
  --encoder FILE      encoder GGUF   (default: <models>/Qwen3-Embedding-4B-Q8_0.gguf)
  --decoder FILE      decoder GGUF   (default: <models>/Qwen3-8B-Q4_K_M.gguf)
  --memory DIR        memory module checkpoint (default: <models>/memorilla/stage3)
  --store DIR         where the collections are kept (default: ./memory-store)
  --user ID           whose memory (default: default)
  --system TEXT       system prompt (default: \"You are a helpful assistant.\")
  --retrieve K        also give the decoder the K stored passages closest to the question, as
                      text (default 3). 0 = the 16 memory tokens alone, as Memorilla itself does
  --grounded          system prompt that tells the model to say so when it does not know,
                      instead of guessing
  --no-memory         ask: answer without the memory, to compare
  --remember          ask, chat: save each question and its answer to the user's memory
  --no-remember       serve: do not (serve saves them by default; a request can also
                      send \"remember\": false)
  --no-chunk          add: one vector per text or file, however long
  --chunk-tokens N    add: most tokens per passage (default 256)
  -n, --max-tokens N  most tokens to generate (default 256)
  --temp F            temperature (default 0: greedy, as Memorilla decodes)
  --threads N         CPU threads (default: all)
  --host H --port N   serve: where to listen (default 127.0.0.1:8000)

FILES for add: .txt/.md (split into passages) or .jsonl (one {\"text\": ...} per line).

SERVER
  POST /v1/chat/completions      \"user\": \"U\" selects the memory; \"memory\": false turns it off;
                                 \"retrieve\": K overrides --retrieve for the request
  POST /v1/memory/documents      {\"user\": \"U\", \"documents\": [\"...\"]} or {\"user\": \"U\", \"text\": \"...\"}
  POST /v1/memory/tokens         {\"user\": \"U\", \"question\": \"...\"} -> the 16 memory vectors and the passages
  POST /v1/memory/list           {\"user\": \"U\", \"limit\": N} -> the newest entries, with their index
  POST /v1/memory/forget         {\"user\": \"U\", \"index\": N} removes one entry
  POST /v1/memory/clear          {\"user\": \"U\"}
  GET  /v1/memory/users
";

struct Args {
    cmd: String,
    models: PathBuf,
    encoder: Option<PathBuf>,
    decoder: Option<PathBuf>,
    memory: Option<PathBuf>,
    store: PathBuf,
    user: String,
    system: String,
    question: Option<String>,
    texts: Vec<String>,
    files: Vec<PathBuf>,
    use_memory: bool,
    remember: Option<bool>,
    chunk: bool,
    chunk_tokens: usize,
    retrieve: usize,
    sp: SampleParams,
    threads: usize,
    host: String,
    port: u16,
    docs: Option<PathBuf>,
    benchmark: Option<String>,
    data: Option<PathBuf>,
    embeddings: Option<PathBuf>,
    scoring: Option<String>,
    offset: usize,
    limit: Option<usize>,
    out: PathBuf,
    fresh: bool,
    system_set: bool,
    retrieve_set: bool,
    max_tokens_set: bool,
    qfile: Option<PathBuf>,
    expect: Option<PathBuf>,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let models = std::env::var("PLENUM_MODELS").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("models"));
    let mut a = Args {
        cmd: it.next().unwrap_or_else(|| "help".into()),
        models,
        encoder: None,
        decoder: None,
        memory: None,
        store: PathBuf::from("memory-store"),
        user: "default".into(),
        system: DEFAULT_SYSTEM_PROMPT.into(),
        question: None,
        texts: Vec::new(),
        files: Vec::new(),
        use_memory: true,
        remember: None,
        chunk: true,
        chunk_tokens: 256,
        retrieve: 3,
        sp: SampleParams { temperature: 0.0, max_tokens: 256, ..Default::default() },
        threads: 0,
        host: "127.0.0.1".into(),
        port: 8000,
        docs: None,
        benchmark: None,
        data: None,
        embeddings: None,
        scoring: None,
        offset: 0,
        limit: None,
        out: PathBuf::from("results"),
        fresh: false,
        system_set: false,
        retrieve_set: false,
        max_tokens_set: false,
        qfile: None,
        expect: None,
    };
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        fn num<T: std::str::FromStr>(name: &str, v: String) -> Result<T, String> {
            v.parse().map_err(|_| format!("{name}: '{v}' is not a number"))
        }
        match arg.as_str() {
            "--models" => a.models = val(&arg)?.into(),
            "--encoder" => a.encoder = Some(val(&arg)?.into()),
            "--decoder" => a.decoder = Some(val(&arg)?.into()),
            "--memory" => a.memory = Some(val(&arg)?.into()),
            "--store" => a.store = val(&arg)?.into(),
            "--user" | "-u" => a.user = val(&arg)?,
            "--system" => {
                a.system = val(&arg)?;
                a.system_set = true;
            }
            "--benchmark" => a.benchmark = Some(val(&arg)?),
            "--data" => a.data = Some(val(&arg)?.into()),
            "--embeddings" => a.embeddings = Some(val(&arg)?.into()),
            "--scoring" => a.scoring = Some(val(&arg)?),
            "--offset" => a.offset = num(&arg, val(&arg)?)?,
            "--limit" => a.limit = Some(num(&arg, val(&arg)?)?),
            "--out" => a.out = val(&arg)?.into(),
            "--fresh" => a.fresh = true,
            "-q" | "--question" => a.question = Some(val(&arg)?),
            "--text" => a.texts.push(val(&arg)?),
            "--grounded" => a.system = GROUNDED_SYSTEM_PROMPT.into(),
            "--no-memory" => a.use_memory = false,
            "--remember" => a.remember = Some(true),
            "--no-remember" => a.remember = Some(false),
            "--no-chunk" => a.chunk = false,
            "--chunk-tokens" => a.chunk_tokens = num(&arg, val(&arg)?)?,
            "--retrieve" => {
                a.retrieve = num(&arg, val(&arg)?)?;
                a.retrieve_set = true;
            }
            "-n" | "--max-tokens" => {
                a.sp.max_tokens = num(&arg, val(&arg)?)?;
                a.max_tokens_set = true;
            }
            "--temp" => a.sp.temperature = num(&arg, val(&arg)?)?,
            "--threads" | "-t" => a.threads = num(&arg, val(&arg)?)?,
            "--host" => a.host = val(&arg)?,
            "--port" => a.port = num(&arg, val(&arg)?)?,
            "--docs" => a.docs = Some(val(&arg)?.into()),
            "--expect" => a.expect = Some(val(&arg)?.into()),
            "--question-file" => a.qfile = Some(val(&arg)?.into()),
            s if s.starts_with('-') => return Err(format!("unknown option {s} (plenum-memory help lists them)")),
            _ => a.files.push(arg.into()),
        }
    }
    Ok(a)
}

fn engine(path: &Path, what: &str) -> Result<Engine, String> {
    let t0 = Instant::now();
    let g = Gguf::open(path)?;
    let model = Model::from_gguf(&g, &LoadOptions::default())?;
    let tok = Tokenizer::from_gguf(&g)?;
    eprintln!("  {what}: {} · {} layers · {} wide · {} · ready in {:.1}s", model.cfg.name, model.cfg.n_layer, model.cfg.n_embd, model.quant_summary, t0.elapsed().as_secs_f64());
    Ok(Engine::new(model, tok, 0))
}

fn module(a: &Args) -> Result<MemoryModule, String> {
    let path = a.memory.clone().unwrap_or_else(|| a.models.join("memorilla").join("stage3"));
    let t0 = Instant::now();
    let m = MemoryModule::load(&path)?;
    eprintln!("  memory module: {} · {} -> {} memory tokens of {} · {} heads · ready in {:.1}s", path.display(), m.dim, m.num_memories, m.out_dim, m.heads, t0.elapsed().as_secs_f64());
    Ok(m)
}

fn stack(a: &Args) -> Result<MemoryStack, String> {
    let module = module(a)?;
    let encoder = engine(&a.encoder.clone().unwrap_or_else(|| a.models.join("Qwen3-Embedding-4B-Q8_0.gguf")), "encoder")?;
    let store = Store::open(&a.store, module.dim)?;
    let mut s = MemoryStack::new(encoder, module, store)?;
    s.chunk_tokens = a.chunk_tokens;
    s.retrieve = a.retrieve;
    Ok(s)
}

fn decoder(a: &Args) -> Result<Engine, String> {
    engine(&a.decoder.clone().unwrap_or_else(|| a.models.join("Qwen3-8B-Q4_K_M.gguf")), "decoder")
}

fn cmd_add(a: &Args) -> Result<(), String> {
    let mut texts = a.texts.clone();
    let mut whole: Vec<String> = Vec::new(); // jsonl rows are documents as they are
    for f in &a.files {
        let body = std::fs::read_to_string(f).map_err(|e| format!("{}: {e}", f.display()))?;
        if f.extension().is_some_and(|e| e == "jsonl") {
            for line in body.lines().filter(|l| !l.trim().is_empty()) {
                let v: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("{}: {e}", f.display()))?;
                match v.get("text").and_then(|t| t.as_str()) {
                    Some(t) => whole.push(t.to_string()),
                    None => return Err(format!("{}: a line has no \"text\"", f.display())),
                }
            }
        } else {
            texts.push(body);
        }
    }
    if texts.is_empty() && whole.is_empty() {
        return Err("nothing to add: give --text \"...\" or files".into());
    }
    let s = stack(a)?;
    let t0 = Instant::now();
    let show = |i: usize, n: usize| {
        eprint!("\r  embedding {}/{n}   ", i + 1);
        std::io::stderr().flush().ok();
    };
    let mut n = s.add(&a.user, &texts, a.chunk, show)?;
    n += s.add(&a.user, &whole, false, show)?;
    eprintln!("\r  added {n} documents to '{}' in {:.1}s · {} in its memory now", a.user, t0.elapsed().as_secs_f64(), s.store.count(&a.user));
    Ok(())
}

fn ask_one(a: &Args, s: &MemoryStack, dec: &Engine, question: &str) -> Result<(), String> {
    let mut sess = dec.new_session();
    let t0 = Instant::now();
    let (answer, st, used) = s.answer(dec, &mut sess, &a.user, question, &a.system, a.use_memory, &a.sp, |piece| {
        print!("{piece}");
        std::io::stdout().flush().is_ok()
    })?;
    println!();
    if a.remember == Some(true) && s.save_exchange(&a.user, question, &answer)? {
        eprintln!("  saved to the memory of '{}'", a.user);
    }
    let how = if used { format!("{} documents -> {} memory tokens", s.store.count(&a.user), s.module.num_memories) } else { "no memory".to_string() };
    eprintln!("  [{how} · {} · {:.1}s in all]", st.line(), t0.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_ask(a: &Args) -> Result<(), String> {
    let q = a.question.clone().ok_or("ask needs a question: -q \"...\"")?;
    let s = stack(a)?;
    let dec = decoder(a)?;
    ask_one(a, &s, &dec, &q)
}

fn cmd_chat(a: &Args) -> Result<(), String> {
    let s = stack(a)?;
    let dec = decoder(a)?;
    eprintln!("  user '{}' · {} documents in memory · /exit leaves", a.user, s.store.count(&a.user));
    let stdin = std::io::stdin();
    loop {
        print!("\n  › ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            q => {
                print!("  ◆ ");
                if let Err(e) = ask_one(a, &s, &dec, q) {
                    eprintln!("  error: {e}");
                }
            }
        }
    }
    Ok(())
}

fn cmd_serve(a: &Args) -> Result<(), String> {
    let s = stack(a)?;
    let dec = decoder(a)?;
    if dec.model.cfg.n_embd != s.module.out_dim {
        return Err(format!("the decoder is {} wide and the memory module writes tokens of {}", dec.model.cfg.n_embd, s.module.out_dim));
    }
    eprintln!("  memory store: {} · {} users", a.store.display(), s.store.users().len());
    let model_id = "plenum-memory".to_string();
    let mut s = s;
    s.remember = a.remember.unwrap_or(true);
    eprintln!("  questions and answers are {} the users' memory", if s.remember { "saved to" } else { "not saved to" });
    serve(dec, ServerConfig { host: a.host.clone(), port: a.port, defaults: a.sp.clone(), think: false, model_id, extension: Some(Arc::new(s)) })
}

/// The stored documents closest to a question (cosine of the embeddings).
fn cmd_search(a: &Args) -> Result<(), String> {
    let q = a.question.clone().ok_or("search needs a question: -q \"...\"")?;
    let s = stack(a)?;
    match s.recall(&a.user, &q, a.retrieve.max(5))? {
        None => println!("'{}' has no documents", a.user),
        Some(r) => {
            for (sc, text) in &r.passages {
                println!("{sc:.4}  {}", text.replace('\n', " | ").chars().take(110).collect::<String>());
            }
        }
    }
    Ok(())
}

fn cmd_users(a: &Args) -> Result<(), String> {
    // the width only matters when reading embeddings; listing needs none
    let store = Store::open(&a.store, 1)?;
    let users = store.users();
    if users.is_empty() {
        println!("no collections in {}", a.store.display());
    }
    for (u, n) in users {
        println!("{u:<32} {n} documents");
    }
    Ok(())
}

fn read_f32(path: &Path) -> Result<Vec<f32>, String> {
    let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// The memory module on given inputs against what PyTorch computed from them.
fn cmd_evaluate(a: &Args) -> Result<(), String> {
    let data = a.data.clone().ok_or("evaluate needs --data (a benchmark folder from tools/import_benchmark.py, or a .jsonl of your own)")?;
    let opts = plenum_memory::eval::EvalOptions {
        benchmark: a.benchmark.clone(),
        data,
        embeddings: a.embeddings.clone(),
        scoring: a.scoring.as_deref().map(plenum_memory::eval::Scoring::parse).transpose()?,
        system: a.system_set.then(|| a.system.clone()),
        max_new_tokens: a.max_tokens_set.then_some(a.sp.max_tokens),
        offset: a.offset,
        limit: a.limit,
        retrieve: if a.retrieve_set { a.retrieve } else { 0 },
        out: a.out.clone(),
        fresh: a.fresh,
        no_memory: !a.use_memory,
    };
    let s = stack(a)?;
    let dec = decoder(a)?;
    let m = plenum_memory::eval::run(&s, &dec, &opts)?;
    println!("{}", serde_json::to_string_pretty(&m).unwrap());
    Ok(())
}

fn cmd_verify(a: &Args) -> Result<(), String> {
    let need = |p: &Option<PathBuf>, n: &str| p.clone().ok_or(format!("verify needs {n}"));
    let (docs, q, want) = (read_f32(&need(&a.docs, "--docs")?)?, read_f32(&need(&a.qfile, "--question-file")?)?, read_f32(&need(&a.expect, "--expect")?)?);
    let m = module(a)?;
    let t0 = Instant::now();
    let got: Vec<f32> = m.forward(&docs, &q)?.concat();
    let secs = t0.elapsed().as_secs_f64();
    if got.len() != want.len() {
        return Err(format!("{} values computed, {} expected", got.len(), want.len()));
    }
    let max_diff = got.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let (dot, na, nb) = got.iter().zip(&want).fold((0f64, 0f64, 0f64), |(d, x, y), (a, b)| (d + (*a as f64) * (*b as f64), x + (*a as f64).powi(2), y + (*b as f64).powi(2)));
    println!("{} documents -> {} values in {secs:.2}s", docs.len() / m.dim, got.len());
    println!("max |rust - pytorch| = {max_diff:.3e}   (largest value {scale:.3})   cosine = {:.9}", dot / (na.sqrt() * nb.sqrt()));
    if max_diff > 2e-3 * scale.max(1.0) {
        return Err("the Rust memory module does not match PyTorch".into());
    }
    println!("the Rust memory module matches PyTorch");
    Ok(())
}

fn main() {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("plenum-memory: {e}");
            std::process::exit(2);
        }
    };
    if a.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(a.threads).build_global();
    }
    let r = match a.cmd.as_str() {
        "add" => cmd_add(&a),
        "ask" => cmd_ask(&a),
        "chat" => cmd_chat(&a),
        "serve" => cmd_serve(&a),
        "users" => cmd_users(&a),
        "search" => cmd_search(&a),
        "clear" => Store::open(&a.store, 1).and_then(|s| s.clear(&a.user)).map(|_| println!("cleared '{}'", a.user)),
        "verify" => cmd_verify(&a),
        "evaluate" => cmd_evaluate(&a),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    if let Err(e) = r {
        eprintln!("plenum-memory: {e}");
        std::process::exit(1);
    }
}
