//! plenum-engine: the command line of the engine.

use plenum_engine::generate::{Engine, Message, SampleParams};
use plenum_engine::gguf::Gguf;
use plenum_engine::model::{LoadOptions, Model};
use plenum_engine::server::{serve, ServerConfig};
use plenum_engine::tokenizer::Tokenizer;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "\
plenum-engine: runs a GGUF language model in place on the CPU.

USAGE
  plenum-engine chat  <model.gguf> [options]     talk to the model in the terminal
  plenum-engine run   <model.gguf> -p \"...\"      answer one prompt and exit
  plenum-engine serve <model.gguf> [options]     OpenAI-compatible server + chat page
  plenum-engine info  <model.gguf>               what is in the file
  plenum-engine selftest                         check the engine on tiny built-in models

OPTIONS
  -p, --prompt TEXT        the prompt (run)
  --system TEXT            system message
  --raw                    run: complete the text as it is, no chat template
  --prof                   run: print where the time of the forward pass went
  --think                  let a Qwen3 model reason before it answers (off by default)
  -n, --max-tokens N       most tokens to generate (default 512)
  --temp F                 temperature, 0 = greedy (default 0.7)
  --top-k N  --top-p F     sampling cut-offs (default 20, 0.8)
  --repeat-penalty F       (default 1.05)
  --seed N                 fixed seed (default: from the clock)
  --ctx N                  context length in tokens (default: the model's, at most 8192)
  --threads N              CPU threads (default: all)
  --expert-cache-mb N      MoE: read experts into a private cache of at most N MB (a hard
                           limit on their RAM). Default 0: experts are used from the
                           mapped file and the OS keeps the used ones in RAM
  --host H  --port N       serve: where to listen (default 127.0.0.1:8000)

Supported GGUF architectures: qwen3, qwen3moe, qwen2, qwen35 (Qwen3.5 / Qwen3.8 dense),
qwen35moe (Qwen3.5 / Qwen3.6 MoE).
Supported tensor types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
";

struct Args {
    cmd: String,
    model: Option<PathBuf>,
    prompt: Option<String>,
    system: Option<String>,
    raw: bool,
    prof: bool,
    think: bool,
    sp: SampleParams,
    ctx: usize,
    threads: usize,
    expert_cache_mb: usize,
    host: String,
    port: u16,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_else(|| "help".into());
    let mut a = Args {
        cmd,
        model: None,
        prompt: None,
        system: None,
        raw: false,
        prof: false,
        think: false,
        sp: SampleParams::default(),
        ctx: 0,
        threads: 0,
        expert_cache_mb: 0,
        host: "127.0.0.1".into(),
        port: 8000,
    };
    while let Some(arg) = it.next() {
        let mut val = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        fn num<T: std::str::FromStr>(name: &str, v: String) -> Result<T, String> {
            v.parse().map_err(|_| format!("{name}: '{v}' is not a number"))
        }
        match arg.as_str() {
            "-p" | "--prompt" => a.prompt = Some(val(&arg)?),
            "--system" => a.system = Some(val(&arg)?),
            "--raw" => a.raw = true,
            "--prof" => a.prof = true,
            "--think" => a.think = true,
            "--no-think" => a.think = false,
            "-n" | "--max-tokens" => a.sp.max_tokens = num(&arg, val(&arg)?)?,
            "--temp" | "--temperature" => a.sp.temperature = num(&arg, val(&arg)?)?,
            "--top-k" => a.sp.top_k = num(&arg, val(&arg)?)?,
            "--top-p" => a.sp.top_p = num(&arg, val(&arg)?)?,
            "--repeat-penalty" => a.sp.repeat_penalty = num(&arg, val(&arg)?)?,
            "--seed" => a.sp.seed = num(&arg, val(&arg)?)?,
            "--ctx" => a.ctx = num(&arg, val(&arg)?)?,
            "--threads" | "-t" => a.threads = num(&arg, val(&arg)?)?,
            "--expert-cache-mb" => a.expert_cache_mb = num(&arg, val(&arg)?)?,
            "--host" => a.host = val(&arg)?,
            "--port" => a.port = num(&arg, val(&arg)?)?,
            "--model" | "-m" => a.model = Some(PathBuf::from(val(&arg)?)),
            s if s.starts_with('-') => return Err(format!("unknown option {s} (plenum-engine help lists them)")),
            _ if a.model.is_none() => a.model = Some(PathBuf::from(arg)),
            _ => return Err(format!("unexpected argument '{arg}'")),
        }
    }
    Ok(a)
}

fn gb(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else {
        format!("{:.0} MB", b as f64 / (1u64 << 20) as f64)
    }
}

fn load(a: &Args) -> Result<Engine, String> {
    let path = a.model.as_ref().ok_or("which model? give the path of a .gguf file")?;
    let t0 = Instant::now();
    let g = Gguf::open(path)?;
    let model = Model::from_gguf(&g, &LoadOptions { expert_cache_bytes: a.expert_cache_mb << 20 })?;
    let tok = Tokenizer::from_gguf(&g)?;
    let c = &model.cfg;
    let mut kind = if c.n_expert > 0 {
        format!("MoE {} experts/layer, {} routed · streaming from disk", c.n_expert, c.n_expert_used)
    } else {
        "dense".to_string()
    };
    if c.delta.is_some() {
        kind.push_str(&format!(" · hybrid: {} attention + {} DeltaNet layers", c.n_attn_layers, c.n_layer - c.n_attn_layers));
    }
    eprintln!("  plenum-engine v{} · {} · {} · {} layers · {}", env!("CARGO_PKG_VERSION"), c.name, c.arch, c.n_layer, kind);
    eprintln!("  blocks: {}", model.quant_summary);
    if c.n_expert > 0 {
        let how = if a.expert_cache_mb == 0 { "used from the mapped file".to_string() } else { format!("private cache up to {}", gb(a.expert_cache_mb << 20)) };
        eprintln!("  resident {} (mapped) · experts on disk {} · {how}", gb(model.dense_bytes), gb(model.expert_bytes));
    } else {
        eprintln!("  resident {} (mapped)", gb(model.dense_bytes));
    }
    let engine = Engine::new(model, tok, a.ctx);
    eprintln!("  ✓ ready in {:.1}s · context {} · {} threads", t0.elapsed().as_secs_f64(), engine.n_ctx, rayon::current_num_threads());
    Ok(engine)
}

fn print_piece(s: &str) -> bool {
    print!("{s}");
    std::io::stdout().flush().is_ok()
}

fn cmd_info(a: &Args) -> Result<(), String> {
    let path = a.model.as_ref().ok_or("which model? give the path of a .gguf file")?;
    let g = Gguf::open(path)?;
    println!("file      {}", path.display());
    println!("GGUF      v{} · {} tensors · {} metadata keys · {}", g.version, g.tensors.len(), g.meta.len(), gb(g.map.len()));
    for k in ["general.architecture", "general.name", "general.size_label", "general.quantized_by", "tokenizer.ggml.model", "tokenizer.ggml.pre"] {
        if let Some(v) = g.str(k) {
            println!("{k:<34} {v}");
        }
    }
    let mut keys: Vec<&String> = g.meta.keys().collect();
    keys.sort();
    for k in keys {
        let v = match g.meta.get(k) {
            Some(plenum_engine::gguf::Value::Int(i)) => Some(i.to_string()),
            Some(plenum_engine::gguf::Value::Float(f)) => Some(f.to_string()),
            Some(plenum_engine::gguf::Value::Bool(b)) => Some(b.to_string()),
            _ => None,
        };
        if let Some(v) = v {
            println!("{k:<34} {v}");
        }
    }
    let mut by_type = std::collections::BTreeMap::new();
    for t in &g.tensors {
        let e = by_type.entry(plenum_engine::quant::type_id_name(t.type_id)).or_insert((0usize, 0usize));
        e.0 += 1;
        e.1 += t.n_elements();
    }
    for (ty, (n, el)) in by_type {
        println!("tensors   {ty:<8} {n:>5}  {:.1} M weights", el as f64 / 1e6);
    }
    match Model::from_gguf(&g, &LoadOptions::default()) {
        Ok(m) => println!("engine    runs: resident {} · experts on disk {}", gb(m.dense_bytes), gb(m.expert_bytes)),
        Err(e) => println!("engine    cannot run this file: {e}"),
    }
    Ok(())
}

fn cmd_run(a: &Args) -> Result<(), String> {
    let prompt = a.prompt.clone().ok_or("run needs a prompt: -p \"...\"")?;
    let eng = load(a)?;
    let mut sess = eng.new_session();
    let (_, st) = if a.raw {
        let ids = eng.tok.encode(&prompt, false);
        eng.generate(&mut sess, &ids, &a.sp, &[], print_piece)?
    } else {
        let mut msgs = Vec::new();
        if let Some(s) = &a.system {
            msgs.push(Message { role: "system".into(), content: s.clone() });
        }
        msgs.push(Message { role: "user".into(), content: prompt });
        eng.chat(&mut sess, &msgs, a.think, &a.sp, &[], print_piece)?
    };
    println!();
    eprintln!("  {}", st.line());
    if let Some(x) = &eng.model.experts {
        eprintln!("  {}", x.stats());
    }
    if a.prof {
        eprintln!("  {}", eng.model.prof.line());
    }
    Ok(())
}

fn cmd_chat(a: &Args) -> Result<(), String> {
    let eng = load(a)?;
    let mut sess = eng.new_session();
    let mut msgs: Vec<Message> = Vec::new();
    if let Some(s) = &a.system {
        msgs.push(Message { role: "system".into(), content: s.clone() });
    }
    let base = msgs.len();
    let mut think = a.think;
    eprintln!("  /reset starts over · /think toggles reasoning · /exit leaves");
    let stdin = std::io::stdin();
    loop {
        print!("\n  › ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/exit" | "/quit" | "/bye" => break,
            "/reset" => {
                msgs.truncate(base);
                eprintln!("  (new conversation)");
                continue;
            }
            "/think" => {
                think = !think;
                eprintln!("  (reasoning {})", if think { "on" } else { "off" });
                continue;
            }
            _ => {}
        }
        msgs.push(Message { role: "user".into(), content: line.to_string() });
        print!("  ◆ ");
        match eng.chat(&mut sess, &msgs, think, &a.sp, &[], print_piece) {
            Ok((text, st)) => {
                println!();
                eprintln!("  [{}]", st.line());
                let (_, answer) = plenum_engine::generate::split_thinking(&text);
                msgs.push(Message { role: "assistant".into(), content: answer });
            }
            Err(e) => {
                println!();
                eprintln!("  error: {e}");
                msgs.pop();
            }
        }
    }
    Ok(())
}

fn cmd_serve(a: &Args) -> Result<(), String> {
    let eng = load(a)?;
    let model_id = a.model.as_ref().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    serve(eng, ServerConfig { host: a.host.clone(), port: a.port, defaults: a.sp.clone(), think: a.think, model_id, extension: None })
}

fn cmd_selftest() -> Result<(), String> {
    let dir = std::env::temp_dir().join(format!("plenum-engine-selftest-{}", std::process::id()));
    let r = plenum_engine::tiny::selfcheck(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    for l in r? {
        println!("{l}");
    }
    println!("selftest passed");
    Ok(())
}

fn main() {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("plenum-engine: {e}");
            std::process::exit(2);
        }
    };
    if a.threads > 0 {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(a.threads).build_global();
    }
    let r = match a.cmd.as_str() {
        "chat" => cmd_chat(&a),
        "run" => cmd_run(&a),
        "serve" => cmd_serve(&a),
        "info" => cmd_info(&a),
        "selftest" => cmd_selftest(),
        "help" | "-h" | "--help" => {
            print!("{USAGE}");
            Ok(())
        }
        other => Err(format!("unknown command '{other}'\n\n{USAGE}")),
    };
    if let Err(e) = r {
        eprintln!("plenum-engine: {e}");
        std::process::exit(1);
    }
}
