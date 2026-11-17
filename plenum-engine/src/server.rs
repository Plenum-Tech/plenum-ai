//! The OpenAI-compatible HTTP server (Rust counterpart of c/openai_server.py), on the
//! standard library alone: /v1/chat/completions (streaming or not), /v1/completions,
//! /v1/models, /health, and a small chat page at /.
//!
//! One model, one conversation slot: requests run one after another, and a request
//! whose messages begin like the previous one reuses its KV cache.

use crate::generate::{split_thinking, Engine, GenStats, Message, SampleParams, Session};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// What a program built on this server adds to it (a memory system, for one).
pub trait Extension: Send + Sync {
    /// Called for every chat request before the prompt is built. It may rewrite the
    /// messages and return latent vectors: one per `<|memory|>` it put in them.
    fn memory(&self, _request: &Value, _messages: &mut Vec<Message>) -> Result<Option<Vec<Vec<f32>>>, String> {
        Ok(None)
    }
    /// Called once a chat answer has been sent in full, on a thread of its own so the
    /// client is not kept waiting: the user's last question as they wrote it (before
    /// `memory` rewrote it) and the answer as sent. Not called for a cancelled answer.
    fn answered(&self, _request: &Value, _question: &str, _answer: &str) {}
    /// Routes of its own: Some((status line, JSON body)) when it answers this one.
    fn route(&self, _method: &str, _path: &str, _body: &[u8]) -> Option<(String, Value)> {
        None
    }
}

pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub defaults: SampleParams,
    pub think: bool,
    pub model_id: String,
    pub extension: Option<Arc<dyn Extension>>,
}

struct Ctx {
    engine: Engine,
    session: Mutex<Session>,
    cfg: ServerConfig,
}

const PAGE: &str = include_str!("chat.html");

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(s: &mut BufReader<TcpStream>) -> Result<Request, String> {
    let mut line = String::new();
    s.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request")?.to_string();
    let path = parts.next().ok_or("no path")?.split('?').next().unwrap_or("/").to_string();
    let mut len = 0usize;
    let mut total = 0usize;
    loop {
        let mut h = String::new();
        let n = s.read_line(&mut h).map_err(|e| e.to_string())?;
        total += n;
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if total > 64 * 1024 {
            return Err("headers too large".into());
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().map_err(|_| "bad Content-Length")?;
            }
        }
    }
    if len > 64 << 20 {
        return Err("body too large".into());
    }
    let mut body = vec![0u8; len];
    s.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, body })
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn respond(s: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    write!(s, "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n", body.len())?;
    s.write_all(body)?;
    s.flush()
}

fn respond_json(s: &mut TcpStream, status: &str, v: &Value) -> std::io::Result<()> {
    respond(s, status, "application/json", v.to_string().as_bytes())
}

fn error(s: &mut TcpStream, status: &str, msg: &str) -> std::io::Result<()> {
    respond_json(s, status, &json!({"error": {"message": msg, "type": "invalid_request_error"}}))
}

/// A message's content: a string, or the list of {type:"text", text} parts.
fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()).or_else(|| p.as_str()))
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn params(ctx: &Ctx, req: &Value) -> (SampleParams, Vec<String>, bool) {
    let mut p = ctx.cfg.defaults.clone();
    let f = |k: &str| req.get(k).and_then(|v| v.as_f64());
    if let Some(v) = f("temperature") {
        p.temperature = v as f32;
    }
    if let Some(v) = f("top_p") {
        p.top_p = v as f32;
    }
    if let Some(v) = f("top_k") {
        p.top_k = v as usize;
    }
    if let Some(v) = f("repeat_penalty").or(f("repetition_penalty")) {
        p.repeat_penalty = v as f32;
    }
    if let Some(v) = f("max_completion_tokens").or(f("max_tokens")) {
        p.max_tokens = v.max(1.0) as usize;
    }
    if let Some(v) = req.get("seed").and_then(|v| v.as_u64()) {
        p.seed = v;
    }
    let stops = match req.get("stop") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    let think = req
        .get("enable_thinking")
        .or_else(|| req.get("chat_template_kwargs").and_then(|k| k.get("enable_thinking")))
        .or_else(|| req.get("think"))
        .and_then(|v| v.as_bool())
        .unwrap_or(ctx.cfg.think);
    (p, stops, think)
}

fn usage(st: &GenStats) -> Value {
    json!({"prompt_tokens": st.prompt_tokens, "completion_tokens": st.completion_tokens, "total_tokens": st.prompt_tokens + st.completion_tokens})
}

fn sse_head(s: &mut TcpStream) -> std::io::Result<()> {
    write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n")?;
    s.flush()
}

fn sse(s: &mut TcpStream, v: &Value) -> bool {
    write!(s, "data: {v}\n\n").and_then(|_| s.flush()).is_ok()
}

fn completions(ctx: &Ctx, s: &mut TcpStream, req: &Value, chat: bool) -> std::io::Result<()> {
    let (p, stops, think) = params(ctx, req);
    let stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let eng = &ctx.engine;
    // latent memory tokens (Memorilla-style): vectors that replace the <|memory|> placeholders
    let mut memory: Option<Vec<Vec<f32>>> = req.get("memory_embeddings").and_then(|m| m.as_array()).map(|rows| {
        rows.iter()
            .map(|r| r.as_array().map(|v| v.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default())
            .collect()
    });
    let prompt_text;
    let mut question = String::new();
    let prompt_ids = if chat {
        let Some(msgs) = req.get("messages").and_then(|m| m.as_array()) else {
            return error(s, "400 Bad Request", "'messages' must be a list");
        };
        let mut messages: Vec<Message> = msgs
            .iter()
            .map(|m| Message {
                role: match m.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                    "developer" => "system".to_string(),
                    r => r.to_string(),
                },
                content: content_text(m.get("content").unwrap_or(&Value::Null)),
            })
            .collect();
        if messages.is_empty() {
            return error(s, "400 Bad Request", "'messages' is empty");
        }
        question = messages.iter().rev().find(|m| m.role == "user").map(|m| m.content.clone()).unwrap_or_default();
        if memory.is_none() {
            if let Some(ext) = &ctx.cfg.extension {
                match ext.memory(req, &mut messages) {
                    Ok(m) => memory = m,
                    Err(e) => return error(s, "400 Bad Request", &e),
                }
            }
        }
        prompt_text = crate::generate::chatml_prompt(&eng.tok, &messages, think);
        eng.tok.encode(&prompt_text, true)
    } else {
        let text = content_text(req.get("prompt").unwrap_or(&Value::Null));
        if text.is_empty() {
            return error(s, "400 Bad Request", "'prompt' is empty");
        }
        let ids = eng.tok.encode(&text, false);
        prompt_text = text;
        ids
    };
    let id = format!("{}-{:x}", if chat { "chatcmpl" } else { "cmpl" }, now() * 1000 + (prompt_ids.len() as u64 % 1000));
    let object = if chat { "chat.completion" } else { "text_completion" };
    let model = ctx.cfg.model_id.as_str();
    let created = now();
    let mut sess = ctx.session.lock().unwrap_or_else(|e| e.into_inner());

    if !stream {
        let res = match &memory {
            Some(m) => eng.generate_with_memory(&mut sess, &prompt_text, m, &p, &stops, |_| true),
            None => eng.generate(&mut sess, &prompt_ids, &p, &stops, |_| true),
        };
        return match res {
            Ok((text, st)) => {
                eprintln!("  {}", st.line());
                let choice = if chat {
                    let (reasoning, answer) = split_thinking(&text);
                    let mut msg = json!({"role": "assistant", "content": answer});
                    if let Some(r) = reasoning {
                        msg["reasoning_content"] = json!(r);
                    }
                    json!({"index": 0, "message": msg, "finish_reason": st.finish})
                } else {
                    json!({"index": 0, "text": text, "finish_reason": st.finish})
                };
                let r = respond_json(s, "200 OK", &json!({"id": id, "object": object, "created": created, "model": model, "choices": [choice], "usage": usage(&st)}));
                drop(sess);
                if chat {
                    notify(ctx, req, &question, &text, st.finish);
                }
                r
            }
            Err(e) => error(s, "400 Bad Request", &e),
        };
    }

    sse_head(s)?;
    let chunk_obj = if chat { "chat.completion.chunk" } else { "text_completion" };
    let chunk = |delta: Value, finish: Value| {
        if chat {
            json!({"id": id, "object": chunk_obj, "created": created, "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        } else {
            json!({"id": id, "object": chunk_obj, "created": created, "model": model, "choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(json!("")), "finish_reason": finish}]})
        }
    };
    if chat {
        sse(s, &chunk(json!({"role": "assistant", "content": ""}), Value::Null));
    }
    let mut out = s.try_clone()?;
    let on_piece = |piece: &str| sse(&mut out, &chunk(json!({"content": piece}), Value::Null));
    let res = match &memory {
        Some(m) => eng.generate_with_memory(&mut sess, &prompt_text, m, &p, &stops, on_piece),
        None => eng.generate(&mut sess, &prompt_ids, &p, &stops, on_piece),
    };
    let mut done = None;
    match res {
        Ok((text, st)) => {
            eprintln!("  {}", st.line());
            done = Some((text, st.finish));
            let fin = if st.finish == "cancel" { "stop" } else { st.finish };
            let mut last = chunk(json!({}), json!(fin));
            last["usage"] = usage(&st);
            sse(s, &last);
        }
        Err(e) => {
            sse(s, &json!({"error": {"message": e, "type": "invalid_request_error"}}));
        }
    }
    write!(s, "data: [DONE]\n\n")?;
    let r = s.flush();
    drop(sess);
    if let (true, Some((text, finish))) = (chat, done) {
        notify(ctx, req, &question, &text, finish);
    }
    r
}

/// Hands a finished chat answer to the extension, on a thread of its own.
fn notify(ctx: &Ctx, req: &Value, question: &str, text: &str, finish: &str) {
    if finish == "cancel" || question.trim().is_empty() {
        return;
    }
    let Some(ext) = ctx.cfg.extension.clone() else { return };
    let (req, q, a) = (req.clone(), question.to_string(), split_thinking(text).1);
    std::thread::spawn(move || ext.answered(&req, &q, &a));
}

fn handle(ctx: &Ctx, stream: TcpStream) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let mut s = stream;
    let req = match read_request(&mut reader) {
        Ok(r) => r,
        Err(e) => {
            let _ = error(&mut s, "400 Bad Request", &e);
            return;
        }
    };
    eprintln!("{} {}", req.method, req.path);
    let c = &ctx.engine.model.cfg;
    let r = match (req.method.as_str(), req.path.as_str()) {
        ("OPTIONS", _) => respond(&mut s, "204 No Content", "text/plain", b""),
        ("GET", "/") | ("GET", "/index.html") => respond(&mut s, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes()),
        ("GET", "/health") | ("GET", "/v1/health") => respond_json(&mut s, "200 OK", &json!({"status": "ok", "model": ctx.cfg.model_id})),
        ("GET", "/v1/models") | ("GET", "/models") => respond_json(
            &mut s,
            "200 OK",
            &json!({"object": "list", "data": [{"id": ctx.cfg.model_id, "object": "model", "created": now(), "owned_by": "plenum-engine",
                "meta": {"architecture": c.arch, "n_ctx": ctx.engine.n_ctx, "n_layer": c.n_layer, "n_embd": c.n_embd, "n_expert": c.n_expert}}]}),
        ),
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") | ("POST", "/v1/completions") | ("POST", "/completions") => {
            match serde_json::from_slice::<Value>(&req.body) {
                Ok(v) => completions(ctx, &mut s, &v, req.path.contains("chat")),
                Err(e) => error(&mut s, "400 Bad Request", &format!("the body is not JSON: {e}")),
            }
        }
        (m, p) => match ctx.cfg.extension.as_ref().and_then(|x| x.route(m, p, &req.body)) {
            Some((status, body)) => respond_json(&mut s, &status, &body),
            None => error(&mut s, "404 Not Found", "no such route"),
        },
    };
    let _ = r;
}

pub fn serve(engine: Engine, cfg: ServerConfig) -> Result<(), String> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
    eprintln!("Starting Plenum AI");
    eprintln!("  Browser:          http://{addr}/");
    eprintln!("  OpenAI base URL:  http://{addr}/v1");
    eprintln!("  model id:         {}", cfg.model_id);
    eprintln!("  stop: press Ctrl+C");
    let session = Mutex::new(engine.new_session());
    let ctx = Arc::new(Ctx { engine, session, cfg });
    for conn in listener.incoming() {
        let Ok(conn) = conn else { continue };
        let ctx = ctx.clone();
        std::thread::spawn(move || handle(&ctx, conn));
    }
    Ok(())
}
