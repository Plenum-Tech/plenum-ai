//! The logits of the last token, for checking the engine against a reference.
//!
//!   cargo run --release --example logits -- model.gguf out.f32 3 17 42 8
//!
//! Writes the logits as little-endian f32. With `--steps`, the tokens are run one at a
//! time instead of as one prompt (the recurrent and cached paths).

use plenum_engine::model::{LoadOptions, Model};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let steps = args.iter().any(|a| a == "--steps");
    let args: Vec<&String> = args.iter().filter(|a| *a != "--steps").collect();
    if args.len() < 3 {
        return Err("usage: logits <model.gguf> <out.f32> <token> [token ...] [--steps]".into());
    }
    let model = Model::load(args[0].as_ref(), &LoadOptions::default())?;
    let toks: Vec<u32> = args[2..].iter().map(|t| t.parse().map_err(|_| format!("'{t}' is not a token id"))).collect::<Result<_, _>>()?;
    let mut st = model.new_state();
    let logits = if steps {
        let mut l = Vec::new();
        for t in &toks {
            l = model.forward(&mut st, &[*t], 4096)?;
        }
        l
    } else {
        model.forward(&mut st, &toks, 4096)?
    };
    let bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(args[1], bytes).map_err(|e| e.to_string())?;
    println!("{} logits written", logits.len());
    Ok(())
}
