//! Latent memory tokens through the library: vectors take the place of `<|memory|>`
//! placeholders in the prompt, the interface a memory module such as Memorilla needs
//! from its frozen decoder.
//!
//! No memory module is here, so the example makes its vectors the simplest way there
//! is: the embedding rows of a sentence the prompt never contains as text. If the
//! model answers from that sentence, the vectors entered the model where they should.
//!
//!   cargo run --release --example memory -- ../models/Qwen3-0.6B-Q8_0.gguf

use plenum_engine::generate::{Engine, SampleParams, MEMORY_PLACEHOLDER};
use plenum_engine::gguf::Gguf;
use plenum_engine::model::{LoadOptions, Model};
use plenum_engine::tokenizer::Tokenizer;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: memory <model.gguf>")?;
    let g = Gguf::open(path.as_ref())?;
    let engine = Engine::new(Model::from_gguf(&g, &LoadOptions::default())?, Tokenizer::from_gguf(&g)?, 0);
    let d = engine.model.cfg.n_embd;

    // the "memory": what a memory module would compute from a document collection
    let fact = "The user's dog is called Biscuit and is a three-year-old beagle.";
    let rows = engine.model.embed_tokens(&engine.tok.encode(fact, false))?;
    let memory: Vec<Vec<f32>> = rows.chunks(d).map(|r| r.to_vec()).collect();

    let ask = |with_memory: bool| -> Result<String, String> {
        let slots = if with_memory { MEMORY_PLACEHOLDER.repeat(memory.len()) } else { String::new() };
        let prompt = format!(
            "<|im_start|>system\nYou are a helpful assistant. {slots}<|im_end|>\n<|im_start|>user\nWhat is my dog's name?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let mem: &[Vec<f32>] = if with_memory { &memory } else { &[] };
        let p = SampleParams { temperature: 0.0, max_tokens: 40, ..Default::default() };
        let mut sess = engine.new_session();
        Ok(engine.generate_with_memory(&mut sess, &prompt, mem, &p, &[], |_| true)?.0)
    };

    println!("{} memory vectors of {d} values\n", memory.len());
    println!("without memory: {}\n", ask(false)?.trim());
    println!("with memory:    {}", ask(true)?.trim());
    Ok(())
}
