//! colibri in Rust: runs GGUF language models in place on the CPU, and streams the
//! experts of mixture-of-experts models from the disk instead of holding them in RAM.
//!
//! ```no_run
//! use colibri::{generate::{Engine, Message, SampleParams}, gguf::Gguf, model::{LoadOptions, Model}, tokenizer::Tokenizer};
//! let g = Gguf::open("model.gguf".as_ref()).unwrap();
//! let engine = Engine::new(Model::from_gguf(&g, &LoadOptions::default()).unwrap(), Tokenizer::from_gguf(&g).unwrap(), 0);
//! let mut sess = engine.new_session();
//! let msgs = [Message { role: "user".into(), content: "Hello!".into() }];
//! let (reply, _stats) = engine.chat(&mut sess, &msgs, false, &SampleParams::default(), &[], |_| true).unwrap();
//! println!("{reply}");
//! ```

pub mod expert;
pub mod generate;
pub mod gguf;
pub mod model;
pub mod quant;
pub mod server;
pub mod tiny;
pub mod tokenizer;
