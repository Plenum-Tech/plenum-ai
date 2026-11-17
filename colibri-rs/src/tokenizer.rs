//! Byte-level BPE tokenizer read from the GGUF metadata (`tokenizer.ggml.*`), the
//! kind Qwen models carry ("gpt2" model, "qwen2" pre-tokenizer). Rust counterpart of
//! c/tok.h, with the vocabulary taken from the model file itself.

use crate::gguf::Gguf;
use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::Mutex;

/// The pre-tokenizer split of the Qwen2/Qwen3 tokenizers (and GPT-4's cl100k family).
const QWEN_SPLIT: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
/// GPT-2's split, for vocabularies that declare another pre-tokenizer.
const GPT2_SPLIT: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

const TYPE_NORMAL: i64 = 1;
const TYPE_CONTROL: i64 = 3;
const TYPE_USER: i64 = 4;

pub struct Tokenizer {
    pub tokens: Vec<String>,
    types: Vec<i64>,
    ids: HashMap<String, u32>,
    ranks: HashMap<String, usize>,
    /// Tokens matched literally in the text before BPE, longest first.
    specials: Vec<(String, u32)>,
    split: Regex,
    byte_char: [char; 256],
    char_byte: HashMap<char, u8>,
    cache: Mutex<HashMap<String, Vec<u32>>>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub add_bos: bool,
    pub chat_template: Option<String>,
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        let model = g.str("tokenizer.ggml.model").unwrap_or("");
        if model != "gpt2" {
            return Err(format!(
                "tokenizer model '{model}' is not supported: this engine reads byte-level BPE ('gpt2') vocabularies"
            ));
        }
        let tokens = g.strs("tokenizer.ggml.tokens").ok_or("GGUF has no tokenizer.ggml.tokens")?.to_vec();
        let types = match g.ints("tokenizer.ggml.token_type") {
            Some(t) if t.len() == tokens.len() => t.to_vec(),
            _ => vec![TYPE_NORMAL; tokens.len()],
        };
        let merges = g.strs("tokenizer.ggml.merges").unwrap_or(&[]);
        let pre = g.str("tokenizer.ggml.pre").unwrap_or("qwen2");
        let split = if pre == "gpt-2" || pre == "gpt2" { GPT2_SPLIT } else { QWEN_SPLIT };
        let tok = Self::build(tokens, types, merges, split)?;
        Ok(Tokenizer {
            bos: g.int("tokenizer.ggml.bos_token_id").map(|v| v as u32),
            eos: g.int("tokenizer.ggml.eos_token_id").map(|v| v as u32),
            add_bos: g.int("tokenizer.ggml.add_bos_token").unwrap_or(0) != 0,
            chat_template: g.str("tokenizer.chat_template").map(|s| s.to_string()),
            ..tok
        })
    }

    pub fn build(tokens: Vec<String>, types: Vec<i64>, merges: &[String], split: &str) -> Result<Tokenizer, String> {
        let mut ids = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            ids.entry(t.clone()).or_insert(i as u32);
        }
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            ranks.entry(m.clone()).or_insert(i);
        }
        let mut specials: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, t)| !t.is_empty() && (types[*i] == TYPE_CONTROL || types[*i] == TYPE_USER))
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        // GPT-2's bytes_to_unicode: printable bytes are themselves, the rest move to U+0100...
        let mut byte_char = ['\0'; 256];
        let mut char_byte = HashMap::with_capacity(256);
        let mut n = 0u32;
        for b in 0..256u32 {
            let cp = if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
                b
            } else {
                n += 1;
                255 + n
            };
            let ch = char::from_u32(cp).unwrap();
            byte_char[b as usize] = ch;
            char_byte.insert(ch, b as u8);
        }
        Ok(Tokenizer {
            tokens,
            types,
            ids,
            ranks,
            specials,
            split: Regex::new(split).map_err(|e| format!("pre-tokenizer regex: {e}"))?,
            byte_char,
            char_byte,
            cache: Mutex::new(HashMap::new()),
            bos: None,
            eos: None,
            add_bos: false,
            chat_template: None,
        })
    }

    pub fn n_vocab(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    /// True for the tokens a model ends its turn with (control tokens).
    pub fn is_control(&self, id: u32) -> bool {
        self.types.get(id as usize).copied() == Some(TYPE_CONTROL)
    }

    fn bpe_word(&self, word: &str, out: &mut Vec<u32>) {
        if let Some(hit) = self.cache.lock().unwrap().get(word) {
            out.extend_from_slice(hit);
            return;
        }
        let mut syms: Vec<String> = word.bytes().map(|b| self.byte_char[b as usize].to_string()).collect();
        let mut key = String::new();
        while syms.len() > 1 {
            let mut best: Option<(usize, usize)> = None; // (rank, position)
            for i in 0..syms.len() - 1 {
                key.clear();
                key.push_str(&syms[i]);
                key.push(' ');
                key.push_str(&syms[i + 1]);
                if let Some(&r) = self.ranks.get(&key) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, at)) = best else { break };
            let (a, b) = (syms[at].clone(), syms[at + 1].clone());
            let mut merged = Vec::with_capacity(syms.len());
            let mut i = 0;
            while i < syms.len() {
                if i + 1 < syms.len() && syms[i] == a && syms[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(std::mem::take(&mut syms[i]));
                    i += 1;
                }
            }
            syms = merged;
        }
        let start = out.len();
        for s in &syms {
            match self.ids.get(s) {
                Some(&id) => out.push(id),
                None => {
                    // A symbol the vocabulary does not hold: fall back to its single bytes.
                    for ch in s.chars() {
                        if let Some(&id) = self.ids.get(ch.to_string().as_str()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
        let mut cache = self.cache.lock().unwrap();
        if cache.len() < 200_000 {
            cache.insert(word.to_string(), out[start..].to_vec());
        }
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for m in self.split.find_iter(text) {
            match m {
                Ok(m) => self.bpe_word(m.as_str(), out),
                Err(_) => break,
            }
        }
    }

    /// Text to token ids. With `parse_special`, control and added tokens written in the
    /// text (`<|im_start|>`, `<think>`...) become their single id.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if !parse_special || self.specials.is_empty() {
            self.encode_plain(text, &mut out);
            return out;
        }
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest special in the rest of the text; the longest one at a tie.
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.specials {
                if let Some(p) = rest.find(s.as_str()) {
                    if best.is_none_or(|(bp, _, _)| p < bp) {
                        best = Some((p, s.len(), *id));
                    }
                }
            }
            match best {
                Some((p, len, id)) => {
                    self.encode_plain(&rest[..p], &mut out);
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    /// The bytes one token stands for.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        if self.types[id as usize] != TYPE_NORMAL {
            return t.as_bytes().to_vec();
        }
        let mut out = Vec::with_capacity(t.len());
        for ch in t.chars() {
            match self.char_byte.get(&ch) {
                Some(&b) => out.push(b),
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&i| self.token_bytes(i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Turns a stream of tokens into text that is always whole UTF-8: a character split
/// across two tokens waits for its second half.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.pending.extend_from_slice(&tok.token_bytes(id));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..ok]).unwrap());
                    match e.error_len() {
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..ok + bad);
                        }
                        None => {
                            self.pending.drain(..ok);
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Tokenizer {
        // bytes h e l o Ġ(space) w r d ! + merges + one control token
        let mut tokens: Vec<String> = "h e l o Ġ w r d !".split(' ').map(String::from).collect();
        let merges: Vec<String> =
            ["l l", "h e", "he ll", "hell o", "Ġ w", "o r", "Ġw or", "l d", "Ġwor ld"].iter().map(|s| s.to_string()).collect();
        for m in &merges {
            tokens.push(m.replace(' ', ""));
        }
        tokens.push("<|eot|>".into());
        let mut types = vec![TYPE_NORMAL; tokens.len()];
        *types.last_mut().unwrap() = TYPE_CONTROL;
        Tokenizer::build(tokens, types, &merges, QWEN_SPLIT).unwrap()
    }

    #[test]
    fn bpe_merges_and_roundtrip() {
        let t = tiny();
        let ids = t.encode("hello world!", true);
        let pieces: Vec<&str> = ids.iter().map(|&i| t.tokens[i as usize].as_str()).collect();
        assert_eq!(pieces, ["hello", "Ġworld", "!"]);
        assert_eq!(t.decode(&ids), "hello world!");
    }

    #[test]
    fn specials_are_single_tokens() {
        let t = tiny();
        let ids = t.encode("hello<|eot|>hello", true);
        assert_eq!(ids.len(), 3);
        assert!(t.is_control(ids[1]));
        assert_eq!(t.decode(&ids), "hello<|eot|>hello");
        // without parse_special the same text is ordinary characters (none in this tiny vocab)
        assert!(t.encode("<|eot|>", false).iter().all(|&i| !t.is_control(i)));
    }

    #[test]
    fn stream_decoder_holds_split_utf8() {
        let t = Tokenizer::build(
            vec!["Ã".into(), "©".into()], // the two bytes of "é" (C3 A9) in byte-level form
            vec![TYPE_NORMAL; 2],
            &[],
            QWEN_SPLIT,
        )
        .unwrap();
        let mut d = StreamDecoder::default();
        assert_eq!(d.push(&t, 0), "");
        assert_eq!(d.push(&t, 1), "é");
    }
}
