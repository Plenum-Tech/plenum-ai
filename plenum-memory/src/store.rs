//! The document collections on disk, one folder per user:
//!
//!   <root>/<user>/docs.jsonl       one {"text": ...} per line, in the order they were added
//!   <root>/<user>/embeddings.f32   the documents' embeddings, little-endian f32, row-aligned
//!
//! The embeddings are the expensive part (one encoder pass per document), so they are
//! computed once, when a document is added, and kept.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Store {
    root: PathBuf,
    pub dim: usize,
    /// One writer at a time: the two files of a collection must stay row-aligned.
    write: Mutex<()>,
}

/// One stored entry, as `list` shows it.
pub struct Entry {
    pub text: String,
    /// "document" (added by hand) or "chat" (a question and its answer, saved after it was asked)
    pub source: String,
    /// Unix seconds when it was stored; 0 for entries from before this was kept.
    pub time: u64,
}

fn parse_entry(line: &str) -> Entry {
    let v = serde_json::from_str::<serde_json::Value>(line).unwrap_or_default();
    Entry {
        text: v.get("text").and_then(|t| t.as_str()).unwrap_or_default().to_string(),
        source: v.get("source").and_then(|t| t.as_str()).unwrap_or("document").to_string(),
        time: v.get("time").and_then(|t| t.as_u64()).unwrap_or(0),
    }
}

/// A user id as a folder name: letters, digits, '_', '-', '.', '@'; anything else becomes '_'.
pub fn user_key(user: &str) -> Result<String, String> {
    let k: String = user.trim().chars().map(|c| if c.is_ascii_alphanumeric() || "_-.@".contains(c) { c } else { '_' }).collect();
    if k.is_empty() || k.chars().all(|c| c == '.') {
        return Err(format!("'{user}' cannot be used as a user id"));
    }
    Ok(k)
}

impl Store {
    pub fn open(root: &Path, dim: usize) -> Result<Store, String> {
        fs::create_dir_all(root).map_err(|e| format!("{}: {e}", root.display()))?;
        Ok(Store { root: root.to_path_buf(), dim, write: Mutex::new(()) })
    }

    fn dir(&self, user: &str) -> Result<PathBuf, String> {
        Ok(self.root.join(user_key(user)?))
    }

    /// Appends one document and its embedding.
    pub fn add(&self, user: &str, text: &str, embedding: &[f32]) -> Result<(), String> {
        self.add_entry(user, text, embedding, "document")
    }

    /// Appends one entry and its embedding, recording where it came from and when.
    pub fn add_entry(&self, user: &str, text: &str, embedding: &[f32], source: &str) -> Result<(), String> {
        let _w = self.write.lock().unwrap_or_else(|e| e.into_inner());
        if embedding.len() != self.dim {
            return Err(format!("the embedding has {} values, the store holds {}", embedding.len(), self.dim));
        }
        let dir = self.dir(user)?;
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let io = |e: std::io::Error| format!("{}: {e}", dir.display());
        // the embedding first: a line in docs.jsonl without its vector would shift every later row
        let mut f = OpenOptions::new().create(true).append(true).open(dir.join("embeddings.f32")).map_err(io)?;
        let bytes: Vec<u8> = embedding.iter().flat_map(|v| v.to_le_bytes()).collect();
        f.write_all(&bytes).map_err(io)?;
        let mut f = OpenOptions::new().create(true).append(true).open(dir.join("docs.jsonl")).map_err(io)?;
        let time = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        writeln!(f, "{}", serde_json::json!({ "text": text, "source": source, "time": time })).map_err(io)?;
        Ok(())
    }

    /// The user's documents and their embeddings ([n, dim]); empty when there are none.
    pub fn load(&self, user: &str) -> Result<(Vec<String>, Vec<f32>), String> {
        let dir = self.dir(user)?;
        let Ok(text) = fs::read_to_string(dir.join("docs.jsonl")) else { return Ok((Vec::new(), Vec::new())) };
        let docs: Vec<String> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<serde_json::Value>(l).ok().and_then(|v| v.get("text").and_then(|t| t.as_str()).map(String::from)).unwrap_or_default())
            .collect();
        let raw = fs::read(dir.join("embeddings.f32")).map_err(|e| format!("{}: {e}", dir.display()))?;
        let n = raw.len() / (4 * self.dim);
        if n < docs.len() {
            return Err(format!("{}: {} documents but {n} embeddings of {} (was the store written with another encoder?)", dir.display(), docs.len(), self.dim));
        }
        let emb: Vec<f32> = raw[..docs.len() * self.dim * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok((docs, emb))
    }

    pub fn count(&self, user: &str) -> usize {
        self.dir(user).ok().and_then(|d| fs::read_to_string(d.join("docs.jsonl")).ok()).map(|t| t.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0)
    }

    /// Every user with a collection, and how many documents each holds.
    pub fn users(&self) -> Vec<(String, usize)> {
        let mut out: Vec<(String, usize)> = fs::read_dir(&self.root)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join("docs.jsonl").exists())
            .map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let n = self.count(&name);
                (name, n)
            })
            .collect();
        out.sort();
        out
    }

    /// The user's entries, in the order they were added.
    pub fn entries(&self, user: &str) -> Result<Vec<Entry>, String> {
        let dir = self.dir(user)?;
        let Ok(text) = fs::read_to_string(dir.join("docs.jsonl")) else { return Ok(Vec::new()) };
        Ok(text.lines().filter(|l| !l.trim().is_empty()).map(parse_entry).collect())
    }

    /// Removes one entry (by its position in `entries`) and its embedding.
    pub fn remove(&self, user: &str, index: usize) -> Result<Entry, String> {
        let _w = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let dir = self.dir(user)?;
        let io = |e: std::io::Error| format!("{}: {e}", dir.display());
        let text = fs::read_to_string(dir.join("docs.jsonl")).map_err(io)?;
        let mut lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if index >= lines.len() {
            return Err(format!("'{user}' has {} entries; there is no entry {index}", lines.len()));
        }
        let mut raw = fs::read(dir.join("embeddings.f32")).map_err(io)?;
        let row = self.dim * 4;
        if raw.len() < lines.len() * row {
            return Err(format!("{}: the embeddings file is shorter than the documents", dir.display()));
        }
        let gone = parse_entry(lines.remove(index));
        raw.drain(index * row..(index + 1) * row);
        raw.truncate(lines.len() * row);
        // write both beside the old ones, then replace: never half a collection on disk
        let mut docs = lines.join("\n");
        if !docs.is_empty() {
            docs.push('\n');
        }
        fs::write(dir.join("embeddings.f32.new"), &raw).map_err(io)?;
        fs::write(dir.join("docs.jsonl.new"), docs).map_err(io)?;
        fs::rename(dir.join("embeddings.f32.new"), dir.join("embeddings.f32")).map_err(io)?;
        fs::rename(dir.join("docs.jsonl.new"), dir.join("docs.jsonl")).map_err(io)?;
        Ok(gone)
    }

    /// Removes a user's collection.
    pub fn clear(&self, user: &str) -> Result<(), String> {
        let _w = self.write.lock().unwrap_or_else(|e| e.into_inner());
        let dir = self.dir(user)?;
        if dir.exists() {
            fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_ids_become_safe_folder_names() {
        assert_eq!(user_key("sam.lee@example.com").unwrap(), "sam.lee@example.com");
        assert_eq!(user_key("../../etc").unwrap(), ".._.._etc");
        assert_eq!(user_key("a/b\\c d").unwrap(), "a_b_c_d");
        assert!(user_key("").is_err());
        assert!(user_key("..").is_err());
    }

    #[test]
    fn documents_and_embeddings_stay_row_aligned() {
        let root = std::env::temp_dir().join(format!("plenum-memory-store-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let st = Store::open(&root, 3).unwrap();
        st.add("u1", "first\nline two", &[1.0, 2.0, 3.0]).unwrap();
        st.add("u1", "second \"quoted\"", &[4.0, 5.0, 6.0]).unwrap();
        st.add("u2", "other user", &[7.0, 8.0, 9.0]).unwrap();
        assert!(st.add("u1", "wrong width", &[1.0]).is_err());
        let (docs, emb) = st.load("u1").unwrap();
        assert_eq!(docs, ["first\nline two", "second \"quoted\""]);
        assert_eq!(emb, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(st.users(), [("u1".to_string(), 2), ("u2".to_string(), 1)]);
        let e = st.entries("u1").unwrap();
        assert_eq!((e[0].source.as_str(), e[1].text.as_str()), ("document", "second \"quoted\""));
        st.add_entry("u1", "q and a", &[0.5, 0.5, 0.5], "chat").unwrap();
        assert_eq!(st.entries("u1").unwrap()[2].source, "chat");
        // removing the middle entry keeps the others aligned with their vectors
        assert_eq!(st.remove("u1", 1).unwrap().text, "second \"quoted\"");
        let (docs, emb) = st.load("u1").unwrap();
        assert_eq!(docs, ["first\nline two", "q and a"]);
        assert_eq!(emb, [1.0, 2.0, 3.0, 0.5, 0.5, 0.5]);
        assert!(st.remove("u1", 5).is_err());
        st.clear("u1").unwrap();
        assert!(st.load("u1").unwrap().0.is_empty());
        assert_eq!(st.count("u2"), 1);
        let _ = fs::remove_dir_all(&root);
    }
}
