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

pub struct Store {
    root: PathBuf,
    pub dim: usize,
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
        Ok(Store { root: root.to_path_buf(), dim })
    }

    fn dir(&self, user: &str) -> Result<PathBuf, String> {
        Ok(self.root.join(user_key(user)?))
    }

    /// Appends one document and its embedding.
    pub fn add(&self, user: &str, text: &str, embedding: &[f32]) -> Result<(), String> {
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
        writeln!(f, "{}", serde_json::json!({ "text": text })).map_err(io)?;
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

    /// Removes a user's collection.
    pub fn clear(&self, user: &str) -> Result<(), String> {
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
        st.clear("u1").unwrap();
        assert!(st.load("u1").unwrap().0.is_empty());
        assert_eq!(st.count("u2"), 1);
        let _ = fs::remove_dir_all(&root);
    }
}
