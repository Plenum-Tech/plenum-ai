//! Expert streaming: the routed experts of a mixture-of-experts model stay on the
//! disk and are read when the router asks for them. This is colibri's core idea
//! (c/expert_store.h, c/olmoe.c): the dense part of the model is resident, an expert
//! is one positioned read, and a cache bounded in bytes keeps the recently used ones.

use crate::quant::GType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Where one layer's expert matrices are in the file. Each is [n_expert] matrices
/// of [rows, cols], one after the other.
#[derive(Clone, Copy, Debug)]
pub struct ExpertTensor {
    pub offset: usize,
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
}

impl ExpertTensor {
    pub fn bytes_per_expert(&self) -> usize {
        self.ty.bytes_for(self.cols) * self.rows
    }
}

/// A matrix's bytes: read into RAM, or a range of the mapped file.
pub enum Bytes {
    Owned(Vec<u8>),
    Mapped(Arc<Mmap>, usize, usize),
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Owned(v) => v,
            Bytes::Mapped(m, off, len) => &m[*off..*off + *len],
        }
    }
}

/// One expert's three matrices, in the file's own bytes.
pub struct Expert {
    pub gate: Bytes,
    pub up: Bytes,
    pub down: Bytes,
}

impl Expert {
    fn bytes(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }
}

struct Cache {
    map: HashMap<(u32, u32), (Arc<Expert>, u64)>,
    bytes: usize,
    tick: u64,
}

pub struct ExpertStore {
    file: File,
    /// Some = mapped mode: experts are ranges of the map, the OS decides what stays in RAM.
    map: Option<Arc<Mmap>>,
    budget: usize,
    cache: Mutex<Cache>,
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub bytes_read: AtomicU64,
}

#[cfg(unix)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(buf, off)
}

#[cfg(windows)]
fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        let n = f.seek_read(&mut buf[done..], off + done as u64)?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "expert read past the end of the file"));
        }
        done += n;
    }
    Ok(())
}

impl ExpertStore {
    /// Two ways to stream. With `map` (budget 0), an expert is a range of the mapped
    /// file: no copy, and the operating system keeps in RAM the experts that are used
    /// and drops the others when memory is short. Without it, an expert is read into
    /// a private cache of at most `budget` bytes (a hard limit on the experts' RAM).
    pub fn open(path: &Path, budget: usize, map: Option<Arc<Mmap>>) -> io::Result<ExpertStore> {
        Ok(ExpertStore {
            file: File::open(path)?,
            map,
            budget,
            cache: Mutex::new(Cache { map: HashMap::new(), bytes: 0, tick: 0 }),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        })
    }

    fn read(&self, t: &ExpertTensor, e: usize) -> io::Result<Bytes> {
        let n = t.bytes_per_expert();
        let mut buf = vec![0u8; n];
        read_at(&self.file, &mut buf, (t.offset + e * n) as u64)?;
        self.bytes_read.fetch_add(n as u64, Ordering::Relaxed);
        Ok(Bytes::Owned(buf))
    }

    pub fn is_mapped(&self) -> bool {
        self.map.is_some()
    }

    /// Expert `e` of `layer`: from the cache, or read from the disk (and the least
    /// recently used ones leave when the budget is passed).
    pub fn get(&self, layer: usize, e: usize, gate: &ExpertTensor, up: &ExpertTensor, down: &ExpertTensor) -> io::Result<Arc<Expert>> {
        if let Some(m) = &self.map {
            let view = |t: &ExpertTensor| {
                let n = t.bytes_per_expert();
                Bytes::Mapped(m.clone(), t.offset + e * n, n)
            };
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::new(Expert { gate: view(gate), up: view(up), down: view(down) }));
        }
        let key = (layer as u32, e as u32);
        {
            let mut c = self.cache.lock().unwrap();
            c.tick += 1;
            let tick = c.tick;
            if let Some((x, used)) = c.map.get_mut(&key) {
                *used = tick;
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(x.clone());
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let x = Arc::new(Expert { gate: self.read(gate, e)?, up: self.read(up, e)?, down: self.read(down, e)? });
        let mut c = self.cache.lock().unwrap();
        c.tick += 1;
        let tick = c.tick;
        c.bytes += x.bytes();
        if let Some((old, _)) = c.map.insert(key, (x.clone(), tick)) {
            c.bytes -= old.bytes();
        }
        while c.bytes > self.budget && c.map.len() > 1 {
            let victim = c.map.iter().filter(|(k, _)| **k != key).min_by_key(|(_, (_, used))| *used).map(|(k, _)| *k);
            match victim {
                Some(k) => {
                    if let Some((old, _)) = c.map.remove(&k) {
                        c.bytes -= old.bytes();
                    }
                }
                None => break,
            }
        }
        Ok(x)
    }

    pub fn resident_bytes(&self) -> usize {
        self.cache.lock().unwrap().bytes
    }

    pub fn stats(&self) -> String {
        let (h, m) = (self.hits.load(Ordering::Relaxed), self.misses.load(Ordering::Relaxed));
        if self.is_mapped() {
            return format!("experts: {h} routed, used from the mapped file (the OS keeps the used ones in RAM)");
        }
        let rate = if h + m > 0 { 100.0 * h as f64 / (h + m) as f64 } else { 0.0 };
        format!(
            "experts: {} hits, {} disk reads ({rate:.1}% cached), {:.1} MB read, {:.1} MB resident",
            h,
            m,
            self.bytes_read.load(Ordering::Relaxed) as f64 / 1048576.0,
            self.resident_bytes() as f64 / 1048576.0
        )
    }
}
