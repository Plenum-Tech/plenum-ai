//! GGUF, read in place: the file is mapped, the metadata parsed, and every tensor is
//! a (type, shape, byte range) view over the map. Nothing is converted or copied.
//! (Rust counterpart of c/gguf.h. A small writer is here too, for tests and selftest.)

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strs(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the row length (the innermost dimension).
    pub dims: Vec<usize>,
    pub type_id: u32,
    /// Absolute byte offset in the file.
    pub offset: usize,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub path: PathBuf,
    pub map: Arc<Mmap>,
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

struct Cur<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.p + n > self.b.len() {
            return Err("GGUF: file ends inside its header".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn scalar(&mut self, t: u32) -> Result<Value, String> {
        Ok(match t {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            _ => return Err(format!("GGUF: unknown metadata type {t}")),
        })
    }
    fn value(&mut self, t: u32) -> Result<Value, String> {
        if t != 9 {
            return self.scalar(t);
        }
        let et = self.u32()?;
        let n = self.u64()? as usize;
        match et {
            8 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.string()?);
                }
                Ok(Value::Strs(v))
            }
            6 | 12 => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    if let Value::Float(f) = self.scalar(et)? {
                        v.push(f);
                    }
                }
                Ok(Value::Floats(v))
            }
            9 => Err("GGUF: nested arrays are not supported".into()),
            _ => {
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    match self.scalar(et)? {
                        Value::Int(i) => v.push(i),
                        Value::Bool(b) => v.push(b as i64),
                        _ => {}
                    }
                }
                Ok(Value::Ints(v))
            }
        }
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: the map is read-only; the file is not expected to change under us.
        let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: mmap: {e}", path.display()))?;
        let mut c = Cur { b: &map, p: 0 };
        if c.take(4)? != b"GGUF" {
            return Err(format!("{}: not a GGUF file", path.display()));
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("GGUF version {version} is not supported (2 and 3 are)"));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut meta = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let k = c.string()?;
            let t = c.u32()?;
            let v = c.value(t)?;
            meta.insert(k, v);
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(c.u64()? as usize);
            }
            let type_id = c.u32()?;
            let offset = c.u64()? as usize;
            tensors.push(TensorInfo { name, dims, type_id, offset });
        }
        let align = match meta.get("general.alignment") {
            Some(Value::Int(a)) if *a > 0 => *a as usize,
            _ => 32,
        };
        let data_start = c.p.div_ceil(align) * align;
        let mut index = HashMap::with_capacity(n_tensors);
        for (i, t) in tensors.iter_mut().enumerate() {
            t.offset += data_start;
            index.insert(t.name.clone(), i);
        }
        Ok(Gguf { path: path.to_path_buf(), map: Arc::new(map), version, meta, tensors, index })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        match self.meta.get(key) {
            Some(Value::Str(s)) => Some(s),
            _ => None,
        }
    }
    pub fn int(&self, key: &str) -> Option<i64> {
        match self.meta.get(key) {
            Some(Value::Int(i)) => Some(*i),
            Some(Value::Bool(b)) => Some(*b as i64),
            Some(Value::Float(f)) => Some(*f as i64),
            _ => None,
        }
    }
    pub fn float(&self, key: &str) -> Option<f64> {
        match self.meta.get(key) {
            Some(Value::Float(f)) => Some(*f),
            Some(Value::Int(i)) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn strs(&self, key: &str) -> Option<&[String]> {
        match self.meta.get(key) {
            Some(Value::Strs(s)) => Some(s),
            _ => None,
        }
    }
    pub fn ints(&self, key: &str) -> Option<&[i64]> {
        match self.meta.get(key) {
            Some(Value::Ints(s)) => Some(s),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------
// Writer: enough of GGUF v3 to build a model file (tests, `plenum-engine selftest`).
// ---------------------------------------------------------------------------------

pub struct Writer {
    kv: Vec<u8>,
    n_kv: u64,
    tensors: Vec<(String, Vec<usize>, u32, Vec<u8>)>,
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub fn new() -> Writer {
        Writer { kv: Vec::new(), n_kv: 0, tensors: Vec::new() }
    }
    fn key(&mut self, k: &str, t: u32) {
        put_str(&mut self.kv, k);
        self.kv.extend_from_slice(&t.to_le_bytes());
        self.n_kv += 1;
    }
    pub fn u32(&mut self, k: &str, v: u32) {
        self.key(k, 4);
        self.kv.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, k: &str, v: f32) {
        self.key(k, 6);
        self.kv.extend_from_slice(&v.to_le_bytes());
    }
    pub fn str(&mut self, k: &str, v: &str) {
        self.key(k, 8);
        put_str(&mut self.kv, v);
    }
    pub fn strs(&mut self, k: &str, v: &[String]) {
        self.key(k, 9);
        self.kv.extend_from_slice(&8u32.to_le_bytes());
        self.kv.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for s in v {
            put_str(&mut self.kv, s);
        }
    }
    pub fn i32s(&mut self, k: &str, v: &[i32]) {
        self.key(k, 9);
        self.kv.extend_from_slice(&5u32.to_le_bytes());
        self.kv.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for x in v {
            self.kv.extend_from_slice(&x.to_le_bytes());
        }
    }
    /// `dims` in ggml order (row length first); `data` already in the type's bytes.
    pub fn tensor(&mut self, name: &str, dims: &[usize], type_id: u32, data: Vec<u8>) {
        self.tensors.push((name.to_string(), dims.to_vec(), type_id, data));
    }
    pub fn tensor_f32(&mut self, name: &str, dims: &[usize], data: &[f32]) {
        let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.tensor(name, dims, 0, bytes);
    }
    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&self.n_kv.to_le_bytes());
        head.extend_from_slice(&self.kv);
        let mut off = 0usize;
        let mut offs = Vec::new();
        for (name, dims, ty, data) in &self.tensors {
            put_str(&mut head, name);
            head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                head.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            head.extend_from_slice(&ty.to_le_bytes());
            head.extend_from_slice(&(off as u64).to_le_bytes());
            offs.push(off);
            off = (off + data.len()).div_ceil(32) * 32;
        }
        let mut f = io::BufWriter::new(File::create(path)?);
        f.write_all(&head)?;
        let pad = head.len().div_ceil(32) * 32 - head.len();
        f.write_all(&vec![0u8; pad])?;
        for (i, (_, _, _, data)) in self.tensors.iter().enumerate() {
            f.write_all(data)?;
            let end = offs[i] + data.len();
            f.write_all(&vec![0u8; end.div_ceil(32) * 32 - end])?;
        }
        f.flush()
    }
}
