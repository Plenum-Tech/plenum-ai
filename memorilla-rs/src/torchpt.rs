//! Reads a PyTorch checkpoint (`torch.save` of a state dict) without PyTorch: the file
//! is a zip archive holding a pickle (`data.pkl`) that describes the tensors and one
//! raw file per storage. Only what a state dict of plain tensors needs is implemented;
//! nothing in the pickle is executed, an unknown callable becomes an opaque value.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
enum V {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<V>),
    List(Vec<V>),
    Dict(Vec<(V, V)>),
    Global(String, String),
    Storage { key: String, dtype: String },
    Tensor { key: String, dtype: String, offset: usize, shape: Vec<usize> },
    Opaque,
    Mark,
}

fn u16le(b: &[u8], o: usize) -> usize {
    u16::from_le_bytes([b[o], b[o + 1]]) as usize
}
fn u32le(b: &[u8], o: usize) -> usize {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) as usize
}
fn u64le(b: &[u8], o: usize) -> usize {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) as usize
}

/// name -> (offset of the data, length), for the stored (uncompressed) entries of a zip.
fn zip_entries(b: &[u8]) -> Result<HashMap<String, (usize, usize)>, String> {
    let bad = || "not a PyTorch checkpoint (no zip directory)".to_string();
    if b.len() < 22 {
        return Err(bad());
    }
    let mut eocd = None;
    let lo = b.len().saturating_sub(22 + 65536);
    for p in (lo..=b.len() - 22).rev() {
        if b[p..p + 4] == [0x50, 0x4b, 0x05, 0x06] {
            eocd = Some(p);
            break;
        }
    }
    let eocd = eocd.ok_or_else(bad)?;
    let (mut n, mut cd) = (u16le(b, eocd + 10), u32le(b, eocd + 16));
    // zip64: the locator sits right before the end record
    if eocd >= 20 && b[eocd - 20..eocd - 16] == [0x50, 0x4b, 0x06, 0x07] {
        let z = u64le(b, eocd - 20 + 8);
        if z + 56 <= b.len() && b[z..z + 4] == [0x50, 0x4b, 0x06, 0x06] {
            n = u64le(b, z + 32);
            cd = u64le(b, z + 48);
        }
    }
    let mut out = HashMap::new();
    let mut p = cd;
    for _ in 0..n {
        if p + 46 > b.len() || b[p..p + 4] != [0x50, 0x4b, 0x01, 0x02] {
            return Err(bad());
        }
        let method = u16le(b, p + 10);
        let (mut size, nl, el, cl, mut local) = (u32le(b, p + 24), u16le(b, p + 28), u16le(b, p + 30), u16le(b, p + 32), u32le(b, p + 42));
        let name = String::from_utf8_lossy(&b[p + 46..p + 46 + nl]).into_owned();
        // zip64 extra field: the 64-bit values, in order, of the fields that are 0xFFFFFFFF
        let mut e = p + 46 + nl;
        let eend = e + el;
        while e + 4 <= eend {
            let (id, len) = (u16le(b, e), u16le(b, e + 2));
            if id == 1 {
                let mut q = e + 4;
                if size == 0xFFFF_FFFF {
                    size = u64le(b, q);
                    q += 8;
                }
                if u32le(b, p + 20) == 0xFFFF_FFFF {
                    q += 8;
                }
                if local == 0xFFFF_FFFF {
                    local = u64le(b, q);
                }
            }
            e += 4 + len;
        }
        if method != 0 {
            return Err(format!("checkpoint entry '{name}' is compressed; torch.save writes them stored"));
        }
        if local + 30 > b.len() {
            return Err(bad());
        }
        let data = local + 30 + u16le(b, local + 26) + u16le(b, local + 28);
        if data + size > b.len() {
            return Err(format!("checkpoint entry '{name}' runs past the end of the file (is the download complete?)"));
        }
        out.insert(name, (data, size));
        p += 46 + nl + el + cl;
    }
    Ok(out)
}

fn reduce(f: V, args: V) -> V {
    let (V::Global(m, n), V::Tuple(a)) = (&f, &args) else { return V::Opaque };
    match (m.as_str(), n.as_str()) {
        ("collections", "OrderedDict") => V::Dict(Vec::new()),
        ("torch._utils", "_rebuild_tensor_v2") | ("torch._utils", "_rebuild_tensor") => match (a.first(), a.get(1), a.get(2)) {
            (Some(V::Storage { key, dtype }), Some(V::Int(off)), Some(V::Tuple(sh))) => V::Tensor {
                key: key.clone(),
                dtype: dtype.clone(),
                offset: *off as usize,
                shape: sh.iter().map(|d| if let V::Int(i) = d { *i as usize } else { 0 }).collect(),
            },
            _ => V::Opaque,
        },
        ("torch._utils", "_rebuild_parameter") => a.first().cloned().unwrap_or(V::Opaque),
        _ => V::Opaque,
    }
}

fn unpickle(b: &[u8]) -> Result<V, String> {
    let mut st: Vec<V> = Vec::new();
    let mut memo: HashMap<usize, V> = HashMap::new();
    let mut p = 0usize;
    let end = |p: usize, n: usize| if p + n > b.len() { Err("pickle ends early".to_string()) } else { Ok(()) };
    let pop = |st: &mut Vec<V>| st.pop().ok_or_else(|| "pickle stack underflow".to_string());
    fn to_mark(st: &mut Vec<V>) -> Result<Vec<V>, String> {
        let at = st.iter().rposition(|v| matches!(v, V::Mark)).ok_or("pickle: no mark")?;
        let items = st.split_off(at + 1);
        st.pop();
        Ok(items)
    }
    loop {
        end(p, 1)?;
        let op = b[p];
        p += 1;
        match op {
            0x80 => p += 1,                                   // PROTO
            0x95 => p += 8,                                   // FRAME
            b'.' => return pop(&mut st),                      // STOP
            b'(' => st.push(V::Mark),
            b'N' => st.push(V::None),
            0x88 => st.push(V::Bool(true)),
            0x89 => st.push(V::Bool(false)),
            b'J' => {
                end(p, 4)?;
                st.push(V::Int(i32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as i64));
                p += 4;
            }
            b'K' => {
                end(p, 1)?;
                st.push(V::Int(b[p] as i64));
                p += 1;
            }
            b'M' => {
                end(p, 2)?;
                st.push(V::Int(u16le(b, p) as i64));
                p += 2;
            }
            0x8a => {
                // LONG1: little-endian two's complement of n bytes
                end(p, 1)?;
                let n = b[p] as usize;
                end(p + 1, n)?;
                let mut v: i64 = 0;
                for (i, byte) in b[p + 1..p + 1 + n].iter().take(8).enumerate() {
                    v |= (*byte as i64) << (8 * i);
                }
                if n > 0 && n < 8 && b[p + n] & 0x80 != 0 {
                    v |= -1i64 << (8 * n);
                }
                st.push(V::Int(v));
                p += 1 + n;
            }
            b'G' => {
                end(p, 8)?;
                st.push(V::Float(f64::from_be_bytes(b[p..p + 8].try_into().unwrap())));
                p += 8;
            }
            b'X' | 0x8c | b'B' | b'C' | b'U' => {
                // BINUNICODE / SHORT_BINUNICODE / BINBYTES / SHORT_BINBYTES / SHORT_BINSTRING
                let short = matches!(op, 0x8c | b'C' | b'U');
                end(p, if short { 1 } else { 4 })?;
                let n = if short { b[p] as usize } else { u32le(b, p) };
                p += if short { 1 } else { 4 };
                end(p, n)?;
                st.push(V::Str(String::from_utf8_lossy(&b[p..p + n]).into_owned()));
                p += n;
            }
            b'c' => {
                // GLOBAL: "module\nname\n"
                let mut parts = Vec::new();
                for _ in 0..2 {
                    let nl = b[p..].iter().position(|&c| c == b'\n').ok_or("pickle: bad GLOBAL")?;
                    parts.push(String::from_utf8_lossy(&b[p..p + nl]).into_owned());
                    p += nl + 1;
                }
                st.push(V::Global(parts[0].clone(), parts[1].clone()));
            }
            0x93 => {
                // STACK_GLOBAL
                let (n, m) = (pop(&mut st)?, pop(&mut st)?);
                match (m, n) {
                    (V::Str(m), V::Str(n)) => st.push(V::Global(m, n)),
                    _ => st.push(V::Opaque),
                }
            }
            b'q' => {
                end(p, 1)?;
                memo.insert(b[p] as usize, st.last().cloned().ok_or("pickle: BINPUT on an empty stack")?);
                p += 1;
            }
            b'r' => {
                end(p, 4)?;
                memo.insert(u32le(b, p), st.last().cloned().ok_or("pickle: LONG_BINPUT on an empty stack")?);
                p += 4;
            }
            0x94 => {
                let n = memo.len();
                memo.insert(n, st.last().cloned().ok_or("pickle: MEMOIZE on an empty stack")?);
            }
            b'h' => {
                end(p, 1)?;
                st.push(memo.get(&(b[p] as usize)).cloned().unwrap_or(V::Opaque));
                p += 1;
            }
            b'j' => {
                end(p, 4)?;
                st.push(memo.get(&u32le(b, p)).cloned().unwrap_or(V::Opaque));
                p += 4;
            }
            b')' => st.push(V::Tuple(Vec::new())),
            b't' => {
                let items = to_mark(&mut st)?;
                st.push(V::Tuple(items));
            }
            0x85 | 0x86 | 0x87 => {
                let n = (op - 0x84) as usize;
                if st.len() < n {
                    return Err("pickle stack underflow".into());
                }
                let items = st.split_off(st.len() - n);
                st.push(V::Tuple(items));
            }
            b']' => st.push(V::List(Vec::new())),
            b'}' => st.push(V::Dict(Vec::new())),
            b'a' => {
                let v = pop(&mut st)?;
                if let Some(V::List(l)) = st.last_mut() {
                    l.push(v);
                }
            }
            b'e' => {
                let items = to_mark(&mut st)?;
                if let Some(V::List(l)) = st.last_mut() {
                    l.extend(items);
                }
            }
            b's' => {
                let (v, k) = (pop(&mut st)?, pop(&mut st)?);
                if let Some(V::Dict(d)) = st.last_mut() {
                    d.push((k, v));
                }
            }
            b'u' => {
                let items = to_mark(&mut st)?;
                if let Some(V::Dict(d)) = st.last_mut() {
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                }
            }
            b'Q' => {
                // BINPERSID: ('storage', <storage type>, key, location, numel)
                let pid = pop(&mut st)?;
                let v = match &pid {
                    V::Tuple(t) => match (t.first(), t.get(1), t.get(2)) {
                        (Some(V::Str(tag)), Some(V::Global(_, ty)), Some(V::Str(key))) if tag == "storage" => V::Storage { key: key.clone(), dtype: ty.clone() },
                        _ => V::Opaque,
                    },
                    _ => V::Opaque,
                };
                st.push(v);
            }
            b'R' => {
                let (args, f) = (pop(&mut st)?, pop(&mut st)?);
                st.push(reduce(f, args));
            }
            0x81 => {
                // NEWOBJ
                pop(&mut st)?;
                pop(&mut st)?;
                st.push(V::Opaque);
            }
            b'b' => {
                pop(&mut st)?; // BUILD: the state (an OrderedDict's _metadata) is not needed
            }
            _ => return Err(format!("pickle opcode 0x{op:02x} is not supported by this reader")),
        }
    }
}

/// Every tensor of a state dict saved with `torch.save`, as f32.
pub fn load_state_dict(path: &Path) -> Result<HashMap<String, Tensor>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: read-only map of a file we do not expect to change.
    let map = unsafe { Mmap::map(&file) }.map_err(|e| format!("{}: {e}", path.display()))?;
    let entries = zip_entries(&map)?;
    let (pk, _) = entries.iter().find(|(k, _)| k.ends_with("data.pkl")).ok_or("checkpoint has no data.pkl")?;
    let prefix = pk.strip_suffix("data.pkl").unwrap_or("").to_string();
    let (po, pl) = entries[pk];
    let root = unpickle(&map[po..po + pl])?;
    let V::Dict(items) = root else { return Err("the checkpoint is not a state dict".into()) };
    let mut out = HashMap::new();
    for (k, v) in items {
        let (V::Str(name), V::Tensor { key, dtype, offset, shape }) = (k, v) else { continue };
        let width = match dtype.as_str() {
            "FloatStorage" => 4,
            "HalfStorage" | "BFloat16Storage" => 2,
            "DoubleStorage" => 8,
            other => return Err(format!("tensor '{name}' is stored as {other}, which this reader does not convert")),
        };
        let &(so, sl) = entries.get(&format!("{prefix}data/{key}")).ok_or_else(|| format!("checkpoint has no storage '{key}' for '{name}'"))?;
        let n: usize = shape.iter().product();
        if (offset + n) * width > sl {
            return Err(format!("tensor '{name}' is larger than its storage"));
        }
        let raw = &map[so + offset * width..so + (offset + n) * width];
        let data: Vec<f32> = match dtype.as_str() {
            "FloatStorage" => raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            "HalfStorage" => raw.chunks_exact(2).map(|c| colibri::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BFloat16Storage" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            _ => raw.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        };
        out.insert(name, Tensor { shape, data });
    }
    if out.is_empty() {
        return Err("the checkpoint holds no tensors".into());
    }
    Ok(out)
}
