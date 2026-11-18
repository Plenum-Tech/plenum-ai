//! Reads safetensors files (a JSON header, then raw little-endian tensors), the format
//! Memorilla publishes its precomputed embeddings in.

use memmap2::Mmap;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub struct SafeTensors {
    map: Mmap,
    data_start: usize,
    tensors: HashMap<String, (String, Vec<usize>, usize, usize)>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<SafeTensors, String> {
        let f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: read-only map of a file that is not expected to change.
        let map = unsafe { Mmap::map(&f) }.map_err(|e| format!("{}: {e}", path.display()))?;
        if map.len() < 8 {
            return Err(format!("{}: not a safetensors file", path.display()));
        }
        let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
        if 8 + n > map.len() {
            return Err(format!("{}: the header runs past the end of the file", path.display()));
        }
        let head: Value = serde_json::from_slice(&map[8..8 + n]).map_err(|e| format!("{}: header: {e}", path.display()))?;
        let mut tensors = HashMap::new();
        for (name, t) in head.as_object().ok_or("safetensors header is not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = t.get("dtype").and_then(|d| d.as_str()).unwrap_or("").to_string();
            let shape = t.get("shape").and_then(|s| s.as_array()).map(|a| a.iter().filter_map(|d| d.as_u64()).map(|d| d as usize).collect()).unwrap_or_default();
            let off = t.get("data_offsets").and_then(|o| o.as_array()).ok_or_else(|| format!("{name}: no data_offsets"))?;
            let (a, b) = (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize);
            if 8 + n + b > map.len() || a > b {
                return Err(format!("{}: tensor {name} runs past the end of the file", path.display()));
            }
            tensors.insert(name.clone(), (dtype, shape, a, b));
        }
        Ok(SafeTensors { map, data_start: 8 + n, tensors })
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.1.as_slice())
    }

    fn raw(&self, name: &str) -> Result<(&str, &[u8]), String> {
        let (dt, _, a, b) = self.tensors.get(name).ok_or_else(|| format!("no tensor '{name}'"))?;
        Ok((dt.as_str(), &self.map[self.data_start + a..self.data_start + b]))
    }

    /// Integer tensor as i64 (I64 or I32).
    pub fn ints(&self, name: &str) -> Result<Vec<i64>, String> {
        let (dt, b) = self.raw(name)?;
        Ok(match dt {
            "I64" => b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect(),
            "I32" => b.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64).collect(),
            other => return Err(format!("tensor '{name}' is {other}, not an integer type")),
        })
    }

    /// Rows `from..to` of a 2-D float tensor (F16, BF16 or F32), as f32.
    pub fn rows(&self, name: &str, from: usize, to: usize) -> Result<Vec<f32>, String> {
        let shape = self.shape(name).ok_or_else(|| format!("no tensor '{name}'"))?.to_vec();
        if shape.len() != 2 || to > shape[0] || from > to {
            return Err(format!("tensor '{name}' {shape:?}: rows {from}..{to} are out of range"));
        }
        let (dt, b) = self.raw(name)?;
        let w = match dt {
            "F16" | "BF16" => 2,
            "F32" => 4,
            other => return Err(format!("tensor '{name}' is {other}, not a float type")),
        };
        let d = shape[1];
        let b = &b[from * d * w..to * d * w];
        Ok(match dt {
            "F16" => b.chunks_exact(2).map(|c| plenum_engine::quant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "BF16" => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            _ => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        })
    }
}
