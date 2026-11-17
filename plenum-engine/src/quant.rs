//! ggml block formats, computed as the file holds them (port of c/ggml_quants.h's
//! exact per-block decoders) and the matmul that runs on top of them.
//!
//! A matrix stays in its quantized bytes (mmap'd, or an expert read from disk). A
//! product decodes one row into a per-thread scratch buffer and takes its dot with
//! every input vector, so a row is decoded once per call however long the prompt is.

use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GType {
    /// The ggml type id as GGUF stores it.
    pub fn from_id(id: u32) -> Option<GType> {
        Some(match id {
            0 => GType::F32,
            1 => GType::F16,
            2 => GType::Q4_0,
            3 => GType::Q4_1,
            6 => GType::Q5_0,
            7 => GType::Q5_1,
            8 => GType::Q8_0,
            10 => GType::Q2K,
            11 => GType::Q3K,
            12 => GType::Q4K,
            13 => GType::Q5K,
            14 => GType::Q6K,
            30 => GType::BF16,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        match self {
            GType::F32 => 0,
            GType::F16 => 1,
            GType::Q4_0 => 2,
            GType::Q4_1 => 3,
            GType::Q5_0 => 6,
            GType::Q5_1 => 7,
            GType::Q8_0 => 8,
            GType::Q2K => 10,
            GType::Q3K => 11,
            GType::Q4K => 12,
            GType::Q5K => 13,
            GType::Q6K => 14,
            GType::BF16 => 30,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GType::F32 => "F32",
            GType::F16 => "F16",
            GType::BF16 => "BF16",
            GType::Q4_0 => "Q4_0",
            GType::Q4_1 => "Q4_1",
            GType::Q5_0 => "Q5_0",
            GType::Q5_1 => "Q5_1",
            GType::Q8_0 => "Q8_0",
            GType::Q2K => "Q2_K",
            GType::Q3K => "Q3_K",
            GType::Q4K => "Q4_K",
            GType::Q5K => "Q5_K",
            GType::Q6K => "Q6_K",
        }
    }

    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            GType::F32 => (1, 4),
            GType::F16 | GType::BF16 => (1, 2),
            GType::Q4_0 => (32, 18),
            GType::Q4_1 => (32, 20),
            GType::Q5_0 => (32, 22),
            GType::Q5_1 => (32, 24),
            GType::Q8_0 => (32, 34),
            GType::Q2K => (256, 84),
            GType::Q3K => (256, 110),
            GType::Q4K => (256, 144),
            GType::Q5K => (256, 176),
            GType::Q6K => (256, 210),
        }
    }

    /// Bytes that `n` elements take. `n` must be a whole number of blocks.
    pub fn bytes_for(self, n: usize) -> usize {
        let (e, b) = self.block();
        n / e * b
    }
}

/// Name of a ggml type id this engine does not compute (for error messages).
pub fn type_id_name(id: u32) -> String {
    match GType::from_id(id) {
        Some(t) => t.name().to_string(),
        None => match id {
            15 => "Q8_K".into(),
            16 => "IQ2_XXS".into(),
            17 => "IQ2_XS".into(),
            18 => "IQ3_XXS".into(),
            19 => "IQ1_S".into(),
            20 => "IQ4_NL".into(),
            21 => "IQ3_S".into(),
            22 => "IQ2_S".into(),
            23 => "IQ4_XS".into(),
            29 => "IQ1_M".into(),
            34 => "TQ1_0".into(),
            35 => "TQ2_0".into(),
            39 => "MXFP4".into(),
            _ => format!("type#{id}"),
        },
    }
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 113u32;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | ((e + 112) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[inline]
fn rd16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
#[inline]
fn rd32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn h(b: &[u8], o: usize) -> f32 {
    f16_to_f32(rd16(b, o))
}

#[inline]
fn scale_min_k4(s: &[u8], g: usize) -> (f32, f32) {
    if g < 4 {
        ((s[g] & 0x3f) as f32, (s[g + 4] & 0x3f) as f32)
    } else {
        (
            ((s[g + 4] & 0x0f) | ((s[g - 4] >> 2) & 0x30)) as f32,
            ((s[g + 4] >> 4) | ((s[g] >> 2) & 0x30)) as f32,
        )
    }
}

fn blk_q4_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qs = &b[2..18];
    for j in 0..16 {
        y[j] = d * ((qs[j] & 0x0f) as i32 - 8) as f32;
        y[j + 16] = d * ((qs[j] >> 4) as i32 - 8) as f32;
    }
}
fn blk_q4_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (h(b, 0), h(b, 2));
    let qs = &b[4..20];
    for j in 0..16 {
        y[j] = d * (qs[j] & 0x0f) as f32 + m;
        y[j + 16] = d * (qs[j] >> 4) as f32 + m;
    }
}
fn blk_q5_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qh = rd32(b, 2);
    let qs = &b[6..22];
    for j in 0..16 {
        let lo = (qs[j] & 0x0f) as i32 | (((qh >> j) & 1) << 4) as i32;
        let hi = (qs[j] >> 4) as i32 | (((qh >> (j + 16)) & 1) << 4) as i32;
        y[j] = d * (lo - 16) as f32;
        y[j + 16] = d * (hi - 16) as f32;
    }
}
fn blk_q5_1(b: &[u8], y: &mut [f32]) {
    let (d, m) = (h(b, 0), h(b, 2));
    let qh = rd32(b, 4);
    let qs = &b[8..24];
    for j in 0..16 {
        let lo = (qs[j] & 0x0f) as i32 | (((qh >> j) & 1) << 4) as i32;
        let hi = (qs[j] >> 4) as i32 | (((qh >> (j + 16)) & 1) << 4) as i32;
        y[j] = d * lo as f32 + m;
        y[j + 16] = d * hi as f32 + m;
    }
}
fn blk_q8_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qs = &b[2..34];
    for j in 0..32 {
        y[j] = (qs[j] as i8) as f32 * d;
    }
}
fn blk_q2_k(b: &[u8], y: &mut [f32]) {
    let sc = &b[0..16];
    let qs = &b[16..80];
    let (d, dmin) = (h(b, 80), h(b, 82));
    for i in 0..256 {
        let (c, s, j, g) = (i >> 7, (i >> 5) & 3, i & 31, i >> 4);
        let q = (qs[c * 32 + j] >> (2 * s)) & 3;
        let dl = d * (sc[g] & 0x0f) as f32;
        let ml = dmin * (sc[g] >> 4) as f32;
        y[i] = dl * q as f32 - ml;
    }
}
fn blk_q3_k(b: &[u8], y: &mut [f32]) {
    let hm = &b[0..32];
    let qs = &b[32..96];
    let s = &b[96..108];
    let d = h(b, 108);
    let mut dl = [0f32; 16];
    for k in 0..16 {
        let l: i32 = (if k < 8 { s[k] & 0x0f } else { s[k - 8] >> 4 }).into();
        let hh = ((s[8 + (k & 3)] >> (2 * (k >> 2))) & 3) as i32;
        dl[k] = d * ((l | (hh << 4)) - 32) as f32;
    }
    for i in 0..256 {
        let (c, sh, j) = (i >> 7, (i >> 5) & 3, i & 31);
        let ql = ((qs[c * 32 + j] >> (2 * sh)) & 3) as i32;
        let hb = ((hm[i & 31] >> (i >> 5)) & 1) as i32;
        let q = ql - ((hb ^ 1) << 2);
        y[i] = dl[i >> 4] * q as f32;
    }
}
fn blk_q4_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let sc = &b[4..16];
    let qs = &b[16..144];
    for g in 0..8 {
        let (s, m) = scale_min_k4(sc, g);
        let (d1, m1) = (d * s, dmin * m);
        let q = &qs[(g >> 1) * 32..(g >> 1) * 32 + 32];
        let sh = (g & 1) * 4;
        let out = &mut y[g * 32..g * 32 + 32];
        for j in 0..32 {
            out[j] = d1 * ((q[j] >> sh) & 0x0f) as f32 - m1;
        }
    }
}
fn blk_q5_k(b: &[u8], y: &mut [f32]) {
    let (d, dmin) = (h(b, 0), h(b, 2));
    let sc = &b[4..16];
    let qh = &b[16..48];
    let qs = &b[48..176];
    for g in 0..8 {
        let (s, m) = scale_min_k4(sc, g);
        let (d1, m1) = (d * s, dmin * m);
        let q = &qs[(g >> 1) * 32..(g >> 1) * 32 + 32];
        let sh = (g & 1) * 4;
        let out = &mut y[g * 32..g * 32 + 32];
        for j in 0..32 {
            let v = ((q[j] >> sh) & 0x0f) | (((qh[j] >> g) & 1) << 4);
            out[j] = d1 * v as f32 - m1;
        }
    }
}
fn blk_q6_k(b: &[u8], y: &mut [f32]) {
    let ql = &b[0..128];
    let qh = &b[128..192];
    let sc = &b[192..208];
    let d = h(b, 208);
    // Two chunks of 128; in each, four runs of 32 that share one qh byte column (2 bits
    // each) and take the low or high nibbles of ql; the scale changes every 16.
    for c in 0..2 {
        let hi = &qh[c * 32..c * 32 + 32];
        for s in 0..4 {
            let lo = &ql[c * 64 + (s & 1) * 32..c * 64 + (s & 1) * 32 + 32];
            let (lsh, hsh) = (4 * (s >> 1), 2 * s);
            let out = &mut y[c * 128 + s * 32..c * 128 + s * 32 + 32];
            for half in 0..2 {
                let scale = d * (sc[c * 8 + s * 2 + half] as i8) as f32;
                let (l, hh, o) = (&lo[half * 16..half * 16 + 16], &hi[half * 16..half * 16 + 16], &mut out[half * 16..half * 16 + 16]);
                for j in 0..16 {
                    let q = (((l[j] >> lsh) & 0x0f) | (((hh[j] >> hsh) & 3) << 4)) as i32 - 32;
                    o[j] = scale * q as f32;
                }
            }
        }
    }
}

/// Decode `out.len()` elements from `src` (a whole number of blocks of `ty`).
pub fn dequant_row(ty: GType, src: &[u8], out: &mut [f32]) {
    let (be, bb) = ty.block();
    match ty {
        GType::F32 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GType::F16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        GType::BF16 => {
            for (o, c) in out.iter_mut().zip(src.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        _ => {
            let f: fn(&[u8], &mut [f32]) = match ty {
                GType::Q4_0 => blk_q4_0,
                GType::Q4_1 => blk_q4_1,
                GType::Q5_0 => blk_q5_0,
                GType::Q5_1 => blk_q5_1,
                GType::Q8_0 => blk_q8_0,
                GType::Q2K => blk_q2_k,
                GType::Q3K => blk_q3_k,
                GType::Q4K => blk_q4_k,
                GType::Q5K => blk_q5_k,
                GType::Q6K => blk_q6_k,
                _ => unreachable!(),
            };
            for (o, c) in out.chunks_exact_mut(be).zip(src.chunks_exact(bb)) {
                f(c, o);
            }
        }
    }
}

/// Sixteen running sums: the shape the compiler turns into two AVX2 (or one AVX-512) lanes.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = 0f32;
    for i in 0..8 {
        s += acc[i] + acc[i + 8];
    }
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// One Q8_0 row times x without decoding it first: the int8 codes are widened and
/// multiplied in place, the block scale applied once per block (the single-token path).
#[inline]
pub fn dot_q8_0(w: &[u8], x: &[f32]) -> f32 {
    let mut tot = [0f32; 8];
    for (blk, xs) in w.chunks_exact(34).zip(x.chunks_exact(32)) {
        let d = h(blk, 0);
        let qs = &blk[2..34];
        let mut acc = [0f32; 8];
        for c in 0..4 {
            let (q, xv) = (&qs[c * 8..c * 8 + 8], &xs[c * 8..c * 8 + 8]);
            for i in 0..8 {
                acc[i] += (q[i] as i8) as f32 * xv[i];
            }
        }
        for i in 0..8 {
            tot[i] += d * acc[i];
        }
    }
    tot.iter().sum()
}

/// The sum of every 16 values of x: what the constant part of a K-quant block
/// (its minimum, or Q6_K's -32 offset) multiplies. Computed once per product.
pub fn sums16(x: &[f32]) -> Vec<f32> {
    x.chunks_exact(16).map(|c| c.iter().sum()).collect()
}

#[inline]
fn sum8(a: [f32; 8]) -> f32 {
    (a[0] + a[4]) + (a[1] + a[5]) + (a[2] + a[6]) + (a[3] + a[7])
}

/// One Q4_K row times x without decoding it. A value is d*sc*nibble - dmin*mn, so a
/// sub-block of 32 gives d*sc*(nibbles . x) - dmin*mn*sum(x): the nibbles are widened
/// and multiplied in place, the minimum meets the precomputed sums (`xs` = sums16(x)).
#[inline]
pub fn dot_q4_k(w: &[u8], x: &[f32], xs: &[f32]) -> f32 {
    let mut tot = 0f32;
    for (bi, (blk, xb)) in w.chunks_exact(144).zip(x.chunks_exact(256)).enumerate() {
        let (d, dmin) = (h(blk, 0), h(blk, 2));
        let sc = &blk[4..16];
        let qs = &blk[16..144];
        let (mut acc, mut macc) = (0f32, 0f32);
        for pair in 0..4 {
            let q = &qs[pair * 32..pair * 32 + 32];
            let xl = &xb[pair * 64..pair * 64 + 32];
            let xh = &xb[pair * 64 + 32..pair * 64 + 64];
            let (mut a, mut b) = ([0f32; 8], [0f32; 8]);
            for c in 0..4 {
                let (qq, l, hh) = (&q[c * 8..c * 8 + 8], &xl[c * 8..c * 8 + 8], &xh[c * 8..c * 8 + 8]);
                for i in 0..8 {
                    a[i] += (qq[i] & 0x0f) as f32 * l[i];
                    b[i] += (qq[i] >> 4) as f32 * hh[i];
                }
            }
            let (s0, m0) = scale_min_k4(sc, 2 * pair);
            let (s1, m1) = scale_min_k4(sc, 2 * pair + 1);
            acc += s0 * sum8(a) + s1 * sum8(b);
            let k = bi * 16 + pair * 4;
            macc += m0 * (xs[k] + xs[k + 1]) + m1 * (xs[k + 2] + xs[k + 3]);
        }
        tot += d * acc - dmin * macc;
    }
    tot
}

/// One Q6_K row times x without decoding it. A value is d*sc*(code - 32), so a run of
/// 16 gives d*sc*((codes . x) - 32*sum(x)) (`xs` = sums16(x)).
#[inline]
pub fn dot_q6_k(w: &[u8], x: &[f32], xs: &[f32]) -> f32 {
    let mut tot = 0f32;
    for (bi, (blk, xb)) in w.chunks_exact(210).zip(x.chunks_exact(256)).enumerate() {
        let ql = &blk[0..128];
        let qh = &blk[128..192];
        let sc = &blk[192..208];
        let d = h(blk, 208);
        let mut acc = 0f32;
        for c in 0..2 {
            let hi = &qh[c * 32..c * 32 + 32];
            for s in 0..4 {
                let lo = &ql[c * 64 + (s & 1) * 32..c * 64 + (s & 1) * 32 + 32];
                let (lsh, hsh) = (4 * (s >> 1), 2 * s);
                let xr = &xb[c * 128 + s * 32..c * 128 + s * 32 + 32];
                for half in 0..2 {
                    let (l, hh, xv) = (&lo[half * 16..half * 16 + 16], &hi[half * 16..half * 16 + 16], &xr[half * 16..half * 16 + 16]);
                    let mut a = [0f32; 8];
                    for cc in 0..2 {
                        for i in 0..8 {
                            let j = cc * 8 + i;
                            let q = ((l[j] >> lsh) & 0x0f) | (((hh[j] >> hsh) & 3) << 4);
                            a[i] += q as f32 * xv[j];
                        }
                    }
                    let k = c * 8 + s * 2 + half;
                    acc += (sc[k] as i8) as f32 * (sum8(a) - 32.0 * xs[bi * 16 + k]);
                }
            }
        }
        tot += d * acc;
    }
    tot
}

/// y += a * x
#[inline]
pub fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (yy, xx) in y.iter_mut().zip(x) {
        *yy += a * xx;
    }
}

/// One row as f32: the file's own bytes when they are f32 and aligned, else decoded into `buf`.
#[inline]
fn row_f32<'a>(ty: GType, bytes: &'a [u8], buf: &'a mut [f32]) -> &'a [f32] {
    if ty == GType::F32 && cfg!(target_endian = "little") {
        // SAFETY: any bit pattern is a valid f32; align_to only yields the aligned middle.
        let (p, m, s) = unsafe { bytes.align_to::<f32>() };
        if p.is_empty() && s.is_empty() {
            return m;
        }
    }
    dequant_row(ty, bytes, buf);
    buf
}

/// y[rows] = W x on the calling thread: for products small enough that the caller
/// runs many of them side by side (the experts of a MoE layer).
pub fn matmul1_serial(w: &[u8], ty: GType, rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    let rb = ty.bytes_for(cols);
    let x = &x[..cols];
    match ty {
        GType::Q8_0 => {
            for (r, yo) in y[..rows].iter_mut().enumerate() {
                *yo = dot_q8_0(&w[r * rb..(r + 1) * rb], x);
            }
        }
        GType::Q4K | GType::Q6K => {
            let xs = sums16(x);
            for (r, yo) in y[..rows].iter_mut().enumerate() {
                let row = &w[r * rb..(r + 1) * rb];
                *yo = if ty == GType::Q4K { dot_q4_k(row, x, &xs) } else { dot_q6_k(row, x, &xs) };
            }
        }
        _ => {
            let mut buf = vec![0f32; cols];
            for (r, yo) in y[..rows].iter_mut().enumerate() {
                *yo = dot(row_f32(ty, &w[r * rb..(r + 1) * rb], &mut buf), x);
            }
        }
    }
}

/// y[s, rows] = x[s, cols] @ W^T, with W [rows, cols] held as `ty` blocks in `w`.
pub fn matmul(w: &[u8], ty: GType, rows: usize, cols: usize, x: &[f32], s: usize, y: &mut [f32]) {
    let rb = ty.bytes_for(cols);
    debug_assert!(w.len() >= rb * rows, "matrix bytes short");
    debug_assert!(x.len() >= s * cols && y.len() >= s * rows);
    let grain = (32768 / (cols * s).max(1)).max(1);
    if s == 1 && matches!(ty, GType::Q8_0 | GType::Q4K | GType::Q6K) {
        // one input vector: the row's codes are multiplied where they are, never decoded
        let x = &x[..cols];
        let xs = if ty == GType::Q8_0 { Vec::new() } else { sums16(x) };
        y[..rows].par_iter_mut().with_min_len(grain).enumerate().for_each(|(r, yo)| {
            let row = &w[r * rb..(r + 1) * rb];
            *yo = match ty {
                GType::Q8_0 => dot_q8_0(row, x),
                GType::Q4K => dot_q4_k(row, x, &xs),
                _ => dot_q6_k(row, x, &xs),
            };
        });
        return;
    }
    if s == 1 {
        y[..rows]
            .par_iter_mut()
            .with_min_len(grain)
            .enumerate()
            .for_each_init(
                || vec![0f32; cols],
                |buf, (r, yo)| {
                    let row = row_f32(ty, &w[r * rb..(r + 1) * rb], buf);
                    *yo = dot(row, &x[..cols]);
                },
            );
        return;
    }
    // Blocked in both directions, so that the inner product runs on data in the cache:
    // the inputs are taken a chunk of tokens at a time (about 512 KB), and for each
    // chunk the rows are decoded a few at a time (about 128 KB) and meet every token of
    // the chunk. Row by row over the whole batch, a long prompt's inputs (megabytes)
    // would be pulled from memory once per row, and twelve threads doing that are
    // limited by memory bandwidth, not by arithmetic. The price is decoding each row
    // once per chunk of tokens instead of once, which is small next to the products.
    let tblk = (131072 / cols.max(1)).clamp(4, 64);
    let rblk = (32768 / cols.max(1)).clamp(1, 32);
    let mut tmp = vec![0f32; rows * tblk.min(s)];
    let mut t0 = 0;
    while t0 < s {
        let tn = tblk.min(s - t0);
        let xs = &x[t0 * cols..(t0 + tn) * cols];
        tmp[..rows * tn].par_chunks_mut(rblk * tn).enumerate().for_each_init(
            || vec![0f32; rblk * cols],
            |buf, (b, out)| {
                let (r0, n) = (b * rblk, out.len() / tn);
                for k in 0..n {
                    dequant_row(ty, &w[(r0 + k) * rb..(r0 + k + 1) * rb], &mut buf[k * cols..(k + 1) * cols]);
                }
                for ti in 0..tn {
                    let xv = &xs[ti * cols..(ti + 1) * cols];
                    for k in 0..n {
                        out[k * tn + ti] = dot(&buf[k * cols..(k + 1) * cols], xv);
                    }
                }
            },
        );
        let tmp = &tmp;
        y[t0 * rows..(t0 + tn) * rows].par_chunks_mut(rows).enumerate().for_each(|(ti, yr)| {
            for (r, o) in yr.iter_mut().enumerate() {
                *o = tmp[r * tn + ti];
            }
        });
        t0 += tn;
    }
}

/// Quantize a row to Q8_0 (what ggml's reference quantizer does). Used by the tests
/// and by `plenum-engine selftest` to build a quantized model without any other tool.
pub fn quantize_q8_0(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() / 32 * 34);
    for blk in src.chunks_exact(32) {
        let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let hd = f32_to_f16(d);
        let dq = f16_to_f32(hd);
        let id = if dq != 0.0 { 1.0 / dq } else { 0.0 };
        out.extend_from_slice(&hd.to_le_bytes());
        for v in blk {
            out.push(((v * id).round().clamp(-127.0, 127.0)) as i8 as u8);
        }
    }
    out
}

pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e);
        let r = (m + 0x1000) >> 13;
        return sign | r as u16;
    }
    let r = ((e as u32) << 10) + ((mant + 0x1000) >> 13);
    sign | r as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 3.1415927, 1e-3, 65504.0, 6e-5, 1e-7] {
            let back = f16_to_f32(f32_to_f16(v));
            assert!((back - v).abs() <= v.abs() * 1e-3 + 1e-7, "{v} -> {back}");
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
    }

    #[test]
    fn q8_0_close() {
        let src: Vec<f32> = (0..64).map(|i| ((i * 37 % 64) as f32 - 31.5) / 17.0).collect();
        let q = quantize_q8_0(&src);
        let mut out = vec![0f32; 64];
        dequant_row(GType::Q8_0, &q, &mut out);
        for (a, b) in src.iter().zip(&out) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    /// Q4_K with every sub-block scale 1 and min 0, d = 1: the values are the nibbles.
    #[test]
    fn q4_k_layout() {
        let mut b = vec![0u8; 144];
        b[0..2].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        b[2..4].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        for g in 0..4 {
            b[4 + g] = 1; // scales of sub-blocks 0..3
            b[4 + 8 + g] = 1; // low nibble: scales of sub-blocks 4..7
        }
        for j in 0..128 {
            b[16 + j] = ((j % 16) as u8) | ((((j + 3) % 16) as u8) << 4);
        }
        let mut y = vec![0f32; 256];
        dequant_row(GType::Q4K, &b, &mut y);
        for g in 0..8 {
            for j in 0..32 {
                let k = (g >> 1) * 32 + j;
                let want = if g & 1 == 0 { k % 16 } else { (k + 3) % 16 };
                assert_eq!(y[g * 32 + j], want as f32, "g{g} j{j}");
            }
        }
    }

    /// Q6_K with scale 1, d = 1: value = 6-bit code - 32.
    #[test]
    fn q6_k_layout() {
        let mut b = vec![0u8; 210];
        for i in 0..16 {
            b[192 + i] = (i as i8 - 7) as u8; // a different (signed) scale per 16
        }
        b[208..210].copy_from_slice(&f32_to_f16(1.0).to_le_bytes());
        // element i has code (i * 7) % 64
        for i in 0..256usize {
            let code = ((i * 7) % 64) as u8;
            let (c, r) = (i >> 7, i & 127);
            let (nib, k) = (r >> 6, r & 63);
            b[c * 64 + k] |= (code & 0x0f) << (4 * nib);
            let (s, j) = (r >> 5, r & 31);
            b[128 + c * 32 + j] |= (code >> 4) << (2 * s);
        }
        let mut y = vec![0f32; 256];
        dequant_row(GType::Q6K, &b, &mut y);
        for i in 0..256usize {
            assert_eq!(y[i], ((i >> 4) as f32 - 7.0) * (((i * 7) % 64) as f32 - 32.0), "element {i}");
        }
    }

    /// The fused kernels (codes multiplied in place) against decode-then-dot, on blocks
    /// of arbitrary bytes: every scale, minimum and code pattern is some valid block.
    #[test]
    fn fused_dots_equal_decoded() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let cols = 512;
        let x: Vec<f32> = (0..cols).map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0).collect();
        let xs = sums16(&x);
        for (ty, bb, dpos) in [(GType::Q4K, 144usize, vec![0usize, 2]), (GType::Q6K, 210, vec![208]), (GType::Q8_0, 34, vec![0])] {
            let n = ty.bytes_for(cols);
            let mut w: Vec<u8> = (0..n).map(|_| (next() >> 32) as u8).collect();
            for blk in w.chunks_exact_mut(bb) {
                for &o in &dpos {
                    let d = 0.01 + (next() >> 40) as f32 / (1u64 << 24) as f32 * 0.1;
                    blk[o..o + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                }
            }
            let mut dec = vec![0f32; cols];
            dequant_row(ty, &w, &mut dec);
            let want = dot(&dec, &x);
            let got = match ty {
                GType::Q4K => dot_q4_k(&w, &x, &xs),
                GType::Q6K => dot_q6_k(&w, &x, &xs),
                _ => dot_q8_0(&w, &x),
            };
            let scale = dec.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum::<f32>();
            assert!((got - want).abs() <= scale * 1e-5, "{}: fused {got} vs decoded {want}", ty.name());
            // and through matmul's single-vector path
            let mut y = [0f32];
            matmul(&w, ty, 1, cols, &x, 1, &mut y);
            assert_eq!(y[0], got, "{}", ty.name());
        }
    }

    #[test]
    fn matmul_batch_equals_single() {
        let (rows, cols, s) = (7, 64, 3);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 31 % 97) as f32 - 48.0) / 50.0).collect();
        let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let x: Vec<f32> = (0..s * cols).map(|i| ((i * 17 % 89) as f32 - 44.0) / 40.0).collect();
        let mut y = vec![0f32; s * rows];
        matmul(&wb, GType::F32, rows, cols, &x, s, &mut y);
        for si in 0..s {
            let mut y1 = vec![0f32; rows];
            matmul(&wb, GType::F32, rows, cols, &x[si * cols..(si + 1) * cols], 1, &mut y1);
            for r in 0..rows {
                let want: f32 = (0..cols).map(|c| w[r * cols + c] * x[si * cols + c]).sum();
                assert!((y[si * rows + r] - want).abs() < 1e-4);
                assert!((y1[r] - want).abs() < 1e-4);
            }
        }
    }
}
