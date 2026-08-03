//! Minimal forward-pass implementation of the trained nets (see
//! `training/train_ppo.py`). Two architectures share one weight-blob
//! loader and output shape:
//!
//! - MLP ("T5N1"/"T5N2"): Linear stem + ReLU, residual blocks
//!   (Linear-ReLU-Linear + skip, LayerNorm, ReLU), then policy / value /
//!   belief heads. ~1.4 MFLOPs per forward.
//! - Attention ("T5N3"): one token per card (features rebuilt here from
//!   the observation, mirroring AttnNet._card_features) plus a CLS token,
//!   pre-LN transformer encoder with a final LayerNorm, per-card-token
//!   policy/belief heads and a CLS value head. ~400 MFLOPs per forward —
//!   fine for raw-policy play, too slow to wrap in determinized search.
//!
//! Weights load from a flat little-endian blob written by
//! `training/export_net.py` (format documented there and in `from_bytes`).

use crate::cards::bullheads;
use crate::obs::OBS_LEN;

pub const POLICY_OUT: usize = crate::cards::NUM_CARDS;
pub const BELIEF_CLASSES: usize = 4; // 3 opponents + stock (4-player nets)
pub const BELIEF_OUT: usize = POLICY_OUT * BELIEF_CLASSES;
const MAGIC_V1: u32 = 0x5435_4E31; // "T5N1": f32 weights
const MAGIC_V2: u32 = 0x5435_4E32; // "T5N2": header gains a dtype field
const MAGIC_V3: u32 = 0x5435_4E33; // "T5N3": attention encoder
const DTYPE_F32: u32 = 0;
const DTYPE_F16: u32 = 1;
const LN_EPS: f32 = 1e-5;
const NUM_CARDS: usize = crate::cards::NUM_CARDS;
const GLOBAL_OFF: usize = 2 * NUM_CARDS; // obs tail fed to the CLS token
const CARD_FEATS: usize = 8;
const TOKENS: usize = NUM_CARDS + 1; // CLS + one per card

/// IEEE 754 half -> single precision (handles subnormals/inf/nan).
fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign << 31
        } else {
            let mut e = 127 - 15 + 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            (sign << 31) | ((e as u32) << 23) | ((f & 0x3FF) << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

struct Linear {
    w: Vec<f32>, // [out][inp] row-major
    b: Vec<f32>,
    out: usize,
    inp: usize,
}

impl Linear {
    fn apply(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.inp);
        debug_assert_eq!(y.len(), self.out);
        for (o, yo) in y.iter_mut().enumerate() {
            let row = &self.w[o * self.inp..(o + 1) * self.inp];
            *yo = self.b[o] + dot(row, x);
        }
    }
}

/// Dot product; explicitly 4-lane SIMD on wasm32 (the attention net is
/// ~400 MFLOPs of these per forward and LLVM does not autovectorize the
/// scalar reduction well there), plain code elsewhere (native builds
/// autovectorize fine).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::wasm32::*;
    let n = a.len().min(b.len());
    let chunks = n / 8;
    let mut acc0 = f32x4_splat(0.0);
    let mut acc1 = f32x4_splat(0.0);
    unsafe {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        for i in 0..chunks {
            let off = i * 8;
            let a0 = v128_load(pa.add(off) as *const v128);
            let b0 = v128_load(pb.add(off) as *const v128);
            let a1 = v128_load(pa.add(off + 4) as *const v128);
            let b1 = v128_load(pb.add(off + 4) as *const v128);
            acc0 = f32x4_add(acc0, f32x4_mul(a0, b0));
            acc1 = f32x4_add(acc1, f32x4_mul(a1, b1));
        }
    }
    let acc = f32x4_add(acc0, acc1);
    let mut sum = f32x4_extract_lane::<0>(acc)
        + f32x4_extract_lane::<1>(acc)
        + f32x4_extract_lane::<2>(acc)
        + f32x4_extract_lane::<3>(acc);
    for i in chunks * 8..n {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (ai, bi) in a.iter().zip(b) {
        acc += ai * bi;
    }
    acc
}

struct ResBlock {
    lin1: Linear,
    lin2: Linear,
    ln_g: Vec<f32>,
    ln_b: Vec<f32>,
}

pub struct NeuralOutput {
    pub policy_logits: [f32; POLICY_OUT],
    pub value: f32,
    /// Row-major [card][class]: 3 relative opponents then stock.
    pub belief_logits: Vec<f32>,
}

struct MlpNet {
    width: usize,
    stem: Linear,
    blocks: Vec<ResBlock>,
    policy: Linear,
    value: Linear,
    belief: Linear,
}

struct AttnLayer {
    in_proj: Linear,  // [3d][d]: packed q, k, v
    out_proj: Linear, // [d][d]
    lin1: Linear,     // [4d][d]
    lin2: Linear,     // [d][4d]
    n1_g: Vec<f32>,
    n1_b: Vec<f32>,
    n2_g: Vec<f32>,
    n2_b: Vec<f32>,
}

struct AttnNet {
    d: usize,
    heads: usize,
    card_emb: Vec<f32>, // [NUM_CARDS][d]
    feat: Linear,       // [d][CARD_FEATS]
    glob: Linear,       // [d][obs_len - GLOBAL_OFF]
    cls: Vec<f32>,      // [d]
    layers: Vec<AttnLayer>,
    fin_g: Vec<f32>,
    fin_b: Vec<f32>,
    policy: Linear, // [1][d], applied per card token
    value: Linear,  // [1][d], applied to CLS
    belief: Linear, // [BELIEF_CLASSES][d], applied per card token
}

enum NetKind {
    Mlp(MlpNet),
    Attn(AttnNet),
}

pub struct NeuralNet {
    obs_len: usize,
    kind: NetKind,
}

impl std::fmt::Debug for NeuralNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            NetKind::Mlp(m) => write!(
                f,
                "NeuralNet(mlp, width={}, blocks={})",
                m.width,
                m.blocks.len()
            ),
            NetKind::Attn(a) => write!(
                f,
                "NeuralNet(attn, d={}, layers={}, heads={})",
                a.d,
                a.layers.len(),
                a.heads
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NeuralError {
    BadMagic,
    BadShape,
    Truncated,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    dtype: u32,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, NeuralError> {
        let end = self.pos + 4;
        if end > self.data.len() {
            return Err(NeuralError::Truncated);
        }
        let v = u32::from_le_bytes(self.data[self.pos..end].try_into().unwrap());
        self.pos = end;
        Ok(v)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, NeuralError> {
        let elem = if self.dtype == DTYPE_F16 { 2 } else { 4 };
        let end = self.pos + elem * n;
        if end > self.data.len() {
            return Err(NeuralError::Truncated);
        }
        let out = if self.dtype == DTYPE_F16 {
            self.data[self.pos..end]
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                .collect()
        } else {
            self.data[self.pos..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        self.pos = end;
        Ok(out)
    }

    fn linear(&mut self, out: usize, inp: usize) -> Result<Linear, NeuralError> {
        Ok(Linear {
            w: self.f32s(out * inp)?,
            b: self.f32s(out)?,
            out,
            inp,
        })
    }
}

impl NeuralNet {
    /// MLP format: u32 magic ("T5N1" = f32; "T5N2" adds a u32 dtype field,
    /// 0 = f32, 1 = f16), u32 width, u32 blocks, u32 obs_len, [u32 dtype,]
    /// then LE tensors in order: stem(w,b); per block lin1(w,b), lin2(w,b),
    /// ln gamma, ln beta; policy(w,b); value(w,b); belief(w,b). Linear
    /// weights are [out][in] row-major (PyTorch convention).
    ///
    /// Attention format ("T5N3"): u32 magic, u32 d_model, u32 layers,
    /// u32 obs_len, u32 dtype, u32 heads, then card_emb; feat(w,b);
    /// glob(w,b); cls; per layer in_proj(w,b), out_proj(w,b), linear1(w,b),
    /// linear2(w,b), norm1(g,b), norm2(g,b); final norm(g,b); policy(w,b);
    /// value(w,b); belief(w,b).
    pub fn from_bytes(data: &[u8]) -> Result<NeuralNet, NeuralError> {
        let mut r = Reader {
            data,
            pos: 0,
            dtype: DTYPE_F32,
        };
        let magic = r.u32()?;
        if magic == MAGIC_V3 {
            return Self::attn_from_reader(r, data.len());
        }
        if magic != MAGIC_V1 && magic != MAGIC_V2 {
            return Err(NeuralError::BadMagic);
        }
        let width = r.u32()? as usize;
        let num_blocks = r.u32()? as usize;
        let obs_len = r.u32()? as usize;
        if magic == MAGIC_V2 {
            let dtype = r.u32()?;
            if dtype != DTYPE_F32 && dtype != DTYPE_F16 {
                return Err(NeuralError::BadShape);
            }
            r.dtype = dtype;
        }
        // Nets exported against older (shorter) observation layouts stay
        // loadable: the layout is append-only, so they read the prefix.
        if obs_len == 0 || obs_len > OBS_LEN || width == 0 || width > 8192 || num_blocks > 64 {
            return Err(NeuralError::BadShape);
        }
        let stem = r.linear(width, obs_len)?;
        let mut blocks = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            blocks.push(ResBlock {
                lin1: r.linear(width, width)?,
                lin2: r.linear(width, width)?,
                ln_g: r.f32s(width)?,
                ln_b: r.f32s(width)?,
            });
        }
        let policy = r.linear(POLICY_OUT, width)?;
        let value = r.linear(1, width)?;
        let belief = r.linear(BELIEF_OUT, width)?;
        if r.pos != data.len() {
            return Err(NeuralError::BadShape);
        }
        Ok(NeuralNet {
            obs_len,
            kind: NetKind::Mlp(MlpNet {
                width,
                stem,
                blocks,
                policy,
                value,
                belief,
            }),
        })
    }

    fn attn_from_reader(mut r: Reader, data_len: usize) -> Result<NeuralNet, NeuralError> {
        let d = r.u32()? as usize;
        let num_layers = r.u32()? as usize;
        let obs_len = r.u32()? as usize;
        let dtype = r.u32()?;
        if dtype != DTYPE_F32 && dtype != DTYPE_F16 {
            return Err(NeuralError::BadShape);
        }
        r.dtype = dtype;
        let heads = r.u32()? as usize;
        if d == 0
            || d > 2048
            || num_layers == 0
            || num_layers > 32
            || heads == 0
            || !d.is_multiple_of(heads)
            || obs_len <= GLOBAL_OFF
            || obs_len > OBS_LEN
        {
            return Err(NeuralError::BadShape);
        }
        let card_emb = r.f32s(NUM_CARDS * d)?;
        let feat = r.linear(d, CARD_FEATS)?;
        let glob = r.linear(d, obs_len - GLOBAL_OFF)?;
        let cls = r.f32s(d)?;
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(AttnLayer {
                in_proj: r.linear(3 * d, d)?,
                out_proj: r.linear(d, d)?,
                lin1: r.linear(4 * d, d)?,
                lin2: r.linear(d, 4 * d)?,
                n1_g: r.f32s(d)?,
                n1_b: r.f32s(d)?,
                n2_g: r.f32s(d)?,
                n2_b: r.f32s(d)?,
            });
        }
        let fin_g = r.f32s(d)?;
        let fin_b = r.f32s(d)?;
        let policy = r.linear(1, d)?;
        let value = r.linear(1, d)?;
        let belief = r.linear(BELIEF_CLASSES, d)?;
        if r.pos != data_len {
            return Err(NeuralError::BadShape);
        }
        Ok(NeuralNet {
            obs_len,
            kind: NetKind::Attn(AttnNet {
                d,
                heads,
                card_emb,
                feat,
                glob,
                cls,
                layers,
                fin_g,
                fin_b,
                policy,
                value,
                belief,
            }),
        })
    }

    pub fn forward(&self, obs: &[f32]) -> NeuralOutput {
        debug_assert!(obs.len() >= self.obs_len);
        match &self.kind {
            NetKind::Mlp(m) => m.forward(&obs[..self.obs_len]),
            NetKind::Attn(a) => a.forward(&obs[..self.obs_len]),
        }
    }

    /// Belief probabilities for one card: 3 relative opponents then stock.
    /// `num_players` beyond 4 is unsupported (net is trained for 4).
    pub fn belief_probs(out: &NeuralOutput, card_idx: usize) -> [f32; BELIEF_CLASSES] {
        let logits = &out.belief_logits[card_idx * BELIEF_CLASSES..(card_idx + 1) * BELIEF_CLASSES];
        softmax4(logits)
    }
}

impl MlpNet {
    fn forward(&self, obs: &[f32]) -> NeuralOutput {
        let w = self.width;
        let mut h = vec![0.0f32; w];
        self.stem.apply(obs, &mut h);
        relu(&mut h);

        let mut tmp1 = vec![0.0f32; w];
        let mut tmp2 = vec![0.0f32; w];
        for block in &self.blocks {
            block.lin1.apply(&h, &mut tmp1);
            relu(&mut tmp1);
            block.lin2.apply(&tmp1, &mut tmp2);
            for i in 0..w {
                tmp2[i] += h[i]; // skip connection
            }
            layer_norm(&tmp2, &block.ln_g, &block.ln_b, &mut h);
            relu(&mut h);
        }

        let mut policy_logits = [0.0f32; POLICY_OUT];
        self.policy.apply(&h, &mut policy_logits);
        let mut value = [0.0f32; 1];
        self.value.apply(&h, &mut value);
        let mut belief_logits = vec![0.0f32; BELIEF_OUT];
        self.belief.apply(&h, &mut belief_logits);
        NeuralOutput {
            policy_logits,
            value: value[0],
            belief_logits,
        }
    }
}

impl AttnNet {
    /// Mirror of `AttnNet._card_features` in training/train_ppo.py: any
    /// drift between the two is caught by tests/test_neural_parity.py.
    fn card_features(obs: &[f32], f: &mut [f32]) {
        for c in 0..NUM_CARDS {
            let row = &mut f[c * CARD_FEATS..(c + 1) * CARD_FEATS];
            row[0] = obs[c];
            row[1] = obs[NUM_CARDS + c];
            row[2] = bullheads((c + 1) as u8) as f32 / 7.0;
            row[3] = (c + 1) as f32 / NUM_CARDS as f32;
            row[4] = 0.0;
            row[5] = 0.0;
            row[6] = 0.0;
            row[7] = 0.0;
        }
        let mut ids = [0usize; 20];
        let mut row_len = [0usize; 4];
        for (i, id) in ids.iter_mut().enumerate() {
            *id = (obs[GLOBAL_OFF + i] * NUM_CARDS as f32).round() as usize;
            if *id > 0 {
                row_len[i / 5] += 1;
            }
        }
        for (i, &id) in ids.iter().enumerate() {
            if id == 0 {
                continue;
            }
            let row = &mut f[(id - 1) * CARD_FEATS..id * CARD_FEATS];
            row[4] = 1.0;
            row[5] = (i / 5) as f32 / 3.0;
            row[6] = (i % 5) as f32 / 4.0;
            row[7] = if (i % 5) + 1 == row_len[i / 5] {
                1.0
            } else {
                0.0
            };
        }
    }

    fn forward(&self, obs: &[f32]) -> NeuralOutput {
        let d = self.d;
        let mut x = vec![0.0f32; TOKENS * d];

        // CLS token: learned vector + projected observation tail.
        self.glob.apply(&obs[GLOBAL_OFF..], &mut x[..d]);
        for (xe, ce) in x[..d].iter_mut().zip(&self.cls) {
            *xe += ce;
        }
        // Card tokens: projected features + learned card embedding.
        let mut feats = vec![0.0f32; NUM_CARDS * CARD_FEATS];
        Self::card_features(obs, &mut feats);
        for c in 0..NUM_CARDS {
            let tok = &mut x[(1 + c) * d..(2 + c) * d];
            self.feat
                .apply(&feats[c * CARD_FEATS..(c + 1) * CARD_FEATS], tok);
            for (te, ee) in tok.iter_mut().zip(&self.card_emb[c * d..(c + 1) * d]) {
                *te += ee;
            }
        }

        let dh = d / self.heads;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut q = vec![0.0f32; TOKENS * d];
        let mut k = vec![0.0f32; TOKENS * d];
        let mut v = vec![0.0f32; TOKENS * d];
        let mut attn = vec![0.0f32; TOKENS * d];
        let mut ln = vec![0.0f32; d];
        let mut qkv = vec![0.0f32; 3 * d];
        let mut ff = vec![0.0f32; 4 * d];
        let mut tmp = vec![0.0f32; d];
        let mut scores = [0.0f32; TOKENS];

        for layer in &self.layers {
            // Pre-LN self-attention block: x += out_proj(attn(norm1(x))).
            for t in 0..TOKENS {
                layer_norm(&x[t * d..(t + 1) * d], &layer.n1_g, &layer.n1_b, &mut ln);
                layer.in_proj.apply(&ln, &mut qkv);
                q[t * d..(t + 1) * d].copy_from_slice(&qkv[..d]);
                k[t * d..(t + 1) * d].copy_from_slice(&qkv[d..2 * d]);
                v[t * d..(t + 1) * d].copy_from_slice(&qkv[2 * d..]);
            }
            attn.fill(0.0);
            for h in 0..self.heads {
                let off = h * dh;
                for i in 0..TOKENS {
                    let qi = &q[i * d + off..i * d + off + dh];
                    for (j, s) in scores.iter_mut().enumerate() {
                        let kj = &k[j * d + off..j * d + off + dh];
                        *s = dot(qi, kj) * scale;
                    }
                    softmax_inplace(&mut scores);
                    let out = i * d + off;
                    for (j, &p) in scores.iter().enumerate() {
                        let vj = &v[j * d + off..j * d + off + dh];
                        for e in 0..dh {
                            attn[out + e] += p * vj[e];
                        }
                    }
                }
            }
            for t in 0..TOKENS {
                layer.out_proj.apply(&attn[t * d..(t + 1) * d], &mut tmp);
                for e in 0..d {
                    x[t * d + e] += tmp[e];
                }
            }
            // Pre-LN feed-forward block: x += lin2(relu(lin1(norm2(x)))).
            for t in 0..TOKENS {
                layer_norm(&x[t * d..(t + 1) * d], &layer.n2_g, &layer.n2_b, &mut ln);
                layer.lin1.apply(&ln, &mut ff);
                relu(&mut ff);
                layer.lin2.apply(&ff, &mut tmp);
                for e in 0..d {
                    x[t * d + e] += tmp[e];
                }
            }
        }

        let mut policy_logits = [0.0f32; POLICY_OUT];
        let mut belief_logits = vec![0.0f32; BELIEF_OUT];
        let mut head_out = [0.0f32; BELIEF_CLASSES];
        let mut value = 0.0f32;
        for t in 0..TOKENS {
            layer_norm(&x[t * d..(t + 1) * d], &self.fin_g, &self.fin_b, &mut ln);
            if t == 0 {
                let mut val = [0.0f32; 1];
                self.value.apply(&ln, &mut val);
                value = val[0];
            } else {
                let c = t - 1;
                let mut logit = [0.0f32; 1];
                self.policy.apply(&ln, &mut logit);
                policy_logits[c] = logit[0];
                self.belief.apply(&ln, &mut head_out);
                belief_logits[c * BELIEF_CLASSES..(c + 1) * BELIEF_CLASSES]
                    .copy_from_slice(&head_out);
            }
        }
        NeuralOutput {
            policy_logits,
            value,
            belief_logits,
        }
    }
}

fn relu(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], out: &mut [f32]) {
    let n = x.len() as f32;
    let mean: f32 = x.iter().sum::<f32>() / n;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
    }
}

fn softmax_inplace(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

fn softmax4(logits: &[f32]) -> [f32; BELIEF_CLASSES] {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0.0f32; BELIEF_CLASSES];
    let mut sum = 0.0;
    for (i, &l) in logits.iter().enumerate() {
        let e = (l - max).exp();
        out[i] = e;
        sum += e;
    }
    for v in &mut out {
        *v /= sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny random-ish net blob for round-trip testing.
    fn tiny_net_bytes(width: usize, blocks: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(MAGIC_V1.to_le_bytes());
        out.extend((width as u32).to_le_bytes());
        out.extend((blocks as u32).to_le_bytes());
        out.extend((OBS_LEN as u32).to_le_bytes());
        let mut state = 0x1234_5678u64;
        let mut push_f32s = |out: &mut Vec<u8>, n: usize| {
            for _ in 0..n {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let v = ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
                out.extend((v * 0.1).to_le_bytes());
            }
        };
        push_f32s(&mut out, width * OBS_LEN + width);
        for _ in 0..blocks {
            push_f32s(&mut out, width * width + width);
            push_f32s(&mut out, width * width + width);
            push_f32s(&mut out, width * 2);
        }
        push_f32s(&mut out, POLICY_OUT * width + POLICY_OUT);
        push_f32s(&mut out, width + 1);
        push_f32s(&mut out, BELIEF_OUT * width + BELIEF_OUT);
        out
    }

    #[test]
    fn roundtrip_and_forward() {
        let bytes = tiny_net_bytes(32, 2);
        let net = NeuralNet::from_bytes(&bytes).unwrap();
        let obs = vec![0.5f32; OBS_LEN];
        let out = net.forward(&obs);
        assert!(out.policy_logits.iter().all(|v| v.is_finite()));
        assert!(out.value.is_finite());
        assert!(out.belief_logits.iter().all(|v| v.is_finite()));
        let probs = NeuralNet::belief_probs(&out, 0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rejects_malformed() {
        assert_eq!(
            NeuralNet::from_bytes(&[0u8; 2]).unwrap_err(),
            NeuralError::Truncated
        );
        assert_eq!(
            NeuralNet::from_bytes(&[0u8; 16]).unwrap_err(),
            NeuralError::BadMagic
        );
        let mut bytes = tiny_net_bytes(16, 1);
        bytes[0] ^= 0xFF;
        assert_eq!(
            NeuralNet::from_bytes(&bytes).unwrap_err(),
            NeuralError::BadMagic
        );
        let mut bytes = tiny_net_bytes(16, 1);
        bytes.pop();
        assert!(NeuralNet::from_bytes(&bytes).is_err());
    }
}
