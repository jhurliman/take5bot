//! Minimal forward-pass implementation of the trained PolicyNet (see
//! `training/train_ppo.py`): Linear stem + ReLU, residual blocks
//! (Linear-ReLU-Linear + skip, LayerNorm, ReLU), then policy / value /
//! belief heads. Pure Rust, no BLAS — a single forward is ~1.4 MFLOPs,
//! microseconds on native and fast enough in WASM.
//!
//! Weights load from a flat little-endian f32 blob written by
//! `training/export_net.py` (format documented there and in `from_bytes`).

use crate::obs::OBS_LEN;

pub const POLICY_OUT: usize = crate::cards::NUM_CARDS;
pub const BELIEF_CLASSES: usize = 4; // 3 opponents + stock (4-player nets)
pub const BELIEF_OUT: usize = POLICY_OUT * BELIEF_CLASSES;
const MAGIC: u32 = 0x5435_4E31; // "T5N1"
const LN_EPS: f32 = 1e-5;

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
            let mut acc = self.b[o];
            for (wi, xi) in row.iter().zip(x) {
                acc += wi * xi;
            }
            *yo = acc;
        }
    }
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

pub struct NeuralNet {
    width: usize,
    stem: Linear,
    blocks: Vec<ResBlock>,
    policy: Linear,
    value: Linear,
    belief: Linear,
}

impl std::fmt::Debug for NeuralNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NeuralNet(width={}, blocks={})",
            self.width,
            self.blocks.len()
        )
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
        let end = self.pos + 4 * n;
        if end > self.data.len() {
            return Err(NeuralError::Truncated);
        }
        let out = self.data[self.pos..end]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
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
    /// Format: u32 magic "T5N1", u32 width, u32 blocks, u32 obs_len, then
    /// f32 LE tensors in order: stem(w,b); per block lin1(w,b), lin2(w,b),
    /// ln gamma, ln beta; policy(w,b); value(w,b); belief(w,b). Linear
    /// weights are [out][in] row-major (PyTorch convention).
    pub fn from_bytes(data: &[u8]) -> Result<NeuralNet, NeuralError> {
        let mut r = Reader { data, pos: 0 };
        if r.u32()? != MAGIC {
            return Err(NeuralError::BadMagic);
        }
        let width = r.u32()? as usize;
        let num_blocks = r.u32()? as usize;
        let obs_len = r.u32()? as usize;
        if obs_len != OBS_LEN || width == 0 || width > 8192 || num_blocks > 64 {
            return Err(NeuralError::BadShape);
        }
        let stem = r.linear(width, OBS_LEN)?;
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
            width,
            stem,
            blocks,
            policy,
            value,
            belief,
        })
    }

    pub fn forward(&self, obs: &[f32]) -> NeuralOutput {
        debug_assert_eq!(obs.len(), OBS_LEN);
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

    /// Belief probabilities for one card: 3 relative opponents then stock.
    /// `num_players` beyond 4 is unsupported (net is trained for 4).
    pub fn belief_probs(out: &NeuralOutput, card_idx: usize) -> [f32; BELIEF_CLASSES] {
        let logits = &out.belief_logits[card_idx * BELIEF_CLASSES..(card_idx + 1) * BELIEF_CLASSES];
        softmax4(logits)
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
        out.extend(MAGIC.to_le_bytes());
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
