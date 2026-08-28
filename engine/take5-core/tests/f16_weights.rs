//! Coverage for the f16 weight-decoding path.
//!
//! The unit tests in `neural.rs` build their fixtures with `tiny_net_bytes`,
//! which emits the V1 header and therefore always `DTYPE_F32`. Every set of
//! weights the project actually ships is f16, so the `elem == 2` branch of
//! `Reader::f32s` had no test at all until this file. It is the branch a
//! byte-chunking mistake would land in, and the failure mode is silent:
//! the net still loads, it just infers from garbage.

use std::path::PathBuf;
use take5_core::{NeuralNet, OBS_LEN};

fn weights_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../web/public/net-attn.t5n")
}

/// Deterministic stand-in for a real observation.
fn fixed_obs() -> Vec<f32> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    (0..OBS_LEN)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 33) as f32) / ((1u64 << 31) as f32)) - 0.5
        })
        .collect()
}

#[test]
fn champion_f16_weights_decode_and_infer() {
    let path = weights_path();
    let bytes =
        std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let net = NeuralNet::from_bytes(&bytes).expect("champion weights should load");
    let out = net.forward(&fixed_obs());

    // 104 cards x 4 classes (3 relative opponents + stock).
    assert_eq!(out.belief_logits.len(), 416, "belief head shape");

    // A mis-chunked f16 decode reads each weight from the wrong byte pair.
    // The net still "loads", so the only thing that catches it is the
    // numbers themselves: garbage weights blow the activations up or
    // produce NaN long before they stay in a plausible logit range.
    assert!(
        out.value.is_finite(),
        "value must be finite, got {}",
        out.value
    );
    assert!(
        out.value.abs() < 100.0,
        "value {} is implausible for a trained net",
        out.value
    );
    for (i, v) in out.policy_logits.iter().enumerate() {
        assert!(v.is_finite(), "policy logit {i} is {v}");
        assert!(v.abs() < 100.0, "policy logit {i} = {v} is implausible");
    }
    for (i, v) in out.belief_logits.iter().enumerate() {
        assert!(v.is_finite(), "belief logit {i} is {v}");
        assert!(v.abs() < 100.0, "belief logit {i} = {v} is implausible");
    }

    // Decoding is pure: the same bytes must give the same numbers.
    let again = NeuralNet::from_bytes(&bytes).expect("reload");
    let out2 = again.forward(&fixed_obs());
    assert_eq!(
        out.value.to_bits(),
        out2.value.to_bits(),
        "value not deterministic"
    );
    assert_eq!(
        out.policy_logits, out2.policy_logits,
        "policy not deterministic"
    );
    assert_eq!(
        out.belief_logits, out2.belief_logits,
        "belief not deterministic"
    );
}
