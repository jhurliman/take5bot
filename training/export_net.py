#!/usr/bin/env python3
"""Export a trained PPO checkpoint to the flat binary format consumed by
the Rust engine (take5-core/src/neural.rs) for native and WASM inference.

MLP format (little-endian): u32 magic "T5N2", u32 width, u32 blocks,
u32 obs_len, u32 dtype, then tensors in order: stem(w,b); per block
lin1(w,b), lin2(w,b), ln gamma, ln beta; policy(w,b); value(w,b);
belief(w,b). Linear weights are [out][in] row-major (PyTorch's layout).

Attention format: u32 magic "T5N3", u32 d_model, u32 layers, u32 obs_len,
u32 dtype, u32 heads, then tensors: card_emb; feat(w,b); glob(w,b); cls;
per layer in_proj(w,b), out_proj(w,b), linear1(w,b), linear2(w,b),
norm1(g,b), norm2(g,b); final norm(g,b); policy(w,b); value(w,b);
belief(w,b).

Example:
  .venv/bin/python training/export_net.py --ckpt training/runs/m4-v1/best.pt \
      --out training/runs/m4-v1/net.t5n
"""

import argparse
import os
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_ppo import AttnNet, build_net

MAGIC_V2 = 0x54354E32  # "T5N2" (MLP; header carries a dtype field)
MAGIC_V3 = 0x54354E33  # "T5N3" (attention encoder)
DTYPES = {"f32": 0, "f16": 1}


def write_tensor(fh, t: torch.Tensor, dtype: str) -> None:
    arr = t.detach().cpu().contiguous().to(torch.float32).numpy()
    if dtype == "f16":
        arr = arr.astype("float16")
    fh.write(arr.tobytes())


def export_attn(net: AttnNet, path: str, dtype: str = "f32") -> int:
    with open(path, "wb") as fh:
        obs_len = net.glob.in_features + AttnNet.GLOBAL_OFF
        heads = net.encoder.layers[0].self_attn.num_heads
        fh.write(
            struct.pack(
                "<IIIIII",
                MAGIC_V3,
                net.width,
                net.blocks,
                obs_len,
                DTYPES[dtype],
                heads,
            )
        )
        write_tensor(fh, net.card_emb.weight, dtype)
        write_tensor(fh, net.feat.weight, dtype)
        write_tensor(fh, net.feat.bias, dtype)
        write_tensor(fh, net.glob.weight, dtype)
        write_tensor(fh, net.glob.bias, dtype)
        write_tensor(fh, net.cls, dtype)
        for layer in net.encoder.layers:
            write_tensor(fh, layer.self_attn.in_proj_weight, dtype)
            write_tensor(fh, layer.self_attn.in_proj_bias, dtype)
            write_tensor(fh, layer.self_attn.out_proj.weight, dtype)
            write_tensor(fh, layer.self_attn.out_proj.bias, dtype)
            write_tensor(fh, layer.linear1.weight, dtype)
            write_tensor(fh, layer.linear1.bias, dtype)
            write_tensor(fh, layer.linear2.weight, dtype)
            write_tensor(fh, layer.linear2.bias, dtype)
            write_tensor(fh, layer.norm1.weight, dtype)
            write_tensor(fh, layer.norm1.bias, dtype)
            write_tensor(fh, layer.norm2.weight, dtype)
            write_tensor(fh, layer.norm2.bias, dtype)
        write_tensor(fh, net.encoder.norm.weight, dtype)
        write_tensor(fh, net.encoder.norm.bias, dtype)
        write_tensor(fh, net.policy.weight, dtype)
        write_tensor(fh, net.policy.bias, dtype)
        write_tensor(fh, net.value.weight, dtype)
        write_tensor(fh, net.value.bias, dtype)
        write_tensor(fh, net.belief.weight, dtype)
        write_tensor(fh, net.belief.bias, dtype)
    return os.path.getsize(path)


def export(net, path: str, dtype: str = "f32") -> int:
    if isinstance(net, AttnNet):
        return export_attn(net, path, dtype)
    with open(path, "wb") as fh:
        obs_len = net.trunk[0].in_features
        fh.write(
            struct.pack(
                "<IIIII", MAGIC_V2, net.width, net.blocks, obs_len, DTYPES[dtype]
            )
        )
        write_tensor(fh, net.trunk[0].weight, dtype)
        write_tensor(fh, net.trunk[0].bias, dtype)
        for block in list(net.trunk)[2:]:
            write_tensor(fh, block.body[0].weight, dtype)
            write_tensor(fh, block.body[0].bias, dtype)
            write_tensor(fh, block.body[2].weight, dtype)
            write_tensor(fh, block.body[2].bias, dtype)
            write_tensor(fh, block.norm.weight, dtype)
            write_tensor(fh, block.norm.bias, dtype)
        write_tensor(fh, net.policy.weight, dtype)
        write_tensor(fh, net.policy.bias, dtype)
        write_tensor(fh, net.value.weight, dtype)
        write_tensor(fh, net.value.bias, dtype)
        write_tensor(fh, net.belief.weight, dtype)
        write_tensor(fh, net.belief.bias, dtype)
    return os.path.getsize(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16"],
        default="f16",
        help="f16 halves the file with negligible strength loss",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", {})
    net = build_net(cfg.get("arch", "mlp"), cfg.get("width", 512), cfg.get("blocks", 2))
    net.load_state_dict(ckpt["model"])
    net.eval()
    size = export(net, args.out, args.dtype)
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
