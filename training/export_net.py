#!/usr/bin/env python3
"""Export a trained PPO checkpoint to the flat binary format consumed by
the Rust engine (take5-core/src/neural.rs) for native and WASM inference.

Format (little-endian): u32 magic "T5N1", u32 width, u32 blocks,
u32 obs_len, then f32 tensors in order: stem(w,b); per block lin1(w,b),
lin2(w,b), ln gamma, ln beta; policy(w,b); value(w,b); belief(w,b).
Linear weights are [out][in] row-major (PyTorch's native layout).

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

from train_ppo import PolicyNet

MAGIC_V2 = 0x54354E32  # "T5N2" (header carries a dtype field)
DTYPES = {"f32": 0, "f16": 1}


def write_tensor(fh, t: torch.Tensor, dtype: str) -> None:
    arr = t.detach().cpu().contiguous().to(torch.float32).numpy()
    if dtype == "f16":
        arr = arr.astype("float16")
    fh.write(arr.tobytes())


def export(net: PolicyNet, path: str, dtype: str = "f32") -> int:
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
    net = PolicyNet(cfg.get("width", 512), cfg.get("blocks", 2))
    net.load_state_dict(ckpt["model"])
    net.eval()
    size = export(net, args.out, args.dtype)
    print(f"wrote {args.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
