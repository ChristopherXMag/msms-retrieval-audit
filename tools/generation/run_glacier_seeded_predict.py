#!/usr/bin/env python3
"""Run GLACIER with seeded RNG and evaluation-safe Graphormer attention.

The released ``glacier/predict_smis_joint.py`` accepts ``--seed`` but does not
apply it.  Its Graphormer attention also passes dropout directly to functional
SDPA during ``eval()``, where functional SDPA does not disable dropout.  The two
issues make separate prediction invocations produce different sparse spectra.

This wrapper preserves the upstream argument surface, applies the requested
seed before model construction, and patches only the Graphormer attention
forward so evaluation uses ``dropout_p=0`` while training behavior is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F


def requested_seed(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args(argv)
    return int(args.seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def wrapper_metadata(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args


def patch_graphormer_eval_dropout() -> Path:
    from ms_pred.graphormer.multihead_attention import MultiheadAttention

    source_path = Path(sys.modules[MultiheadAttention.__module__].__file__).resolve()

    def eval_safe_forward(self, query, attn_bias):
        target_length, batch_size, embedding_dim = query.size()
        query_key_value = self.qkv_proj(query)
        query_tensor, key_tensor, value_tensor = query_key_value.chunk(3, dim=-1)
        query_tensor = query_tensor.view(
            target_length, batch_size, self.num_heads, self.head_dim
        ).permute(1, 2, 0, 3)
        key_tensor = key_tensor.view(
            -1, batch_size, self.num_heads, self.head_dim
        ).permute(1, 2, 0, 3)
        value_tensor = value_tensor.view(
            -1, batch_size, self.num_heads, self.head_dim
        ).permute(1, 2, 0, 3)
        attention_bias = attn_bias.view(
            batch_size, self.num_heads, target_length, -1
        )
        attention = F.scaled_dot_product_attention(
            query_tensor,
            key_tensor,
            value_tensor,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attention = attention.permute(2, 0, 1, 3).reshape(
            target_length, batch_size, embedding_dim
        )
        return self.out_proj(attention)

    MultiheadAttention.forward = eval_safe_forward
    return source_path


def main() -> None:
    metadata = wrapper_metadata(sys.argv[1:])
    seed = int(metadata.seed)
    pl.seed_everything(seed, workers=True)
    attention_source = patch_graphormer_eval_dropout()
    print("Applied GLACIER eval-safe SDPA dropout patch")
    from ms_pred.glacier.predict_smis_joint import predict

    predict()
    protocol_path = metadata.save_dir / "wrapper_protocol.json"
    protocol = {
        "schema": 1,
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
        "seed": seed,
        "batch_size": int(metadata.batch_size),
        "checkpoint": str(metadata.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(metadata.checkpoint.resolve()),
        "upstream_attention_source": str(attention_source),
        "upstream_attention_source_sha256": sha256_file(attention_source),
        "patch": (
            "Graphormer functional SDPA uses module dropout during training and "
            "dropout_p=0 during eval"
        ),
        "torch_version": torch.__version__,
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
