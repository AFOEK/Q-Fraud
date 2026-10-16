#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--bundle",
    type=Path,
    required=True
)
parser.add_argument(
    "--ids-csv",
    type=Path,
    required=True
)
parser.add_argument(
    "--out",
    type=Path,
    default=Path("v5_fresh_inference.csv")
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=512
)

args = parser.parse_args()
root = args.bundle.resolve()

sys.path.insert(
    0,
    str(root / "source/src/quantum_1m")
)

from v5_architecture import QGuardV5

query = pd.read_csv(
    args.ids_csv,
    dtype={"account_id": "string"}
)

if query.columns.tolist() != ["account_id"]:
    raise ValueError(
        "CSV must contain only account_id. "
        "This utility uses frozen neural inputs."
    )

if (
    query.account_id.isna().any()
    or not query.account_id.is_unique
    or len(query) == 0
):
    raise ValueError(
        "Account IDs must be nonempty and unique"
    )

known = pd.read_parquet(
    root / "inference_inputs/account_ids.parquet",
    columns=["account_id"]
)

known_index = pd.Index(
    known.account_id.astype("string")
)

indices = known_index.get_indexer(
    query.account_id
)

if (indices < 0).any():
    raise ValueError(
        "Unknown account IDs: "
        + str(
            query.account_id[
                indices < 0
            ].head(10).tolist()
        )
    )

names = (
    "tab", "seq", "graph_self",
    "graph_device", "graph_token",
    "graph_links"
)

data = {
    name: np.load(
        root / f"inference_inputs/{name}.npy",
        mmap_mode="r",
        allow_pickle=False
    )
    for name in names
}

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

result = query.copy()

for kind in ("classical_strong", "quantum"):
    mean_scores = np.zeros(
        (len(indices), 3),
        dtype=np.float32
    )

    for seed in (17, 42, 99):
        model = QGuardV5(
            int(data["tab"].shape[1]),
            kind
        ).to(device)

        checkpoint_path = (
            root / f"models/neural_v5/"
            f"seed_{seed}/{kind}/{kind}_best.pt"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            checkpoint["model"],
            strict=True
        )
        model.eval()

        parts = []

        with torch.inference_mode():
            for start in range(
                0,
                len(indices),
                args.batch_size
            ):
                rows = indices[
                    start:start + args.batch_size
                ]

                batch = {
                    name: torch.from_numpy(
                        np.ascontiguousarray(
                            data[name][rows],
                            dtype=np.float32
                        )
                    ).to(device)
                    for name in names
                }

                parts.append(
                    torch.sigmoid(
                        model(batch)
                    ).cpu().numpy()
                )

        mean_scores += np.concatenate(
            parts,
            axis=0
        )

    mean_scores /= 3.0

    expected = np.load(
        root / "predictions"
        / f"{kind}_ensemble_scores_1m.npy",
        mmap_mode="r"
    )[indices]

    error = float(np.max(
        np.abs(mean_scores - expected)
    ))

    if error > 0.001:
        raise RuntimeError(
            f"Checkpoint reproduction failed: "
            f"{kind}, error={error}"
        )

    prefix = (
        "classical_v5"
        if kind == "classical_strong"
        else "quantum_v5"
    )

    for i, task in enumerate(
        ("fraud", "aml", "protection")
    ):
        result[f"{prefix}_{task}_score"] = (
            mean_scores[:, i]
        )

    print(
        f"[PASS] {kind}: max_error={error:.8f}",
        flush=True
    )

args.out.parent.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    args.out,
    index=False
)

print(
    "[PASS] FRESH KNOWN-ACCOUNT V5 INFERENCE:",
    args.out,
    flush=True
)
