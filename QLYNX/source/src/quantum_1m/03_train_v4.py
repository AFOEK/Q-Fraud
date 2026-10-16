#!/usr/bin/env python3
"""Q-Guard V4: GRU + one-hop GraphSAGE-style encoder + optional 4-qubit VQC."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

TASKS = ("fraud", "aml", "protection")
INPUTS = (
    "tab", "seq", "graph_self",
    "graph_device", "graph_token", "graph_links"
)


def apply_ry(state, angle, qubit):
    n = state.shape[0]
    x = state.reshape(n, 1 << qubit, 2, 1 << (3 - qubit))

    a = x[:, :, 0, :]
    b = x[:, :, 1, :]

    c = torch.cos(angle * 0.5).reshape(n, 1, 1)
    s = torch.sin(angle * 0.5).reshape(n, 1, 1)

    return torch.stack(
        (c * a - s * b, s * a + c * b),
        dim=2
    ).reshape(n, 16)


class FourQubitVQC(nn.Module):
    def __init__(self, layers=2):
        super().__init__()

        self.layers = layers

        self.theta = nn.Parameter(
            torch.empty(layers, 4).uniform_(-0.1, 0.1)
        )
        self.scale = nn.Parameter(torch.ones(layers, 4))

        edges = ((0, 1), (1, 2), (2, 3), (3, 0))

        for edge, (control, target) in enumerate(edges):
            indices = np.arange(16, dtype=np.int64)
            mask = (
                indices & (1 << (3 - control))
            ) != 0

            permutation = np.where(
                mask,
                indices ^ (1 << (3 - target)),
                indices
            )

            self.register_buffer(
                f"cnot_{edge}",
                torch.from_numpy(permutation)
            )

        signs = np.empty((4, 16), dtype=np.float32)

        for q in range(4):
            signs[q] = 1 - 2 * (
                (np.arange(16) >> (3 - q)) & 1
            )

        self.register_buffer(
            "z_signs",
            torch.from_numpy(signs)
        )

    def forward(self, angles):
        batch = angles.shape[0]

        state = torch.zeros(
            (batch, 16),
            device=angles.device,
            dtype=angles.dtype
        )
        state[:, 0] = 1.0

        for layer in range(self.layers):
            for q in range(4):
                rotation = (
                    angles[:, q] * self.scale[layer, q]
                    + self.theta[layer, q]
                )
                state = apply_ry(state, rotation, q)

            for edge in range(4):
                state = state.index_select(
                    1,
                    getattr(self, f"cnot_{edge}")
                )

        return (state * state) @ self.z_signs.T


class QGuardV4(nn.Module):
    def __init__(self, tab_dim, kind):
        super().__init__()

        self.kind = kind

        self.tab = nn.Sequential(
            nn.Linear(tab_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 64),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=3,
            hidden_size=24,
            batch_first=True
        )

        self.graph_proj = nn.Linear(8, 32)

        self.sage = nn.Sequential(
            nn.Linear(66, 32),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(120, 64),
            nn.ReLU()
        )

        self.angle_proj = nn.Linear(64, 4)

        if kind == "quantum":
            self.head4 = FourQubitVQC(layers=2)
        else:
            self.head4 = nn.Sequential(
                nn.Linear(4, 16),
                nn.Tanh(),
                nn.Linear(16, 4),
                nn.Tanh()
            )

        self.out = nn.Linear(68, 3)

    def forward(self, batch):
        tab = self.tab(batch["tab"])

        _, hidden = self.gru(batch["seq"])
        temporal = hidden[-1]

        self_emb = torch.relu(
            self.graph_proj(batch["graph_self"])
        )

        device_emb = torch.relu(
            self.graph_proj(batch["graph_device"])
        )

        token_emb = torch.relu(
            self.graph_proj(batch["graph_token"])
        )

        links = batch["graph_links"]

        dm = (links[:, 0:1] > 0).to(links.dtype)
        tm = (links[:, 1:2] > 0).to(links.dtype)

        neighbours = (
            device_emb * dm + token_emb * tm
        ) / (dm + tm).clamp_min(1)

        graph = self.sage(
            torch.cat(
                (self_emb, neighbours, links),
                dim=-1
            )
        )

        fused = self.fusion(
            torch.cat(
                (tab, temporal, graph),
                dim=-1
            )
        )

        angles = math.pi * torch.tanh(
            self.angle_proj(fused)
        )

        output4 = self.head4(angles)

        return self.out(
            torch.cat((fused, output4), dim=-1)
        )


def average_precision(targets, scores):
    positives = int(np.count_nonzero(targets))

    if positives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    y = targets[order].astype(np.int64)
    s = scores[order]

    last = np.r_[
        np.flatnonzero(np.diff(s) != 0),
        len(s) - 1
    ]

    tp = np.cumsum(y)[last]
    delta = np.diff(np.r_[0, tp])

    return float(
        np.sum(
            (delta / positives) * (tp / (last + 1))
        )
    )


def best_threshold(targets, scores):
    positives = int(targets.sum())

    if positives == 0:
        return 0.5

    order = np.argsort(-scores, kind="stable")
    y = targets[order].astype(np.int64)
    s = scores[order]

    last = np.r_[
        np.flatnonzero(np.diff(s) != 0),
        len(s) - 1
    ]

    tp = np.cumsum(y)[last]

    f1 = 2 * tp / np.maximum(
        positives + last + 1, 1
    )

    return float(s[last[int(np.argmax(f1))]])


def metrics(labels, scores, thresholds):
    results = {}

    for j, task in enumerate(TASKS):
        y = labels[:, j].astype(bool)
        predicted = scores[:, j] >= thresholds[j]

        tp = int((y & predicted).sum())
        fp = int((~y & predicted).sum())
        fn = int((y & ~predicted).sum())

        results[task] = {
            "auprc": average_precision(y, scores[:, j]),
            "threshold": float(thresholds[j]),
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "positives": int(y.sum())
        }

    return results


def load_data(root):
    data = {
        name: np.load(
            root / f"{name}.npy",
            mmap_mode="r"
        )
        for name in INPUTS
    }

    for name in ("labels", "split"):
        data[name] = np.load(
            root / f"{name}.npy",
            mmap_mode="r"
        )

    n = len(data["labels"])

    assert n == 1_199_885
    assert data["labels"].shape == (n, 3)
    assert data["seq"].shape == (n, 3, 3)
    assert data["graph_self"].shape[1] == 8
    assert data["graph_links"].shape[1] == 2
    assert all(len(a) == n for a in data.values())

    return data


def make_batch(data, indices, device):
    batch = {}

    for name in INPUTS:
        array = np.ascontiguousarray(
            data[name][indices],
            dtype=np.float32
        )

        batch[name] = torch.from_numpy(
            array
        ).to(device)

    return batch


def inference(model, data, indices, device, batch_size):
    model.eval()

    output = np.empty(
        (len(indices), 3),
        dtype=np.float32
    )

    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]

            batch = make_batch(data, ids, device)
            scores = torch.sigmoid(model(batch))

            output[start:start + len(ids)] = (
                scores.cpu().numpy()
            )

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)

    parser.add_argument(
        "--kind",
        choices=["classical", "quantum"],
        required=True
    )

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--infer-batch", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument(
        "--train-source",
        choices=["all", "selected"],
        default="all"
    )

    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--no-full-export", action="store_true")

    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    seed = 20261016

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on this node")

    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")

    properties = torch.cuda.get_device_properties(0)

    print(
        f"[GPU] {properties.name}, "
        f"{properties.total_memory / 2**30:.2f} GiB",
        flush=True
    )

    print("[LOAD] Prepared tensors", flush=True)
    data = load_data(args.data)

    split = np.asarray(data["split"])

    train_all = np.flatnonzero(split == 0)
    val_all = np.flatnonzero(split == 1)
    test = np.flatnonzero(split == 2)

    if args.train_source == "selected":
        train = np.load(
            args.data / "train_selected.npy"
        )
        assert np.all(split[train] == 0)
    else:
        train = train_all

    if args.max_train > 0:
        train = train[:args.max_train]

    val = (
        val_all[:args.max_val]
        if args.max_val > 0
        else val_all
    )

    print(
        f"[DATA] Train={len(train):,}, "
        f"Val={len(val):,}, Test={len(test):,}",
        flush=True
    )

    model = QGuardV4(
        data["tab"].shape[1],
        args.kind
    ).to(device)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print(
        f"[MODEL] {args.kind}, parameters={parameters:,}",
        flush=True
    )

    train_labels = np.asarray(
        data["labels"][train],
        dtype=np.float32
    )

    positive_counts = train_labels.sum(axis=0)

    positive_weights = np.sqrt(
        (len(train) - positive_counts)
        / np.maximum(positive_counts, 1)
    )

    positive_weights = np.clip(
        positive_weights, 1.0, 12.0
    )

    print(
        "[TRAIN] Positive counts:",
        positive_counts.astype(int).tolist(),
        flush=True
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weights,
            device=device,
            dtype=torch.float32
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.001
    )

    val_labels = np.asarray(
        data["labels"][val],
        dtype=np.uint8
    )

    checkpoint = args.out / f"{args.kind}_best.pt"

    best_ap = -1.0
    best_epoch = 0
    history = []

    rng = np.random.default_rng(seed)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        order = rng.permutation(train)

        running_loss = 0.0
        steps = 0

        for start in range(
            0, len(order), args.batch_size
        ):
            indices = order[
                start:start + args.batch_size
            ]

            batch = make_batch(
                data, indices, device
            )

            targets = torch.from_numpy(
                np.array(
                    data["labels"][indices],
                    dtype=np.float32,
                    copy=True
                )
            ).to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = loss_fn(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Nonfinite loss at epoch {epoch}"
                )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 3.0
            )

            optimizer.step()

            running_loss += float(
                loss.detach().cpu()
            )
            steps += 1

        val_scores = inference(
            model, data, val, device,
            args.infer_batch
        )

        ap = [
            average_precision(
                val_labels[:, j],
                val_scores[:, j]
            )
            for j in range(3)
        ]

        mean_ap = float(np.nanmean(ap))

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, steps),
            "validation_auprc": dict(zip(TASKS, ap)),
            "mean_auprc": mean_ap
        }

        history.append(row)

        print(
            f"[EPOCH {epoch:02d}] "
            f"loss={row['train_loss']:.5f} "
            f"fraud_AP={ap[0]:.5f} "
            f"aml_AP={ap[1]:.5f} "
            f"protection_AP={ap[2]:.5f} "
            f"mean={mean_ap:.5f} "
            f"elapsed={time.time()-started:.1f}s",
            flush=True
        )

        if mean_ap > best_ap + 1e-5:
            best_ap = mean_ap
            best_epoch = epoch

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "kind": args.kind,
                    "tab_dim": data["tab"].shape[1]
                },
                checkpoint
            )

        elif epoch - best_epoch >= args.patience:
            print(
                "[EARLY STOP] No validation improvement",
                flush=True
            )
            break

    print(
        f"[SELECT] Epoch={best_epoch}, "
        f"validation mean AUPRC={best_ap:.5f}",
        flush=True
    )

    saved = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(saved["model"])

    val_scores = inference(
        model, data, val, device,
        args.infer_batch
    )

    thresholds = [
        best_threshold(
            val_labels[:, j],
            val_scores[:, j]
        )
        for j in range(3)
    ]

    report = {
        "status": "PASSED",
        "kind": args.kind,
        "training_accounts": len(train),
        "validation_accounts": len(val),
        "test_accounts": len(test),
        "trainable_parameters": parameters,
        "best_epoch": best_epoch,
        "qubits": 4 if args.kind == "quantum" else 0,
        "graph": "Precomputed transductive one-hop GraphSAGE-style aggregates",
        "validation": metrics(
            val_labels, val_scores, thresholds
        ),
        "history": history,
        "test_evaluated": False,
        "score_is_calibrated_probability": False
    }

    # The full evaluation is omitted during smoke testing.
    if args.max_train == 0 and args.max_val == 0:
        print("[EVAL] Final test evaluation", flush=True)

        test_scores = inference(
            model, data, test, device,
            args.infer_batch
        )

        test_labels = np.asarray(
            data["labels"][test],
            dtype=np.uint8
        )

        report["test_diagnostic"] = metrics(
            test_labels,
            test_scores,
            thresholds
        )

        report["test_evaluated"] = True

        if not args.no_full_export:
            print(
                "[EXPORT] Scoring all 1,199,885 accounts",
                flush=True
            )

            all_indices = np.arange(len(split))

            full_scores = inference(
                model,
                data,
                all_indices,
                device,
                args.infer_batch
            )

            np.save(
                args.out / f"{args.kind}_scores_1m.npy",
                full_scores
            )

            report["full_population_scores_shape"] = (
                list(full_scores.shape)
            )

    report["cuda_max_allocated_gib"] = (
        torch.cuda.max_memory_allocated() / 2**30
    )

    (args.out / f"{args.kind}_metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    print(
        json.dumps(
            {
                "kind": args.kind,
                "best_epoch": best_epoch,
                "validation": report["validation"],
                "test": report.get("test_diagnostic")
            },
            indent=2
        ),
        flush=True
    )

    print(
        f"[PASS] Q-GUARD V4 {args.kind.upper()}",
        flush=True
    )


if __name__ == "__main__":
    main()
