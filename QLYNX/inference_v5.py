"""Inference adapter for the Q-Guard V5 release: known-account V5 and feature-level LightGBM."""
from __future__ import annotations

import json
import runpy
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TASKS = ("fraud", "aml", "protection")
OPERATIONAL_WEIGHTS = {
    "fraud": "models/models_fraud_v3/fraud_weight_4.txt",
    "aml": "models/models_context_v2/aml_lightgbm.txt",
    "protection": "models/models_context_v2/protection_lightgbm.txt",
}


def infer_known_accounts(account_ids: list[str]) -> pd.DataFrame:
    """Run all six V5 neural checkpoints on already-preprocessed, known accounts."""
    ids = [str(x).strip() for x in account_ids]
    if not ids or any(not x for x in ids):
        raise ValueError("Provide at least one nonempty account ID.")
    if len(ids) != len(set(ids)):
        raise ValueError("Account IDs must be unique.")
    if len(ids) > 100:
        raise ValueError("Select at most 100 accounts per run (interactive prototype).")

    # Compact standalone distribution has 130 aligned prepared demo rows.
    # Full 1.2M-account original bundle provides the frozen mmap arrays.
    if not (ROOT / "inference_inputs/account_ids.parquet").exists():
        return _infer_demo_accounts(ids)

    script = ROOT / "source/src/dashboard/13_infer_v5_known_accounts.py"
    with tempfile.TemporaryDirectory(prefix="qguard_live_") as temp:
        input_csv = Path(temp) / "account_ids.csv"
        output_csv = Path(temp) / "v5_scores.csv"
        pd.DataFrame({"account_id": ids}).to_csv(input_csv, index=False)
        completed = subprocess.run(
            [sys.executable, str(script), "--bundle", str(ROOT),
             "--ids-csv", str(input_csv), "--out", str(output_csv),
             "--batch-size", "64"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError((completed.stderr or completed.stdout)[-3000:])
        return pd.read_csv(output_csv, dtype={"account_id": "string"})


def _infer_demo_accounts(ids: list[str]) -> pd.DataFrame:
    """Execute all SIX authentic V5 weights on identical stored 130-account tensors.

    The demo arrays were exported from the prepared model features. This
    subset intentionally does not support unseen account IDs or raw events.
    """
    import importlib.util
    import torch

    with np.load(ROOT / "demo/v4_demo_inputs.npz", allow_pickle=False) as data:
        known_ids = pd.Index(data["account_ids"].astype(str))
        indices = known_ids.get_indexer(ids)
        if (indices < 0).any():
            bad = [x for x, idx in zip(ids, indices) if idx < 0]
            raise ValueError(
                "This standalone demo contains 130 frozen account feature rows. "
                "Unknown account IDs: " + str(bad[:5])
            )
        fields = ("tab", "seq", "graph_self", "graph_device", "graph_token", "graph_links")
        batch = {name: torch.from_numpy(np.ascontiguousarray(data[name][indices], dtype=np.float32))
                 for name in fields}

    spec = importlib.util.spec_from_file_location(
        "qlynx_v5_architecture", ROOT / "source/src/quantum_1m/v5_architecture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = pd.DataFrame({"account_id": ids})
    for kind, label in (("classical_strong", "classical_v5"), ("quantum", "quantum_v5")):
        values = []
        for seed in (17, 42, 99):
            model = module.QGuardV5(batch["tab"].shape[1], kind)
            state = torch.load(
                ROOT / f"models/neural_v5/seed_{seed}/{kind}/{kind}_best.pt",
                map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            model.eval()
            with torch.inference_mode():
                values.append(torch.sigmoid(model(batch)).numpy())
        mean_scores = np.mean(np.stack(values, axis=0), axis=0)
        for j, task in enumerate(TASKS):
            result[f"{label}_{task}_score"] = mean_scores[:, j]
    return result


@lru_cache(maxsize=1)
def feature_builder():
    """Reuse the exact original feature engineering from training."""
    source = ROOT / "source/src/ml_1m/14_train_risk.py"
    return runpy.run_path(str(source))["features"]


@lru_cache(maxsize=3)
def get_booster(task: str):
    return lgb.Booster(model_file=str(ROOT / OPERATIONAL_WEIGHTS[task]))


def infer_feature_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify correctly prepared 90-day aggregate account features (not raw events)."""
    if not 0 < len(frame) <= 5000:
        raise ValueError("Upload between 1 and 5,000 account records.")
    if "account_id" not in frame:
        raise ValueError("Missing account_id.")
    if frame.account_id.isna().any() or not frame.account_id.astype(str).is_unique:
        raise ValueError("account_id values must be present and unique.")

    X = feature_builder()(frame)
    manifest = json.loads((ROOT / "dashboard_manifest.json").read_text())
    scored = pd.DataFrame({"account_id": frame.account_id.astype(str).to_numpy()})
    for task in TASKS:
        model = get_booster(task)
        missing = sorted(set(model.feature_name()) - set(X.columns))
        if missing:
            raise ValueError(f"Missing {task} features: {', '.join(missing[:12])}")
        probs = np.asarray(model.predict(X[model.feature_name()], num_threads=2))
        threshold = manifest["operational_models"][task]["threshold"]
        scored[f"{task}_score"] = probs
        scored[f"{task}_alert"] = probs >= threshold
    scored["alert_count"] = scored[[f"{t}_alert" for t in TASKS]].sum(axis=1)
    return scored.sort_values("alert_count", ascending=False, kind="stable")
