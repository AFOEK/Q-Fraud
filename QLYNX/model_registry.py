"""QLYNX model registry and inference; all model families in the V5 release.

Fresh LightGBM = compatible engineered features, never arbitrary raw events.
Neural V4 = packaged 130-account demo inputs only.
Neural V5 = frozen prepared features for known snapshot IDs.
"""
from __future__ import annotations

import importlib.util
import json
import runpy
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from inference_v5 import infer_known_accounts

ROOT = Path(__file__).resolve().parent
TASKS = ("fraud", "aml", "protection")
TASK_LABELS = {"fraud": "Fraud", "aml": "Anti-Money Laundering", "protection": "Player Protection"}

# Explicit mappings, not arbitrary user-supplied file paths.
LGBM = {
    **{f"V1 · {task.upper()}": (task, f"models/models/{task}_lightgbm.txt") for task in TASKS},
    **{f"V2 · {task.upper()}": (task, f"models/models_context_v2/{task}_lightgbm.txt") for task in TASKS},
    **{f"V3 · Fraud (weight {weight}×)": ("fraud", f"models/models_fraud_v3/fraud_weight_{weight}.txt") for weight in (1, 2, 4, 8)},
}
OPERATIONAL = {
    "fraud": "V3 · Fraud (weight 4×)",
    "aml": "V2 · AML",
    "protection": "V2 · PROTECTION",
}
NEURAL_CHECKPOINTS = {
    "V4 · Classical": "models/neural_v4/classical_best.pt",
    "V4 · Quantum": "models/neural_v4/quantum_best.pt",
    **{
        f"V5 · {kind} · seed {seed}": f"models/neural_v5/seed_{seed}/{filename}/{filename}_best.pt"
        for kind, filename in (("Classical", "classical_strong"), ("Quantum", "quantum"))
        for seed in (17, 42, 99)
    },
}

@lru_cache(maxsize=1)
def manifest():
    return json.loads((ROOT / "dashboard_manifest.json").read_text(encoding="utf-8"))

@lru_cache(maxsize=1)
def feature_builder():
    return runpy.run_path(str(ROOT / "source/src/ml_1m/14_train_risk.py"))["features"]

@lru_cache(maxsize=20)
def booster(name: str):
    if name not in LGBM:
        raise ValueError(f"Unknown model {name!r}")
    path = ROOT / LGBM[name][1]
    return lgb.Booster(model_file=str(path))

@lru_cache(maxsize=1)
def expected_upload_columns():
    return pd.read_csv(ROOT / "demo/demo_accounts_upload.csv", nrows=0).columns.tolist()


def prepare_features(frame: pd.DataFrame):
    missing = sorted(set(expected_upload_columns()) - set(frame.columns))
    if missing:
        raise ValueError("Missing engineered input columns: " + ", ".join(missing[:20]))
    if len(frame) < 1 or len(frame) > 5000:
        raise ValueError("Select between 1 and 5,000 engineered account rows.")
    if frame.account_id.isna().any() or not frame.account_id.astype(str).is_unique:
        raise ValueError("account_id must be non-null and unique.")
    return feature_builder()(frame)


def score_lightgbm(frame: pd.DataFrame, names: list[str] | None = None, *, explanations: bool = False):
    """Return all requested original model scores + *operational* alert flags.

    Contributions are LightGBM log-odds raw-margin contributions, NOT explanations for neural models.
    """
    names = list(LGBM) if names is None else list(names)
    if not names or any(n not in LGBM for n in names):
        raise ValueError("Choose valid LightGBM models.")
    X = prepare_features(frame)
    output = pd.DataFrame({"account_id": frame.account_id.astype(str).to_numpy()})
    evidence = []
    for name in names:
        task, _ = LGBM[name]
        model = booster(name)
        missing = sorted(set(model.feature_name()) - set(X.columns))
        if missing:
            raise ValueError(f"{name} missing features: {missing}")
        selected = X[model.feature_name()]
        scores = np.asarray(model.predict(selected, num_threads=2), dtype=float)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Non-finite results from {name}")
        output[name] = scores
        if OPERATIONAL[task] == name:
            threshold = float(manifest()["operational_models"][task]["threshold"])
            output[f"{task}_score"] = scores
            output[f"{task}_alert"] = scores >= threshold
        if explanations:
            contrib = np.asarray(model.predict(selected, pred_contrib=True, num_threads=2))
            for i in range(len(selected)):
                weights = contrib[i, :-1]
                for j in np.argsort(np.abs(weights))[::-1][:5]:
                    evidence.append({
                        "account_id": str(frame.account_id.iloc[i]),
                        "model": name, "risk_category": task,
                        "feature": model.feature_name()[j],
                        "observed_value": str(selected.iloc[i, j]),
                        "contribution_log_odds": float(weights[j]),
                    })
    if all(OPERATIONAL[t] in names for t in TASKS):
        output["alert_count"] = output[[f"{t}_alert" for t in TASKS]].sum(axis=1)
        output["priority"] = np.select(
            [output.alert_count >= 2, output.alert_count == 1],
            ["Critical", "High"], default="No current alert")
    return output, pd.DataFrame(evidence)


@lru_cache(maxsize=1)
def _v4_module():
    path = ROOT / "source/src/quantum_1m/03_train_v4.py"
    spec = importlib.util.spec_from_file_location("qguard_v4_architecture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_v4_demo(account_ids: list[str]):
    """Score the stored V4 demonstration cases using BOTH actual V4 checkpoints."""
    with np.load(ROOT / "demo/v4_demo_inputs.npz", allow_pickle=False) as data:
        index = pd.Index(data["account_ids"].astype(str))
        indices = index.get_indexer([str(a) for a in account_ids])
        if len(indices) == 0 or len(indices) != len(set(account_ids)):
            raise ValueError("Provide unique demo account IDs.")
        if (indices < 0).any():
            raise ValueError("V4 is supported for the 130 bundled demo accounts only.")
        keys = ("tab", "seq", "graph_self", "graph_device", "graph_token", "graph_links")
        batch = {key: torch.from_numpy(np.ascontiguousarray(data[key][indices], dtype=np.float32)) for key in keys}
        results = pd.DataFrame({"account_id": [str(a) for a in account_ids]})
        module = _v4_module()
        for kind in ("classical", "quantum"):
            checkpoint = torch.load(ROOT / f"models/neural_v4/{kind}_best.pt", map_location="cpu", weights_only=True)
            model = module.QGuardV4(batch["tab"].shape[1], kind)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            with torch.inference_mode():
                values = torch.sigmoid(model(batch)).numpy()
            expected = data[f"{kind}_reference"][indices]
            max_error = float(np.max(np.abs(values - expected)))
            if max_error > .001:
                raise RuntimeError(f"V4 {kind} checkpoint mismatch: {max_error:.6f}")
            for j, task in enumerate(TASKS):
                results[f"v4_{kind}_{task}"] = values[:, j]
        return results


def model_status():
    return pd.DataFrame(
        [{"Model": name, "Family": "LightGBM", "Status": "Available" if (ROOT / path).exists() else "Missing"}
         for name, (_, path) in LGBM.items()] +
        [{"Model": name, "Family": "Neural", "Status": "Available" if (ROOT / path).exists() else "Missing"}
         for name, path in NEURAL_CHECKPOINTS.items()]
    )
