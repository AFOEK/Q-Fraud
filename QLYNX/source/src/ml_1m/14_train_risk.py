#!/usr/bin/env python3
"""Q-Guard 1M retrospective classical risk detection."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)


TASKS = {
    "fraud": "fraud_injected",
    "aml": "aml_injected",
    "protection": "player_protection_injected",
}

EXCLUDE = {
    "account_id",
    "person_id",
    "household_id",
    "fraud_injected",
    "aml_injected",
    "player_protection_injected",
    "benign_control_injected",
}


def values(df, column):
    return (
        pd.to_numeric(
            df[column], errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )


def features(df):
    x = df.drop(
        columns=[
            col for col in EXCLUDE
            if col in df.columns
        ]
    ).copy()

    for col in ("home_region", "income_band"):
        x[col] = x[col].astype("category")

    for col in x.columns.difference(
        ["home_region", "income_band"]
    ):
        x[col] = pd.to_numeric(
            x[col], errors="coerce"
        ).astype("float32")

    ratios = [
        (
            "wager_growth_ratio",
            "wager_last30",
            "wager_first30",
        ),
        (
            "deposit_growth_ratio",
            "deposit_last30",
            "deposit_first30",
        ),
        (
            "wager_share_recent",
            "wager_last30",
            "wager_total",
        ),
        (
            "deposit_share_recent",
            "deposit_last30",
            "deposit_total",
        ),
        (
            "withdraw_to_deposit",
            "withdrawal_total",
            "deposit_total",
        ),
        (
            "wager_to_deposit",
            "wager_total",
            "deposit_total",
        ),
        (
            "avg_wager_growth",
            "avg_wager_last30",
            "avg_wager_first30",
        ),
        (
            "away_session_share",
            "away_region_sessions",
            "session_count",
        ),
        (
            "deposit_per_session",
            "deposit_count",
            "session_count",
        ),
    ]

    for name, numerator, denominator in ratios:
        x[name] = (
            values(df, numerator)
            / (values(df, denominator) + 1.0)
        )

    return x


def fixed_rule_scores(df, task):
    v = lambda column: values(df, column)

    if task == "fraud":
        tests = [
            v("distinct_devices") >= 3,
            v("distinct_regions") >= 2,
            v("away_region_sessions") >= 2,
            v("devices_last30") >= 2,
            v("distinct_payment_tokens") >= 3,
        ]

    elif task == "aml":
        tests = [
            v("deposit_count") >= 8,
            v("withdrawal_count") >= 4,
            v("withdrawal_total")
            >= 0.8 * (v("deposit_total") + 1),
            v("payment_deposits") >= 8,
            v("payment_withdrawals") >= 4,
        ]

    else:
        tests = [
            v("wager_last30")
            >= 2 * (v("wager_first30") + 1),

            v("avg_wager_last30")
            >= 1.5 * (v("avg_wager_first30") + 1),

            v("deposit_last30")
            >= 2 * (v("deposit_first30") + 1),

            v("deposit_count_last30") >= 8,

            v("wager_last30")
            > v("deposit_last30"),
        ]

    return np.sum(
        np.stack(tests, axis=0),
        axis=0
    ).astype(np.float32)


def choose_threshold(y, score):
    precision, recall, thresholds = (
        precision_recall_curve(y, score)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    return float(
        thresholds[np.argmax(f1)]
    )


def metrics(y, score, threshold):
    pred = (
        score >= threshold
    ).astype(np.uint8)

    tn, fp, fn, tp = [
        int(i)
        for i in confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()
    ]

    k = max(
        1, int(np.ceil(0.01 * len(y)))
    )

    top = np.argsort(
        -score, kind="stable"
    )[:k]

    return {
        "n": len(y),
        "positive": int(np.sum(y)),
        "threshold": threshold,
        "precision": float(
            tp / max(1, tp + fp)
        ),
        "recall": float(
            tp / max(1, tp + fn)
        ),
        "f1": float(
            f1_score(
                y, pred, zero_division=0
            )
        ),
        "average_precision": float(
            average_precision_score(y, score)
        ),
        "roc_auc": (
            float(roc_auc_score(y, score))
            if len(np.unique(y)) > 1
            else None
        ),
        "mcc": float(
            matthews_corrcoef(y, pred)
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision_at_top_1pct": float(
            np.mean(y[top])
        ),
        "recall_at_top_1pct": float(
            np.sum(y[top])
            / max(1, np.sum(y))
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--task", choices=TASKS, required=True
    )
    ap.add_argument(
        "--features", type=Path, required=True
    )
    ap.add_argument(
        "--splits", type=Path, required=True
    )
    ap.add_argument(
        "--benign-labels",
        type=Path,
        required=True
    )
    ap.add_argument(
        "--output", type=Path, required=True
    )
    ap.add_argument(
        "--threads", type=int, default=16
    )

    args = ap.parse_args()
    args.output.mkdir(
        parents=True, exist_ok=True
    )

    target = TASKS[args.task]

    print(
        "[LOAD] Account features and splits",
        flush=True
    )

    df = pd.read_parquet(args.features)

    splits = pd.read_parquet(
        args.splits,
        columns=["account_id", "split"]
    )

    assert df.account_id.is_unique
    assert splits.account_id.is_unique

    df = df.merge(
        splits,
        on="account_id",
        how="left",
        validate="one_to_one",
        sort=False
    )

    assert not df.split.isna().any()

    y = df[target].to_numpy(
        dtype=np.uint8
    )

    tr = (
        df.split == "train"
    ).to_numpy()

    va = (
        df.split == "val"
    ).to_numpy()

    te = (
        df.split == "test"
    ).to_numpy()

    assert np.sum(y[tr]) > 100
    assert np.sum(y[va]) > 20
    assert np.sum(y[te]) > 20

    x = features(
        df.drop(columns=["split"])
    )

    baseline = fixed_rule_scores(
        df, args.task
    )

    print(
        f"[TRAIN] {args.task}: "
        f"{int(tr.sum()):,} train, "
        f"{int(va.sum()):,} validation, "
        f"{int(te.sum()):,} test, "
        f"{x.shape[1]} features",
        flush=True
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        colsample_bytree=0.9,
        subsample=0.85,
        subsample_freq=1,
        reg_lambda=2.0,
        n_jobs=args.threads,
        random_state=20261016,
        verbosity=-1,
    )

    model.fit(
        x.loc[tr],
        y[tr],
        eval_set=[
            (x.loc[va], y[va])
        ],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=40,
                verbose=True
            ),
            lgb.log_evaluation(period=50),
        ],
    )

    scores = model.predict_proba(
        x,
        num_threads=args.threads
    )[:, 1].astype(np.float32)

    # Both thresholds are selected on validation only.
    ml_threshold = choose_threshold(
        y[va], scores[va]
    )

    rule_threshold = choose_threshold(
        y[va], baseline[va]
    )

    ml_results = metrics(
        y[te], scores[te], ml_threshold
    )

    rule_results = metrics(
        y[te],
        baseline[te],
        rule_threshold
    )

    print(
        "[EVAL] ML:",
        json.dumps(ml_results),
        flush=True
    )

    print(
        "[EVAL] RULES:",
        json.dumps(rule_results),
        flush=True
    )

    benign = pd.read_parquet(
        args.benign_labels,
        columns=[
            "account_id",
            "lookalike_type"
        ]
    )

    if benign.account_id.duplicated().any():
        raise ValueError(
            "Expected one benign scenario "
            "per account"
        )

    benign_type = (
        df.account_id.map(
            pd.Series(
                benign.lookalike_type.to_numpy(),
                index=benign.account_id
            )
        )
        .fillna("none")
    )

    lookalikes = {}

    for code in (
        "B01", "B02", "B03", "B04"
    ):
        mask = (
            te
            & (
                benign_type.to_numpy()
                == code
            )
        )

        n = int(np.sum(mask))

        if n == 0:
            raise ValueError(
                f"No held-out accounts for {code}"
            )

        assert np.sum(y[mask]) == 0

        lookalikes[code] = {
            "accounts": n,
            "ml_false_positive_rate": float(
                np.mean(
                    scores[mask] >= ml_threshold
                )
            ),
            "rules_false_positive_rate": float(
                np.mean(
                    baseline[mask]
                    >= rule_threshold
                )
            ),
        }

    # Dashboard export: no hidden target labels.
    dashboard = pd.DataFrame({
        "account_id": df.account_id.to_numpy(),
        "risk_score": scores,
        "baseline_rule_score": baseline,
        "ml_alert": scores >= ml_threshold,
        "baseline_alert": (
            baseline >= rule_threshold
        ),
    })

    dashboard.to_parquet(
        args.output
        / f"dashboard_{args.task}.parquet",
        index=False
    )

    # Private evaluation output; never expose
    # synthetic ground truth as an inference input.
    holdout = pd.DataFrame({
        "account_id": (
            df.loc[te, "account_id"].to_numpy()
        ),
        "actual_synthetic_label": y[te],
        "risk_score": scores[te],
        "baseline_rule_score": baseline[te],
        "lookalike_type": (
            benign_type.loc[te].to_numpy()
        ),
    })

    holdout.to_parquet(
        args.output
        / f"test_{args.task}_private.parquet",
        index=False
    )

    model.booster_.save_model(
        str(
            args.output
            / f"{args.task}_lightgbm.txt"
        )
    )

    importance = pd.DataFrame({
        "feature": x.columns,
        "importance": (
            model.booster_.feature_importance(
                importance_type="gain"
            )
        ),
    }).sort_values(
        "importance", ascending=False
    )

    importance.to_csv(
        args.output
        / f"{args.task}_feature_importance.csv",
        index=False
    )

    report = {
        "task": args.task,
        "target": target,
        "dataset": (
            "Q-Guard 1M Risk v3, "
            "full-90-day retrospective features"
        ),
        "model": "LightGBM binary CPU",
        "cpu_threads": args.threads,
        "best_iteration": int(
            model.best_iteration_
        ),
        "train_count": int(tr.sum()),
        "validation_count": int(va.sum()),
        "test_count": int(te.sum()),
        "synthetic_test_results": {
            "model": ml_results,
            "rules": rule_results,
        },
        "benign_lookalike_test": lookalikes,
        "thresholds_tuned_only_on": (
            "validation"
        ),
        "limitations": [
            "Not real-world validated",
            "Uncalibrated model scores",
            "Retrospective 90-day detection",
        ],
    }

    (
        args.output
        / f"metrics_{args.task}.json"
    ).write_text(
        json.dumps(report, indent=2)
    )

    print(
        f"[PASS] {args.task} "
        "model and exports saved",
        flush=True
    )


if __name__ == "__main__":
    main()
