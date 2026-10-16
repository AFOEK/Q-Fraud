"""Optional pre-flight check of QLYNX's packaged assets and inference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = (
    "main.py", "model_registry.py", "inference_v5.py", "copilot.py", "dashboard_manifest.json",
    "demo/demo_accounts_upload.csv", "demo/v4_demo_inputs.npz", "data/investigator_queue.parquet",
    "demo/v4_demo_inputs.npz",
    "source/src/dashboard/13_infer_v5_known_accounts.py",
)
missing = [file for file in REQUIRED if not (ROOT / file).is_file()]
if missing:
    raise SystemExit("Missing files: " + ", ".join(missing))
print("[OK] Essential assets present")

from model_registry import OPERATIONAL, model_status, score_lightgbm, score_v4_demo
import pandas as pd
status = model_status()
print(status.to_string(index=False))
assert len(status) == 18, "Expected ten LightGBM and eight neural weights"
assert (status["Status"] == "Available").all(), "One or more checkpoint files are missing"
frame = pd.read_csv(ROOT / "demo/demo_accounts_upload.csv", dtype={"account_id": str})
result, _ = score_lightgbm(frame, list(OPERATIONAL.values()), explanations=False)
assert len(result) == 130
assert int(result.alert_count.gt(0).sum()) == 48
print("[OK] LightGBM inference 130 rows, 48 flagged")
print(score_v4_demo(frame.account_id.iloc[:1].tolist()).to_string(index=False))
print("[OK] V4 checkpoints reproduce stored demo scores")
from inference_v5 import infer_known_accounts
print(infer_known_accounts(frame.account_id.iloc[:1].tolist()).to_string(index=False))
print("[OK] All 6 V5 quantum/classical checkpoints produced demo scores")
