# QLYNX — Anti-Money Laundering, Fraud & Player Protection

**Ready-to-run Streamlit project for Windows**, containing 18 authentic trained artifacts
(10 LightGBM, 2 neural V4, 6 neural V5), 130 demo accounts, and a 20,000-case synthetic investigation queue.

## Launch — no manual code setup

1. Extract the downloaded **QLYNX_READY_TO_RUN.zip** completely.
2. Open its `qguard_dashboard_v5` folder.
3. **Double-click `RUN_QLYNX.bat`.**

The script creates a Python virtual environment and installs packages automatically on first launch.
Python 3.11 or 3.12 and internet access for the first dependency installation are needed.
Open **http://localhost:8501** if your browser does not open automatically.

## Optional direct start (VS Code / Conda)

```powershell
conda create -n qlynx python=3.11 -y
conda activate qlynx
python -m pip install -r requirements-dashboard.txt
python -m streamlit run main.py
```

## Test path for your demo

- **Command Centre**: display the 130 synthetic demo cases, or choose the exported 20K queue.
- **Data Intake**: click **Run fraud + AML + player protection** with no CSV selected to run the 3 operational LightGBM models. Expected: 48/130 flagged.
- **Model Laboratory**: run all 10 LightGBM models, then run V4 and V5 quantum/classical checkpoints on sample ID `A0108836`.
- **LLM Investigator Copilot**: generate a case-note using a local Ollama model, or use deterministic fallback.
- **Decisions & Audit**: human-reviewed actions logged locally to `audit/decisions.jsonl`.

## Optional local LLM — Ollama

Install Ollama and run `ollama pull qwen2.5:3b`. On Windows, Ollama normally starts as a service. If no server is running, start `ollama serve` in another console.
Open `LLM Investigator Copilot` and click **Generate with Ollama**.
No LLM fine-tuning or external API key is required; the LLM writes case notes and never changes predictions.

## Full 1.2 million account V5 inference using your original ZIP

The compact build is intended for hackathon demos. To enable V5 predictions on *any account already in the frozen 1.2M release*, copy the `inference_inputs/` and `predictions/` folders from your uploaded original `qguard_dashboard_v5_complete_20261016.zip` into this same `qguard_dashboard_v5/` folder. The adapter automatically switches to the verified original V5 known-account inference script, including checkpoint/prediction checks. The `data/risk_scores_1m.parquet` is optional and not required for a live demo; 20K queue remains available.

## Important limitations

Model scores are **not calibrated probabilities of harm/crime**. The six-qubit model uses a classical state-vector simulator. The dataset is entirely synthetic, and arbitrary new raw transactions **cannot** be scored without matching feature preprocessing and graph inputs. No actual account suspension occurs. A model attribution for LightGBM is *not* an explanation of a quantum VQC output.


## QLYNX UI Refresh — Risk Colors & Live Graph

- Navigation buttons now fill the sidebar width with a compact, active-page highlight.
- **Fraud = red**, **AML = orange**, **Player Protection = green** in overview cards, queue tables, and account score cards.
- The investigation queue computes **Critical / High / Review / Watch / Routine** triage from each task's saved score threshold. This only changes display priority; it never changes risk model predictions or operational alert flags.
- **Relationship Explorer** has two modes:
  1. **Aggregate evidence**: automatically creates an interactive overview for a chosen account, based on available counts such as distinct devices, networks, tokens and shared-device indicators. Dotted edges indicate a summary category, NOT a real account-to-device relationship.
  2. **Observed account-resource network**: upload a sanitized CSV with exact columns `account_id,entity_type,entity_id`, where `entity_type` is `device`, `network` or `payment`. The graph builds from actual uploaded edges, showing the selected account and limited 2-hop shared-resource neighbours. The original model ZIP contains no raw account-to-resource edge list.

To launch: `RUN_QLYNX.bat` on Windows or `python -m streamlit run main.py` in a correctly configured Python environment. The launcher checks NetworkX and installs requirements when missing.
