# Supplied checkpoint integration — IMPORTANT

This package includes the **verified uploaded quantum fusion model** (`checkpoints/Quantum_best.pt`), a **classical fusion-only baseline** (`checkpoints/classical_fusion_best.pt`) and uploaded validation CSV reports (`reports/`). The uploaded file named `classical_best.pt` **was not an encoder checkpoint**: its state dict contained only `fusion.net.*` and `decoder.classifier.*` keys. In particular, it had **no `temporal.*` or `graph_encoder.*` weights**. The correct encoder checkpoint **is not included**.

**To recover it from your still-running training notebook:** after your trained `classical_model` is loaded with the best encoder weights, run:

```python
from pathlib import Path
import torch

out = Path('checkpoints/encoder_best.pt')
out.parent.mkdir(parents=True, exist_ok=True)
encoder_state = classical_model.state_dict()
assert any(k.startswith('temporal.') for k in encoder_state)
assert any(k.startswith('graph_encoder.') for k in encoder_state)
torch.save(encoder_state, out)
```

Copy `encoder_best.pt` into this dashboard's `checkpoints/` directory. If the notebook kernel is no longer alive, locate a backup/full checkpoint, or retrain the encoder; saved cached embeddings can support a limited fixed-dataset export but not general inference on unseen accounts.

Why this likely happened: on Windows, `classical_best.pt` and `Classical_best.pt` are the **same path** because names are case-insensitive. Saving the fusion head later may overwrite the full encoder checkpoint. Use unambiguous filenames: `encoder_best.pt`, `classical_fusion_best.pt`, and `Quantum_best.pt`.

**Still missing:** raw/ dataset files for actual account-by-account prediction (or a separately generated real model case export). Aggregate model CSV reports do not include individual case scores.

The app shows the uploaded validation metrics while the encoder is missing, and disables trained-model inference until an encoder checkpoint can be verified. Neither model is a trained fraud/AML action classifier.

---

# Q-Guard trained-model investigator dashboard

This edition **does not show fabricated demo scores**. It can reconstruct the 50K generator's observable features, load the **actual trained** GRU + heterogeneous GraphSAGE + four-qubit PennyLane VQC checkpoints, score account-link candidates and display an evidence-backed review queue.

**Model scope:** Q-Guard currently learns *common account ownership*, **not fraud, AML or safer-gambling scenarios**. It does not have a trained six-scenario risk head. The dashboard suggests review actions and records human decisions; it does not automatically ban, suspend, restrict or file a report. NLP drafts are not evidence of wrongdoing.

## Expected layout

```text
qguard_model_connected/
├── app.py
├── inference.py
├── requirements.txt
├── checkpoints/
│   ├── classical_best.pt    # trained GRU + GraphSAGE checkpoint
│   └── Quantum_best.pt      # trained VQC + pair classifier checkpoint
├── exports/                 # real model predictions appear here
└── audit/
```

Dataset root can be somewhere else on your computer:

```text
/path/to/risk_v3/
├── manifest.json
├── raw/
│   ├── accounts.parquet
│   ├── transactions.parquet
│   ├── sessions.parquet
│   ├── payment_events.parquet
│   └── networks.parquet
└── private/                  # Not used by inference
```

The features and `MAX_SEQ_LEN=30` match the source `main.ipynb` the user provided. The VQC depth is inferred from `Quantum_best.pt`, so 2- and 3-layer weight shapes can be loaded. The decoder supports either the no-dropout or inserted-dropout model state dictionary. If the latest training notebook uses different activations, feature preprocessing, embedding dimensions or circuit architecture, update `inference.py` to match it **exactly** before interpreting scores.

## Install and run

Run in the Conda environment that already executes your notebook (or install required dependencies):

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The sidebar has a **Run trained GRU + GraphSAGE + VQC** button. Point it at the dataset and checkpoints. `max_candidates=2000` is a demo-time partial scan of the higher-evidence candidates. Set `0` for all discovered candidates (may run considerably longer).

For offline command-line inference:

```bash
python inference.py --data /path/to/risk_v3 --checkpoints ./checkpoints --out ./exports/cases.csv --max-candidates 2000
streamlit run app.py
```

The inference script reads only `raw/` and `manifest.json`, **never private/ ownership or scenario truth**. It constructs the account-feature tensors and relationship graph, recovers the trained network weights, and scores observable candidate pairs.

`exports/inference_manifest.json` records how many candidates were scored. The output includes source label, account IDs, linkage model score, shared resource evidence and limited transaction-activity summaries.

**Important:** The output is an *uncalibrated common-ownership link score*, not a calibrated fraud probability. The displayed action is a deterministic review recommendation, not another trained decision model. Suspected fraud/AML and policy enforcement need additional ground-truth scenario labels, separate model training, legal/policy decision rules and authorized human approval.

## Optional NLP via local Ollama

```bash
ollama pull qwen2.5:3b
ollama serve
```

Enable `Use local Ollama NLP` in the sidebar. The model gets only a structured evidence packet of observable relationships and transaction summaries. The note-generator is prompted to mention innocent explanations, not invent evidence, and not order punitive actions. Users must verify the generated output.

## Deployment limits

This implementation reproduces the supplied notebook's CPU/PyG full-graph computation and can be resource intensive even at 50K. It is **not** a production-scale inference engine for the proposed 1M-person population. At 1M use partitioned candidate tables, out-of-core preprocessing, graph neighborhood sampling and batch embedding materialization. The JSONL audit storage is for synthetic hackathon demonstration only, without authorization or tamper protection.
