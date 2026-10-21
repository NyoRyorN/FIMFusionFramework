# src/training — Training and evaluation loop management

The training/evaluation control layer that both experiment entrypoints build on. `Train` and
`Test` implement the per-epoch loops; `Run` orchestrates an experiment end-to-end — building
either a single encoder or an Early/Middle/Late fusion model (via `build_model.py`), then
training and evaluating it.

**Used by:**
- `experiments/exam_each_seed.py` — single-action, multi-seed experiments (single-encoder config).
- `experiments/exam_fusion.py` — multi-action fusion experiments (fusion config).

## Files

| File | Role |
|---|---|
| `train.py` | `Train`: per-epoch training loop (loss, optimizer step, metric logging). |
| `test.py` | `Test`: evaluation loop; computes metrics, confusion matrices, and GradCAM weights (skeleton pre-fusion). |
| `run.py` | `Run`: orchestrates one experiment — build model (single encoder, or Early/Middle/Late fusion dispatched by `config["fusion"]["type"]`) → train → test. |
