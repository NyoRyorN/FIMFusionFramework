# src/model/params — Model parameter helpers

In-house helpers that derive model wiring parameters from the config. Used by the fusion
models (via `training/run.py`) to split multi-action inputs correctly.

## Files

| File | Role |
|---|---|
| `separate_channels.py` | Computes how input channels are separated per action for the GCN / fusion structure (`separate_channels`). |
