# src/model/attention/components — Attention building blocks

The primitive attention modules that `attention_part.py` composes into the encoder's
attention stage. Part of the proposed method (this project).

## Files

| File | Role |
|---|---|
| `attention.py` | Base attention module: computes attention weights and outputs the attended features. |
| `st_joint_attention.py` | ST-joint attention: joint spatial-temporal attention that integrates over frames and joints together. |
