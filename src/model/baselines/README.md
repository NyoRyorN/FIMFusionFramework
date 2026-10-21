# src/model/baselines — Ported baseline models

PyTorch ports of two external baseline models. Selected by `model.name` and configured in
`config/model_ablation/`. All run with `fusion.type: late_probability` (early and middle
fusion are rejected for baselines); GradCAM is not supported for baselines. Each file's module
docstring records the official source repo and the adaptations made during porting.

## Files

| File | Model | Source paper / repo |
|---|---|---|
| `litemv.py` | LITEMV (LITE MultiVariate) | DeepRehabPile — Ismail-Fawaz et al., arXiv:2409.02869 (github.com/MSD-IRIMAS/DeepRehabPile) |
| `convtran.py` | ConvTran | Foumani et al., DMKD 2024 (github.com/Navidfoumani/ConvTran) |

## Run

```bash
source path.sh
python experiments/exam_fusion.py -c config/model_ablation/litemv.yaml
python experiments/exam_fusion.py -c config/model_ablation/convtran.yaml
```
