# src/model/baselines — 移植ベースラインモデル

外部ベースライン 2 モデルの PyTorch 移植。`model.name` で選択し、`config/model_ablation/`
で設定する。いずれも `fusion.type: late_probability` で動作し（ベースラインでは
early / middle は拒否される）、GradCAM は非対応。各ファイルのモジュール docstring
に公式リポジトリと移植時の変更点を記録している。

## ファイル

| ファイル | モデル | 元論文 / リポジトリ |
|---|---|---|
| `litemv.py` | LITEMV (LITE MultiVariate) | DeepRehabPile — Ismail-Fawaz et al., arXiv:2409.02869（github.com/MSD-IRIMAS/DeepRehabPile） |
| `convtran.py` | ConvTran | Foumani et al., DMKD 2024（github.com/Navidfoumani/ConvTran） |

## 実行方法

```bash
source path.sh
python experiments/exam_fusion.py -c config/model_ablation/litemv.yaml
python experiments/exam_fusion.py -c config/model_ablation/convtran.yaml
```
