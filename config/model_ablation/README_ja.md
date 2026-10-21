# model_ablation — モデル / モジュールのアブレーションとベースライン

このフォルダの config は、パイプラインの他部分を固定したまま **どのエンコーダ
モデルを使うか** を比較する。`model` セクション以外のキーはすべて同一で、
1 つのアブレーションバッチとして実行される。いずれも
`experiments/exam_fusion.py` で実行する。

提案モデルのモジュール変種（`model.name` と attention `mode` で選択）と、2 つの
ベースラインモデルを含む。

## 提案モデルのモジュール変種

| Config | `model.name` | Attention mode |
|---|---|---|
| `stgclsatt.yaml` | STGCLSAtt（提案手法） | ST-sequential |
| `stam.yaml` | STGCNAtt | ST-sequential |
| `stgcn.yaml` | STGCN | ST-sequential |
| `stgcn_agcbilstm.yaml` | STGCAGLS | ST-sequential |
| `stgcn_bilstm.yaml` | STGCLSTM | ST-joint |
| `stgcn_bilstm_stjoint.yaml` | STGCLSAtt | ST-joint |
| `stgcn_stjoint.yaml` | STGCNAtt | ST-joint |

## ベースライン

`src/model/baselines/` に PyTorch 移植。いずれも
`fusion.type: late_probability` を使用し（ベースラインでは early / middle は
拒否される）、GradCAM は非対応。

| Config | `model.name` | 出典 |
|---|---|---|
| `litemv.yaml` | LITEMV | DeepRehabPile（Ismail-Fawaz et al., arXiv:2409.02869） |
| `convtran.yaml` | ConvTran | DeepRehabPile（Foumani et al., DMKD 2024） |

## 実行方法

```bash
source path.sh

# 提案モデルのモジュール変種
python experiments/exam_fusion.py -c config/model_ablation/stgclsatt.yaml
python experiments/exam_fusion.py -c config/model_ablation/stam.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_agcbilstm.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_bilstm.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_bilstm_stjoint.yaml
python experiments/exam_fusion.py -c config/model_ablation/stgcn_stjoint.yaml

# ベースライン
python experiments/exam_fusion.py -c config/model_ablation/litemv.yaml
python experiments/exam_fusion.py -c config/model_ablation/convtran.yaml
```
