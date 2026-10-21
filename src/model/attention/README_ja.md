# src/model/attention — attention サブネットと CAM/GradCAM 融合可視化

提案エンコーダの attention 段と、fusion 予測を解釈するためのクラス活性化マップ可視化を
含む。attention は 2 モードあり、モデル変種に応じて選択される。**ST-sequential**
（空間 → 時間）と **ST-joint**（空間時間を同時に扱う）。基本 attention ブロックは
`components/` に置く。

**出典:** attention サブネットは提案手法（本プロジェクト）の一部。CAM/GradCAM 可視化は
Grad-CAM（Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization," ICCV 2017）に準拠する。

## ファイル

| ファイル | 役割 |
|---|---|
| `attention_part.py` | `Attention_part` / `Attention_part_fusion`。GCN+RNN 特徴に対する attention ヘッド。重みの凍結と、解釈のための attention 重み出力に対応。 |
| `cam_fusion.py` | 複数アクション fusion 向けに空間 (T, V) 次元で行う融合前 Grad-CAM / Class Activation Mapping。`reports/skeleton_plots/` で可視化される重みを生成する。 |

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`components/`](components/) | attention 構成要素。基本 attention・ST-joint attention。 |
