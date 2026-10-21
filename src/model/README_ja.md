# src/model — ネットワーク定義

すべてのニューラルネットワーク定義。提案 **STGCLSATT** エンコーダ、複数アクションを統合
する 3 種の Fusion モデル、エンコーダファクトリ、移植ベースラインを含む。エンコーダは
GCN → RNN → attention のスタックで、各段は `config/model_ablation/` のモデルアブレーション
で差し替えられるようそれぞれサブフォルダ（`gcn/`・`rnn/`・`attention/`）に分かれる。

## ファイル

| ファイル | 役割 | 出典 |
|---|---|---|
| `stgclsatt.py` | `STGCLSATT` — 提案する空間時間グラフ + LSTM + attention エンコーダ。自作モジュール変種（STGCN / STGCNAtt / STGCLSTM / …）もパラメータで表現。 | 提案手法（本プロジェクト, TNSRE 2026）。ST-GCN・AGC-LSTM が基盤（`gcn/`・`rnn/` 参照）。 |
| `early_fusion.py` | Early fusion。単一エンコーダの前段で複数アクションを入力段で統合（チャンネル結合 / グラフ拡張）。隣接行列可視化の要素ばらつきは、行列診断用の母集団的な記述統計として扱う。 | 提案手法（本プロジェクト）。 |
| `middle_fusion.py` | Middle fusion。アクションごとのエンコーダを中間特徴段で融合。 | 提案手法（本プロジェクト）。 |
| `late_fusion.py` | Late fusion。アクションごとのエンコーダを特徴/確率レベルで平均により統合。 | 提案手法（本プロジェクト）。 |
| `build_model.py` | エンコーダファクトリ。`config["model"]["name"]` をエンコーダクラスに解決し、ベースラインを判定（`is_baseline`）。 | — |
| `mask_token.py` | Fusion モデルで欠損アクションのマスクを扱う共通ヘルパ: zero padding、pad_mask（存在サンプルのみ順伝搬）、modality dropout。 | 提案手法（本プロジェクト）。 |

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`gcn/`](gcn/) | 空間時間グラフ畳み込みサブネット（ST-GCN）。 |
| [`rnn/`](rnn/) | 再帰サブネット。LSTM・AGC-LSTM。 |
| [`attention/`](attention/) | attention サブネットと CAM/GradCAM の融合可視化。 |
| [`baselines/`](baselines/) | 外部ベースライン 2 モデルの PyTorch 移植。 |
| [`params/`](params/) | モデルパラメータヘルパ（fusion 用のチャンネル分割）。 |

## ベースラインと元論文

ベースラインは `model.name` で選択し、`config/model_ablation/` で設定する。

| モデル | 出典 |
|---|---|
| LITEMV | DeepRehabPile（Ismail-Fawaz et al., arXiv:2409.02869） |
| ConvTran | DeepRehabPile（Foumani et al., DMKD 2024） |
