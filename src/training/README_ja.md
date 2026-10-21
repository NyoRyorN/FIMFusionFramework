# src/training — 学習・評価ループ管理

両実験エントリが基盤とする学習/評価制御層。`Train` と `Test` がエポックごとのループを実装
し、`Run` が実験全体を統括する（単一エンコーダ、または `build_model.py` 経由で構築する
Early/Middle/Late Fusion モデルのいずれかを構築し、学習 → 評価まで行う）。

**利用元:**
- `experiments/exam_each_seed.py` — 単一アクション・複数 seed 実験（単一エンコーダ用 config）。
- `experiments/exam_fusion.py` — 複数アクション fusion 実験（fusion 用 config）。

## ファイル

| ファイル | 役割 |
|---|---|
| `train.py` | `Train`。エポックごとの学習ループ(損失・オプティマイザ更新・指標ログ)。 |
| `test.py` | `Test`。評価ループ。指標・混同行列・GradCAM 重み(skeleton pre-fusion)を計算する。 |
| `run.py` | `Run`。1つの実験を統括(モデル構築 — 単一エンコーダ、または `config["fusion"]["type"]` で分岐する Early/Middle/Late Fusion — → 学習 → 評価)。 |
