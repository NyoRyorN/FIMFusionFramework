# src — コアライブラリ（モデル・データ・学習・評価）

`experiments/` の実験エントリスクリプトが import するすべてを収めた Python パッケージ。
ニューラルネットワーク定義・骨格データパイプライン・学習/評価ループ・後処理/可視化
ヘルパを含む。ここのファイルを直接実験実行することはなく、`experiments/exam_fusion.py`
と `experiments/exam_each_seed.py` がこれらのモジュールを組み合わせて利用する。

以下の各サブフォルダは、ファイル一覧を記した `README.md` / `README_ja.md` を持つ。

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`data/`](data/) | 骨格データパイプライン。生データ前処理・HDF5 データセット生成・PyTorch `Dataset`/`DataLoader` フィーダ。 |
| [`model/`](model/) | ネットワーク定義。提案 STGCLSATT エンコーダ・Early/Middle/Late Fusion モデル・GCN/RNN/attention サブネット・移植ベースライン。 |
| [`training/`](training/) | 学習・評価ループ管理（`Train` / `Test` / `Run`）。両実験エントリから利用される。 |
| [`plot/`](plot/) | 評価指標計算と図生成（クラス別ボックスプロット・指標計算・サマリ CSV 出力）。 |
| [`utils/`](utils/) | 実験ユーティリティ。名称/ラベル定義・プロットフォント・混同行列・実験後削除。 |

## トップレベルスクリプト

| ファイル | 役割 |
|---|---|
| `ensemble_mean.py` | 複数実験の予測を平均してアンサンブル精度を算出する。 |
| `results_process.py` | 実験後処理パイプライン（不要ファイル削除 → サマリ出力 → ボックスプロット）。 |
| `results_process_batch.py` | 特定ディレクトリを対象とした実験結果後処理。 |
