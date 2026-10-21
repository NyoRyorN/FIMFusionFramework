# reports — 結果集計と可視化

`results/` の実験出力に対する事後分析と図生成。実験終了後に手動で実行するスクリプト群で、
モデル学習は行わない。各サブフォルダは `README.md` / `README_ja.md` を持つ。

## トップレベルスクリプト

| ファイル | 役割 |
|---|---|
| `combine_results.py` | 複数パスの結果 CSV（対話入力）を 1 つの表に統合する。 |
| `ablation_process.py` | 一連の実験ディレクトリに対するアブレーション後処理（集計 → 描画）を駆動する。 |

## サブフォルダ

| サブフォルダ | 役割 |
|---|---|
| [`calculates/`](calculates/) | fold ごとの `PredictTruelabel.csv` からの被験者横断変数の集計。 |
| [`ablation_boxplot/`](ablation_boxplot/) | 被験者横断サマリからのアブレーション box/violin プロットとクラス別プロット。 |
| [`check_fim_dist/`](check_fim_dist/) | FIM スコア分布のチェック/出力。 |
| [`skeleton_plots/`](skeleton_plots/) | 骨格描画と Attention/GradCAM 可視化（動画・サブプロット・分布統計）。 |
