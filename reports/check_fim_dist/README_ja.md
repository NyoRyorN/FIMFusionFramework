# reports/check_fim_dist — FIM 分布チェック

データセット全体における FIM スコア分布のチェックと出力。

## ファイル

| ファイル | 役割 |
|---|---|
| `fimdist_outputs.py` | 項目ごとの FIM スコア分布を集計・出力する（`src/utils/names.py` の `FIMitems()` を使用）。 |
| `missing_motion_class_view.py` | 指定した1つの FIM 項目について `data/Distributions/MissingMotionCounts.csv` のスコア 1-7 行を任意クラス（例: `1-6,7`）に統合して表示する。動作ごとの件数・欠損率に加え、`MotionStatus.csv` + `all_fim.csv` から算出した「少なくとも1動作欠損」の被験者数（`AnyMissing` 行）も出力する。出力全体が `--any-tasks` で指定した対象動作（デフォルト全11動作）に限定される。詳細は `docs/missing_motion_ja.md` を参照。 |
