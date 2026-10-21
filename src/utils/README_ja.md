# src/utils — 実験ユーティリティ

パイプライン全体で共有される横断的ヘルパ。正規の名称/ラベル定義、プロットフォント設定、
混同行列描画、実験後のクリーンアップ。

## ファイル

| ファイル | 役割 |
|---|---|
| `names.py` | 正規定義。アクション `Tasks()`・`FIMitems()`・`SimpleJoints()`。config やデータセットで使うインデックス/ラベルの単一の情報源。 |
| `decides.py` | 判定/選択ヘルパ（config から実験オプションを解決するなど）。 |
| `make_confusion_matrix.py` | 予測/真ラベルから混同行列を構築・描画する。 |
| `set_plt_font.py` | 図を統一するために matplotlib フォントを設定する。 |
| `delete_after.py` | `DeleteAfter`。実験後に中間ファイルを削除するクリーンアップヘルパ。 |
