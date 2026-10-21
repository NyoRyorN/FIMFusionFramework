# src/data/preprocessing/order — 生ファイル/ディレクトリ整理スクリプト

生の NEDO データツリーを整理・再構成する単機能スクリプト群。`../orderliness_data.py` から
統括的に呼び出され、それぞれが 1 つのファイル/ディレクトリ操作を担う。

## ファイル

| ファイル | 役割 |
|---|---|
| `change_dir_name.py` | CSV マッピングで指定したデータディレクトリをリネームする。 |
| `change_task_name.py` | CSV マッピングで指定したタスクファイルをリネームする。 |
| `delete_all_parts.py` | データセットに不要な all-parts データファイルを削除する。 |
| `delete_duplicate_files.py` | 重複ファイルを検出して削除する。 |
| `delete_some_parts.py` | CSV ファイルから指定フレーム範囲を削除し、特定区間を除去する。 |
| `delete_unused_files.py` | NEDO データディレクトリから不要ファイルを削除する。 |
| `move_delete_dirs.py` | NEDO ツリー内の不要ディレクトリを移動/削除する。 |
| `move_files_each_people.py` | NEDO ファイルを被験者ごとのディレクトリに集約する。 |
| `move_not_fim_file.py` | 対応する FIM 記録のないファイルを専用の場所へ移動する。 |
| `rename.py` | NEDO のファイル名を英語表記に標準化する。 |
