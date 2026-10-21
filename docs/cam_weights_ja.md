# CAM Weight 可視化パイプライン

このドキュメントでは、`reports/skeleton_plots/` 配下の CAM / GradCAM 重み可視化スクリプトを説明します。

- `makeWeightPlots.py`
- `makeWeightPlotsBulk.py`
- `makeWeightPlotsBulkCompact.py`
- `makeWeightDists.py`
- `plot2D.py`
- `plot3D.py`

## 概要

実験時に重みを出力しておくと、これらのスクリプトで「どの骨格点が予測に強く寄与したか」を可視化できます。

1. **フレーム単位の骨格動画 / 画像** - 1 サンプルの 2D / 3D 骨格系列に重みを重ねて描画します。
2. **Fusion 用の一括表示動画 / 画像** - 1 被験者に対して、利用可能な複数動作を 1 つの 2D / 3D 図にまとめて表示します。
3. **GradCAM 代表値 CSV** - GradCAM 重みを action、joint、身体部位 group の単位で集計し、seed 単位 CSV として保存します。

対応する実験 entrypoint はスクリプトごとに異なります。

| スクリプト | 対応する実験出力 |
|---|---|
| `makeWeightPlots.py` | `experiments/exam_each_seed.py` と `experiments/exam_fusion.py` |
| `makeWeightPlotsBulk.py` | `experiments/exam_fusion.py` のみ |
| `makeWeightPlotsBulkCompact.py` | `experiments/exam_fusion.py` のみ |
| `makeWeightDists.py` | `experiments/exam_fusion.py` のみ |

`plot2D.py` と `plot3D.py` は、`makeWeightPlots.py` から呼び出される低レベル描画関数です。
Bulk 系スクリプトは複数動作を同時に描画するため独自の combined 描画関数を持っていますが、
処理の考え方は同じです。`Graph(skl_type).neighbor_link` から骨格 graph の辺を取得し、
各フレームの関節重みを `Reds` カラーマップへ正規化して、MP4 またはフレーム単位 SVG として保存します。

---

## 実行前提

### 実験設定

実験時に重みを出力する設定にしておく必要があります。

```yaml
output_weights: true
model:
  cam_type: gradcam   # weights_CAM.h5 が必要な場合
```

Attention 重みは、選択した model が Attention 重みを返せる場合のみ保存されます。
GradCAM 重みは、`model.cam_type: gradcam` を指定し、model 側が GradCAM 経路を実行できる場合のみ保存されます。

### Fold 単位で必要なファイル

可視化スクリプトは、実験結果ディレクトリから以下を読み込みます。

| ファイル | 使用するスクリプト | 内容 |
|---|---|---|
| `weights_Att.h5` | `makeWeightPlots.py` のみ | Attention 重み。Bulk 系と分布可視化では未対応です。 |
| `weights_CAM.h5` | `makeWeightPlots.py` と、`exam_fusion.py` のみ対応の Bulk / 分布可視化スクリプト | GradCAM 重み。`makeWeightPlotsBulk.py`, `makeWeightPlotsBulkCompact.py`, `makeWeightDists.py` で必須です。 |
| `testDir.csv` | 動画 / 画像系 | test sample の `Directory` と開始 frame、または動作別開始 frame を対応づけます。`makeWeightDists.py` では使用しません。 |
| `PredictTruelabel.csv` | `makeWeightDists.py` | `Directory` と推論結果列を取得し、代表値 CSV に内部結合します。`Prob_` で始まる予測確率列は保存しません。 |
| 骨格 CSV | 動画 / 画像系 | 元骨格座標の `p_x.csv`, `p_y.csv`, `p_z.csv` です。 |

### スクリプトごとの対応範囲

`makeWeightPlotsBulk.py`, `makeWeightPlotsBulkCompact.py`, `makeWeightDists.py` は、現在の実装では GradCAM のみ、
かつ `exam_fusion.py` の出力のみに対応しています。これらは fusion 形式の `weights_CAM.h5` を読み込み、
`weights_Att.h5` や single-action の `exam_each_seed.py` 結果ディレクトリは処理しません。

`makeWeightPlots.py` は柔軟なスクリプトで、存在する場合は `weights.h5`, `weights_Att.h5`, `weights_CAM.h5` を順に処理します。
また、single-action 出力と fusion 出力の両方に対応する path 分岐を持っています。

### 骨格データの前提

現在の entrypoint は簡略 17 関節の骨格をデフォルトで使用します。

| 設定値 | デフォルト |
|---|---|
| `headers` | `SimpleJoints()` |
| `skl_type` | `simple_ntu-rgb+d` |
| 入力骨格データ | `data/ConciseData` |
| `is_center` | `False` |

コード内には full NTU layout や centered data 用の `plot_data_dict` もありますが、
現在の `__main__` ブロックでは simple skeleton 側に固定されています。

実行前に repository root で環境を読み込んでください。

```bash
source ./path.sh
```

---

## スクリプト

### `reports/skeleton_plots/makeWeightPlots.py` - 1 sample ごとの骨格重み可視化

各 test sample について、利用可能な重みファイルごとに 2D / 3D 動画または SVG frame 群を作成します。

**実行方法:**

```bash
python reports/skeleton_plots/makeWeightPlots.py
```

対話入力:

| 入力 | 内容 |
|---|---|
| root path | `Ex_...` で始まる実験 root。 |
| task | single-action mode のみ。動作 index `1--11`。 |
| item | FIM item index `0--12`。 |
| seed | `100` などの seed 文字列。 |
| K-Fold / fold index | root path に `kfold` が含まれる場合、または fusion prompt で `y` を選んだ場合に `Fold0` から `Fold4` を指定します。 |
| mode | `2D`, `3D`, `Both`。不正値の場合は `Both`。 |
| output format | `mp4`, `svg`。不正値の場合は `mp4`。 |

root path の文字列で処理対象のディレクトリ構造を切り替えます。

| root path 条件 | 処理対象 |
|---|---|
| `multi-model` または `Step` を含む | Fusion mode: `rootpath/<Item>/Stratified_Seed<seed>/[FoldN]/` |
| それ以外 | Single-action mode: `rootpath/<Task>_to_<Item>/Stratified_Seed<seed>/[FoldN]/` |

以下の重みファイルを順に探します。

| shift | 入力ファイル | 出力ディレクトリ prefix |
|---|---|---|
| 空文字 | `weights.h5` | `Weights/` |
| `_Att` | `weights_Att.h5` | `Weights_Att/` |
| `_CAM` | `weights_CAM.h5` | `Weights_CAM/` |

存在しない重みファイルは `No weights: <path>` を表示してスキップします。

---

### `reports/skeleton_plots/makeWeightPlotsBulk.py` - `exam_fusion.py` 用の combined GradCAM 可視化

`exam_fusion.py` の出力を対象に、1 test subject に対して `weights_CAM.h5` 内に存在する各動作を
1 つの frame 内に subplot としてまとめます。

**実行方法:**

```bash
python reports/skeleton_plots/makeWeightPlotsBulk.py
```

対話入力:

| 入力 | 内容 |
|---|---|
| root path | `Ex_...` で始まる `exam_fusion.py` の結果 root。 |
| item | FIM item index `0--12`。 |
| seed | `100` などの seed 文字列。 |
| K-Fold / fold index | 必要に応じて `Fold0` から `Fold4` を指定します。 |
| mode | `2D`, `3D`, `Both`。 |
| output format | `mp4`, `svg`。 |

このスクリプトは GradCAM のみ、かつ `experiments/exam_fusion.py` が出力する fusion result layout のみ対応です。
読み込む構造は以下です。

```text
rootpath/
└── <Item>/
    └── Stratified_Seed<seed>/
        └── [FoldN]/
            ├── testDir.csv
            └── weights_CAM.h5
```

各 test `Directory` について、HDF5 group の key を global action index として扱います。
対応する骨格系列を読み込み、`testDir.csv` の動作別開始 frame で切り出し、
その被験者内の全動作で共通の min/max color scale を使って描画します。

---

### `reports/skeleton_plots/makeWeightPlotsBulkCompact.py` - `exam_fusion.py` 用の compact combined GradCAM 可視化

`makeWeightPlotsBulk.py` と同じく GradCAM のみ、かつ `exam_fusion.py` の出力のみを対象にしますが、
より詰めた canvas に描画します。

| 違い | `makeWeightPlotsBulk.py` | `makeWeightPlotsBulkCompact.py` |
|---|---|---|
| 2D layout | 動作ごとに subplot | 1 つの axes に各動作を grid 配置 |
| 3D layout | 動作ごとに subplot | 1 つの 3D axes に各骨格を grid 配置 |
| centering の順序 | trim 前に centering | trim 後、描画対象の先頭 frame で centering |
| 処理範囲 | 単一 item/seed/fold | 3 モード: item ディレクトリ内の全 seed、root 以下の全 seed、対話入力による単一指定 |

subplot 版では余白が大きい場合や、複数動作をより密に比較したい場合に使います。

**実行方法:**

```bash
python reports/skeleton_plots/makeWeightPlotsBulkCompact.py
```

起動直後にプロット設定を確認した後、モード選択を求めます。

| 入力 | 値 | 内容 |
|---|---|---|
| プロットモード | `2D`, `3D`, `Both`（デフォルト） | 生成する combined view を選択します。 |
| 出力形式 | `mp4`（デフォルト）, `svg` | 動画またはフレームごとの SVG シーケンスを生成します。 |

| モード | 入力 | 内容 |
|---|---|---|
| **1** | item ディレクトリパス | `Stratified_Seed*` を直下に持つ親ディレクトリ。見つかった全 seed を順番に処理します。K-Fold は seed ごとに自動検出されます。 |
| **2** | root ディレクトリパス | 任意の祖先ディレクトリ。`weights_CAM.h5` を持つ（直下または `Fold*/` 内に）すべての `Stratified_Seed*` ディレクトリを再帰的に探して処理します。 |
| **3** | root → item → seed → fold の対話入力 | 元の対話モード。`exam_fusion.py` の結果 root、FIM item インデックス `0--12`、seed 文字列、オプションの fold インデックスを入力します。 |

モード 1・2 では K-Fold を**自動検出**します。`Stratified_Seed<seed>/` 直下に `Fold*` が存在すれば列挙して処理し、存在しなければ seed ディレクトリ自体を単一 fold として扱います。

**出力先:**

| モード | 出力先 |
|---|---|
| 1 または 2 | `Stratified_Seed<seed>/CAM_Animations/[FoldN/]<data_dir>_{2D,3D}_combined.{mp4,…}` |
| 3 | `rootpath/Weights_CAM/<Item>_Seed<seed>[_FoldN]/<data_dir>_{2D,3D}_combined.{mp4,…}` |

---

### `reports/skeleton_plots/makeWeightDists.py` - `exam_fusion.py` 用の GradCAM 分布解析

このスクリプトも `exam_fusion.py` 出力の GradCAM 重みを処理しますが、フレーム単位の骨格動画は生成しません。
代わりに、全 fold の subject 行を縦に結合し、seed 単位の未正規化代表値 CSV、
label group 別の action violin plot、正解 class 別の skeleton + body-part group violin plot を作成します。

**実行方法:**

```bash
python reports/skeleton_plots/makeWeightDists.py
```

起動直後にモード選択を求めます。

| モード | 入力 | 内容 |
|---|---|---|
| **1** | item ディレクトリパス | `Stratified_Seed*` を直下に持つディレクトリを指定します。配下のすべての seed を順に処理します。 |
| **2** | 任意の root ディレクトリパス | 指定ディレクトリ以下を再帰的に探索し、`weights_CAM.h5` を持つすべての `Stratified_Seed*` を処理します。 |
| **3** | root path → item → seed | 従来の対話モード。`exam_fusion.py` 結果 root、FIM item index `0--12`、seed 文字列の順に入力します。 |

K-Fold レイアウトは **自動検出** します。`Stratified_Seed<seed>/` 直下に `Fold*` サブディレクトリが存在すれば列挙・処理し、
存在しなければ seed directory 自体を単一 fold として扱います。K-Fold を問う対話入力はありません。

このスクリプトは GradCAM のみ、かつ `experiments/exam_fusion.py` が出力する fusion result layout のみ対応です。
`weights_CAM.h5` と `PredictTruelabel.csv` を読み込みます。subject 名は `PredictTruelabel.csv` の `Directory` から取得し、
`testDir.csv` は読み込みません。

動画系との主な違いは、重みを以下の粒度に集約する点です。

| 粒度 | 列の形 | 意味 |
|---|---|---|
| Action | `<Action>` | 1 subject / 1 action につき、全 frame・全 joint を集約した scalar。 |
| Joint | `<Action>_<Joint>` | 1 subject / 1 action / 1 joint につき、frame 方向を集約した scalar。 |
| Body-part group | `<Action>_<Group>` | 1 subject / 1 action / 1 body-part group につき、group 内 joint と frame を集約した scalar。 |

各粒度について `Max`, `Mean`, `Median` を算出し、`CAM_Dists` に代表値ごとの CSV を 3 本保存します。
CSV の値は未正規化のまま保存し、plot 用の min-max 正規化は可視化時だけ適用します。

---

## 内部処理

### Step 1 - 重みファイルの読み込み

`makeWeightPlots.py` は single-action の HDF5 dataset、または fusion の action index 別 group を読み込みます。

Bulk 系と分布可視化は、`exam_fusion.py` が出力する fusion 形式の `weights_CAM.h5` のみを読み込みます。

```text
weights_CAM.h5
└── <Directory>/
    ├── <ActionIndex>
    ├── <ActionIndex>
    └── ...
```

`exam_fusion.py` 出力では、action-index key を `src/utils/names.py` の `Tasks()` で動作名に変換します。

### Step 2 - 重みと骨格 frame の整合

重みは `(T, V)` に整形します。`T` は frame 数、`V` は joint 数です。
重み系列が既定 frame 長の `150` より短い場合は、行を 4 回繰り返して中央を切り出します。
これは `makeWeightPlots.py` で Attention 系の短い系列を扱うための処理です。

`makeWeightDists.py` ではこのパディングを **適用しません**。
GradCAM の生の出力長をそのまま使用することで、代表値統計が実際のモデル出力を反映します。

骨格座標は以下から読み込みます。

```text
<data_path>/<Directory>/<Task>/
├── p_x.csv
├── p_y.csv
└── p_z.csv
```

開始 frame は `testDir.csv` から取得します。

| 実験種別 | 開始 frame の列 |
|---|---|
| Single-action の `makeWeightPlots.py` | `FrameStart` |
| Fusion 系 | 動作名ごとの列 |

### Step 3 - 2D / 3D の frame-level plot

`plot2D.py` は `networkx.Graph` を作成し、各 joint を `(x, y)` に配置します。
`Graph(skl_type).neighbor_link` の辺を描画し、frame ごとの重みを `Reds` カラーマップへ変換します。

| format | 出力 |
|---|---|
| `mp4` | `<Task>_2D.mp4` |
| `svg` | `figures_2D/` 配下の frame 単位 SVG |

`plot3D.py` は同じ辺と色付けを使い、座標 `(x, z, y)` の 3D scatter として描画します。

| format | 出力 |
|---|---|
| `mp4` | `<Task>_3D.mp4` |
| `svg` | `figures_3D/` 配下の frame 単位 SVG |

Bulk 系は同等の combined 関数を使います。

| 関数 | 出力 |
|---|---|
| `PlotAllTasks2D_Combined` | `<Directory>_2D_combined.mp4` または `figures_2D_combined/frameNNN_2d_combined.svg` |
| `PlotAllTasks3D_Combined` | `<Directory>_3D_combined.mp4` または `figures_3D_combined/frameNNN_3d_combined.svg` |

### Step 4 - GradCAM 代表値 CSV の算出

`makeWeightDists.py` は各 fold の `PredictTruelabel.csv` から `Directory` を取得し、
`weights_CAM.h5` 内に同名 group が存在する subject だけを対象にします。
`testDir.csv` は使用しません。`PredictTruelabel.csv` に存在しても `weights_CAM.h5` に存在しない subject 行はスキップします。

各 subject / action の HDF5 配列は、まず `new_weights` として `(T, V)` に整形されます。

| 記号 | 意味 |
|---|---|
| `T` | GradCAM の生の出力 frame 数。パディングや切り出しは適用せずそのまま使用します。 |
| `V` | 骨格 joint 数。現在の entrypoint では `SimpleJoints()` がデフォルトです。 |

frame 長の調整やパディングは行いません。NaN を明示的に除外する処理もないため、
`new_weights` に NaN が含まれる場合は保存される代表値にも NaN が伝播します。

3 つの代表値は、いずれも調整後の `new_weights` から正規化なしで計算されます。

| 統計量 | 算出内容 | 出力ファイル |
|---|---|---|
| `Max` | 集約対象内で最大の GradCAM 値です。最も強く反応した 1 点を表します。 | `GradCAM_Max.csv` |
| `Mean` | 集約対象全体の算術平均です。全体的な重みの大きさを表します。 | `GradCAM_Mean.csv` |
| `Median` | 必要に応じて集約対象を 1 次元に潰した後の中央値です。外れ値の影響を受けにくい中心値を表します。 | `GradCAM_Median.csv` |

各 CSV は、全 fold の subject 行を縦方向に結合した seed 単位の table です。
fold 間で subject は重複しない想定ですが、重複が見つかった場合も行は削除せず警告のみ出します。

| 列 prefix | 列名 | 1 subject / 1 action における集約対象 | 代表値の計算 |
|---|---|---|---|
| `Action_` | `Action_<Action>` | 全 frame・全 joint、つまり `(T, V)` 行列全体。 | `Max = max(new_weights.values)`, `Mean = mean(new_weights.values)`, `Median = median(new_weights.values)`。 |
| `Joint_` | `Joint_<Action>_<Joint>` | 1 joint の全 frame、つまり長さ `T` の vector。 | `Max = max(new_weights[Joint])`, `Mean = mean(new_weights[Joint])`, `Median = median(new_weights[Joint])`。 |
| `BodypartGroup_` | `BodypartGroup_<Action>_<Group>` | 1 身体部位 group に含まれる joint の全 frame、つまり `(T, |Group|)` の部分行列。 | `Max = max(new_weights[group_joints].values)`, `Mean = mean(new_weights[group_joints].values)`, `Median = median(new_weights[group_joints].values)`。 |

各 CSV の 1 行は `Directory` 1 件に対応し、CSV index 名は `Subject` です。
先頭列には各 fold の `PredictTruelabel.csv` に含まれる列から `Prob_` で始まる予測確率列を除いた列が付与され、
その後ろに Action / Joint / BodypartGroup の代表値列が続き、最終列に `Fold` 列が追加されます。
`Fold` 列には `Fold0` のような fold directory 名から取り出した番号を保存します。
CSV に保存される代表値は subject 間平均ではなく、各 subject ごとの未正規化値です。
保存直前に、`Directory` 値に相当する index を昇順に並べ替えます。

身体部位 group は以下です。

| Group | Joints |
|---|---|
| `LeftArm` | `ShoulderLeft`, `ElbowLeft`, `WristLeft` |
| `RightArm` | `ShoulderRight`, `ElbowRight`, `WristRight` |
| `Torso` | `SpineBase`, `SpineMid`, `SpineShoulder` |
| `LeftLeg` | `HipLeft`, `KneeLeft`, `AnkleLeft` |
| `RightLeg` | `HipRight`, `KneeRight`, `AnkleRight` |

実際の計算では、現在の `headers` に存在する joint だけを使用します。
選択中の骨格 layout に group 内 joint が 1 つも存在しない場合、その group はスキップされます。

### Step 5 - Action label group violin plot の作成

CSV 保存後、`Action_` 列から `GradCAM_Action_LabelGroup_Violin.png` と `.svg` を作成します。
`Max`, `Mean`, `Median` の action 代表値は、各 subject 行に対して全動作の `Action_` 列をまとめて
min-max 正規化した後に plot します。CSV に保存されている action 代表値自体は未正規化のままです。

Figure は `(True, Predicted) = (0, 0), (0, 1), (1, 0), (1, 1)` の 4 panel で、
正解群だけでなく誤分類群も残して表示します。各 panel では動作ごとに `Max`, `Mean`, `Median`
の seaborn violin を並べ、`inner_kws` を使って violin 内に boxplot を表示します。
さらに各 violin には subject 方向の平均を別 marker として重ねます。

### Step 6 - 正解 class 別 skeleton / violin plot の作成

`makeWeightDists.py` は CSV 保存後、同じ seed 単位 table から `GradCAM_Skeleton_{Max,Mean,Median}.png`
と `.svg` を作成します。画像は各統計量につき 1 枚で、列は各動作、行は以下の 4 段です。

| 行 | 内容 |
|---|---|
| 1 | `True == Predicted == 0` の subject だけを使った Class 0 の joint 平均 skeleton plot。 |
| 2 | 同じ Class 0 subject の body-part group violin plot。 |
| 3 | `True == Predicted == 1` の subject だけを使った Class 1 の joint 平均 skeleton plot。 |
| 4 | 同じ Class 1 subject の body-part group violin plot。 |

Joint plot では、まず各 subject 行に対して全動作・全 joint の `Joint_` 列をまとめて min-max 正規化します。
その後、各 class で正解している subject のみを抽出し、動作ごと・joint ごとに subject 方向の平均を取って
skeleton node color として描画します。色 range は全 class で共通にはせず、Class 0 行、Class 1 行の中で
それぞれの動作列に対して統一します。

Body-part group violin plot でも、各 subject 行に対して全動作・全 group の `BodypartGroup_` 列をまとめて
min-max 正規化します。各 seaborn violin は 1 subject を 1 点として扱う subject 方向の分布で、
`inner_kws` による boxplot を内部に表示し、その group の subject 平均を別 marker として重ねます。
BodyGroup ごとの violin 色は `reports/ablation_boxplot/ablation_cross_subject_variables.py` の
ablation plot に近い高コントラスト palette に合わせています。
x 軸の group 順は `BODYPART_GROUP_DISPLAY_ORDER` に従い、骨格の前面視での配置に合わせて
`RightArm`, `RightLeg`, `Torso`, `LeftLeg`, `LeftArm` の順（視聴者の左 = 被験者の右側）で並べます。

骨格の bone エッジは body-part group に対応した色で描画します。
両端の joint が **同じ** group に属するエッジはそのグループの palette 色を使い、
group をまたぐエッジ（例: ShoulderLeft → SpineShoulder）はグレーで描画します。
`nx.draw_networkx_edges` には `edgelist` を `Graph.neighbor_link` と同じ順序で明示的に渡し、
各色が正しい bone に対応するようにしています。

### Step 7 - Seed 単位 CSV / plot の保存

K-Fold の場合は `Stratified_Seed<seed>/` 直下にある `Fold*` ディレクトリの DataFrame を縦方向に結合してから保存します。
非 K-Fold の場合も同じ保存形式を使い、単一 seed directory の内容を `CAM_Dists` に保存します。
CSV と各種 violin / skeleton plot はどちらも同じ `CAM_Dists` に保存します。

---

## 出力ファイル

### `makeWeightPlots.py`

Single-action の出力:

```text
rootpath/
└── Weights[_Att|_CAM]/
    └── <Task>_to_<Item>_Seed<seed>[_FoldN]/
        └── <Directory>/
            ├── <TaskIndex>_<Task>_2D.mp4
            ├── <TaskIndex>_<Task>_3D.mp4
            └── <TaskIndex>_<Task>/
                ├── figures_2D/frameNNN.svg
                └── figures_3D/frameNNN_3d.svg
```

Fusion の出力:

```text
rootpath/
└── Weights[_Att|_CAM]/
    └── <Item>_Seed<seed>[_FoldN]/
        └── <Directory>/
            ├── <ActionIndex>_<Task>_2D.mp4
            ├── <ActionIndex>_<Task>_3D.mp4
            └── <ActionIndex>_<Task>/
                ├── figures_2D/frameNNN.svg
                └── figures_3D/frameNNN_3d.svg
```

`.mp4` は `output_fmt` が `mp4` の場合のみ作成されます。
`figures_*` ディレクトリは `output_fmt` が `svg` の場合のみ作成されます。

### `makeWeightPlotsBulk.py` / `makeWeightPlotsBulkCompact.py` (`exam_fusion.py` のみ)

```text
rootpath/
└── Weights_CAM/
    └── <Item>_Seed<seed>[_FoldN]/
        ├── <Directory>_2D_combined.mp4
        ├── <Directory>_3D_combined.mp4
        └── <Directory>/
            ├── figures_2D_combined/frameNNN_2d_combined.svg
            └── figures_3D_combined/frameNNN_3d_combined.svg
```

### `makeWeightDists.py` (`exam_fusion.py` のみ)

Seed 単位の出力:

```text
rootpath/
└── <Item>/
    └── Stratified_Seed<seed>/
        └── CAM_Dists/
            ├── GradCAM_Max.csv
            ├── GradCAM_Mean.csv
            ├── GradCAM_Median.csv
            ├── GradCAM_Action_LabelGroup_Violin.png
            ├── GradCAM_Action_LabelGroup_Violin.svg
            ├── GradCAM_Skeleton_Max.png
            ├── GradCAM_Skeleton_Max.svg
            ├── GradCAM_Skeleton_Mean.png
            ├── GradCAM_Skeleton_Mean.svg
            ├── GradCAM_Skeleton_Median.png
            └── GradCAM_Skeleton_Median.svg
```

K-Fold の場合も fold ごとの `Weight_Distributions/` は作成せず、全 fold の subject 行を縦に結合した
3 つの CSV、action label group violin plot、統計量別の skeleton / violin plot を `CAM_Dists/` に保存します。

---

## 実行時の処理と担当ファイル / 関数

| 処理 | ファイル / 関数 | 役割 |
|---|---|---|
| root, item, seed, fold, mode, output format の選択 | `makeWeightPlots.py::makePlotsSingle`, `makeWeightPlots.py::makePlotsFusion`, `makeWeightPlotsBulk.py::makePlotsFusionBulk`, `makeWeightPlotsBulkCompact.py::makePlotsFusionBulk`, `makeWeightDists.py::__main__` | 対話入力を受け取り、読み込み対象の result path を組み立てます。BulkCompact と Dists の `__main__` はどちらも item ディレクトリ / 再帰 root / 手動 item+seed の 3 モードを提供します。 |
| seed ディレクトリの解決（BulkCompact および Dists） | `makeWeightPlotsBulkCompact.py::_list_seed_dirs`, `makeWeightPlotsBulkCompact.py::_find_seed_dirs`, `makeWeightDists.py::_list_seed_dirs`, `makeWeightDists.py::_find_seed_dirs` | Mode 1 は item ディレクトリ直下の `Stratified_Seed*` を列挙し、Mode 2 は `weights_CAM.h5` を持つすべての seed ディレクトリを再帰的に探します。 |
| 1 seed ディレクトリの処理（BulkCompact および Dists） | `makeWeightPlotsBulkCompact.py::_process_seed_dir_animate`, `makeWeightDists.py::_process_seed_dir` | `Fold*` サブディレクトリをスキャンして K-Fold を自動検出し、per-fold 処理関数に委譲します。 |
| HDF5 重みの読み込みと整形 | `makeWeightPlots.py::makeWeightPlots`, `makeWeightPlotsBulk*.py::_process_single_fold_animate`, `makeWeightDists.py::_process_single_fold` | 保存された配列を `(T, V)` に変換し、joint name を付与します。Bulk / BulkCompact / Dists は fusion 形式の `weights_CAM.h5` のみを読み込みます。Dists はパディングなしで CAM 生の出力長を使用します。 |
| 骨格座標の読み込み | `makeWeightPlots.py::makeWeightPlots`, `makeWeightPlotsBulk*.py::_process_single_fold_animate` | デフォルトでは `data/ConciseData` から `p_x.csv`, `p_y.csv`, `p_z.csv` を読み込みます。 |
| 1 action の 2D frame 描画 | `plot2D.py::Plot2D` | 2D skeleton graph を描画し、MP4 または SVG frame を保存します。 |
| 1 action の 3D frame 描画 | `plot3D.py::Plot3D` | 3D skeleton scatter / edge frame を描画し、MP4 または SVG frame を保存します。 |
| Fusion の combined frame 描画 | `makeWeightPlotsBulk.py::PlotAllTasks2D_Combined`, `makeWeightPlotsBulk.py::PlotAllTasks3D_Combined`, `makeWeightPlotsBulkCompact.py::PlotAllTasks2D_Combined`, `makeWeightPlotsBulkCompact.py::PlotAllTasks3D_Combined` | 1 subject の複数動作を共通 color scale で同時に描画します。 |
| GradCAM 代表値の計算 | `makeWeightDists.py::_process_single_fold` | 各 fold の `weights_CAM.h5` と `PredictTruelabel.csv` から、未正規化の Max, Mean, Median を action / joint / body-part group の粒度で計算します。 |
| Seed 単位 CSV の保存 | `makeWeightDists.py::_save_seed_representative_csvs` | 全 fold の subject 行を縦に結合し、`CAM_Dists/GradCAM_{Max,Mean,Median}.csv` を保存します。 |
| Action label group violin plot の作成 | `makeWeightDists.py::_plot_label_group_violins` | seed 単位の `Action_` table を subject 内 min-max 正規化し、`True` / `Predicted` の 4 group すべてについて action 別 violin plot を `CAM_Dists/GradCAM_Action_LabelGroup_Violin.{png,svg}` に保存します。 |
| 正解 class 別 skeleton / violin plot の作成 | `makeWeightDists.py::_plot_skeleton_weights` | seed 単位 table を subject 内 min-max 正規化し、正解 subject だけで joint 平均 skeleton と body-part group violin plot を `CAM_Dists/GradCAM_Skeleton_{Max,Mean,Median}.{png,svg}` に保存します。 |
