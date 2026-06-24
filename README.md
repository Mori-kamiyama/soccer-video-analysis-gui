# サッカー動画分析 GUI ツールセット

鳥瞰図撮影したサッカー動画から、選手の位置・移動・チーム分析を行うためのGUIツール群です。

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)

## 🎯 主な機能

- **選手検出・トラッキング**: 動画から選手を自動検出し、ID付きで追跡
- **ボックス修正GUI**: 検出結果の手動修正・チーム色割り当て
- **選手ID管理**: トラッキングIDの結合・分割・追加
- **ボールアノテーション**: ボール位置の手動指定
- **ピッチ変換**: 画像座標からピッチ座標へのホモグラフィ変換
- **チーム分類**: ユニフォーム色による自動チーム分類
- **分析可視化**: ヒートマップ、ボロノイ図、チーム重心など

## 🚀 クイックスタート

### 前提条件

- Python 3.12+
- macOS（Tkinter対応）
- uv（推奨）または pip

### セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/Mori-kamiyama/soccer-video-analysis-gui.git
cd soccer-video-analysis-gui

# 仮想環境を作成
uv venv --python 3.12
# または: python3.12 -m venv .venv

# 依存パッケージをインストール
source .venv/bin/activate
pip install -r requirements.txt
```

### GUIツールの起動

| ツール | 説明 | 起動コマンド |
|-------|------|-------------|
| **選手トラッキングGUI** | 選手検出結果の確認・ID割り当て修正 | `./run_gui.sh` |
| **ボックス修正GUI** | バウンディングボックスの修正・チーム色設定 | `./run_box_gui.sh` |
| **選手IDエディター** | トラッキングIDの結合・分割・新規追加 | `./run_id_gui.sh` |
| **ボールアノテーション** | ボール位置の手動アノテーション | `./run_ball_gui.sh` |

## 📁 ディレクトリ構成

```
.
├── src/                          # ソースコード
│   ├── player_tracker_gui.py     # 選手トラッキングGUI
│   ├── photo_annotator.py        # ボックス修正GUI
│   ├── player_id_editor.py       # 選手IDエディター
│   ├── ball_annotator.py         # ボールアノテーションGUI
│   ├── player_merge_gui.py       # プレイヤーマージGUI
│   ├── pitch_corner_gui.py       # ピッチコーナー指定GUI
│   └── step*.py                  # パイプライン処理スクリプト
├── outputs/                      # 出力データ（CSV/JSON）
│   ├── tracks.csv                # トラッキングデータ
│   ├── players_hi.csv            # 選手データ（高解像度）
│   ├── ball_positions.csv        # ボール位置データ
│   └── segments/                 # セグメント別データ
├── run_*.sh                      # 起動スクリプト
└── requirements.txt              # 依存パッケージ
```

## 🛠️ パイプライン処理フロー

### ステップ1: フレーム抽出
```bash
python src/step1_extract_frames.py --video "match.mp4" --fps 2
```

### ステップ2: 選手検出
```bash
python src/step2_detect_players.py --source frames/
```

### ステップ3: トラッキング
```bash
python src/step4_track_players.py
```

### ステップ4: GUIで修正
```bash
# 検出結果の確認・修正
./run_gui.sh

# ボックス位置の微調整
./run_box_gui.sh

# IDの結合・分割
./run_id_gui.sh
```

### ステップ5: ピッチ座標変換
```bash
python src/step5_transform_to_pitch.py
```

### ステップ6: 分析・可視化
```bash
python src/step6_basic_analysis.py
python src/step8_visualize_voronoi_movement.py
```

## 🎮 GUIツール詳細

### 1. 選手トラッキングGUI (`run_gui.sh`)

トラッキング結果を時系列で確認し、IDの誤割り当てを修正します。

**主な機能:**
- フレーム単位の選手位置確認
- IDのクリックによる選択・変更
- トラッキングの分割・結合

**ショートカット:**
- `←` `→` : 前後のフレーム
- `Space` : 再生/停止
- `Delete` : 選択したトラックを削除

### 2. ボックス修正GUI (`run_box_gui.sh`)

検出されたバウンディングボックスの位置・サイズを修正し、チーム色を割り当てます。

**主な機能:**
- ボックスのドラッグ移動・リサイズ
- チーム色の割り当て（チームA/チームB/不明）
- 複数フレームの一括修正

### 3. 選手IDエディター (`run_id_gui.sh`)

トラッキングIDを統合・分割したり、新しいIDを追加します。

**主な機能:**
- 複数IDの結合（同一選手の分離したトラックを統合）
- IDの分割（異なる選手が同じIDに割り当てられた場合）
- 新規IDの追加

### 4. ボールアノテーションGUI (`run_ball_gui.sh`)

ボール位置を手動でアノテーションします。

**主な機能:**
- フレーム単位のボール位置指定
- キーフレームによる補間
- 高解像度/低解像度モード切り替え

## 📊 データフォーマット

### トラッキングデータ (`tracks.csv`)
```csv
frame,track_id,x1,y1,x2,y2,confidence,team
0,1,100,200,150,300,0.95,A
0,2,300,200,350,300,0.92,B
```

### 選手位置データ (`players_hi.csv`)
```csv
frame,track_id,x_img,y_img,x_pitch,y_pitch,team
0,1,125,250,34.2,18.5,A
0,2,325,250,42.8,20.1,B
```

### ボール位置データ (`ball_positions.csv`)
```csv
frame,x,y,visible
0,400,300,1
1,405,302,1
```

## 🔧 設定ファイル

### `pitch_points.json`
ピッチの四隅・基準点を指定し、画像座標からピッチ座標への変換を行います。

```json
{
  "image_points": [[100, 200], [900, 200], [900, 700], [100, 700]],
  "pitch_points": [[0, 0], [105, 0], [105, 68], [0, 68]]
}
```

### `roster.json`
選手名簿と背番号を管理します。

```json
{
  "team_a": ["10", "11", "7", "8", "9"],
  "team_b": ["1", "2", "3", "4", "5"]
}
```

## 📈 分析出力

処理が完了すると、`outputs/`以下に以下のファイルが生成されます：

| ファイル名 | 内容 |
|-----------|------|
| `player_distances.csv` | 選手ごとの移動距離 |
| `team_centroids.csv` | チーム重心の時系列データ |
| `heatmap.png` | 選手のヒートマップ |
| `trajectory.png` | 選手軌跡の可視化 |
| `dashboard.html` | 分析結果ダッシュボード |

## 📝 注意事項

- **動画ファイル**: リポジトリには含まれていません（各自で用意してください）
- **フレーム画像**: `frames/`ディレクトリは`.gitignore`で除外されています
- **容量**: CSV/JSONデータは含まれていますが、画像・動画は別途生成が必要です

## 🔗 関連リポジトリ

- このツールセットは、[Moondream API](https://moondream.ai/)を使用した選手検出と、[ByteTrack](https://github.com/ifzhang/ByteTrack)ベースのトラッキングを組み合わせています。

## 📄 ライセンス

MIT License

## 🤝 コントリビューション

バグ報告・機能リクエストは[Issues](https://github.com/Mori-kamiyama/soccer-video-analysis-gui/issues)へお願いします。
