# 競輪予測AI「パソ子」

競輪レースの着順を予測するAIシステムです。過去のレースデータを学習し、各選手の「3着以内に入る確率（入着確率）」を予測します。

## 🎯 主な機能

### 1. 予測モデル
- **二値分類**: 各選手が3着以内に入るか否かを予測
- **確率スコア**: 0〜1の入着確率を出力
- **カテゴリ別モデル**: 車立て・グレード別に最適化されたモデルを使用
  - 7車立て用
  - 9車立て用
  - ガールズケイリン用
  - チャレンジレース用
  - G3（記念）特化型
  - G1特化型

### 2. 独自指標
- **A率**: ランキング最上位選手（A選手）の3着以内確率
- **CT値（カラータイマー値）**: 上位3名で決まる確率の高さを示す指標
  - 高い（青）＝本命決着傾向
  - 低い（赤）＝荒れる傾向
- **KS値**: B率 - C率。AとBの実力差を示す

### 3. ゾーン分類
| ゾーン | 条件 | 推奨買い目 |
|--------|------|------------|
| ガチゾーン | A率≥0.85, CT≥70 | 3連複1点 |
| ブルーゾーン | CT≥60 | 3連複2点 |
| トワイライトゾーン | CT≥50, A率≤0.80 | 3連複3点 |
| レッドゾーン | CT<50 | 穴狙い推奨 |

### 4. イカサマサイコロ投票
確率を重みとした抽選で、中穴〜大穴を狙う買い目を生成する実験的機能。

## 📁 プロジェクト構造

```
keirin_ai/
├── main.py                 # メインアプリケーション
├── requirements.txt        # 必要パッケージ
├── README.md              # このファイル
├── src/
│   ├── __init__.py
│   ├── config.py          # 設定・定数
│   ├── utils.py           # ユーティリティ関数
│   ├── preprocessing.py   # データ前処理
│   ├── model.py           # 機械学習モデル
│   ├── indicators.py      # 指標算出
│   ├── output.py          # 出力・可視化
│   ├── ikasama_dice.py    # イカサマサイコロ
│   └── scrapers/
│       ├── __init__.py
│       ├── base_scraper.py        # スクレイパー基底クラス
│       ├── keirin_jp_scraper.py   # keirin.jp用
│       ├── netkeiba_scraper.py    # netkeiba用
│       ├── oddspark_scraper.py    # oddspark用
│       └── integrated_scraper.py  # 統合スクレイパー
├── data/
│   ├── raw/               # スクレイピングした生データ
│   └── processed/         # 前処理済みデータ
├── models/                # 学習済みモデル
├── output/                # 予測結果出力
└── logs/                  # ログファイル
```

## 🚀 セットアップ

### 1. 環境準備

```powershell
# Pythonの仮想環境を作成
python -m venv venv
.\venv\Scripts\Activate.ps1

# パッケージインストール
pip install -r requirements.txt
```

### 2. Chrome WebDriver（Selenium使用時）

Seleniumを使用する場合は、Chrome WebDriverが自動的にインストールされます。
手動でインストールする場合は、[ChromeDriver](https://chromedriver.chromium.org/)からダウンロードしてください。

## 📖 使用方法

### データ取得

```powershell
# 特定日のデータを取得
python main.py scrape --date 2025-02-04

# 期間指定でデータを取得（学習用）
python main.py scrape --start 2024-01-01 --end 2024-12-31
```

### モデル学習

```powershell
# 全カテゴリのモデルを学習
python main.py train --entries data/raw/entries.csv --results data/raw/results.csv

# 特定カテゴリのみ学習
python main.py train --entries data/raw/entries.csv --results data/raw/results.csv --category 9car
```

### 予測実行

```powershell
# 本日の予測（Excelで出力）
python main.py predict --date 2025-02-04

# CSV形式で出力
python main.py predict --date 2025-02-04 --output csv

# テキスト形式（コンソール出力）
python main.py predict --date 2025-02-04 --output text

# 既存データを使用（再スクレイピングしない）
python main.py predict --date 2025-02-04 --no-scrape
```

### イカサマサイコロ

```powershell
# 特定レースのサイコロを振る
python main.py dice --date 2025-02-04 --race 5

# 会場を指定
python main.py dice --date 2025-02-04 --race 5 --venue 立川

# シミュレーション回数を指定
python main.py dice --date 2025-02-04 --race 5 --simulations 500
```

### ゾーン分析

```powershell
# 本日のゾーン分布を確認
python main.py zones --date 2025-02-04
```

## 📊 出力例

### 予測レポート

```
##############################################################
# 競輪予想AI「パソ子」 予想レポート
# 日付: 2025-02-04
##############################################################

【本日のゾーン分布】
  ガチゾーン: 3レース
  ブルーゾーン: 8レース
  トワイライトゾーン: 12レース
  レッドゾーン: 5レース

==============================================================
【ガチゾーン】本命決着濃厚！
==============================================================
==================================================
立川 5R
日付: 2025-02-04
==================================================
【ガチゾーン】

A率: 89.2%
CT値: 75.3
KS値: 0.152

▼ 予想ランキング
------------------------------
A: 1番 山田太郎 (89.2%)
B: 5番 鈴木一郎 (74.0%)
C: 3番 佐藤次郎 (62.1%)
...

▼ 推奨買い目
------------------------------
  3連複 1-5-3
  3連単 1→5→3
==================================================
```

### Excel出力

- 各レースの予想印（A〜G）
- A率、CT値、KS値
- ゾーン別の背景色
  - ガチゾーン: 濃い青
  - ブルーゾーン: 薄い青
  - トワイライトゾーン: 薄い黄
  - レッドゾーン: 赤

## ⚙️ 設定カスタマイズ

`src/config.py` で各種パラメータを調整できます：

### ゾーン閾値

```python
@dataclass
class ZoneThresholds:
    gachi_a_rate: float = 0.85     # ガチゾーンのA率閾値
    gachi_ct_value: float = 70.0   # ガチゾーンのCT値閾値
    blue_ct_value: float = 60.0    # ブルーゾーンのCT値閾値
    twilight_ct_min: float = 50.0  # トワイライトゾーンのCT値下限
```

### スクレイピング設定

```python
@dataclass
class ScrapingConfig:
    request_interval: float = 1.5  # リクエスト間隔（秒）
    max_retries: int = 5           # リトライ回数
    use_selenium: bool = True      # Selenium使用
    headless: bool = True          # ヘッドレスモード
```

### モデル設定

```python
@dataclass
class ModelConfig:
    lgbm_params: Dict = field(default_factory=lambda: {
        "objective": "binary",
        "n_estimators": 500,
        "learning_rate": 0.05,
        ...
    })
```

## 🔧 特徴量

モデルで使用する主な特徴量：

| 特徴量 | 説明 |
|--------|------|
| car_number | 車番 |
| waku | 枠番 |
| bank_type | バンク周長（333/400/500） |
| competition_score | 競走得点 |
| back_count | バック回数 |
| win_rate | 勝率 |
| second_rate | 2連対率 |
| third_rate | 3連対率 |
| rank_class_num | 級班（SS=7, S1=6, ...） |
| line_position_num | ライン位置（先頭/番手/三番手） |
| line_formation_num | ライン構成（3分戦/4分戦等） |
| flag4 | 独自フラグ（先頭＋得点1位＋バック1位） |
| score_rank_in_race | レース内得点順位 |
| back_rank_in_race | レース内バック回数順位 |

## ⚠️ 注意事項

1. **スクレイピング制限**: 競輪サイトのスクレイピングは利用規約に従ってください。過度なアクセスは避けてください。

2. **予測精度**: 本システムは参考情報であり、投票を推奨するものではありません。競輪の購入は自己責任で行ってください。

3. **データ更新**: 出走表データは当日の変更（選手交代、欠場等）に対応できない場合があります。

4. **モデル更新**: 定期的にモデルを再学習することで予測精度を維持できます。

## 📄 ライセンス

このプロジェクトは個人利用を目的としています。商用利用には別途ライセンスが必要です。

## 🙏 謝辞

- 競輪データは各競輪ポータルサイトから取得しています
- 機械学習にはLightGBM、scikit-learnを使用しています

---

**競輪予測AI「パソ子」** - 2025年開発
