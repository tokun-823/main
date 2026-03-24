# 競輪予測AI「パソ子」v2.1

競輪レースデータのスクレイピングから予測・出力まで一貫して行うAIシステムです。

## 機能概要

| 機能 | 説明 |
|------|------|
| データ収集 | 2018〜2025年（デフォルト）の全レースデータを自動スクレイピング |
| Excel出力 | 出走表・レース結果・学習データを自動整形・色分けExcelへ保存 |
| モデル学習 | カテゴリ別（7車・9車・ガールズ・チャレンジ・G3・G1）RandomForest |
| 予測指標 | A率・CT値・KS値の算出、ゾーン分類（ガチ/Blue/Twilight/Red） |
| 出力 | カラーExcel・統計・グラフ・イカサマサイコロ投票 |
| 中断再開 | `.scrape_progress.json` で進捗を保存、途中から再開可能 |

---

## インストール

```bash
pip install -r requirements.txt
```

---

## クイックスタート

### デモ実行（サンプルデータ）
```bash
python main.py
# または
python main.py --mode demo
```

### 全期間スクレイピング → 学習 → 予測（2018〜2025年）
```bash
python main.py --mode scrape
```

### 年範囲を指定してスクレイピング
```bash
python main.py --mode scrape --start-year 2022 --end-year 2023
```

### 前回の続きから再開（デフォルト動作）
```bash
python main.py --mode scrape       # 自動で続きから再開
python main.py --mode scrape --reset   # 最初からやり直す
```

### スクレイピング済みExcelから学習
```bash
python main.py --mode train --input "./scraped_data/training_data_*.xlsx"
```

### 学習済みモデルで予測のみ
```bash
python main.py --mode predict --input ./scraped_data/training_data_2024.xlsx
```

---

## スクレイパー単体使用

```bash
# 2018〜2025年 全期間取得
python scraper.py --mode full

# 年範囲指定
python scraper.py --mode full --start-year 2020 --end-year 2022

# 進捗をリセットして最初から
python scraper.py --mode full --reset

# 1日分の出走表
python scraper.py --mode card --date 2024-12-01

# 1日分の結果
python scraper.py --mode result --date 2024-11-30

# 進捗確認
python scraper.py --mode progress

# ダミーデータでExcel書き込みテスト
python scraper.py --mode test
```

---

## Pythonコードから使用

```python
from main import PasukoAI

pasuko = PasukoAI()

# ① スクレイピング → 学習 → 予測 → 出力（フルパイプライン）
predictions = pasuko.run_full_pipeline_with_scraping(
    start_year=2018,
    end_year=2025,
)

# ② スクレイピングのみ実行
saved_paths = pasuko.run_scraping(start_year=2022, end_year=2023)

# ③ 既存Excelから学習 → 予測 → 出力
predictions = pasuko.run_pipeline_from_excel(
    excel_pattern="./scraped_data/training_data_*.xlsx"
)

# ④ デモ（サンプルデータ）
predictions = pasuko.run_full_pipeline(data_source='sample', is_excel=False)
```

```python
# スクレイパー単体使用
from scraper import KeirinDataPipeline

pipeline = KeirinDataPipeline(output_dir="./scraped_data")

# 全期間
paths = pipeline.run_full_history(start_year=2018, end_year=2025)

# 1日分
df, path = pipeline.run_race_card("2024-12-01")
df, path = pipeline.run_race_result("2024-11-30")

# パソ子AIに渡せる形式でExcelをロード
df = pipeline.load_excel_for_pasuko("./scraped_data/training_data_2024.xlsx")
```

---

## プロジェクト構成

```
keirin_ai_pasuko/
├── main.py               メインスクリプト（統合パイプライン・CLI）
├── scraper.py            スクレイピング & Excel書き込みモジュール
├── config.py             設定ファイル（SCRAPER_CONFIG含む）
├── data_processing.py    データ収集・前処理・特徴量生成
├── model.py              機械学習モデル（RandomForest）
├── prediction.py         予測・指標計算・ゾーン分類
├── output_generator.py   Excel・グラフ出力
├── examples.py           使用例
├── quickstart.py         クイックスタートスクリプト
├── requirements.txt      依存ライブラリ
├── README.md             このファイル
├── PROJECT_SUMMARY.md    プロジェクト概要
├── INSTALL.txt           インストール手順
├── models/               学習済みモデルファイル（*.pkl）
├── output/               予測結果・グラフ出力先
└── scraped_data/         スクレイピング結果Excel保存先
    ├── training_data_2018.xlsx
    ├── training_data_2019.xlsx
    ├── ...
    ├── training_data_2025.xlsx
    └── .scrape_progress.json   中断再開用進捗ファイル
```

---

## 実行モード詳細

| モード | コマンド | 説明 |
|--------|----------|------|
| `demo` | `python main.py` | サンプルデータで動作確認 |
| `scrape` | `python main.py --mode scrape` | スクレイピング→学習→予測→出力 |
| `train` | `python main.py --mode train --input <path>` | Excel読み込み→学習→予測→出力 |
| `predict` | `python main.py --mode predict --input <path>` | 学習済みモデルで予測のみ |

---

## 出力ファイル

| ファイル | 内容 |
|----------|------|
| `output/pasuko_predictions_summary.xlsx` | 予測サマリー（A率・CT・KS・ゾーン） |
| `output/pasuko_predictions_detail.xlsx` | 全選手の詳細予測スコア |
| `output/pasuko_statistics.xlsx` | ゾーン別・A率分布統計 |
| `output/pasuko_zone_distribution.png` | ゾーン分布グラフ |
| `scraped_data/training_data_YYYY.xlsx` | 年単位の学習データ |

---

## ゾーン分類基準

| ゾーン | 条件 | 推奨戦略 |
|--------|------|----------|
| 🔵 **ガチゾーン** | A率≥0.8 かつ CT値≥70 | 3連複1点 |
| 💙 **ブルーゾーン** | CT値≥70 | 3連複2点 |
| 🌅 **トワイライト** | CT値 50〜70 | 3連複3点 |
| 🔴 **レッドゾーン** | CT値<50 | 穴狙い・万車券候補 |

---

## 注意事項

- 実際のスクレイピングはサイト規約を遵守してください。
- 大量データ取得時はリクエスト間隔（デフォルト1.5秒）を守ってください。
- 取得中断時は `.scrape_progress.json` が残り、次回実行時に自動再開します。
- 機械学習モデルの精度はデータ量・品質に大きく依存します。

---

## 技術スタック

- **Python** 3.7+
- **scikit-learn** RandomForest
- **pandas** / **numpy** データ処理
- **openpyxl** Excel出力
- **requests** / **BeautifulSoup4** スクレイピング
- **matplotlib** グラフ生成
- **joblib** モデル保存

---

バージョン: v2.1 | 完成日: 2026-03-08
