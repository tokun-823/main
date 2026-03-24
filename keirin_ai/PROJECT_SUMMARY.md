# 競輪予測AI「パソ子」プロジェクト概要 v2.1

## バージョン履歴

| バージョン | 日付 | 変更内容 |
|------------|------|----------|
| v1.0.0 | 2026-01-11 | 初版リリース |
| v2.0.0 | 2026-01-11 | scraper.py 追加（スクレイピング + Excel書き込み） |
| v2.1.0 | 2026-03-08 | scraper.py を main.py / config.py に完全統合 |

---

## v2.1 主な変更点

### config.py
- `SCRAPER_CONFIG` セクションを新設
  - デフォルト取得範囲: 2018年〜2025年
  - リクエスト間隔・リトライ・タイムアウト設定
  - 中断再開フラグ
- `DATA_CONFIG.end_year` を 2025 に更新（7年間）
- `PIPELINE_MODE` セクション追加（実行モード切り替え）
- `LOGGING_CONFIG` セクション追加（ログレベル・ファイル設定）

### main.py
- **CLIコマンドライン引数** 対応（argparse）
  - `--mode {demo,scrape,train,predict}`
  - `--start-year` / `--end-year`（scrape モード用）
  - `--input`（train/predict モード用）
  - `--reset`（スクレイピング進捗リセット）
- **新規メソッド追加**（PasukoAI クラス）
  - `run_scraping()` : scraper.py を呼び出してデータ収集
  - `load_scraped_excels()` : スクレイピング済みExcelを一括読み込み
  - `run_full_pipeline_with_scraping()` : スクレイピング→学習→予測→出力
  - `run_pipeline_from_excel()` : 既存Excelから学習→予測→出力
  - `_preprocess_df()` : 前処理共通ロジックを切り出し

---

## データフロー

```
[スクレイピング]
  scraper.py
  KeirinScraper.collect_full_history(2018〜2025)
        │
        ▼
  scraped_data/training_data_YYYY.xlsx  (年単位)
        │
        ▼
[読み込み・結合]
  PasukoAI.load_scraped_excels()
  → 全年分を結合して1つのDataFrame
        │
        ▼
[前処理・特徴量生成]
  KeirinDataPreprocessor.clean_data()
  FeatureEngineering.create_features()
        │
        ▼
[モデル学習]
  MultiModelManager.train_all_models()
  カテゴリ別: RACE_9 / G3 / GIRLS / CHALLENGE / G1 / RACE_7
        │
        ▼
[予測・指標計算]
  BatchPredictor.predict_races()
  → A率 / CT値 / KS値 / ゾーン分類
        │
        ▼
[出力]
  ExcelOutputGenerator → summary / detail / statistics
  VisualizationGenerator → zone_distribution.png
```

---

## ファイル構成

| ファイル | 行数 | 役割 |
|----------|------|------|
| main.py | ~350行 | 統合パイプライン・CLI |
| scraper.py | ~1,322行 | スクレイピング・Excel書き込み |
| config.py | ~130行 | 全設定（SCRAPER_CONFIG追加） |
| data_processing.py | ~200行 | 前処理・特徴量生成 |
| model.py | ~300行 | RandomForestモデル管理 |
| prediction.py | ~350行 | 予測・A率・CT値・KS値・ゾーン |
| output_generator.py | ~400行 | Excel・グラフ出力 |

---

## 実行モード

```bash
# デモ（サンプルデータ）
python main.py

# 全期間スクレイピング（2018〜2025年）
python main.py --mode scrape

# 年指定スクレイピング
python main.py --mode scrape --start-year 2022 --end-year 2023

# 既存Excelから学習
python main.py --mode train --input "./scraped_data/training_data_*.xlsx"

# 予測のみ
python main.py --mode predict --input ./scraped_data/training_data_2024.xlsx
```

---

## 実装済み仕様チェックリスト

| 仕様項目 | 実装状況 |
|----------|----------|
| 2018〜2025年 全レースデータ取得 | ✅ scraper.py + config.py SCRAPER_CONFIG |
| 落車・失格レース除外 | ✅ accident_flag フィルタリング |
| 年単位Excel分割保存 | ✅ training_data_YYYY.xlsx |
| 中断・再開機能 | ✅ .scrape_progress.json |
| カテゴリ別モデル（7車/9車/Girls/G3/G1等） | ✅ MultiModelManager |
| A率・CT値・KS値計算 | ✅ prediction.py |
| ゾーン分類（ガチ/Blue/Twilight/Red） | ✅ prediction.py |
| カラーExcel出力 | ✅ output_generator.py |
| イカサマサイコロ投票 | ✅ IkasamaDice クラス |
| CLIコマンドライン | ✅ argparse (main.py / scraper.py) |
| main.py にscraper統合 | ✅ v2.1で完了 |

