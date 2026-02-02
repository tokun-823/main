# 競馬予測AIシステム

## 概要
競馬の過去データを収集・分析し、LightGBMによる機械学習モデルで予測を行うシステムです。

## 必要環境
- Python 3.11.4推奨
- Chrome/Chromium（Selenium/Playwright用）

## インストール
```bash
pip install -r requirements.txt
playwright install chromium
```

## ディレクトリ構成
```
keiba_ai/
├── config.yaml              # 設定ファイル
├── requirements.txt         # 必要パッケージ
├── common/                  # 共通ライブラリ
│   └── src/
│       ├── scraping.py      # スクレイピング関数群
│       └── create_raw_df.py # HTMLからRawデータ作成
├── data/                    # データ保存用
│   ├── html/               # スクレイピングしたHTML
│   │   ├── race/
│   │   ├── horse/
│   │   ├── pedigree/
│   │   └── jra_odds/
│   ├── raw_df/             # 加工前CSV
│   └── 01_preprocessed/    # 前処理済みデータ
├── v3/                      # バージョン3実装
│   └── src/
│       ├── preprocessing.py      # データ前処理
│       ├── feature_engineering.py # 特徴量作成
│       ├── train.py              # モデル学習
│       ├── evaluation.py         # 回収率シミュレーション
│       ├── prediction.py         # 本番予測
│       └── jra_scraping.py       # JRAリアルタイムオッズ
├── models/                  # 学習済みモデル
├── results/                 # 予測結果・シミュレーション結果
└── scripts/                 # 実行スクリプト
    ├── collect_data.py      # データ収集実行
    ├── train_model.py       # モデル学習実行
    ├── run_prediction.py    # 予測実行
    └── run_realtime.py      # リアルタイム運用
```

## 使い方

### 1. データ収集
```bash
python scripts/collect_data.py --start_year 2020 --end_year 2024
```

### 2. モデル学習
```bash
python scripts/train_model.py
```

### 3. 予測実行
```bash
python scripts/run_prediction.py --date 20240701
```

### 4. リアルタイム運用
```bash
python scripts/run_realtime.py
```

## 注意事項
- スクレイピングは1秒以上の間隔を空けること
- サーバー負荷に配慮した利用をお願いします
- 予測結果は参考情報であり、馬券購入は自己責任でお願いします
