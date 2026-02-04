# 🚤 ボートレース予測AI

LightGBM + 量子回帰による競艇3連単予測システム

## 機能

- **データ収集**: 公式サイトからのLZHファイルダウンロード + オッズスクレイピング
- **確率推定**: LightGBM多クラス分類による120通りの3連単確率推定
- **期待値予測**: 分位点回帰（10%, 50%, 80%）による期待値予測
- **購入戦略**: Kelly Criterion / Horse Kelly による最適ベットサイズ計算
- **可視化**: Streamlit ダッシュボード、Sunburst チャート
- **LLMアシスタント**: Ollama + Gemma2 による分析アシスタント、Discord Bot

## セットアップ

### 必要条件

- Python 3.10+
- Ollama（LLMアシスタント機能を使用する場合）

### インストール

```bash
# リポジトリをクローン
cd kyoutei_ai

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係をインストール
pip install -r requirements.txt

# 環境変数を設定
cp .env.example .env
# .envファイルを編集してDISCORD_BOT_TOKENなどを設定
```

### Ollama のセットアップ（オプション）

```bash
# Ollama をインストール（https://ollama.ai）
# モデルをダウンロード
ollama pull gemma2:2b
```

## 使い方

### コマンド一覧

```bash
# ヘルプを表示
python main.py --help

# データをダウンロード
python main.py download --start 2018 --end 2025

# ETLパイプラインを実行
python main.py etl

# モデルを学習
python main.py train

# ハイパーパラメータ最適化を有効にして学習
python main.py train --optimize

# レース予測
python main.py predict 20250101 01 1  # 桐生1R

# ダッシュボードを起動
python main.py dashboard

# LLMアシスタントを起動
python main.py assistant

# Discord Botを起動
python main.py discord-bot

# バックテスト
python main.py backtest --start 20240101 --end 20241231

# システム情報を表示
python main.py info
```

### ワークフロー例

```bash
# 1. データをダウンロード
python main.py download --type lzh  # まずLZHファイルのみ

# 2. ETL実行
python main.py etl

# 3. モデル学習
python main.py train

# 4. バックテストで検証
python main.py backtest --start 20240101 --end 20241231

# 5. 予測を実行
python main.py predict 20250101 01 1 --bankroll 50000

# 6. ダッシュボードで詳細確認
python main.py dashboard
```

## プロジェクト構成

```
kyoutei_ai/
├── main.py                    # CLIアプリケーション
├── requirements.txt           # 依存関係
├── .env.example              # 環境変数テンプレート
├── config/
│   ├── __init__.py
│   └── settings.py           # 設定
├── src/
│   ├── __init__.py
│   ├── data_collection/      # データ収集
│   │   ├── __init__.py
│   │   ├── lzh_downloader.py # LZHファイルダウンロード
│   │   ├── scraper.py        # Webスクレイパー
│   │   └── curl_scraper.py   # curl-cffiスクレイパー
│   ├── etl/                  # ETLパイプライン
│   │   ├── __init__.py
│   │   ├── parser.py         # 固定長データパーサー
│   │   ├── database.py       # DuckDB管理
│   │   └── pipeline.py       # ETLパイプライン
│   ├── features/             # 特徴量エンジニアリング
│   │   ├── __init__.py
│   │   └── feature_engineering.py
│   ├── models/               # 機械学習モデル
│   │   ├── __init__.py
│   │   ├── probability_model.py  # 確率推定モデル
│   │   └── ev_model.py           # 期待値予測モデル
│   ├── betting/              # 購入戦略
│   │   ├── __init__.py
│   │   └── kelly.py          # Kelly Criterion
│   ├── visualization/        # 可視化
│   │   ├── __init__.py
│   │   ├── charts.py         # Plotlyチャート
│   │   └── dashboard.py      # Streamlitダッシュボード
│   └── assistant/            # LLMアシスタント
│       ├── __init__.py
│       ├── llm_assistant.py  # Ollamaアシスタント
│       └── discord_bot.py    # Discord Bot
├── data/                     # データディレクトリ
│   ├── lzh/                  # LZHファイル
│   ├── extracted/            # 展開済みデータ
│   └── boatrace.duckdb       # データベース
├── models/                   # 学習済みモデル
└── logs/                     # ログファイル
```

## データソース

- **LZHファイル**: 競艇公式サイト（http://www1.mbrace.or.jp/）から番組・結果データ
- **オッズ・直前情報**: https://www.boatrace.jp/ からスクレイピング

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| ML | LightGBM, Scikit-learn |
| DB | DuckDB |
| スクレイピング | curl-cffi, Playwright, BeautifulSoup4 |
| Web UI | Streamlit, Plotly |
| LLM | Ollama, Gemma2 |
| Bot | Discord.py |
| CLI | Typer, Rich |

## アンチスクレイピング対策

公式サイトはBot対策が強化されています。本システムでは以下の対策を実装しています:

- **curl-cffi**: TLSフィンガープリント偽装
- **Playwright**: WebDriver検出回避
- **User-Agent ローテーション**: 複数ブラウザのUAを切り替え
- **スマートディレイ**: リクエスト数に応じた動的待機
- **キャッシュ**: 同一URLへの重複リクエスト回避

## ライセンス

このプロジェクトは教育・研究目的で作成されています。
実際の投票には十分な検証とリスク管理をお願いします。

## 注意事項

- ギャンブルは自己責任で行ってください
- 本システムは利益を保証するものではありません
- スクレイピングは対象サイトの利用規約を確認してください
- 過度なリクエストを避け、サーバーに負荷をかけないでください
