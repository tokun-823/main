"""
ボートレース予測AI システム設定
"""
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# プロジェクトルートパス
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

# ディレクトリ作成
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class ScrapingConfig(BaseModel):
    """スクレイピング設定"""
    # 基本URL
    base_url: str = "https://www.boatrace.jp"
    official_data_url: str = "https://www.boatrace.jp/owpc/pc/extra/data/download.html"
    
    # リクエスト間隔（秒）- サーバー負荷対策
    request_interval: float = Field(default=2.0, ge=0.5, le=10.0)
    retry_interval: float = Field(default=5.0, ge=1.0, le=30.0)
    max_retries: int = Field(default=5, ge=1, le=10)
    
    # 同時接続数
    max_concurrent_requests: int = Field(default=3, ge=1, le=10)
    
    # タイムアウト（秒）
    request_timeout: int = Field(default=30, ge=10, le=120)
    
    # プロキシ設定
    use_proxy: bool = False
    proxy_list: list[str] = Field(default_factory=list)
    
    # ヘッダー設定
    rotate_user_agent: bool = True
    
    # データ取得年範囲
    start_year: int = 2018
    end_year: int = 2025


class DatabaseConfig(BaseModel):
    """データベース設定"""
    db_path: str = str(DATA_DIR / "boatrace.duckdb")
    memory_limit: str = "4GB"
    threads: int = 4


class ModelConfig(BaseModel):
    """機械学習モデル設定"""
    # LightGBM基本設定
    lgb_params: dict = Field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 120,  # 3連単の組み合わせ数
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42
    })
    
    # 確率推定モデル設定
    probability_model_path: str = str(MODEL_DIR / "probability_model.txt")
    
    # 期待値予測モデル設定（Quantile Regression）
    quantiles: list[float] = [0.1, 0.5, 0.8]
    ev_model_path: str = str(MODEL_DIR / "ev_model.txt")
    
    # 学習設定
    train_test_split_ratio: float = 0.2
    validation_ratio: float = 0.1
    early_stopping_rounds: int = 100
    num_boost_round: int = 5000


class BettingConfig(BaseModel):
    """賭け戦略設定"""
    # ケリー基準設定
    kelly_fraction: float = Field(default=0.25, ge=0.0, le=1.0)  # フラクショナルケリー
    min_bet_amount: int = 100  # 最小購入額
    max_bet_ratio: float = Field(default=0.1, ge=0.01, le=0.5)  # 資金に対する最大購入比率
    
    # 期待値フィルター
    min_expected_value: float = Field(default=1.0, ge=0.5, le=2.0)
    
    # リスク管理
    max_daily_loss_ratio: float = Field(default=0.2, ge=0.05, le=0.5)
    
    # 購入対象券種
    ticket_types: list[str] = ["3連単", "3連複", "2連単", "2連複"]


class LLMConfig(BaseModel):
    """LLMアシスタント設定"""
    model_name: str = "gemma2:2b"
    ollama_base_url: str = "http://localhost:11434"
    discord_token: Optional[str] = os.getenv("DISCORD_BOT_TOKEN")
    max_context_length: int = 4096


class AppConfig(BaseModel):
    """アプリケーション全体設定"""
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    betting: BettingConfig = Field(default_factory=BettingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)


# グローバル設定インスタンス
config = AppConfig()


# ボートレース場コード
RACECOURSE_CODES = {
    "01": "桐生",
    "02": "戸田",
    "03": "江戸川",
    "04": "平和島",
    "05": "多摩川",
    "06": "浜名湖",
    "07": "蒲郡",
    "08": "常滑",
    "09": "津",
    "10": "三国",
    "11": "びわこ",
    "12": "住之江",
    "13": "尼崎",
    "14": "鳴門",
    "15": "丸亀",
    "16": "児島",
    "17": "宮島",
    "18": "徳山",
    "19": "下関",
    "20": "若松",
    "21": "芦屋",
    "22": "福岡",
    "23": "唐津",
    "24": "大村"
}

# 3連単組み合わせ（120通り）
def generate_trifecta_combinations():
    """3連単の全組み合わせを生成"""
    combinations = []
    for first in range(1, 7):
        for second in range(1, 7):
            if second == first:
                continue
            for third in range(1, 7):
                if third == first or third == second:
                    continue
                combinations.append((first, second, third))
    return combinations

TRIFECTA_COMBINATIONS = generate_trifecta_combinations()
TRIFECTA_TO_INDEX = {combo: idx for idx, combo in enumerate(TRIFECTA_COMBINATIONS)}
INDEX_TO_TRIFECTA = {idx: combo for idx, combo in enumerate(TRIFECTA_COMBINATIONS)}
