# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- 設定ファイル
===================================
各種パラメータ、パス、閾値などの設定を管理
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# ============================================================
# ディレクトリ設定
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"           # スクレイピングした生データ
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # 前処理済みデータ
MODEL_DIR = BASE_DIR / "models"           # 学習済みモデル
OUTPUT_DIR = BASE_DIR / "output"          # 予測結果出力
LOG_DIR = BASE_DIR / "logs"               # ログファイル

# ディレクトリ作成
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================
# 列挙型定義
# ============================================================
class BankType(Enum):
    """バンク周長区分"""
    BANK_333 = "333"   # 33バンク（333m）
    BANK_400 = "400"   # 400バンク（400m）
    BANK_500 = "500"   # 500バンク（500m）


class GradeType(Enum):
    """グレード区分"""
    GP = "GP"          # グランプリ
    G1 = "G1"          # G1
    G2 = "G2"          # G2
    G3 = "G3"          # G3（記念）
    F1 = "F1"          # F1
    F2 = "F2"          # F2


class RaceCategory(Enum):
    """レースカテゴリ（モデル分割用）"""
    SEVEN_CAR = "7car"           # 7車立て用
    NINE_CAR = "9car"            # 9車立て用
    GIRLS = "girls"              # ガールズケイリン用
    CHALLENGE = "challenge"      # チャレンジレース用
    G3_SPECIAL = "g3_special"    # G3（記念）特化型
    G1_SPECIAL = "g1_special"    # G1特化型


class ZoneType(Enum):
    """レース分類ゾーン"""
    GACHI = "ガチゾーン"             # A率高＋CT値高 → 3連複1点
    BLUE = "ブルーゾーン"            # CT値高 → 3連複2点
    TWILIGHT = "トワイライトゾーン"   # 中間 → 3連複3点
    RED = "レッドゾーン"             # CT値低 → 荒れやすい


# ============================================================
# スクレイピング設定
# ============================================================
@dataclass
class ScrapingConfig:
    """スクレイピング設定"""
    # 基本URL
    base_urls: Dict[str, str] = field(default_factory=lambda: {
        "keirin_jp": "https://keirin.jp",
        "keirin_netkeiba": "https://keirin.netkeiba.com",
        "gamboo": "https://www.gamboo.jp",
        "oddspark": "https://www.oddspark.com/keirin",
        "chariloto": "https://www.chariloto.com",
    })
    
    # リクエスト設定
    request_timeout: int = 30
    max_retries: int = 5
    retry_delay: float = 2.0       # リトライ間隔（秒）
    request_interval: float = 1.5  # リクエスト間隔（秒）
    
    # User-Agent設定（複数用意してローテーション）
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ])
    
    # Selenium設定
    use_selenium: bool = True
    headless: bool = True
    selenium_wait_time: int = 10
    
    # プロキシ設定（必要に応じて）
    use_proxy: bool = False
    proxy_list: List[str] = field(default_factory=list)


# ============================================================
# モデル設定
# ============================================================
@dataclass
class ModelConfig:
    """機械学習モデル設定"""
    # 学習データ期間
    train_start_year: int = 2018
    train_end_year: int = 2025
    
    # モデルパラメータ（LightGBM）
    lgbm_params: Dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "random_state": 42,
    })
    
    # 交差検証
    cv_folds: int = 5
    
    # 特徴量リスト
    feature_columns: List[str] = field(default_factory=lambda: [
        "waku",                  # 枠番
        "car_number",            # 車番
        "bank_type",             # バンク周長区分
        "race_grade",            # グレード
        "line_position",         # ライン位置（先頭/番手/三番手）
        "line_formation",        # ライン構成（3分戦/4分戦等）
        "competition_score",     # 競走得点
        "back_count",            # バック回数
        "win_rate",              # 勝率
        "second_rate",           # 2連対率
        "third_rate",            # 3連対率
        "flag4",                 # 独自フラグ4（先頭＋得点1位＋バック1位）
        "avg_start_timing",      # 平均スタートタイミング
        "gear_ratio",            # ギア倍率
        "age",                   # 年齢
        "rank_class",            # 級班
        "line_strength",         # ライン強度
        "score_rank_in_race",    # レース内得点順位
        "back_rank_in_race",     # レース内バック回数順位
    ])


# ============================================================
# ゾーン分類閾値
# ============================================================
@dataclass
class ZoneThresholds:
    """ゾーン分類の閾値設定"""
    # ガチゾーン閾値
    gachi_a_rate: float = 0.85     # A率がこれ以上
    gachi_ct_value: float = 70.0   # CT値がこれ以上
    
    # ブルーゾーン閾値
    blue_ct_value: float = 60.0    # CT値がこれ以上
    
    # トワイライトゾーン閾値
    twilight_ct_min: float = 50.0  # CT値がこれ以上
    twilight_a_rate_max: float = 0.80  # A率がこれ以下
    
    # レッドゾーン閾値（トワイライト以下）
    red_ct_value: float = 50.0     # CT値がこれ以下


# ============================================================
# 出力設定
# ============================================================
@dataclass
class OutputConfig:
    """出力設定"""
    # Excel出力設定
    excel_row_height: int = 20
    excel_col_width: int = 12
    
    # 背景色設定（RGB）
    color_gachi: str = "#0066FF"      # ガチゾーン（濃い青）
    color_blue: str = "#66B2FF"       # ブルーゾーン（薄い青）
    color_twilight: str = "#FFFF99"   # トワイライトゾーン（薄い黄）
    color_red: str = "#FF6666"        # レッドゾーン（赤）
    
    # 予想印
    rank_symbols: List[str] = field(default_factory=lambda: [
        "A", "B", "C", "D", "E", "F", "G", "H", "I"
    ])


# ============================================================
# イカサマサイコロ設定
# ============================================================
@dataclass
class DiceConfig:
    """イカサマサイコロ投票設定"""
    # 確率の加工方法
    power_factor: float = 2.0      # 確率を何乗するか
    
    # 抽選回数
    num_draws: int = 100           # シミュレーション回数
    
    # 上位N着を選出
    top_n: int = 3


# ============================================================
# メイン設定クラス
# ============================================================
@dataclass
class Config:
    """統合設定クラス"""
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    zone: ZoneThresholds = field(default_factory=ZoneThresholds)
    output: OutputConfig = field(default_factory=OutputConfig)
    dice: DiceConfig = field(default_factory=DiceConfig)


# デフォルト設定インスタンス
config = Config()


# ============================================================
# バンク情報マッピング
# ============================================================
BANK_INFO = {
    # 33バンク（333m）
    "前橋": BankType.BANK_333,
    "松戸": BankType.BANK_333,
    "千葉": BankType.BANK_333,
    "川崎": BankType.BANK_333,
    "伊東温泉": BankType.BANK_333,
    "小田原": BankType.BANK_333,
    "富山": BankType.BANK_333,
    "大垣": BankType.BANK_333,
    "奈良": BankType.BANK_333,
    "和歌山": BankType.BANK_333,
    "福井": BankType.BANK_333,
    "岸和田": BankType.BANK_333,
    "玉野": BankType.BANK_333,
    "広島": BankType.BANK_333,
    "防府": BankType.BANK_333,
    "松山": BankType.BANK_333,
    "高知": BankType.BANK_333,
    "小倉": BankType.BANK_333,
    "久留米": BankType.BANK_333,
    "武雄": BankType.BANK_333,
    "佐世保": BankType.BANK_333,
    "別府": BankType.BANK_333,
    "熊本": BankType.BANK_333,
    
    # 400バンク（400m）
    "函館": BankType.BANK_400,
    "青森": BankType.BANK_400,
    "いわき平": BankType.BANK_400,
    "弥彦": BankType.BANK_400,
    "西武園": BankType.BANK_400,
    "京王閣": BankType.BANK_400,
    "立川": BankType.BANK_400,
    "静岡": BankType.BANK_400,
    "名古屋": BankType.BANK_400,
    "岐阜": BankType.BANK_400,
    "四日市": BankType.BANK_400,
    "京都向日町": BankType.BANK_400,
    "高松": BankType.BANK_400,
    "豊橋": BankType.BANK_400,
    "松阪": BankType.BANK_400,
    
    # 500バンク（500m）
    "宇都宮": BankType.BANK_500,
    "大宮": BankType.BANK_500,
    "取手": BankType.BANK_500,
}


# ============================================================
# ロギング設定
# ============================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    "rotation": "10 MB",
    "retention": "7 days",
}
