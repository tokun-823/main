"""
競輪予測AI「パソ子」設定ファイル

v2.1 変更点:
  - SCRAPER_CONFIG セクション追加（2018〜2025年全期間取得設定）
  - DATA_CONFIG の end_year を 2025 に更新
"""

# データ設定
DATA_CONFIG = {
    'training_years': 7,       # 学習データの年数（2018〜2025 = 7年）
    'start_year': 2018,
    'end_year': 2025,           # ★ 2025年まで拡張
    'exclude_accidents': True,  # 落車・失格を除外
}

# =====================================================================
# スクレイパー設定（scraper.py と連携）
# =====================================================================
SCRAPER_CONFIG = {
    # -------- 取得範囲 --------
    'start_year':  2018,    # ★ デフォルト開始年
    'end_year':    2025,    # ★ デフォルト終了年
    'start_date':  '2018-01-01',
    'end_date':    '2025-12-31',

    # -------- 出力先 --------
    'output_dir':  './scraped_data',   # スクレイプ結果の保存先
    'split_by':    'year',             # 'year'（年単位）or 'month'（月単位）

    # -------- リクエスト制御 --------
    'request_interval': 1.5,   # 通常リクエスト間隔（秒）
    'retry_interval':   5.0,   # リトライ待機（秒）
    'max_retries':      3,     # 最大リトライ回数
    'request_timeout':  15,    # タイムアウト（秒）

    # -------- 中断・再開 --------
    'resume': True,            # True = 進捗ファイルから再開

    # -------- フィルタ --------
    'exclude_accidents': True,  # 落車・失格レースを除外
}

# バンク分類
BANK_CATEGORIES = {
    'BANK_333': '333バンク',
    'BANK_400': '400バンク',
    'BANK_500': '500バンク'
}

# レースカテゴリ（モデル分割用）
RACE_CATEGORIES = {
    'RACE_7': '7車立て',
    'RACE_9': '9車立て',
    'GIRLS': 'ガールズケイリン',
    'CHALLENGE': 'チャレンジレース',
    'G3': 'G3記念',
    'G1': 'G1'
}

# モデル設定
MODEL_CONFIG = {
    'algorithm': 'RandomForest',  # または 'XGBoost', 'LightGBM'
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
}

# 特徴量設定
FEATURE_COLUMNS = [
    'frame_number',       # 枠番
    'car_number',         # 車番
    'player_id',          # 選手ID
    'bank_category',      # バンク区分
    'line_formation',     # ライン構成
    'race_score',         # 競走得点
    'back_count',         # バック回数
    'flag4',              # 独自フラグ4
]

# 指標計算設定
INDICATOR_CONFIG = {
    'a_rate_threshold': {
        'high': 0.8,  # A率が高いとされる閾値
        'very_high': 0.9
    },
    'ct_value_threshold': {
        'blue_zone': 70,      # ブルーゾーン
        'twilight_zone': 50,  # トワイライトゾーン
        'red_zone': 50        # レッドゾーン（以下）
    },
    'ks_value_threshold': {
        'high': 1.0,      # KS値が高い閾値
        'very_high': 1.3
    }
}

# ゾーン分類設定
ZONE_CONFIG = {
    'GACHI': {
        'name': 'ガチゾーン',
        'color': '#0066CC',
        'a_rate_min': 0.8,
        'ct_value_min': 70,
        'recommendation': '3連複1点'
    },
    'BLUE': {
        'name': 'ブルーゾーン',
        'color': '#3399FF',
        'a_rate_min': 0.0,
        'ct_value_min': 70,
        'recommendation': '3連複2点'
    },
    'TWILIGHT': {
        'name': 'トワイライトゾーン',
        'color': '#FFCC66',
        'a_rate_max': 0.8,
        'ct_value_min': 50,
        'ct_value_max': 70,
        'recommendation': '3連複3点（ABC, ABD, ACD）'
    },
    'RED': {
        'name': 'レッドゾーン',
        'color': '#FF3333',
        'ct_value_max': 50,
        'recommendation': '穴狙い・万車券候補'
    }
}

# 出力設定
OUTPUT_CONFIG = {
    'output_dir': './output',
    'prediction_file': 'predictions.xlsx',
    'visualization_file': 'race_zones.xlsx',
    'enable_color': True,
    'enable_recommendations': True
}

# イカサマサイコロ設定
DICE_CONFIG = {
    'weight_power': 2,  # 確率を何乗するか
    'num_trials': 1000,  # 試行回数
    'enable': True
}

# =====================================================================
# パイプライン全体の実行モード
# =====================================================================
PIPELINE_MODE = {
    # 'demo'    : サンプルデータで動作確認（デフォルト）
    # 'scrape'  : スクレイピング → 学習 → 予測 → 出力
    # 'train'   : 既存Excelから学習 → 予測 → 出力
    # 'predict' : 学習済みモデルで予測のみ
    'mode': 'demo',

    # 'train' / 'predict' モード時の入力Excelファイル（glob パターン可）
    'input_excel': './scraped_data/training_data_*.xlsx',
}

# =====================================================================
# ロギング設定
# =====================================================================
LOGGING_CONFIG = {
    'level': 'INFO',              # DEBUG / INFO / WARNING / ERROR
    'log_file': './pasuko.log',   # ログファイルパス（None で無効）
    'encoding': 'utf-8',
}
