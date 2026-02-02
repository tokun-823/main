"""
共通ユーティリティ関数
ファイル操作、設定読み込み、ロギングなど
"""

import os
import yaml
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """ロギング設定
    
    Args:
        log_level: ログレベル
        log_file: ログファイルパス（Noneの場合はコンソールのみ）
    
    Returns:
        設定済みlogger
    """
    logger = logging.getLogger("keiba_ai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラ（指定時）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str = "config.yaml") -> dict:
    """設定ファイルの読み込み
    
    Args:
        config_path: 設定ファイルのパス
    
    Returns:
        設定辞書
    """
    project_root = get_project_root()
    full_path = project_root / config_path
    
    with open(full_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """プロジェクトルートディレクトリを取得
    
    Returns:
        プロジェクトルートのPathオブジェクト
    """
    # このファイルからの相対パスでプロジェクトルートを特定
    current_file = Path(__file__).resolve()
    # common/src/utils.py -> common/src -> common -> project_root
    project_root = current_file.parent.parent.parent
    return project_root


def ensure_dir(dir_path: str) -> Path:
    """ディレクトリが存在しなければ作成
    
    Args:
        dir_path: ディレクトリパス（相対パスの場合はプロジェクトルートからの相対）
    
    Returns:
        作成/確認されたディレクトリのPathオブジェクト
    """
    path = Path(dir_path)
    if not path.is_absolute():
        path = get_project_root() / path
    
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj: Any, file_path: str) -> None:
    """オブジェクトをpickle形式で保存
    
    Args:
        obj: 保存するオブジェクト
        file_path: 保存先パス
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(file_path: str) -> Any:
    """pickle形式のファイルを読み込み
    
    Args:
        file_path: ファイルパス
    
    Returns:
        読み込んだオブジェクト
    """
    path = Path(file_path)
    if not path.is_absolute():
        path = get_project_root() / path
    
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_date_range(start_date: str, end_date: str) -> list:
    """日付範囲のリストを生成
    
    Args:
        start_date: 開始日（YYYYMMDD形式）
        end_date: 終了日（YYYYMMDD形式）
    
    Returns:
        日付文字列のリスト
    """
    from datetime import timedelta
    
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime('%Y%m%d'))
        current += timedelta(days=1)
    
    return dates


def parse_race_id(race_id: str) -> dict:
    """レースIDをパース
    
    12桁のレースID: YYYYPPDDRRNN
    - YYYY: 年
    - PP: 競馬場コード
    - DD: 開催回
    - RR: 開催日
    - NN: レース番号
    
    Args:
        race_id: 12桁のレースID
    
    Returns:
        パースされた情報の辞書
    """
    return {
        'year': race_id[0:4],
        'place_code': race_id[4:6],
        'kai': race_id[6:8],
        'day': race_id[8:10],
        'race_num': race_id[10:12],
        'race_id': race_id
    }


# 競馬場コードの対応表
PLACE_CODE_MAP = {
    '01': '札幌',
    '02': '函館',
    '03': '福島',
    '04': '新潟',
    '05': '東京',
    '06': '中山',
    '07': '中京',
    '08': '京都',
    '09': '阪神',
    '10': '小倉'
}


def place_code_to_name(code: str) -> str:
    """競馬場コードを名前に変換"""
    return PLACE_CODE_MAP.get(code, '不明')


def place_name_to_code(name: str) -> str:
    """競馬場名をコードに変換"""
    reverse_map = {v: k for k, v in PLACE_CODE_MAP.items()}
    return reverse_map.get(name, '00')
