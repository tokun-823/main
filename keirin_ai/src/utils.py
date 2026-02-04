# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- ユーティリティ関数
==========================================
共通で使用するユーティリティ関数群
"""

import os
import sys
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import pickle

import pandas as pd
import numpy as np
from loguru import logger

from .config import (
    LOG_DIR, LOGGING_CONFIG, BankType, GradeType, 
    RaceCategory, ZoneType, BANK_INFO
)


# ============================================================
# ロギング設定
# ============================================================
def setup_logger(name: str = "keirin_ai") -> None:
    """ロガーのセットアップ"""
    log_file = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
    )
    logger.add(
        log_file,
        level=LOGGING_CONFIG["level"],
        format=LOGGING_CONFIG["format"],
        rotation=LOGGING_CONFIG["rotation"],
        retention=LOGGING_CONFIG["retention"],
    )


# ============================================================
# 日付・時間関連
# ============================================================
def parse_date(date_str: str) -> Optional[datetime]:
    """日付文字列をdatetimeに変換"""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y年%m月%d日",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None


def get_date_range(start_date: str, end_date: str) -> List[str]:
    """開始日から終了日までの日付リストを生成"""
    start = parse_date(start_date)
    end = parse_date(end_date)
    if not start or not end:
        return []
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return date_list


# ============================================================
# バンク・グレード判定
# ============================================================
def get_bank_type(venue: str) -> BankType:
    """会場名からバンク周長区分を取得"""
    return BANK_INFO.get(venue, BankType.BANK_400)


def get_bank_type_numeric(venue: str) -> int:
    """会場名からバンク周長区分を数値で取得（特徴量用）"""
    bank = get_bank_type(venue)
    mapping = {
        BankType.BANK_333: 333,
        BankType.BANK_400: 400,
        BankType.BANK_500: 500,
    }
    return mapping.get(bank, 400)


def get_grade_type(grade_str: str) -> GradeType:
    """グレード文字列からGradeTypeを取得"""
    grade_str = grade_str.upper().strip()
    if "GP" in grade_str:
        return GradeType.GP
    elif "G1" in grade_str or "GI" in grade_str:
        return GradeType.G1
    elif "G2" in grade_str or "GII" in grade_str:
        return GradeType.G2
    elif "G3" in grade_str or "GIII" in grade_str:
        return GradeType.G3
    elif "F1" in grade_str or "FI" in grade_str:
        return GradeType.F1
    elif "F2" in grade_str or "FII" in grade_str:
        return GradeType.F2
    return GradeType.F1


def get_race_category(
    num_cars: int,
    grade: GradeType,
    is_girls: bool = False,
    is_challenge: bool = False
) -> RaceCategory:
    """レースカテゴリを判定"""
    if is_girls:
        return RaceCategory.GIRLS
    if is_challenge:
        return RaceCategory.CHALLENGE
    if grade == GradeType.G1 or grade == GradeType.GP:
        return RaceCategory.G1_SPECIAL
    if grade == GradeType.G3:
        return RaceCategory.G3_SPECIAL
    if num_cars == 7:
        return RaceCategory.SEVEN_CAR
    return RaceCategory.NINE_CAR


# ============================================================
# ゾーン分類
# ============================================================
def classify_zone(
    a_rate: float,
    ct_value: float,
    gachi_a: float = 0.85,
    gachi_ct: float = 70.0,
    blue_ct: float = 60.0,
    twilight_ct: float = 50.0,
    twilight_a_max: float = 0.80,
) -> ZoneType:
    """A率とCT値からゾーンを分類"""
    # ガチゾーン: A率が高く、CT値も高い
    if a_rate >= gachi_a and ct_value >= gachi_ct:
        return ZoneType.GACHI
    
    # ブルーゾーン: CT値が高い
    if ct_value >= blue_ct:
        return ZoneType.BLUE
    
    # トワイライトゾーン: 中間領域
    if ct_value >= twilight_ct and a_rate <= twilight_a_max:
        return ZoneType.TWILIGHT
    
    # レッドゾーン: CT値が低い
    return ZoneType.RED


# ============================================================
# ライン構成解析
# ============================================================
def parse_line_formation(line_info: str) -> Tuple[int, List[List[int]]]:
    """
    ライン情報を解析
    
    Returns:
        Tuple[int, List[List[int]]]: (分戦数, ラインごとの車番リスト)
    
    Example:
        "123/45/6789" -> (3, [[1,2,3], [4,5], [6,7,8,9]])
    """
    if not line_info:
        return 0, []
    
    # 区切り文字で分割
    lines = []
    for sep in ["/", "-", "｜", "|"]:
        if sep in line_info:
            parts = line_info.split(sep)
            lines = [[int(c) for c in p if c.isdigit()] for p in parts]
            break
    
    if not lines:
        # 区切りなしの場合は1ライン
        lines = [[int(c) for c in line_info if c.isdigit()]]
    
    return len(lines), lines


def get_line_position(car_number: int, lines: List[List[int]]) -> str:
    """車番のライン内での位置を取得"""
    for line in lines:
        if car_number in line:
            idx = line.index(car_number)
            if idx == 0:
                return "先頭"
            elif idx == 1:
                return "番手"
            else:
                return "三番手以降"
    return "単騎"


# ============================================================
# 特徴量計算
# ============================================================
def calculate_flag4(
    is_line_head: bool,
    score_rank: int,
    back_rank: int
) -> int:
    """
    フラグ4を計算
    ライン先頭 + 競走得点1位 + バック回数1位 → 1
    """
    if is_line_head and score_rank == 1 and back_rank == 1:
        return 1
    return 0


def calculate_line_strength(
    line: List[int],
    scores: Dict[int, float]
) -> float:
    """ラインの強度を計算（ライン内選手の競走得点合計）"""
    return sum(scores.get(car, 0) for car in line)


# ============================================================
# 指標計算
# ============================================================
def calculate_ct_value(
    a_rate: float,
    b_rate: float,
    c_rate: float
) -> float:
    """
    CT値（カラータイマー値）を計算
    上位3名で決まる確率の高さを示す指標
    """
    # 全員がプラス判定（0.5以上）の場合は高い値
    # マイナス判定が含まれると低い値
    
    # 基本計算: 各確率を正規化して合計
    positive_threshold = 0.5
    
    # 各選手がプラス判定かどうかでスコア調整
    a_adj = a_rate if a_rate >= positive_threshold else a_rate * 0.5
    b_adj = b_rate if b_rate >= positive_threshold else b_rate * 0.5
    c_adj = c_rate if c_rate >= positive_threshold else c_rate * 0.5
    
    # CT値 = 調整後の確率の合計 * 100 / 3
    ct = (a_adj + b_adj + c_adj) * 100 / 3
    
    # 追加調整: A,B,C全員がプラス判定ならボーナス
    if a_rate >= positive_threshold and b_rate >= positive_threshold and c_rate >= positive_threshold:
        ct += 10
    
    return min(100, max(0, ct))


def calculate_ks_value(b_rate: float, c_rate: float) -> float:
    """
    KS値を計算
    B率 - C率（AとBの実力差を示す）
    """
    return b_rate - c_rate


# ============================================================
# データ保存・読込
# ============================================================
def save_pickle(obj: Any, filepath: Union[str, Path]) -> None:
    """オブジェクトをpickleで保存"""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved pickle: {filepath}")


def load_pickle(filepath: Union[str, Path]) -> Any:
    """pickleファイルを読み込み"""
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded pickle: {filepath}")
    return obj


def save_json(data: Dict, filepath: Union[str, Path]) -> None:
    """辞書をJSONで保存"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved JSON: {filepath}")


def load_json(filepath: Union[str, Path]) -> Dict:
    """JSONファイルを読み込み"""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON: {filepath}")
    return data


# ============================================================
# スクレイピングユーティリティ
# ============================================================
def random_sleep(min_sec: float = 1.0, max_sec: float = 3.0) -> None:
    """ランダムな時間待機"""
    sleep_time = random.uniform(min_sec, max_sec)
    time.sleep(sleep_time)


def generate_cache_key(url: str, params: Optional[Dict] = None) -> str:
    """URLとパラメータからキャッシュキーを生成"""
    cache_str = url
    if params:
        cache_str += json.dumps(params, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()


# ============================================================
# データ変換
# ============================================================
def safe_float(value: Any, default: float = 0.0) -> float:
    """安全にfloatに変換"""
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """安全にintに変換"""
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default


def normalize_score(
    value: float,
    min_val: float,
    max_val: float
) -> float:
    """値を0-1に正規化"""
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


# ============================================================
# 検証ユーティリティ
# ============================================================
def validate_race_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """レースデータの検証"""
    errors = []
    
    # 必須列の確認
    required_cols = ["car_number", "competition_score"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"必須列 '{col}' が存在しません")
    
    # 車番の重複チェック
    if "car_number" in df.columns:
        if df["car_number"].duplicated().any():
            errors.append("車番に重複があります")
    
    # 欠損値チェック
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        errors.append(f"欠損値が含まれる列: {null_cols}")
    
    return len(errors) == 0, errors


# ============================================================
# 初期化
# ============================================================
setup_logger()
