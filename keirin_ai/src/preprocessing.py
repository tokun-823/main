# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- データ前処理モジュール
============================================
特徴量エンジニアリング、ライン構成解析、データクリーニング
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

from .config import (
    PROCESSED_DATA_DIR, BankType, GradeType, RaceCategory,
    ModelConfig, BANK_INFO
)
from .utils import (
    get_bank_type_numeric, get_grade_type, get_race_category,
    calculate_flag4, calculate_line_strength, parse_line_formation,
    safe_float, safe_int
)


class DataPreprocessor:
    """
    データ前処理クラス
    
    主な機能:
    - 欠損値処理
    - 特徴量エンジニアリング
    - フラグ4の計算
    - ライン構成の解析
    - カテゴリ変数のエンコーディング
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.output_dir = PROCESSED_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # エンコーディングマッピング
        self.rank_class_map = {
            "SS": 7, "S1": 6, "S2": 5,
            "A1": 4, "A2": 3, "A3": 2,
            "L1": 1,  # ガールズ
        }
        
        self.line_position_map = {
            "先頭": 3, "番手": 2, "三番手以降": 1, "単騎": 0
        }
        
        self.line_formation_map = {
            "2分戦": 4, "3分戦": 3, "4分戦": 2, "細切れ": 1
        }
    
    def preprocess(
        self,
        entries_df: pd.DataFrame,
        results_df: Optional[pd.DataFrame] = None,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        データの前処理を実行
        
        Args:
            entries_df: 出走表データ
            results_df: 結果データ（学習時のみ）
            is_training: 学習用かどうか
            
        Returns:
            pd.DataFrame: 前処理済みデータ
        """
        logger.info(f"Preprocessing {len(entries_df)} entries")
        
        df = entries_df.copy()
        
        # 1. 基本的なクリーニング
        df = self._clean_data(df)
        
        # 2. 目的変数の作成（学習時のみ）
        if is_training and results_df is not None:
            df = self._add_target(df, results_df)
        
        # 3. 特徴量エンジニアリング
        df = self._engineer_features(df)
        
        # 4. カテゴリ変数のエンコーディング
        df = self._encode_categories(df)
        
        # 5. フラグ4の計算
        df = self._calculate_flag4(df)
        
        # 6. レース内順位の計算
        df = self._calculate_rankings(df)
        
        # 7. 欠損値の処理
        df = self._handle_missing(df)
        
        logger.info(f"Preprocessed data shape: {df.shape}")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的なデータクリーニング"""
        # 車番がない行を削除
        df = df.dropna(subset=["car_number"])
        df["car_number"] = df["car_number"].astype(int)
        
        # 数値列の型変換
        numeric_cols = [
            "competition_score", "back_count", "win_rate",
            "second_rate", "third_rate", "age", "gear_ratio"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 重複の削除
        df = df.drop_duplicates(subset=["race_id", "car_number"], keep="first")
        
        return df
    
    def _add_target(
        self,
        df: pd.DataFrame,
        results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """目的変数（3着以内フラグ）を追加"""
        # 結果データから3着以内の車番を抽出
        top3_dict = {}
        for _, row in results_df.iterrows():
            race_id = row["race_id"]
            top3 = set()
            for i in range(1, 4):
                col = f"rank{i}"
                if col in row and pd.notna(row[col]):
                    top3.add(int(row[col]))
            top3_dict[race_id] = top3
        
        # 目的変数を作成
        def is_top3(row):
            race_id = row["race_id"]
            car_num = row["car_number"]
            if race_id in top3_dict:
                return 1 if car_num in top3_dict[race_id] else 0
            return np.nan
        
        df["target"] = df.apply(is_top3, axis=1)
        
        # 目的変数がないレースを削除
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
        
        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量エンジニアリング"""
        # バンク区分を数値化
        if "venue_name" in df.columns:
            df["bank_type"] = df["venue_name"].apply(get_bank_type_numeric)
        elif "bank_type" not in df.columns:
            df["bank_type"] = 400  # デフォルト
        
        # グレードを数値化
        if "grade" in df.columns:
            df["grade_num"] = df["grade"].apply(self._grade_to_num)
        
        # 車立て数（同一レースの選手数）
        if "race_id" in df.columns:
            df["num_cars"] = df.groupby("race_id")["car_number"].transform("count")
        
        # 競走得点の平均からの偏差
        if "competition_score" in df.columns and "race_id" in df.columns:
            df["score_deviation"] = df.groupby("race_id")["competition_score"].transform(
                lambda x: x - x.mean()
            )
        
        # 勝率・連対率・3連対率の合成指標
        rate_cols = ["win_rate", "second_rate", "third_rate"]
        if all(col in df.columns for col in rate_cols):
            df["combined_rate"] = (
                df["win_rate"] * 0.5 +
                df["second_rate"] * 0.3 +
                df["third_rate"] * 0.2
            )
        
        # 年齢グループ
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 30, 35, 40, 50, 100],
                labels=[1, 2, 3, 4, 5, 6]
            ).astype(float)
        
        return df
    
    def _grade_to_num(self, grade: str) -> int:
        """グレードを数値化"""
        if pd.isna(grade):
            return 3
        grade = str(grade).upper()
        if "GP" in grade:
            return 7
        elif "G1" in grade or "GI" in grade:
            return 6
        elif "G2" in grade or "GII" in grade:
            return 5
        elif "G3" in grade or "GIII" in grade:
            return 4
        elif "F1" in grade or "FI" in grade:
            return 3
        elif "F2" in grade or "FII" in grade:
            return 2
        return 3
    
    def _encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ変数のエンコーディング"""
        # 級班
        if "rank_class" in df.columns:
            df["rank_class_num"] = df["rank_class"].map(self.rank_class_map).fillna(3)
        
        # ライン位置
        if "line_position" in df.columns:
            df["line_position_num"] = df["line_position"].map(self.line_position_map).fillna(0)
        
        # ライン構成
        if "line_formation" in df.columns:
            df["line_formation_num"] = df["line_formation"].map(self.line_formation_map).fillna(1)
        
        # ガールズ・チャレンジフラグ
        if "is_girls" in df.columns:
            df["is_girls"] = df["is_girls"].astype(int)
        if "is_challenge" in df.columns:
            df["is_challenge"] = df["is_challenge"].astype(int)
        
        return df
    
    def _calculate_flag4(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        フラグ4を計算
        ライン先頭 + 競走得点1位 + バック回数1位 → 1
        """
        df["flag4"] = 0
        
        if "line_position" not in df.columns or "competition_score" not in df.columns:
            return df
        
        for race_id in df["race_id"].unique():
            race_mask = df["race_id"] == race_id
            race_df = df.loc[race_mask]
            
            # 競走得点1位の車番
            if "competition_score" in race_df.columns:
                score_rank1 = race_df.loc[
                    race_df["competition_score"] == race_df["competition_score"].max(),
                    "car_number"
                ].values
            else:
                score_rank1 = []
            
            # バック回数1位の車番
            if "back_count" in race_df.columns:
                back_rank1 = race_df.loc[
                    race_df["back_count"] == race_df["back_count"].max(),
                    "car_number"
                ].values
            else:
                back_rank1 = []
            
            # フラグ4を設定
            for idx in race_df.index:
                car = df.loc[idx, "car_number"]
                position = df.loc[idx, "line_position"] if "line_position" in df.columns else ""
                
                is_head = position == "先頭"
                is_score1 = car in score_rank1
                is_back1 = car in back_rank1
                
                if is_head and is_score1 and is_back1:
                    df.loc[idx, "flag4"] = 1
        
        return df
    
    def _calculate_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """レース内での順位を計算"""
        # 競走得点順位
        if "competition_score" in df.columns and "race_id" in df.columns:
            df["score_rank_in_race"] = df.groupby("race_id")["competition_score"].rank(
                ascending=False, method="min"
            )
        
        # バック回数順位
        if "back_count" in df.columns and "race_id" in df.columns:
            df["back_rank_in_race"] = df.groupby("race_id")["back_count"].rank(
                ascending=False, method="min"
            )
        
        return df
    
    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値の処理"""
        # 数値列の欠損値を中央値で補完
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """モデル学習に使用する特徴量列のリスト"""
        return [
            "car_number",
            "waku",
            "bank_type",
            "grade_num",
            "num_cars",
            "competition_score",
            "score_deviation",
            "back_count",
            "win_rate",
            "second_rate",
            "third_rate",
            "combined_rate",
            "rank_class_num",
            "line_position_num",
            "line_formation_num",
            "flag4",
            "score_rank_in_race",
            "back_rank_in_race",
            "age",
            "age_group",
            "gear_ratio",
            "is_girls",
            "is_challenge",
        ]
    
    def filter_by_category(
        self,
        df: pd.DataFrame,
        category: RaceCategory
    ) -> pd.DataFrame:
        """カテゴリでデータをフィルタリング"""
        if category == RaceCategory.SEVEN_CAR:
            return df[df["num_cars"] == 7]
        elif category == RaceCategory.NINE_CAR:
            return df[df["num_cars"] == 9]
        elif category == RaceCategory.GIRLS:
            return df[df["is_girls"] == 1]
        elif category == RaceCategory.CHALLENGE:
            return df[df["is_challenge"] == 1]
        elif category == RaceCategory.G3_SPECIAL:
            return df[df["grade_num"] == 4]
        elif category == RaceCategory.G1_SPECIAL:
            return df[df["grade_num"] >= 6]
        return df
    
    def save_processed(self, df: pd.DataFrame, name: str) -> Path:
        """前処理済みデータを保存"""
        output_file = self.output_dir / f"{name}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info(f"Saved processed data to {output_file}")
        return output_file
    
    def load_processed(self, name: str) -> pd.DataFrame:
        """前処理済みデータを読み込み"""
        input_file = self.output_dir / f"{name}.csv"
        df = pd.read_csv(input_file, encoding="utf-8-sig")
        logger.info(f"Loaded processed data from {input_file}")
        return df


class LineAnalyzer:
    """
    ライン構成解析クラス
    
    競輪特有のライン構成を解析し、特徴量を生成
    """
    
    def __init__(self):
        pass
    
    def analyze_line_from_text(self, line_text: str) -> Dict:
        """
        ライン情報テキストを解析
        
        Args:
            line_text: ライン情報（例: "123/45/6789"）
            
        Returns:
            Dict: 解析結果
        """
        num_lines, lines = parse_line_formation(line_text)
        
        result = {
            "num_lines": num_lines,
            "lines": lines,
            "formation_type": self._get_formation_type(num_lines),
            "line_sizes": [len(line) for line in lines],
        }
        
        return result
    
    def _get_formation_type(self, num_lines: int) -> str:
        """分戦タイプを判定"""
        if num_lines == 2:
            return "2分戦"
        elif num_lines == 3:
            return "3分戦"
        elif num_lines == 4:
            return "4分戦"
        else:
            return "細切れ"
    
    def calculate_line_features(
        self,
        df: pd.DataFrame,
        scores_col: str = "competition_score"
    ) -> pd.DataFrame:
        """
        ライン関連の特徴量を計算
        
        Args:
            df: 選手データ
            scores_col: 得点列名
            
        Returns:
            pd.DataFrame: 特徴量追加後のデータ
        """
        df = df.copy()
        
        # レースごとにライン強度を計算
        for race_id in df["race_id"].unique():
            race_mask = df["race_id"] == race_id
            race_df = df.loc[race_mask]
            
            # ライン情報がある場合
            if "line_cars" in race_df.columns:
                for idx in race_df.index:
                    line_cars = df.loc[idx, "line_cars"]
                    if isinstance(line_cars, list) and len(line_cars) > 0:
                        # ラインの競走得点合計
                        line_scores = race_df.loc[
                            race_df["car_number"].isin(line_cars),
                            scores_col
                        ].sum()
                        df.loc[idx, "line_strength"] = line_scores
                        df.loc[idx, "line_size"] = len(line_cars)
        
        return df
    
    def detect_anomaly_line(self, lines: List[List[int]]) -> bool:
        """
        変則的な並び（競り、6車並び等）を検出
        
        Returns:
            bool: 変則的な場合True
        """
        total_cars = sum(len(line) for line in lines)
        
        # 6車以上の長いラインがある
        for line in lines:
            if len(line) >= 6:
                return True
        
        # 車立て数との不一致
        if total_cars > 9:
            return True
        
        return False
