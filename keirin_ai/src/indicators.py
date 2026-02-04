# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- 予測指標算出モジュール
============================================
A率、CT値、KS値の算出とゾーン分類
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

from .config import ZoneThresholds, ZoneType, config
from .utils import calculate_ct_value, calculate_ks_value, classify_zone


@dataclass
class RacePrediction:
    """レース予測結果"""
    race_id: str
    date: str
    venue_name: str
    race_no: int
    
    # 予想ランキング
    rankings: pd.DataFrame  # A, B, C, D... の順
    
    # 指標
    a_rate: float           # A選手の3着以内確率
    b_rate: float           # B選手の3着以内確率
    c_rate: float           # C選手の3着以内確率
    ct_value: float         # CT値
    ks_value: float         # KS値
    
    # ゾーン分類
    zone: ZoneType
    
    # 推奨買い目
    recommended_bets: List[str]


class IndicatorCalculator:
    """
    予測指標算出クラス
    
    A率、CT値、KS値を算出し、レースをゾーン分類
    """
    
    def __init__(self, thresholds: Optional[ZoneThresholds] = None):
        self.thresholds = thresholds or config.zone
    
    def calculate_indicators(
        self,
        race_df: pd.DataFrame,
        proba_col: str = "pred_proba",
        rank_col: str = "rank_order",
    ) -> Dict[str, float]:
        """
        レースの各種指標を算出
        
        Args:
            race_df: ランキング済みレースデータ
            proba_col: 確率列名
            rank_col: ランク順位列名
            
        Returns:
            Dict: 各種指標
        """
        # ランク順にソート
        sorted_df = race_df.sort_values(rank_col).reset_index(drop=True)
        
        # 上位3名の確率を取得
        a_rate = sorted_df.loc[0, proba_col] if len(sorted_df) > 0 else 0.0
        b_rate = sorted_df.loc[1, proba_col] if len(sorted_df) > 1 else 0.0
        c_rate = sorted_df.loc[2, proba_col] if len(sorted_df) > 2 else 0.0
        
        # CT値を算出
        ct_value = self._calculate_ct_value(a_rate, b_rate, c_rate)
        
        # KS値を算出
        ks_value = calculate_ks_value(b_rate, c_rate)
        
        return {
            "a_rate": a_rate,
            "b_rate": b_rate,
            "c_rate": c_rate,
            "ct_value": ct_value,
            "ks_value": ks_value,
        }
    
    def _calculate_ct_value(
        self,
        a_rate: float,
        b_rate: float,
        c_rate: float
    ) -> float:
        """
        CT値（カラータイマー値）を算出
        
        上位3名で決まる確率の高さを示す独自指標
        - 全員がプラス判定（0.5以上）なら高い値
        - マイナス判定が含まれると低い値
        """
        positive_threshold = 0.5
        
        # 各選手の判定
        a_positive = a_rate >= positive_threshold
        b_positive = b_rate >= positive_threshold
        c_positive = c_rate >= positive_threshold
        
        # プラス判定の数
        num_positive = sum([a_positive, b_positive, c_positive])
        
        # 基本スコア: 確率の加重平均
        base_score = a_rate * 0.5 + b_rate * 0.3 + c_rate * 0.2
        
        # CT値計算
        if num_positive == 3:
            # 全員プラス → 高いCT値
            ct = base_score * 100 + 20
        elif num_positive == 2:
            # 2人プラス → 中程度
            ct = base_score * 100 + 10
        elif num_positive == 1:
            # 1人プラス → 低め
            ct = base_score * 100
        else:
            # 全員マイナス → 非常に低い
            ct = base_score * 100 - 10
        
        # KS値による調整（AとBの差が大きいほどボーナス）
        ab_diff = a_rate - b_rate
        if ab_diff > 0.2:
            ct += 5
        
        return min(100, max(0, ct))
    
    def classify_zone(
        self,
        a_rate: float,
        ct_value: float
    ) -> ZoneType:
        """
        レースをゾーン分類
        
        Args:
            a_rate: A率
            ct_value: CT値
            
        Returns:
            ZoneType: ゾーン分類
        """
        th = self.thresholds
        
        # ガチゾーン: A率が高く、CT値も高い
        if a_rate >= th.gachi_a_rate and ct_value >= th.gachi_ct_value:
            return ZoneType.GACHI
        
        # ブルーゾーン: CT値が高い
        if ct_value >= th.blue_ct_value:
            return ZoneType.BLUE
        
        # トワイライトゾーン: 中間領域
        if ct_value >= th.twilight_ct_min and a_rate <= th.twilight_a_rate_max:
            return ZoneType.TWILIGHT
        
        # レッドゾーン: CT値が低い
        return ZoneType.RED
    
    def generate_recommended_bets(
        self,
        race_df: pd.DataFrame,
        zone: ZoneType,
    ) -> List[str]:
        """
        ゾーンに基づいて推奨買い目を生成
        
        Args:
            race_df: ランキング済みレースデータ
            zone: ゾーン分類
            
        Returns:
            List[str]: 推奨買い目リスト
        """
        sorted_df = race_df.sort_values("rank_order").reset_index(drop=True)
        
        if len(sorted_df) < 3:
            return []
        
        # 上位選手の車番
        a_car = int(sorted_df.loc[0, "car_number"])
        b_car = int(sorted_df.loc[1, "car_number"])
        c_car = int(sorted_df.loc[2, "car_number"])
        d_car = int(sorted_df.loc[3, "car_number"]) if len(sorted_df) > 3 else None
        
        bets = []
        
        if zone == ZoneType.GACHI:
            # ガチゾーン: 3連複1点
            bets.append(f"3連複 {a_car}-{b_car}-{c_car}")
            bets.append(f"3連単 {a_car}→{b_car}→{c_car}")
        
        elif zone == ZoneType.BLUE:
            # ブルーゾーン: 3連複2点
            bets.append(f"3連複 {a_car}-{b_car}-{c_car}")
            if d_car:
                bets.append(f"3連複 {a_car}-{b_car}-{d_car}")
        
        elif zone == ZoneType.TWILIGHT:
            # トワイライトゾーン: 3連複3点
            bets.append(f"3連複 {a_car}-{b_car}-{c_car}")
            if d_car:
                bets.append(f"3連複 {a_car}-{b_car}-{d_car}")
                bets.append(f"3連複 {a_car}-{c_car}-{d_car}")
        
        else:  # RED
            # レッドゾーン: 荒れ予想、広く買う
            bets.append("※荒れる可能性高 - 穴狙い推奨")
            if d_car:
                bets.append(f"3連複 {a_car}-{b_car}-{d_car}")
                bets.append(f"3連複 {a_car}-{c_car}-{d_car}")
                bets.append(f"3連複 {b_car}-{c_car}-{d_car}")
        
        return bets
    
    def process_race(
        self,
        race_df: pd.DataFrame,
        race_info: Dict,
    ) -> RacePrediction:
        """
        1レース分の予測結果を処理
        
        Args:
            race_df: 予測済みレースデータ（ランキング付き）
            race_info: レース基本情報
            
        Returns:
            RacePrediction: 完全な予測結果
        """
        # 指標算出
        indicators = self.calculate_indicators(race_df)
        
        # ゾーン分類
        zone = self.classify_zone(
            indicators["a_rate"],
            indicators["ct_value"]
        )
        
        # 推奨買い目
        recommended_bets = self.generate_recommended_bets(race_df, zone)
        
        return RacePrediction(
            race_id=race_info.get("race_id", ""),
            date=race_info.get("date", ""),
            venue_name=race_info.get("venue_name", ""),
            race_no=race_info.get("race_no", 0),
            rankings=race_df,
            a_rate=indicators["a_rate"],
            b_rate=indicators["b_rate"],
            c_rate=indicators["c_rate"],
            ct_value=indicators["ct_value"],
            ks_value=indicators["ks_value"],
            zone=zone,
            recommended_bets=recommended_bets,
        )
    
    def process_all_races(
        self,
        predictions_df: pd.DataFrame,
    ) -> List[RacePrediction]:
        """
        全レースの予測結果を処理
        
        Args:
            predictions_df: 全レースの予測済みデータ
            
        Returns:
            List[RacePrediction]: 全レースの予測結果
        """
        results = []
        
        for race_id in predictions_df["race_id"].unique():
            race_df = predictions_df[predictions_df["race_id"] == race_id].copy()
            
            # レース情報を抽出
            race_info = {
                "race_id": race_id,
                "date": race_df["date"].iloc[0] if "date" in race_df.columns else "",
                "venue_name": race_df["venue_name"].iloc[0] if "venue_name" in race_df.columns else "",
                "race_no": race_df["race_no"].iloc[0] if "race_no" in race_df.columns else 0,
            }
            
            # ランキング順にソート
            if "rank_order" in race_df.columns:
                race_df = race_df.sort_values("rank_order")
            elif "adjusted_score" in race_df.columns:
                race_df = race_df.sort_values("adjusted_score", ascending=False)
                race_df["rank_order"] = range(1, len(race_df) + 1)
            
            prediction = self.process_race(race_df, race_info)
            results.append(prediction)
        
        logger.info(f"Processed {len(results)} races")
        return results
    
    def summarize_predictions(
        self,
        predictions: List[RacePrediction]
    ) -> pd.DataFrame:
        """
        予測結果をサマリーDataFrameに変換
        
        Args:
            predictions: 予測結果リスト
            
        Returns:
            pd.DataFrame: サマリーデータ
        """
        records = []
        
        for pred in predictions:
            rankings = pred.rankings
            
            # 上位3選手の情報
            top3 = rankings.head(3)
            
            record = {
                "race_id": pred.race_id,
                "date": pred.date,
                "venue_name": pred.venue_name,
                "race_no": pred.race_no,
                "A_car": int(top3.iloc[0]["car_number"]) if len(top3) > 0 else None,
                "B_car": int(top3.iloc[1]["car_number"]) if len(top3) > 1 else None,
                "C_car": int(top3.iloc[2]["car_number"]) if len(top3) > 2 else None,
                "A_rate": pred.a_rate,
                "B_rate": pred.b_rate,
                "C_rate": pred.c_rate,
                "CT_value": pred.ct_value,
                "KS_value": pred.ks_value,
                "zone": pred.zone.value,
                "recommended_bet": pred.recommended_bets[0] if pred.recommended_bets else "",
            }
            
            # A, B, C選手の詳細
            if "player_name" in rankings.columns:
                record["A_name"] = rankings.iloc[0]["player_name"] if len(rankings) > 0 else ""
                record["B_name"] = rankings.iloc[1]["player_name"] if len(rankings) > 1 else ""
                record["C_name"] = rankings.iloc[2]["player_name"] if len(rankings) > 2 else ""
            
            records.append(record)
        
        return pd.DataFrame(records)


class HighConfidenceFilter:
    """
    高信頼度レースのフィルタリング
    """
    
    def __init__(
        self,
        min_a_rate: float = 0.7,
        min_ct_value: float = 50.0,
    ):
        self.min_a_rate = min_a_rate
        self.min_ct_value = min_ct_value
    
    def filter_gachi_races(
        self,
        predictions: List[RacePrediction]
    ) -> List[RacePrediction]:
        """ガチゾーンのレースのみ抽出"""
        return [p for p in predictions if p.zone == ZoneType.GACHI]
    
    def filter_high_confidence(
        self,
        predictions: List[RacePrediction]
    ) -> List[RacePrediction]:
        """高信頼度レースを抽出"""
        return [
            p for p in predictions
            if p.a_rate >= self.min_a_rate and p.ct_value >= self.min_ct_value
        ]
    
    def filter_by_ks_value(
        self,
        predictions: List[RacePrediction],
        min_ks: float = 0.1,
    ) -> List[RacePrediction]:
        """KS値で絞り込み（AとBの差が大きいレース）"""
        return [p for p in predictions if p.ks_value >= min_ks]
