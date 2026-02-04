# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- イカサマサイコロ投票機能
=============================================
重み付き抽選による買い目生成
中穴〜大穴を狙うための実験的機能
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
from loguru import logger

from .config import DiceConfig, config


@dataclass
class DiceResult:
    """サイコロ抽選結果"""
    first: int       # 1着予想車番
    second: int      # 2着予想車番
    third: int       # 3着予想車番
    
    def to_sanrentan(self) -> str:
        """3連単形式"""
        return f"{self.first}→{self.second}→{self.third}"
    
    def to_sanrenpuku(self) -> str:
        """3連複形式"""
        cars = sorted([self.first, self.second, self.third])
        return f"{cars[0]}-{cars[1]}-{cars[2]}"


class IkasamaDice:
    """
    イカサマサイコロ
    
    AIが算出した確率を重みとして利用し、
    ランダム性を持たせつつ有力選手が出やすい
    「重み付き抽選」で買い目を生成
    
    特徴:
    - 確率を2乗等で加工し、強い選手がより選ばれやすくする
    - random.choices による重み付き抽選
    - 本命（ABC）以外の中穴〜大穴（AB-D, C-AB等）を狙う
    """
    
    def __init__(self, dice_config: Optional[DiceConfig] = None):
        self.config = dice_config or config.dice
    
    def _adjust_weights(
        self,
        probas: List[float],
        power: Optional[float] = None,
    ) -> List[float]:
        """
        確率を加工して重みを作成
        
        Args:
            probas: 各選手の確率リスト
            power: 累乗の指数（デフォルト: config.power_factor）
            
        Returns:
            List[float]: 調整後の重みリスト
        """
        if power is None:
            power = self.config.power_factor
        
        # 確率を累乗して差を強調
        weights = [p ** power for p in probas]
        
        # 正規化（合計が1になるように）
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            # 全て0の場合は均等
            weights = [1.0 / len(probas)] * len(probas)
        
        return weights
    
    def roll_once(
        self,
        car_numbers: List[int],
        probas: List[float],
        power: Optional[float] = None,
    ) -> DiceResult:
        """
        1回抽選を実行
        
        Args:
            car_numbers: 車番リスト
            probas: 各車番の確率リスト
            power: 累乗の指数
            
        Returns:
            DiceResult: 抽選結果
        """
        if len(car_numbers) < 3:
            raise ValueError("At least 3 cars required")
        
        # 重みを調整
        weights = self._adjust_weights(probas, power)
        
        # 1着を抽選
        first_idx = random.choices(range(len(car_numbers)), weights=weights, k=1)[0]
        first = car_numbers[first_idx]
        
        # 1着を除外して2着を抽選
        remaining_cars = [c for i, c in enumerate(car_numbers) if i != first_idx]
        remaining_weights = [w for i, w in enumerate(weights) if i != first_idx]
        
        # 重みを再正規化
        total = sum(remaining_weights)
        if total > 0:
            remaining_weights = [w / total for w in remaining_weights]
        else:
            remaining_weights = [1.0 / len(remaining_cars)] * len(remaining_cars)
        
        second_idx = random.choices(range(len(remaining_cars)), weights=remaining_weights, k=1)[0]
        second = remaining_cars[second_idx]
        
        # 1着・2着を除外して3着を抽選
        remaining_cars2 = [c for c in remaining_cars if c != second]
        remaining_weights2 = [w for i, w in enumerate(remaining_weights) if remaining_cars[i] != second]
        
        # 重みを再正規化
        total = sum(remaining_weights2)
        if total > 0:
            remaining_weights2 = [w / total for w in remaining_weights2]
        else:
            remaining_weights2 = [1.0 / len(remaining_cars2)] * len(remaining_cars2)
        
        third_idx = random.choices(range(len(remaining_cars2)), weights=remaining_weights2, k=1)[0]
        third = remaining_cars2[third_idx]
        
        return DiceResult(first=first, second=second, third=third)
    
    def simulate(
        self,
        car_numbers: List[int],
        probas: List[float],
        num_simulations: Optional[int] = None,
        power: Optional[float] = None,
    ) -> Dict[str, int]:
        """
        複数回シミュレーションを実行し、出目の頻度を集計
        
        Args:
            car_numbers: 車番リスト
            probas: 各車番の確率リスト
            num_simulations: シミュレーション回数
            power: 累乗の指数
            
        Returns:
            Dict[str, int]: 3連複パターンごとの出現回数
        """
        if num_simulations is None:
            num_simulations = self.config.num_draws
        
        results = []
        for _ in range(num_simulations):
            result = self.roll_once(car_numbers, probas, power)
            results.append(result.to_sanrenpuku())
        
        # 頻度を集計
        counter = Counter(results)
        return dict(counter)
    
    def get_recommended_bets(
        self,
        car_numbers: List[int],
        probas: List[float],
        top_n: int = 5,
        num_simulations: Optional[int] = None,
        power: Optional[float] = None,
    ) -> List[Tuple[str, int, float]]:
        """
        推奨買い目を取得
        
        Args:
            car_numbers: 車番リスト
            probas: 各車番の確率リスト
            top_n: 上位何点を返すか
            num_simulations: シミュレーション回数
            power: 累乗の指数
            
        Returns:
            List[Tuple[str, int, float]]: (買い目, 出現回数, 確率)のリスト
        """
        if num_simulations is None:
            num_simulations = self.config.num_draws
        
        # シミュレーション実行
        freq_dict = self.simulate(car_numbers, probas, num_simulations, power)
        
        # 頻度順にソート
        sorted_bets = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        
        # 上位N件を返す
        results = []
        for bet, count in sorted_bets[:top_n]:
            prob = count / num_simulations
            results.append((bet, count, prob))
        
        return results
    
    def roll_for_race(
        self,
        race_df: pd.DataFrame,
        proba_col: str = "pred_proba",
        power: Optional[float] = None,
    ) -> DiceResult:
        """
        レースデータから抽選を実行
        
        Args:
            race_df: レースデータ（車番と確率を含む）
            proba_col: 確率列名
            power: 累乗の指数
            
        Returns:
            DiceResult: 抽選結果
        """
        car_numbers = race_df["car_number"].tolist()
        probas = race_df[proba_col].tolist()
        
        return self.roll_once(car_numbers, probas, power)
    
    def get_varied_bets(
        self,
        race_df: pd.DataFrame,
        proba_col: str = "pred_proba",
        num_bets: int = 5,
        power_range: Tuple[float, float] = (1.0, 3.0),
    ) -> List[str]:
        """
        異なるパワー値で多様な買い目を生成
        
        Args:
            race_df: レースデータ
            proba_col: 確率列名
            num_bets: 生成する買い目数
            power_range: パワー値の範囲
            
        Returns:
            List[str]: 買い目リスト（重複なし）
        """
        car_numbers = race_df["car_number"].tolist()
        probas = race_df[proba_col].tolist()
        
        bets = set()
        
        # 異なるパワー値で抽選
        powers = np.linspace(power_range[0], power_range[1], num_bets * 2)
        
        for power in powers:
            result = self.roll_once(car_numbers, probas, power)
            bets.add(result.to_sanrenpuku())
            
            if len(bets) >= num_bets:
                break
        
        return list(bets)[:num_bets]


class MiddleHoleStrategy:
    """
    中穴狙い戦略
    
    ガチガチの本命（ABC）以外の、
    中穴〜大穴（AB-D, C-AB等）を狙う
    """
    
    def __init__(self, dice: Optional[IkasamaDice] = None):
        self.dice = dice or IkasamaDice()
    
    def generate_middle_hole_bets(
        self,
        race_df: pd.DataFrame,
        proba_col: str = "pred_proba",
    ) -> List[str]:
        """
        中穴買い目を生成
        
        戦略:
        1. A, B選手を軸に
        2. C以外の選手（D, E）を3着に
        
        Args:
            race_df: ランキング済みレースデータ
            proba_col: 確率列名
            
        Returns:
            List[str]: 中穴買い目リスト
        """
        # ランク順にソート
        sorted_df = race_df.sort_values("rank_order").reset_index(drop=True)
        
        if len(sorted_df) < 4:
            return []
        
        # 上位選手の車番
        a_car = int(sorted_df.loc[0, "car_number"])
        b_car = int(sorted_df.loc[1, "car_number"])
        c_car = int(sorted_df.loc[2, "car_number"])
        d_car = int(sorted_df.loc[3, "car_number"])
        e_car = int(sorted_df.loc[4, "car_number"]) if len(sorted_df) > 4 else None
        
        bets = []
        
        # パターン1: A-B-D（Cを外す）
        bets.append(self._format_sanrenpuku([a_car, b_car, d_car]))
        
        # パターン2: A-C-D（Bを外す）
        bets.append(self._format_sanrenpuku([a_car, c_car, d_car]))
        
        # パターン3: B-C-D（Aを外す）
        bets.append(self._format_sanrenpuku([b_car, c_car, d_car]))
        
        # パターン4: A-B-E（C, Dを外す）
        if e_car:
            bets.append(self._format_sanrenpuku([a_car, b_car, e_car]))
        
        # パターン5: A-D-E（B, Cを外す）
        if e_car:
            bets.append(self._format_sanrenpuku([a_car, d_car, e_car]))
        
        return bets
    
    def generate_upset_bets(
        self,
        race_df: pd.DataFrame,
    ) -> List[str]:
        """
        大穴買い目を生成
        
        戦略:
        - 上位3選手（A, B, C）を1人だけ含む
        - D, E, F を中心に
        
        Args:
            race_df: ランキング済みレースデータ
            
        Returns:
            List[str]: 大穴買い目リスト
        """
        sorted_df = race_df.sort_values("rank_order").reset_index(drop=True)
        
        if len(sorted_df) < 6:
            return []
        
        # 上位選手
        a_car = int(sorted_df.loc[0, "car_number"])
        b_car = int(sorted_df.loc[1, "car_number"])
        c_car = int(sorted_df.loc[2, "car_number"])
        
        # 下位選手
        d_car = int(sorted_df.loc[3, "car_number"])
        e_car = int(sorted_df.loc[4, "car_number"])
        f_car = int(sorted_df.loc[5, "car_number"])
        
        bets = []
        
        # Aだけを含む穴目
        bets.append(self._format_sanrenpuku([a_car, d_car, e_car]))
        bets.append(self._format_sanrenpuku([a_car, d_car, f_car]))
        bets.append(self._format_sanrenpuku([a_car, e_car, f_car]))
        
        # Bだけを含む穴目
        bets.append(self._format_sanrenpuku([b_car, d_car, e_car]))
        
        # Cだけを含む穴目
        bets.append(self._format_sanrenpuku([c_car, d_car, e_car]))
        
        # 大穴：上位3人を含まない
        bets.append(self._format_sanrenpuku([d_car, e_car, f_car]))
        
        return bets
    
    def _format_sanrenpuku(self, cars: List[int]) -> str:
        """3連複フォーマット"""
        sorted_cars = sorted(cars)
        return f"{sorted_cars[0]}-{sorted_cars[1]}-{sorted_cars[2]}"
    
    def get_strategy_bets(
        self,
        race_df: pd.DataFrame,
        strategy: str = "middle",
    ) -> List[str]:
        """
        戦略に応じた買い目を取得
        
        Args:
            race_df: レースデータ
            strategy: 戦略（"honmei", "middle", "upset"）
            
        Returns:
            List[str]: 買い目リスト
        """
        sorted_df = race_df.sort_values("rank_order").reset_index(drop=True)
        
        if strategy == "honmei":
            # 本命: A-B-C
            if len(sorted_df) < 3:
                return []
            a = int(sorted_df.loc[0, "car_number"])
            b = int(sorted_df.loc[1, "car_number"])
            c = int(sorted_df.loc[2, "car_number"])
            return [self._format_sanrenpuku([a, b, c])]
        
        elif strategy == "middle":
            return self.generate_middle_hole_bets(race_df)
        
        elif strategy == "upset":
            return self.generate_upset_bets(race_df)
        
        else:
            return []


class DiceVoting:
    """
    イカサマサイコロ投票システム
    
    複数回のシミュレーションに基づいて
    投票先を決定
    """
    
    def __init__(
        self,
        dice: Optional[IkasamaDice] = None,
        strategy: Optional[MiddleHoleStrategy] = None,
    ):
        self.dice = dice or IkasamaDice()
        self.strategy = strategy or MiddleHoleStrategy(self.dice)
    
    def vote(
        self,
        race_df: pd.DataFrame,
        budget: int = 1000,
        bet_unit: int = 100,
        num_simulations: int = 100,
    ) -> List[Dict]:
        """
        投票先を決定
        
        Args:
            race_df: レースデータ
            budget: 予算（円）
            bet_unit: 1点あたりの金額（円）
            num_simulations: シミュレーション回数
            
        Returns:
            List[Dict]: 投票リスト
        """
        car_numbers = race_df["car_number"].tolist()
        probas = race_df["pred_proba"].tolist()
        
        # シミュレーション
        freq_dict = self.dice.simulate(car_numbers, probas, num_simulations)
        
        # 予算内で購入
        max_bets = budget // bet_unit
        
        # 頻度順にソート
        sorted_bets = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        
        votes = []
        for bet, count in sorted_bets[:max_bets]:
            prob = count / num_simulations
            votes.append({
                "bet": bet,
                "amount": bet_unit,
                "frequency": count,
                "probability": prob,
            })
        
        return votes
    
    def vote_with_strategy(
        self,
        race_df: pd.DataFrame,
        strategies: List[str] = ["honmei", "middle"],
        budget: int = 1000,
        bet_unit: int = 100,
    ) -> List[Dict]:
        """
        戦略ベースの投票
        
        Args:
            race_df: レースデータ
            strategies: 使用する戦略リスト
            budget: 予算
            bet_unit: 1点あたりの金額
            
        Returns:
            List[Dict]: 投票リスト
        """
        all_bets = []
        
        for strategy in strategies:
            bets = self.strategy.get_strategy_bets(race_df, strategy)
            all_bets.extend(bets)
        
        # 重複除去
        unique_bets = list(set(all_bets))
        
        # 予算内で割り当て
        max_bets = budget // bet_unit
        selected_bets = unique_bets[:max_bets]
        
        votes = []
        for bet in selected_bets:
            votes.append({
                "bet": bet,
                "amount": bet_unit,
                "strategy": "mixed",
            })
        
        return votes
