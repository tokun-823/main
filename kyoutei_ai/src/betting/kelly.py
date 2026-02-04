"""
資金管理・賭け戦略
ケリー基準・ホースケリー（排他ケリー）の実装
"""
import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config


@dataclass
class BetRecommendation:
    """賭け推奨"""
    combination: Tuple[int, int, int]  # 買い目
    probability: float  # 推定確率
    odds: float  # オッズ
    expected_value: float  # 期待値
    kelly_fraction: float  # ケリー基準による投資比率
    bet_amount: int  # 推奨購入金額（100円単位）
    ev_q10: float = 0  # 期待値下限
    ev_q50: float = 0  # 期待値中央
    ev_q80: float = 0  # 期待値上限


@dataclass
class BettingPlan:
    """購入計画"""
    race_id: str
    bankroll: int  # 資金
    total_bet: int  # 総賭け金
    bets: List[BetRecommendation] = field(default_factory=list)
    remaining: int = 0  # 余剰資金
    expected_return: float = 0  # 期待リターン
    risk_of_ruin: float = 0  # 破産リスク


class KellyCriterion:
    """ケリー基準による資金配分"""
    
    def __init__(
        self,
        kelly_fraction: float = None,
        min_bet: int = None,
        max_bet_ratio: float = None
    ):
        self.kelly_fraction = kelly_fraction or config.betting.kelly_fraction
        self.min_bet = min_bet or config.betting.min_bet_amount
        self.max_bet_ratio = max_bet_ratio or config.betting.max_bet_ratio
    
    def simple_kelly(self, probability: float, odds: float) -> float:
        """
        単純ケリー基準
        f = p - (1-p)/b
        where:
            f = 投資比率
            p = 勝率
            b = 実質オッズ（オッズ - 1）
        """
        if odds <= 1:
            return 0
        
        b = odds - 1
        f = probability - (1 - probability) / b
        
        return max(0, f)
    
    def fractional_kelly(self, probability: float, odds: float) -> float:
        """フラクショナルケリー（リスク低減版）"""
        base_kelly = self.simple_kelly(probability, odds)
        return base_kelly * self.kelly_fraction
    
    def calculate_bet(
        self,
        bankroll: int,
        probability: float,
        odds: float
    ) -> int:
        """購入金額を計算"""
        
        fraction = self.fractional_kelly(probability, odds)
        
        # 最大購入比率の制限
        fraction = min(fraction, self.max_bet_ratio)
        
        # 金額計算
        amount = int(bankroll * fraction)
        
        # 100円単位に丸め（切り捨て）
        amount = (amount // 100) * 100
        
        # 最小購入額チェック
        if amount < self.min_bet:
            return 0
        
        return amount


class HorseKelly:
    """
    ホースケリー（排他ケリー）
    複数の排反事象（同時に1つしか当たらない）に対する最適資金配分
    ラグランジュの未定乗数法を使用
    """
    
    def __init__(
        self,
        kelly_fraction: float = None,
        min_bet: int = None,
        max_bet_ratio: float = None
    ):
        self.kelly_fraction = kelly_fraction or config.betting.kelly_fraction
        self.min_bet = min_bet or config.betting.min_bet_amount
        self.max_bet_ratio = max_bet_ratio or config.betting.max_bet_ratio
    
    def optimize_allocation(
        self,
        probabilities: np.ndarray,
        odds: np.ndarray,
        bankroll: int
    ) -> np.ndarray:
        """
        ホースケリーによる最適配分を計算
        
        ラグランジュの未定乗数法で解く問題:
            maximize: E[log(W)] = Σ p_i * log(1 + f_i * (o_i - 1) - Σ_{j≠i} f_j)
            subject to: Σ f_i <= 1, f_i >= 0
        
        Args:
            probabilities: 各買い目の勝率 (n,)
            odds: 各買い目のオッズ (n,)
            bankroll: 資金
        
        Returns:
            fractions: 各買い目への投資比率 (n,)
        """
        
        n = len(probabilities)
        
        if n == 0:
            return np.array([])
        
        # 期待値が1以上の買い目のみ
        ev = probabilities * odds
        valid_mask = ev >= 1.0
        
        if not np.any(valid_mask):
            return np.zeros(n)
        
        valid_probs = probabilities[valid_mask]
        valid_odds = odds[valid_mask]
        n_valid = len(valid_probs)
        
        def negative_log_growth(f):
            """対数資産成長率の負値（最小化用）"""
            total_bet = np.sum(f)
            
            # 各イベントが起きた時の対数リターン
            log_returns = []
            for i in range(n_valid):
                # イベント i が起きた時のリターン
                win_return = f[i] * valid_odds[i] + (1 - total_bet)
                if win_return <= 0:
                    return 1e10  # ペナルティ
                log_returns.append(valid_probs[i] * np.log(win_return))
            
            # どのイベントも起きなかった時
            none_prob = max(0, 1 - np.sum(valid_probs))
            if none_prob > 0 and (1 - total_bet) > 0:
                log_returns.append(none_prob * np.log(1 - total_bet))
            
            return -np.sum(log_returns)
        
        # 初期値
        x0 = np.ones(n_valid) / (n_valid + 1)
        
        # 制約条件
        constraints = [
            {'type': 'ineq', 'fun': lambda f: 1 - np.sum(f)},  # Σf <= 1
        ]
        
        # 境界条件
        bounds = [(0, self.max_bet_ratio) for _ in range(n_valid)]
        
        # 最適化
        result = minimize(
            negative_log_growth,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if result.success:
            optimal_fractions = result.x
        else:
            logger.warning(f"Optimization failed: {result.message}")
            # フォールバック: 単純な比例配分
            optimal_fractions = (valid_probs * valid_odds - 1) / np.sum(valid_probs * valid_odds - 1 + 1e-10)
            optimal_fractions = np.clip(optimal_fractions, 0, self.max_bet_ratio)
        
        # フラクショナルケリー適用
        optimal_fractions *= self.kelly_fraction
        
        # 結果を元の配列に戻す
        result_fractions = np.zeros(n)
        result_fractions[valid_mask] = optimal_fractions
        
        return result_fractions
    
    def calculate_bets(
        self,
        predictions: List[Dict],
        bankroll: int,
        min_ev: float = 1.0
    ) -> BettingPlan:
        """
        購入計画を作成
        
        Args:
            predictions: 予測結果のリスト
                [{'combination': (1,2,3), 'probability': 0.05, 'odds': 30.0, ...}, ...]
            bankroll: 資金
            min_ev: 最小期待値閾値
        
        Returns:
            BettingPlan: 購入計画
        """
        
        # 期待値でフィルタリング
        valid_preds = [
            p for p in predictions
            if p['probability'] * p.get('prelim_odds', p.get('odds', 0)) >= min_ev
            and p.get('prelim_odds', p.get('odds', 0)) > 0
        ]
        
        if not valid_preds:
            return BettingPlan(
                race_id="",
                bankroll=bankroll,
                total_bet=0,
                remaining=bankroll
            )
        
        # 配列に変換
        probs = np.array([p['probability'] for p in valid_preds])
        odds = np.array([p.get('prelim_odds', p.get('odds', 0)) for p in valid_preds])
        
        # 最適配分を計算
        fractions = self.optimize_allocation(probs, odds, bankroll)
        
        # 購入金額を計算
        bets = []
        total_bet = 0
        
        for i, pred in enumerate(valid_preds):
            amount = int(bankroll * fractions[i])
            amount = (amount // 100) * 100  # 100円単位
            
            if amount >= self.min_bet:
                ev = pred['probability'] * pred.get('prelim_odds', pred.get('odds', 0))
                
                bet = BetRecommendation(
                    combination=pred['combination'],
                    probability=pred['probability'],
                    odds=pred.get('prelim_odds', pred.get('odds', 0)),
                    expected_value=ev,
                    kelly_fraction=fractions[i],
                    bet_amount=amount,
                    ev_q10=pred.get('ev_low', ev),
                    ev_q50=pred.get('ev_mid', ev),
                    ev_q80=pred.get('ev_high', ev)
                )
                bets.append(bet)
                total_bet += amount
        
        # 期待リターン計算
        expected_return = sum(b.probability * b.odds * b.bet_amount for b in bets)
        
        plan = BettingPlan(
            race_id="",
            bankroll=bankroll,
            total_bet=total_bet,
            bets=bets,
            remaining=bankroll - total_bet,
            expected_return=expected_return
        )
        
        return plan


class RiskManager:
    """リスク管理"""
    
    def __init__(
        self,
        max_daily_loss_ratio: float = None
    ):
        self.max_daily_loss_ratio = max_daily_loss_ratio or config.betting.max_daily_loss_ratio
        self.daily_pnl = 0
        self.initial_bankroll = 0
    
    def start_day(self, bankroll: int):
        """日次セッション開始"""
        self.initial_bankroll = bankroll
        self.daily_pnl = 0
    
    def update_pnl(self, amount: int):
        """損益更新"""
        self.daily_pnl += amount
    
    def can_bet(self) -> bool:
        """賭けが可能か判定"""
        if self.initial_bankroll == 0:
            return True
        
        loss_ratio = -self.daily_pnl / self.initial_bankroll
        return loss_ratio < self.max_daily_loss_ratio
    
    def get_available_bankroll(self, current_bankroll: int) -> int:
        """利用可能な資金を取得"""
        if not self.can_bet():
            return 0
        
        # 最大損失額を考慮
        max_loss = int(self.initial_bankroll * self.max_daily_loss_ratio)
        remaining_loss_capacity = max_loss + self.daily_pnl
        
        return min(current_bankroll, remaining_loss_capacity)


class BettingSimulator:
    """賭けシミュレーター"""
    
    def __init__(self, initial_bankroll: int = 100000):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history = []
    
    def reset(self):
        """リセット"""
        self.bankroll = self.initial_bankroll
        self.history = []
    
    def simulate_bet(
        self,
        plan: BettingPlan,
        actual_result: Tuple[int, int, int]
    ) -> Dict:
        """1レースのシミュレーション"""
        
        total_bet = plan.total_bet
        payout = 0
        
        # 的中判定
        for bet in plan.bets:
            if bet.combination == actual_result:
                payout = int(bet.bet_amount * bet.odds)
                break
        
        # 損益計算
        pnl = payout - total_bet
        self.bankroll += pnl
        
        result = {
            'total_bet': total_bet,
            'payout': payout,
            'pnl': pnl,
            'bankroll': self.bankroll,
            'hit': payout > 0
        }
        
        self.history.append(result)
        
        return result
    
    def get_statistics(self) -> Dict:
        """統計を取得"""
        if not self.history:
            return {}
        
        total_bets = sum(h['total_bet'] for h in self.history)
        total_payout = sum(h['payout'] for h in self.history)
        hits = sum(1 for h in self.history if h['hit'])
        
        return {
            'total_races': len(self.history),
            'total_bets': total_bets,
            'total_payout': total_payout,
            'total_pnl': total_payout - total_bets,
            'hit_rate': hits / len(self.history) if self.history else 0,
            'roi': total_payout / total_bets if total_bets > 0 else 0,
            'final_bankroll': self.bankroll,
            'max_drawdown': self._calculate_max_drawdown()
        }
    
    def _calculate_max_drawdown(self) -> float:
        """最大ドローダウンを計算"""
        if not self.history:
            return 0
        
        bankrolls = [self.initial_bankroll] + [h['bankroll'] for h in self.history]
        peak = bankrolls[0]
        max_dd = 0
        
        for br in bankrolls:
            if br > peak:
                peak = br
            dd = (peak - br) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd
