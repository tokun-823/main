# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」- イカサマサイコロ投票モジュール
Weighted Random Betting ("Rigged Dice") Module for Keirin Prediction AI "Pasoko"

AIの確率を「重み（Weight）」として利用した重み付きランダム抽選機能
"""

import random
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations, permutations


class IkasamaDice:
    """
    イカサマサイコロクラス
    
    AIが出力した確率を重みとして使用し、
    重み付きランダム抽選で車券を生成する
    
    重要: 確率を「2乗」した数値を重みとして使用することで、
    強い選手がより高頻度で出現する「当たるイカサマサイコロ」を実現
    """
    
    def __init__(self, power=2.0):
        """
        Args:
            power: 確率の累乗数（デフォルト: 2.0）
                   高いほど強い選手の出現率が上がる
        """
        self.power = power
        random.seed(None)  # ランダムシード
    
    def _calculate_weights(self, probabilities, use_power=True):
        """
        重みを計算
        
        Args:
            probabilities: 確率リスト
            use_power: べき乗を適用するか
            
        Returns:
            list: 重みリスト
        """
        if use_power:
            # 確率を2乗（または指定のべき乗）
            weights = [max(0.001, p) ** self.power for p in probabilities]
        else:
            weights = [max(0.001, p) for p in probabilities]
        
        # 正規化
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        return weights
    
    def roll_single(self, players, probabilities, use_power=True):
        """
        サイコロを1回振る（1名選出）
        
        Args:
            players: 選手リスト（車番など）
            probabilities: 各選手の確率リスト
            use_power: べき乗を適用するか
            
        Returns:
            選出された選手
        """
        weights = self._calculate_weights(probabilities, use_power)
        selected = random.choices(players, weights=weights, k=1)
        return selected[0]
    
    def roll_multiple(self, players, probabilities, k=3, unique=True, use_power=True):
        """
        サイコロを複数回振る（複数名選出）
        
        Args:
            players: 選手リスト
            probabilities: 各選手の確率リスト
            k: 選出人数
            unique: 重複を許可しないか
            use_power: べき乗を適用するか
            
        Returns:
            list: 選出された選手リスト
        """
        weights = self._calculate_weights(probabilities, use_power)
        
        if unique:
            # 重複なし: 1人ずつ選んで除外
            selected = []
            remaining_players = list(players)
            remaining_weights = list(weights)
            
            for _ in range(min(k, len(players))):
                if not remaining_players:
                    break
                
                # 重みを再正規化
                total_w = sum(remaining_weights)
                normalized_weights = [w / total_w for w in remaining_weights] if total_w > 0 else [1/len(remaining_weights)] * len(remaining_weights)
                
                # 選出
                idx = random.choices(range(len(remaining_players)), weights=normalized_weights, k=1)[0]
                selected.append(remaining_players[idx])
                
                # 除外
                remaining_players.pop(idx)
                remaining_weights.pop(idx)
            
            return selected
        else:
            # 重複あり
            return random.choices(players, weights=weights, k=k)
    
    def generate_trifecta_bet(self, df, race_id, use_power=True):
        """
        3連複の車券を生成
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            use_power: べき乗を適用するか
            
        Returns:
            tuple: 3連複の組み合わせ
        """
        race_df = df[df['race_id'] == race_id]
        
        if len(race_df) < 3:
            return None
        
        # プラスの確率の選手のみ対象
        positive_df = race_df[race_df['pred_label'] == 1].copy()
        
        if len(positive_df) < 3:
            # プラスが3人未満の場合は上位から補充
            positive_df = race_df.sort_values('pred_score', ascending=False).head(3)
        
        players = positive_df['車番'].tolist() if '車番' in positive_df.columns else list(range(1, len(positive_df) + 1))
        probabilities = positive_df['pred_proba'].tolist()
        
        # 3名選出
        selected = self.roll_multiple(players, probabilities, k=3, unique=True, use_power=use_power)
        
        return tuple(sorted(selected))
    
    def generate_exacta_bet(self, df, race_id, use_power=True):
        """
        3連単の車券を生成（順序あり）
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            use_power: べき乗を適用するか
            
        Returns:
            tuple: 3連単の組み合わせ（1着, 2着, 3着）
        """
        race_df = df[df['race_id'] == race_id]
        
        if len(race_df) < 3:
            return None
        
        players = race_df['車番'].tolist() if '車番' in race_df.columns else list(range(1, len(race_df) + 1))
        probabilities = race_df['pred_proba'].tolist()
        
        # 1着を選出（より強い重み）
        weights_1st = self._calculate_weights(probabilities, use_power=True)
        # さらに1着は重みを強くする
        weights_1st = [w ** 1.5 for w in weights_1st]
        total = sum(weights_1st)
        weights_1st = [w / total for w in weights_1st]
        
        first = random.choices(players, weights=weights_1st, k=1)[0]
        
        # 残りから2着を選出
        remaining_players = [p for p in players if p != first]
        remaining_probas = [probabilities[players.index(p)] for p in remaining_players]
        weights_2nd = self._calculate_weights(remaining_probas, use_power=True)
        
        second = random.choices(remaining_players, weights=weights_2nd, k=1)[0]
        
        # 残りから3着を選出
        remaining_players = [p for p in remaining_players if p != second]
        remaining_probas = [probabilities[players.index(p)] for p in remaining_players]
        weights_3rd = self._calculate_weights(remaining_probas, use_power=True)
        
        third = random.choices(remaining_players, weights=weights_3rd, k=1)[0]
        
        return (first, second, third)
    
    def generate_multiple_bets(self, df, race_id, n_bets=5, bet_type='trifecta', use_power=True):
        """
        複数の車券を生成
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            n_bets: 生成する車券数
            bet_type: 'trifecta' or 'exacta'
            use_power: べき乗を適用するか
            
        Returns:
            list: 車券リスト
        """
        bets = []
        
        for _ in range(n_bets):
            if bet_type == 'trifecta':
                bet = self.generate_trifecta_bet(df, race_id, use_power)
            else:
                bet = self.generate_exacta_bet(df, race_id, use_power)
            
            if bet:
                bets.append(bet)
        
        return bets
    
    def generate_bets_by_zone(self, df, race_id, ct_value=None):
        """
        ゾーン（CT値）に応じた車券生成
        
        CT値が低い（荒れる）レースではサイコロを振る回数を増やす
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            ct_value: CT値（Noneの場合はDFから取得）
            
        Returns:
            list: 車券リスト
        """
        race_df = df[df['race_id'] == race_id]
        
        if len(race_df) == 0:
            return []
        
        if ct_value is None:
            ct_value = race_df['ct_value'].iloc[0] if 'ct_value' in race_df.columns else 0.5
        
        # CT値に応じて試行回数を決定
        if ct_value >= 0.8:
            n_bets = 1  # ガチゾーン: 1点勝負
        elif ct_value >= 0.6:
            n_bets = 3  # ブルー〜トワイライト
        elif ct_value >= 0.4:
            n_bets = 5  # 波乱含み
        else:
            n_bets = 10  # レッドゾーン: 広く
        
        bets = self.generate_multiple_bets(df, race_id, n_bets=n_bets, bet_type='trifecta')
        
        return bets
    
    def analyze_bet_distribution(self, bets, n_players=7):
        """
        車券の出現分布を分析
        
        Args:
            bets: 車券リスト
            n_players: 選手数
            
        Returns:
            dict: 各選手の出現回数
        """
        all_numbers = []
        for bet in bets:
            all_numbers.extend(bet)
        
        counter = Counter(all_numbers)
        
        # 出現率を計算
        total = len(all_numbers)
        distribution = {}
        
        for i in range(1, n_players + 1):
            count = counter.get(i, 0)
            rate = count / total if total > 0 else 0
            distribution[i] = {'count': count, 'rate': rate}
        
        return distribution
    
    def print_bet_analysis(self, bets, n_players=7):
        """車券分析結果を表示"""
        dist = self.analyze_bet_distribution(bets, n_players)
        
        print("\n【車券分布分析】")
        print(f"生成した車券数: {len(bets)}")
        print("\n車番別出現率:")
        
        for num in sorted(dist.keys()):
            info = dist[num]
            bar = '█' * int(info['rate'] * 50)
            print(f"  {num}番: {info['count']:3d}回 ({info['rate']*100:5.1f}%) {bar}")
    
    def simulation(self, df, race_id, n_simulations=1000, bet_type='trifecta'):
        """
        シミュレーション実行
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            n_simulations: シミュレーション回数
            bet_type: 車券タイプ
            
        Returns:
            dict: シミュレーション結果
        """
        bets = []
        
        for _ in range(n_simulations):
            if bet_type == 'trifecta':
                bet = self.generate_trifecta_bet(df, race_id)
            else:
                bet = self.generate_exacta_bet(df, race_id)
            
            if bet:
                bets.append(bet)
        
        # 最頻出の組み合わせ
        bet_counter = Counter([tuple(sorted(b) if bet_type == 'trifecta' else b) for b in bets])
        most_common = bet_counter.most_common(10)
        
        return {
            'total_simulations': n_simulations,
            'unique_combinations': len(bet_counter),
            'most_common': most_common
        }
    
    def print_simulation_result(self, result):
        """シミュレーション結果を表示"""
        print(f"\n【シミュレーション結果】")
        print(f"総試行回数: {result['total_simulations']}")
        print(f"ユニーク組み合わせ数: {result['unique_combinations']}")
        print("\n最頻出の組み合わせ TOP10:")
        
        for i, (combo, count) in enumerate(result['most_common'], 1):
            rate = count / result['total_simulations'] * 100
            print(f"  {i:2d}. {combo}: {count:4d}回 ({rate:5.2f}%)")


class AdaptiveDice(IkasamaDice):
    """
    適応型イカサマサイコロ
    
    過去の的中データを基に重みを動的に調整
    """
    
    def __init__(self, base_power=2.0):
        super().__init__(power=base_power)
        self.hit_history = []
        self.adaptive_factor = 1.0
    
    def update_history(self, bet, actual_result, is_hit):
        """
        的中履歴を更新
        
        Args:
            bet: 購入した車券
            actual_result: 実際の結果
            is_hit: 的中したか
        """
        self.hit_history.append({
            'bet': bet,
            'result': actual_result,
            'hit': is_hit
        })
        
        # 直近の的中率で適応係数を調整
        recent = self.hit_history[-50:]
        if len(recent) >= 10:
            hit_rate = sum(1 for h in recent if h['hit']) / len(recent)
            
            if hit_rate < 0.1:
                # 的中率が低い場合: 分散を増やす
                self.adaptive_factor = 0.8
            elif hit_rate > 0.3:
                # 的中率が高い場合: 集中させる
                self.adaptive_factor = 1.2
            else:
                self.adaptive_factor = 1.0
    
    def _calculate_weights(self, probabilities, use_power=True):
        """適応係数を考慮した重み計算"""
        base_weights = super()._calculate_weights(probabilities, use_power)
        
        # 適応係数を適用
        adapted_weights = [w ** self.adaptive_factor for w in base_weights]
        
        # 正規化
        total = sum(adapted_weights)
        if total > 0:
            adapted_weights = [w / total for w in adapted_weights]
        
        return adapted_weights


if __name__ == "__main__":
    # テスト
    print("イカサマサイコロテスト")
    
    dice = IkasamaDice(power=2.0)
    
    # ダミーデータ
    test_players = [1, 2, 3, 4, 5, 6, 7]
    test_probas = [0.85, 0.70, 0.55, 0.40, 0.25, 0.15, 0.10]
    
    # 単発テスト
    print("\n単発抽選テスト（100回）:")
    results = [dice.roll_single(test_players, test_probas) for _ in range(100)]
    counter = Counter(results)
    
    for num in sorted(counter.keys()):
        bar = '█' * counter[num]
        print(f"  {num}番: {counter[num]:3d}回 {bar}")
    
    # 3連複生成テスト
    print("\n3連複生成テスト（50回）:")
    test_df = pd.DataFrame({
        'race_id': ['test'] * 7,
        '車番': test_players,
        'pred_proba': test_probas,
        'pred_label': [1, 1, 1, 0, 0, 0, 0],
        'pred_score': [0.85, 0.70, 0.55, -0.40, -0.25, -0.15, -0.10]
    })
    
    bets = dice.generate_multiple_bets(test_df, 'test', n_bets=50, bet_type='trifecta')
    dice.print_bet_analysis(bets, n_players=7)
    
    # シミュレーション
    print("\n1000回シミュレーション:")
    sim_result = dice.simulation(test_df, 'test', n_simulations=1000)
    dice.print_simulation_result(sim_result)
