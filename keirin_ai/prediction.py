"""
予測・指標算出モジュール
AI予測結果から独自指標（A率、CT値、KS値）を算出し、ゾーン分類を行う
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import random


class RacePrediction:
    """レース予測クラス"""
    
    def __init__(self, race_id: str, race_data: pd.DataFrame, probabilities: np.ndarray):
        """
        Args:
            race_id: レースID
            race_data: レースの選手データ（DataFrame）
            probabilities: AIが予測した各選手の3着以内確率
        """
        self.race_id = race_id
        self.race_data = race_data.copy()
        self.race_data['probability'] = probabilities
        self.race_data['prediction'] = (probabilities >= 0.5).astype(int)
        
        # ランキング生成
        self._create_ranking()
        
        # 指標算出
        self.a_rate = self._calculate_a_rate()
        self.ct_value = self._calculate_ct_value()
        self.ks_value = self._calculate_ks_value()
        
        # ゾーン分類
        self.zone = self._classify_zone()
    
    def _create_ranking(self):
        """
        選手ランキングの生成
        確率が高い順にA, B, C, D, E, F, G...と付与
        0と判定された選手は確率をマイナスにしてソート順を下げる
        """
        # 予測が0の選手は確率を負にする
        adjusted_prob = self.race_data['probability'].copy()
        adjusted_prob[self.race_data['prediction'] == 0] *= -1
        
        # ソート
        self.race_data['adjusted_probability'] = adjusted_prob
        self.race_data = self.race_data.sort_values('adjusted_probability', ascending=False)
        
        # ランク記号付与
        rank_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.race_data['rank'] = rank_labels[:len(self.race_data)]
        
        # インデックスをリセット
        self.race_data = self.race_data.reset_index(drop=True)
    
    def _calculate_a_rate(self) -> float:
        """
        A率の算出
        最上位選手（A）の3着以内確率
        """
        a_player = self.race_data[self.race_data['rank'] == 'A']
        if len(a_player) > 0:
            return float(a_player['probability'].iloc[0])
        return 0.0
    
    def _calculate_ct_value(self) -> float:
        """
        CT値（カラータイマー値）の算出
        上位3名（A, B, C）で決まる確率の高さを示す独自指標
        
        ロジック:
        - A, B, Cが全てプラス判定（prediction=1）の場合: 高い値
        - マイナス判定が含まれる場合: 低い値
        - B率 - C率も考慮
        """
        top3 = self.race_data[self.race_data['rank'].isin(['A', 'B', 'C'])]
        
        if len(top3) < 3:
            return 0.0
        
        a_prob = top3[top3['rank'] == 'A']['probability'].iloc[0]
        b_prob = top3[top3['rank'] == 'B']['probability'].iloc[0]
        c_prob = top3[top3['rank'] == 'C']['probability'].iloc[0]
        
        a_pred = top3[top3['rank'] == 'A']['prediction'].iloc[0]
        b_pred = top3[top3['rank'] == 'B']['prediction'].iloc[0]
        c_pred = top3[top3['rank'] == 'C']['prediction'].iloc[0]
        
        # 基本CT値：確率の合計
        base_ct = (a_prob + b_prob + c_prob) / 3 * 100
        
        # 全員プラス判定ならボーナス
        if a_pred == 1 and b_pred == 1 and c_pred == 1:
            bonus = 20
        elif a_pred == 1 and b_pred == 1:
            bonus = 10
        elif a_pred == 1:
            bonus = 5
        else:
            bonus = 0
        
        ct_value = base_ct + bonus
        
        return round(ct_value, 2)
    
    def _calculate_ks_value(self) -> float:
        """
        KS値の算出
        B率 - C率
        
        大きい値 = AとBの実力が突出している
        """
        top3 = self.race_data[self.race_data['rank'].isin(['A', 'B', 'C'])]
        
        if len(top3) < 3:
            return 0.0
        
        b_prob = top3[top3['rank'] == 'B']['probability'].iloc[0]
        c_prob = top3[top3['rank'] == 'C']['probability'].iloc[0]
        
        ks_value = b_prob - c_prob
        
        return round(ks_value, 4)
    
    def _classify_zone(self) -> str:
        """
        ゾーン分類
        
        ガチゾーン: A率 >= 0.8 かつ CT値 >= 70
        ブルーゾーン: CT値 >= 70
        トワイライトゾーン: 50 <= CT値 < 70 かつ A率 < 0.8
        レッドゾーン: CT値 < 50
        """
        if self.a_rate >= 0.8 and self.ct_value >= 70:
            return 'GACHI'
        elif self.ct_value >= 70:
            return 'BLUE'
        elif 50 <= self.ct_value < 70 and self.a_rate < 0.8:
            return 'TWILIGHT'
        else:
            return 'RED'
    
    def get_top_players(self, n: int = 3) -> pd.DataFrame:
        """上位N人の選手情報を取得"""
        return self.race_data.head(n)
    
    def get_recommendation(self) -> str:
        """ゾーンに応じた推奨買い目を取得"""
        recommendations = {
            'GACHI': '3連複1点（ABC）',
            'BLUE': '3連複2点（ABC, ABD）',
            'TWILIGHT': '3連複3点（ABC, ABD, ACD）',
            'RED': '穴狙い・万車券候補'
        }
        return recommendations.get(self.zone, '不明')
    
    def to_dict(self) -> Dict:
        """予測結果を辞書形式で返す"""
        top3 = self.get_top_players(3)
        
        return {
            'race_id': self.race_id,
            'a_rate': self.a_rate,
            'ct_value': self.ct_value,
            'ks_value': self.ks_value,
            'zone': self.zone,
            'recommendation': self.get_recommendation(),
            'top_a': top3[top3['rank'] == 'A']['car_number'].iloc[0] if len(top3) >= 1 else None,
            'top_b': top3[top3['rank'] == 'B']['car_number'].iloc[0] if len(top3) >= 2 else None,
            'top_c': top3[top3['rank'] == 'C']['car_number'].iloc[0] if len(top3) >= 3 else None,
        }


class IkasamaDice:
    """イカサマサイコロ投票クラス"""
    
    def __init__(self, weight_power: float = 2.0):
        """
        Args:
            weight_power: 確率を何乗するか（2.0なら2乗）
        """
        self.weight_power = weight_power
    
    def generate_bet(self, race_data: pd.DataFrame, num_bets: int = 1) -> List[Tuple]:
        """
        重み付き抽選で買い目を生成
        
        Args:
            race_data: ランク付けされた選手データ
            num_bets: 生成する買い目の数
            
        Returns:
            [(1着, 2着, 3着), ...] のリスト
        """
        # 確率を重みとして使用（weight_power乗する）
        probabilities = race_data['probability'].values
        weights = np.power(probabilities, self.weight_power)
        
        # 負の重みを0に
        weights = np.maximum(weights, 0)
        
        # 正規化
        weights_sum = weights.sum()
        if weights_sum == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights_sum
        
        car_numbers = race_data['car_number'].values
        
        bets = []
        for _ in range(num_bets):
            # 重複なしで3台抽選
            selected = np.random.choice(
                car_numbers,
                size=3,
                replace=False,
                p=weights
            )
            bets.append(tuple(selected))
        
        return bets
    
    def generate_multiple_bets(self, race_data: pd.DataFrame, 
                              num_trials: int = 100) -> pd.DataFrame:
        """
        複数回試行して買い目の出現頻度を分析
        
        Args:
            race_data: ランク付けされた選手データ
            num_trials: 試行回数
            
        Returns:
            買い目と出現回数のDataFrame
        """
        bets = self.generate_bet(race_data, num_bets=num_trials)
        
        # 出現頻度をカウント
        bet_counts = {}
        for bet in bets:
            sorted_bet = tuple(sorted(bet))
            bet_counts[sorted_bet] = bet_counts.get(sorted_bet, 0) + 1
        
        # DataFrameに変換
        df_bets = pd.DataFrame([
            {'bet': f"{b[0]}-{b[1]}-{b[2]}", 'count': c, 'probability': c/num_trials}
            for b, c in sorted(bet_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        return df_bets


class BatchPredictor:
    """バッチ予測クラス"""
    
    def __init__(self, model_manager):
        """
        Args:
            model_manager: MultiModelManagerインスタンス
        """
        self.model_manager = model_manager
    
    def predict_races(self, races_df: pd.DataFrame, 
                     race_id_col: str = 'race_id',
                     race_category_col: str = 'race_category') -> Dict[str, RacePrediction]:
        """
        複数レースの予測を一括実行
        
        Args:
            races_df: 全レースのデータ
            race_id_col: レースID列名
            race_category_col: レースカテゴリ列名
            
        Returns:
            {race_id: RacePrediction} の辞書
        """
        predictions = {}
        
        # レースごとにグループ化
        for race_id, race_data in races_df.groupby(race_id_col):
            # レースカテゴリを取得
            race_category = race_data[race_category_col].iloc[0]
            
            # 対応するモデルを取得
            model = self.model_manager.get_model(race_category)
            
            if model is None or not model.is_trained:
                print(f"警告: レース{race_id}のモデルが存在しないかまたは未学習です")
                continue
            
            # 特徴量の準備
            feature_cols = model.feature_names
            X = race_data[feature_cols]
            
            # 予測実行
            probabilities = model.predict_proba(X)
            
            # RacePredictionオブジェクト作成
            prediction = RacePrediction(race_id, race_data, probabilities)
            predictions[race_id] = prediction
        
        return predictions
    
    def create_summary_dataframe(self, predictions: Dict[str, RacePrediction]) -> pd.DataFrame:
        """
        予測結果をサマリーDataFrameに変換
        """
        summary_data = []
        
        for race_id, pred in predictions.items():
            summary_data.append(pred.to_dict())
        
        df_summary = pd.DataFrame(summary_data)
        return df_summary


if __name__ == '__main__':
    # テスト実行
    print("=== 予測・指標算出モジュールテスト ===")
    
    # サンプルデータ
    race_data = pd.DataFrame({
        'car_number': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'player_name': [f'選手{i}' for i in range(1, 10)],
    })
    
    # ダミー確率
    probabilities = np.array([0.85, 0.75, 0.65, 0.45, 0.35, 0.25, 0.15, 0.10, 0.05])
    
    # 予測オブジェクト作成
    prediction = RacePrediction('R001', race_data, probabilities)
    
    print("\n【予測結果】")
    print(prediction.race_data[['car_number', 'player_name', 'probability', 'prediction', 'rank']])
    
    print(f"\nA率: {prediction.a_rate:.4f}")
    print(f"CT値: {prediction.ct_value:.2f}")
    print(f"KS値: {prediction.ks_value:.4f}")
    print(f"ゾーン: {prediction.zone}")
    print(f"推奨: {prediction.get_recommendation()}")
    
    # イカサマサイコロテスト
    print("\n【イカサマサイコロ】")
    dice = IkasamaDice(weight_power=2.0)
    bets = dice.generate_bet(prediction.race_data, num_bets=5)
    print("生成された買い目（5パターン）:")
    for i, bet in enumerate(bets, 1):
        print(f"{i}. {bet[0]}-{bet[1]}-{bet[2]}")
    
    # 頻度分析
    df_bets = dice.generate_multiple_bets(prediction.race_data, num_trials=1000)
    print("\n買い目出現頻度（トップ10）:")
    print(df_bets.head(10))
