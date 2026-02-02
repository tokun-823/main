"""
回収率シミュレーション・評価モジュール
フェーズ2-10, 2-11: 払い戻しテーブル整形、回収率シミュレーション
フェーズ4-22: 全券種対応の期待値シミュレーション
"""

import itertools
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.src.utils import load_config, get_project_root, ensure_dir, setup_logging

logger = setup_logging()


class PayoutProcessor:
    """払い戻しデータ処理クラス"""
    
    @staticmethod
    def parse_payout_table(payout_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """払い戻しテーブルを券種別に整形
        
        Args:
            payout_df: 払い戻しデータ
        
        Returns:
            券種別のDataFrame辞書
        """
        result = {}
        
        bet_types = ['単勝', '複勝', '枠連', '馬連', 'ワイド', '馬単', '三連複', '三連単']
        
        for bet_type in bet_types:
            type_df = payout_df[payout_df['bet_type'] == bet_type].copy()
            if len(type_df) > 0:
                result[bet_type] = type_df
        
        return result
    
    @staticmethod
    def expand_multiple_winners(payout_df: pd.DataFrame) -> pd.DataFrame:
        """複数的中の払い戻しを展開（ワイド、3連複など）
        
        Args:
            payout_df: 払い戻しデータ
        
        Returns:
            展開後のDataFrame
        """
        # 既に展開済みの場合はそのまま返す
        return payout_df


class ReturnSimulator:
    """回収率シミュレーションクラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.sim_config = self.config.get('simulation', {})
        self.threshold_config = self.sim_config.get('threshold', {})
    
    def simulate_win_bet(
        self,
        race_df: pd.DataFrame,
        payout_df: pd.DataFrame,
        pred_column: str = 'pred_proba',
        top_n: int = 1,
        min_probability: float = None
    ) -> Dict[str, Any]:
        """単勝シミュレーション
        
        Args:
            race_df: レース結果（予測確率含む）
            payout_df: 払い戻しデータ
            pred_column: 予測確率カラム
            top_n: 購入する上位頭数
            min_probability: 足切り確率
        
        Returns:
            シミュレーション結果
        """
        min_probability = min_probability or self.threshold_config.get('min_probability', 0.01)
        
        total_bet = 0
        total_return = 0
        hit_count = 0
        bet_count = 0
        
        for race_id in race_df['race_id'].unique():
            race_data = race_df[race_df['race_id'] == race_id].copy()
            
            # 足切り
            race_data = race_data[race_data[pred_column] >= min_probability]
            
            if len(race_data) == 0:
                continue
            
            # 予測確率上位N頭を選択
            race_data = race_data.nlargest(top_n, pred_column)
            
            for _, row in race_data.iterrows():
                total_bet += 100  # 100円賭け
                bet_count += 1
                
                # 1着かどうか
                if row.get('rank') == 1:
                    # 払い戻し確認
                    win_payout = payout_df[
                        (payout_df['race_id'] == race_id) &
                        (payout_df['bet_type'] == '単勝')
                    ]
                    
                    if len(win_payout) > 0:
                        payout = win_payout['payout'].values[0]
                        total_return += payout
                        hit_count += 1
        
        # 結果計算
        hit_rate = hit_count / bet_count if bet_count > 0 else 0
        return_rate = total_return / total_bet if total_bet > 0 else 0
        
        result = {
            'bet_type': '単勝',
            'top_n': top_n,
            'total_bet': total_bet,
            'total_return': total_return,
            'profit': total_return - total_bet,
            'bet_count': bet_count,
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'return_rate': return_rate,
        }
        
        logger.info(f"単勝シミュレーション (Top{top_n}): 回収率={return_rate*100:.1f}%, 的中率={hit_rate*100:.1f}%")
        
        return result
    
    def simulate_place_bet(
        self,
        race_df: pd.DataFrame,
        payout_df: pd.DataFrame,
        pred_column: str = 'pred_proba',
        top_n: int = 3,
        min_probability: float = None
    ) -> Dict[str, Any]:
        """複勝シミュレーション
        
        Args:
            race_df: レース結果
            payout_df: 払い戻しデータ
            pred_column: 予測確率カラム
            top_n: 購入する上位頭数
            min_probability: 足切り確率
        
        Returns:
            シミュレーション結果
        """
        min_probability = min_probability or self.threshold_config.get('min_probability', 0.01)
        
        total_bet = 0
        total_return = 0
        hit_count = 0
        bet_count = 0
        
        for race_id in race_df['race_id'].unique():
            race_data = race_df[race_df['race_id'] == race_id].copy()
            race_data = race_data[race_data[pred_column] >= min_probability]
            
            if len(race_data) == 0:
                continue
            
            race_data = race_data.nlargest(top_n, pred_column)
            
            for _, row in race_data.iterrows():
                total_bet += 100
                bet_count += 1
                
                # 3着以内かどうか
                if row.get('rank') <= 3:
                    # 払い戻し確認
                    place_payout = payout_df[
                        (payout_df['race_id'] == race_id) &
                        (payout_df['bet_type'] == '複勝') &
                        (payout_df['horse_numbers'].astype(str) == str(int(row.get('horse_number', 0))))
                    ]
                    
                    if len(place_payout) > 0:
                        payout = place_payout['payout'].values[0]
                        total_return += payout
                        hit_count += 1
        
        hit_rate = hit_count / bet_count if bet_count > 0 else 0
        return_rate = total_return / total_bet if total_bet > 0 else 0
        
        result = {
            'bet_type': '複勝',
            'top_n': top_n,
            'total_bet': total_bet,
            'total_return': total_return,
            'profit': total_return - total_bet,
            'bet_count': bet_count,
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'return_rate': return_rate,
        }
        
        logger.info(f"複勝シミュレーション (Top{top_n}): 回収率={return_rate*100:.1f}%, 的中率={hit_rate*100:.1f}%")
        
        return result
    
    def simulate_box_bet(
        self,
        race_df: pd.DataFrame,
        payout_df: pd.DataFrame,
        bet_type: str,
        pred_column: str = 'pred_proba',
        top_n: int = 3,
        min_probability: float = None
    ) -> Dict[str, Any]:
        """ボックス買いシミュレーション
        
        Args:
            race_df: レース結果
            payout_df: 払い戻しデータ
            bet_type: 馬券種類（馬連、ワイド、三連複、三連単）
            pred_column: 予測確率カラム
            top_n: ボックス対象頭数
            min_probability: 足切り確率
        
        Returns:
            シミュレーション結果
        """
        min_probability = min_probability or self.threshold_config.get('min_probability', 0.01)
        
        total_bet = 0
        total_return = 0
        hit_count = 0
        race_count = 0
        
        for race_id in race_df['race_id'].unique():
            race_data = race_df[race_df['race_id'] == race_id].copy()
            race_data = race_data[race_data[pred_column] >= min_probability]
            
            if len(race_data) < top_n:
                continue
            
            race_count += 1
            
            # 上位N頭を取得
            top_horses = race_data.nlargest(top_n, pred_column)
            horse_numbers = top_horses['horse_number'].astype(int).tolist()
            
            # 組み合わせ数を計算
            if bet_type in ['馬連', 'ワイド']:
                combinations = list(itertools.combinations(horse_numbers, 2))
            elif bet_type == '三連複':
                combinations = list(itertools.combinations(horse_numbers, 3))
            elif bet_type == '三連単':
                combinations = list(itertools.permutations(horse_numbers, 3))
            elif bet_type == '馬単':
                combinations = list(itertools.permutations(horse_numbers, 2))
            else:
                continue
            
            bet_amount = len(combinations) * 100
            total_bet += bet_amount
            
            # 払い戻し確認
            race_payout = payout_df[
                (payout_df['race_id'] == race_id) &
                (payout_df['bet_type'] == bet_type)
            ]
            
            for _, payout_row in race_payout.iterrows():
                winning_str = str(payout_row['horse_numbers'])
                
                # 馬番の組み合わせを解析
                if bet_type in ['三連複', '三連単']:
                    winning_nums = [int(x.strip()) for x in winning_str.replace('-', ' ').replace('→', ' ').split()]
                else:
                    winning_nums = [int(x.strip()) for x in winning_str.replace('-', ' ').replace('→', ' ').split()]
                
                # 的中判定
                if bet_type in ['馬連', 'ワイド', '三連複']:
                    winning_set = set(winning_nums)
                    for comb in combinations:
                        if set(comb) == winning_set:
                            total_return += payout_row['payout']
                            hit_count += 1
                            break
                elif bet_type in ['馬単', '三連単']:
                    winning_tuple = tuple(winning_nums)
                    if winning_tuple in combinations:
                        total_return += payout_row['payout']
                        hit_count += 1
        
        hit_rate = hit_count / race_count if race_count > 0 else 0
        return_rate = total_return / total_bet if total_bet > 0 else 0
        
        result = {
            'bet_type': bet_type,
            'box_type': f'{top_n}頭ボックス',
            'total_bet': total_bet,
            'total_return': total_return,
            'profit': total_return - total_bet,
            'race_count': race_count,
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'return_rate': return_rate,
        }
        
        logger.info(f"{bet_type} {top_n}頭BOX: 回収率={return_rate*100:.1f}%, 的中率={hit_rate*100:.1f}%")
        
        return result


class ExpectedValueCalculator:
    """期待値計算クラス
    フェーズ4-22: 全券種対応の期待値シミュレーション
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.ev_thresholds = self.config.get('simulation', {}).get('expected_value_thresholds', {})
    
    def calculate_win_expected_value(
        self,
        pred_proba: float,
        odds: float
    ) -> float:
        """単勝の期待値を計算
        
        Args:
            pred_proba: 予測確率
            odds: オッズ
        
        Returns:
            期待値
        """
        return pred_proba * odds
    
    def calculate_quinella_expected_value(
        self,
        pred_proba_1: float,
        pred_proba_2: float,
        odds: float
    ) -> float:
        """馬連の期待値を計算（近似式）
        
        合成確率 = P(A勝) * P(B勝|A勝でない) + P(B勝) * P(A勝|B勝でない)
        近似: P(A) * P(B) / (1 - P(A)) + P(B) * P(A) / (1 - P(B))
        
        Args:
            pred_proba_1: 馬1の予測確率
            pred_proba_2: 馬2の予測確率
            odds: 馬連オッズ
        
        Returns:
            期待値
        """
        # 1着確率から2着以内確率を推定
        # 簡易的な近似
        p1 = pred_proba_1
        p2 = pred_proba_2
        
        # 合成確率（両方が1-2着に入る確率）
        # P(1着A, 2着B) + P(1着B, 2着A)
        if p1 >= 1 or p2 >= 1:
            return 0
        
        prob = p1 * (p2 / (1 - p1)) + p2 * (p1 / (1 - p2))
        prob = min(prob, 1.0)  # 1を超えないように
        
        return prob * odds
    
    def calculate_wide_expected_value(
        self,
        pred_proba_1: float,
        pred_proba_2: float,
        odds: float
    ) -> float:
        """ワイドの期待値を計算（近似式）
        
        両方が3着以内に入る確率
        
        Args:
            pred_proba_1: 馬1の予測確率
            pred_proba_2: 馬2の予測確率
            odds: ワイドオッズ
        
        Returns:
            期待値
        """
        # 3着以内確率（1着確率の約2.5倍と近似）
        p1_place = min(pred_proba_1 * 2.5, 1.0)
        p2_place = min(pred_proba_2 * 2.5, 1.0)
        
        # 両方が3着以内
        prob = p1_place * p2_place
        
        return prob * odds
    
    def calculate_trifecta_expected_value(
        self,
        pred_proba_1: float,
        pred_proba_2: float,
        pred_proba_3: float,
        odds: float
    ) -> float:
        """三連複の期待値を計算
        
        Args:
            pred_proba_1, 2, 3: 予測確率
            odds: 三連複オッズ
        
        Returns:
            期待値
        """
        # 3着以内確率
        p1 = min(pred_proba_1 * 2.5, 1.0)
        p2 = min(pred_proba_2 * 2.5, 1.0)
        p3 = min(pred_proba_3 * 2.5, 1.0)
        
        # 3頭全てが3着以内
        prob = p1 * p2 * p3 * 6  # 順列を考慮した補正
        prob = min(prob, 1.0)
        
        return prob * odds
    
    def filter_by_expected_value(
        self,
        predictions_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        bet_type: str = '単勝'
    ) -> pd.DataFrame:
        """期待値でフィルタリング
        
        Args:
            predictions_df: 予測データ
            odds_df: オッズデータ
            bet_type: 券種
        
        Returns:
            期待値閾値を超える買い目
        """
        threshold = self.ev_thresholds.get(bet_type, 1.0)
        
        results = []
        
        if bet_type == '単勝':
            for _, row in predictions_df.iterrows():
                odds = odds_df[
                    (odds_df['race_id'] == row['race_id']) &
                    (odds_df['horse_number'] == row['horse_number'])
                ]['win_odds'].values
                
                if len(odds) > 0:
                    ev = self.calculate_win_expected_value(row['pred_proba'], odds[0])
                    if ev >= threshold:
                        results.append({
                            'race_id': row['race_id'],
                            'horse_number': row['horse_number'],
                            'pred_proba': row['pred_proba'],
                            'odds': odds[0],
                            'expected_value': ev,
                            'bet_type': bet_type
                        })
        
        return pd.DataFrame(results)


class ComparisonSimulator:
    """予測モデル vs 人気順 比較シミュレーション"""
    
    @staticmethod
    def compare_with_popularity(
        race_df: pd.DataFrame,
        payout_df: pd.DataFrame,
        pred_column: str = 'pred_proba',
        top_n: int = 3
    ) -> Dict[str, Dict]:
        """予測モデルと人気順の比較
        
        Args:
            race_df: レース結果
            payout_df: 払い戻しデータ
            pred_column: 予測確率カラム
            top_n: 購入頭数
        
        Returns:
            比較結果
        """
        simulator = ReturnSimulator()
        
        # 予測モデルでのシミュレーション
        model_result = simulator.simulate_win_bet(
            race_df, payout_df, pred_column=pred_column, top_n=top_n
        )
        
        # 人気順でのシミュレーション
        popularity_result = simulator.simulate_win_bet(
            race_df, payout_df, pred_column='popularity', top_n=top_n
        )
        
        # 比較
        comparison = {
            'model': model_result,
            'popularity': popularity_result,
            'improvement': {
                'return_rate_diff': model_result['return_rate'] - popularity_result['return_rate'],
                'profit_diff': model_result['profit'] - popularity_result['profit'],
            }
        }
        
        logger.info(f"モデル vs 人気順 (Top{top_n}):")
        logger.info(f"  モデル回収率: {model_result['return_rate']*100:.1f}%")
        logger.info(f"  人気順回収率: {popularity_result['return_rate']*100:.1f}%")
        logger.info(f"  改善幅: {comparison['improvement']['return_rate_diff']*100:.1f}pt")
        
        return comparison


def run_simulation_pipeline(
    prediction_result_path: str,
    payout_data_path: str,
    output_path: str,
    config: Optional[dict] = None
) -> Dict[str, Any]:
    """シミュレーションパイプライン
    
    Args:
        prediction_result_path: 予測結果のパス
        payout_data_path: 払い戻しデータのパス
        output_path: 結果出力パス
        config: 設定辞書
    
    Returns:
        シミュレーション結果
    """
    config = config or load_config()
    
    # データ読み込み
    logger.info("データ読み込み...")
    pred_df = pd.read_csv(prediction_result_path)
    payout_df = pd.read_csv(payout_data_path)
    
    simulator = ReturnSimulator(config)
    results = {}
    
    # 単勝シミュレーション
    logger.info("単勝シミュレーション...")
    for top_n in [1, 2, 3]:
        key = f'win_top{top_n}'
        results[key] = simulator.simulate_win_bet(pred_df, payout_df, top_n=top_n)
    
    # 複勝シミュレーション
    logger.info("複勝シミュレーション...")
    for top_n in [1, 2, 3]:
        key = f'place_top{top_n}'
        results[key] = simulator.simulate_place_bet(pred_df, payout_df, top_n=top_n)
    
    # ボックス買いシミュレーション
    logger.info("ボックス買いシミュレーション...")
    box_configs = self.config.get('simulation', {}).get('box_purchase', {}).get('top_n', [3, 4, 5])
    
    for bet_type in ['馬連', 'ワイド', '三連複']:
        for top_n in box_configs:
            key = f'{bet_type}_box{top_n}'
            results[key] = simulator.simulate_box_bet(
                pred_df, payout_df, bet_type=bet_type, top_n=top_n
            )
    
    # 人気順との比較
    logger.info("人気順との比較...")
    comparison = ComparisonSimulator.compare_with_popularity(pred_df, payout_df, top_n=1)
    results['comparison'] = comparison
    
    # 結果を保存
    output_dir = ensure_dir(Path(output_path).parent)
    results_df = pd.DataFrame([
        {**v, 'simulation': k} for k, v in results.items() 
        if isinstance(v, dict) and 'return_rate' in v
    ])
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"結果保存完了: {output_path}")
    
    return results
