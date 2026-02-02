"""
回収率シミュレーション実行スクリプト

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --prediction_path results/predictions/test_predictions.csv
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.src.utils import load_config, setup_logging, ensure_dir
from v3.src.evaluation import (
    ReturnSimulator, 
    ExpectedValueCalculator,
    ComparisonSimulator
)

import pandas as pd

logger = setup_logging()


def run_simulation(prediction_path: str, payout_path: str, config: dict):
    """シミュレーションを実行"""
    logger.info("シミュレーション開始")
    
    # データ読み込み
    pred_df = pd.read_csv(prediction_path)
    payout_df = pd.read_csv(payout_path)
    
    simulator = ReturnSimulator(config)
    results = {}
    
    # 単勝シミュレーション
    logger.info("=" * 50)
    logger.info("単勝シミュレーション")
    logger.info("=" * 50)
    
    for top_n in [1, 2, 3]:
        result = simulator.simulate_win_bet(pred_df, payout_df, top_n=top_n)
        results[f'win_top{top_n}'] = result
    
    # 複勝シミュレーション
    logger.info("=" * 50)
    logger.info("複勝シミュレーション")
    logger.info("=" * 50)
    
    for top_n in [1, 2, 3]:
        result = simulator.simulate_place_bet(pred_df, payout_df, top_n=top_n)
        results[f'place_top{top_n}'] = result
    
    # ボックス買いシミュレーション
    logger.info("=" * 50)
    logger.info("ボックス買いシミュレーション")
    logger.info("=" * 50)
    
    for bet_type in ['馬連', 'ワイド', '三連複']:
        for top_n in [3, 4, 5]:
            result = simulator.simulate_box_bet(
                pred_df, payout_df, 
                bet_type=bet_type, 
                top_n=top_n
            )
            results[f'{bet_type}_box{top_n}'] = result
    
    # 人気順との比較
    logger.info("=" * 50)
    logger.info("予測モデル vs 人気順")
    logger.info("=" * 50)
    
    comparison = ComparisonSimulator.compare_with_popularity(pred_df, payout_df, top_n=1)
    results['comparison'] = comparison
    
    return results


def print_summary(results: dict):
    """結果サマリーを出力"""
    logger.info("=" * 60)
    logger.info("シミュレーション結果サマリー")
    logger.info("=" * 60)
    
    print(f"\n{'券種/戦略':<20} {'的中率':>10} {'回収率':>10} {'収支':>15}")
    print("-" * 60)
    
    for key, result in results.items():
        if isinstance(result, dict) and 'return_rate' in result:
            bet_type = result.get('bet_type', key)
            top_n = result.get('top_n', result.get('box_type', ''))
            hit_rate = result.get('hit_rate', 0) * 100
            return_rate = result.get('return_rate', 0) * 100
            profit = result.get('profit', 0)
            
            label = f"{bet_type} {top_n}"
            print(f"{label:<20} {hit_rate:>9.1f}% {return_rate:>9.1f}% {profit:>14,.0f}円")
    
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description='回収率シミュレーションスクリプト')
    parser.add_argument('--prediction_path', type=str, help='予測結果CSVのパス')
    parser.add_argument('--payout_path', type=str, help='払い戻しデータCSVのパス')
    parser.add_argument('--output', type=str, default='results/simulation_results.csv', help='結果出力パス')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # デフォルトパス
    if args.prediction_path is None:
        # テストデータの予測結果を使用
        prediction_path = project_root / 'results' / 'test_predictions.csv'
    else:
        prediction_path = Path(args.prediction_path)
    
    if args.payout_path is None:
        payout_path = project_root / 'data' / 'raw_df' / 'payouts.csv'
    else:
        payout_path = Path(args.payout_path)
    
    if not prediction_path.exists():
        logger.error(f"予測結果が見つかりません: {prediction_path}")
        logger.error("先に予測を実行してください")
        return
    
    if not payout_path.exists():
        logger.warning(f"払い戻しデータが見つかりません: {payout_path}")
        logger.warning("ダミーデータを使用します（結果は参考値）")
        
        # ダミーの払い戻しデータを作成
        pred_df = pd.read_csv(prediction_path)
        payout_df = pd.DataFrame({
            'race_id': pred_df['race_id'].unique(),
            'bet_type': '単勝',
            'horse_numbers': '1',
            'payout': 500,
            'popularity': 1
        })
    else:
        payout_df = None
    
    try:
        if payout_df is None:
            results = run_simulation(str(prediction_path), str(payout_path), config)
        else:
            # ダミーデータの場合
            pred_df = pd.read_csv(prediction_path)
            simulator = ReturnSimulator(config)
            results = {
                'win_top1': simulator.simulate_win_bet(pred_df, payout_df, top_n=1),
                'win_top3': simulator.simulate_win_bet(pred_df, payout_df, top_n=3),
            }
        
        print_summary(results)
        
        # 結果を保存
        output_path = project_root / args.output
        ensure_dir(output_path.parent)
        
        results_list = []
        for key, result in results.items():
            if isinstance(result, dict) and 'return_rate' in result:
                result['simulation_name'] = key
                results_list.append(result)
        
        if results_list:
            results_df = pd.DataFrame(results_list)
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"結果保存: {output_path}")
        
        logger.info("シミュレーション完了")
        
    except KeyboardInterrupt:
        logger.info("中断されました")
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == '__main__':
    main()
