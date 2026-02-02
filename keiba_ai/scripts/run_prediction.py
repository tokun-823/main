"""
予測実行スクリプト

Usage:
    python scripts/run_prediction.py --race_id 202405050811
    python scripts/run_prediction.py --date 20240701
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.src.utils import load_config, setup_logging, ensure_dir
from common.src.scraping import RaceIdScraper
from v3.src.prediction import RacePredictor, run_prediction_pipeline

logger = setup_logging()


def predict_single_race(model_path: str, race_id: str, config: dict):
    """単一レースの予測"""
    logger.info(f"レース予測: {race_id}")
    
    result = run_prediction_pipeline(model_path, race_id, config)
    
    if result.empty:
        logger.error("予測失敗")
        return
    
    # 結果を保存
    results_dir = ensure_dir(project_root / 'results' / 'predictions')
    result_path = results_dir / f'{race_id}_prediction.csv'
    result.to_csv(result_path, index=False, encoding='utf-8-sig')
    
    logger.info(f"結果保存: {result_path}")
    
    # 推奨馬を表示
    predictor = RacePredictor(model_path, config)
    recommendations = predictor.get_recommended_bets(result, top_n=5)
    
    logger.info("=" * 50)
    logger.info("予測結果 Top 5:")
    logger.info("=" * 50)
    
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        horse_num = row.get('horse_number', '?')
        horse_name = row.get('horse_name', '不明')
        prob = row.get('pred_proba', 0)
        logger.info(f"  {i}. 馬番{horse_num} {horse_name}: {prob:.3f}")


def predict_date(model_path: str, date: str, config: dict):
    """指定日の全レースを予測"""
    logger.info(f"日付指定予測: {date}")
    
    # レースIDを取得
    scraper = RaceIdScraper(config, use_selenium=True)
    race_ids = scraper.get_race_ids_from_date(date)
    
    if not race_ids:
        logger.error(f"レースが見つかりません: {date}")
        return
    
    logger.info(f"{len(race_ids)} レースを予測")
    
    predictor = RacePredictor(model_path, config)
    
    all_results = []
    
    for race_id in race_ids:
        try:
            result = predictor.predict_race(race_id)
            if not result.empty:
                all_results.append(result)
                
                # 上位馬を表示
                top3 = result.nlargest(3, 'pred_proba')
                logger.info(f"レース {race_id}:")
                for _, row in top3.iterrows():
                    logger.info(f"  馬番{row.get('horse_number', '?')}: {row.get('pred_proba', 0):.3f}")
        
        except Exception as e:
            logger.error(f"予測エラー ({race_id}): {e}")
    
    # 全結果を保存
    if all_results:
        import pandas as pd
        combined = pd.concat(all_results, ignore_index=True)
        
        results_dir = ensure_dir(project_root / 'results' / 'predictions')
        result_path = results_dir / f'{date}_all_predictions.csv'
        combined.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"全結果保存: {result_path}")


def main():
    parser = argparse.ArgumentParser(description='予測実行スクリプト')
    parser.add_argument('--race_id', type=str, help='予測対象レースID（12桁）')
    parser.add_argument('--date', type=str, help='予測対象日（YYYYMMDD形式）')
    parser.add_argument('--model', type=str, default='models/lightgbm_model.pkl', help='モデルパス')
    
    args = parser.parse_args()
    
    if not args.race_id and not args.date:
        parser.error('--race_id または --date を指定してください')
    
    config = load_config()
    
    # モデルパス
    model_path = project_root / args.model
    if not model_path.exists():
        logger.error(f"モデルが見つかりません: {model_path}")
        logger.error("先に scripts/train_model.py を実行してください")
        return
    
    try:
        if args.race_id:
            predict_single_race(str(model_path), args.race_id, config)
        elif args.date:
            predict_date(str(model_path), args.date, config)
        
        logger.info("予測完了")
        
    except KeyboardInterrupt:
        logger.info("中断されました")
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == '__main__':
    main()
