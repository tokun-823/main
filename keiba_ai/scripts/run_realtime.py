"""
リアルタイム運用スクリプト
レース当日に自動でスクレイピングと予測を実行

Usage:
    python scripts/run_realtime.py
    python scripts/run_realtime.py --date 20240701
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.src.utils import load_config, setup_logging, ensure_dir
from common.src.scraping import RaceIdScraper
from v3.src.prediction import RacePredictor, RaceScheduler
from v3.src.jra_scraping import JRAOddsScraper

logger = setup_logging()


async def fetch_realtime_odds(race_id: str, race_url: str, config: dict):
    """リアルタイムオッズを取得"""
    scraper = JRAOddsScraper(config)
    
    try:
        odds_data = await scraper.get_all_odds(race_url, race_id)
        await scraper.save_odds(odds_data, race_id)
        return odds_data
    finally:
        await scraper._close_browser()


def run_realtime_prediction(model_path: str, race_id: str, config: dict):
    """リアルタイム予測を実行"""
    logger.info(f"リアルタイム予測: {race_id}")
    
    predictor = RacePredictor(model_path, config)
    
    # 予測実行
    result = predictor.predict_race(race_id)
    
    if result.empty:
        logger.error(f"予測失敗: {race_id}")
        return
    
    # 推奨買い目を取得
    recommendations = predictor.get_recommended_bets(result, top_n=5)
    
    # 結果を表示
    logger.info("=" * 50)
    logger.info(f"レース {race_id} 予測結果")
    logger.info("=" * 50)
    
    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        horse_num = row.get('horse_number', '?')
        horse_name = row.get('horse_name', '不明')
        prob = row.get('pred_proba', 0)
        logger.info(f"  {i}. 馬番{horse_num} {horse_name}: 確率={prob:.3f}")
    
    # 結果を保存
    results_dir = ensure_dir(project_root / 'results' / 'realtime')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = results_dir / f'{race_id}_{timestamp}.csv'
    result.to_csv(result_path, index=False, encoding='utf-8-sig')
    
    return result


def run_scheduled_mode(model_path: str, date: str, config: dict):
    """スケジュールモードで実行"""
    logger.info(f"スケジュールモード開始: {date}")
    
    predictor = RacePredictor(model_path, config)
    scheduler = RaceScheduler(config)
    
    # レースタイムテーブルを取得
    timetable = scheduler.get_timetable(date)
    
    if not timetable:
        logger.warning("レースが見つかりません")
        return
    
    logger.info(f"{len(timetable)} レースをスケジュール")
    
    # 予測タスクをスケジュール
    scheduler.schedule_predictions(timetable, predictor)
    
    # スケジューラ開始
    scheduler.start()


def run_immediate_mode(model_path: str, date: str, config: dict):
    """即時実行モード"""
    logger.info(f"即時実行モード: {date}")
    
    # レースIDを取得
    scraper = RaceIdScraper(config, use_selenium=True)
    race_ids = scraper.get_race_ids_from_date(date)
    
    if not race_ids:
        logger.error(f"レースが見つかりません: {date}")
        return
    
    logger.info(f"{len(race_ids)} レースを処理")
    
    predictor = RacePredictor(model_path, config)
    
    all_results = []
    
    for race_id in race_ids:
        try:
            result = run_realtime_prediction(model_path, race_id, config)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            logger.error(f"エラー ({race_id}): {e}")
    
    # 全結果を統合保存
    if all_results:
        import pandas as pd
        combined = pd.concat(all_results, ignore_index=True)
        
        results_dir = ensure_dir(project_root / 'results' / 'realtime')
        result_path = results_dir / f'{date}_all_realtime.csv'
        combined.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"全結果保存: {result_path}")


def main():
    parser = argparse.ArgumentParser(description='リアルタイム運用スクリプト')
    parser.add_argument('--date', type=str, help='対象日（YYYYMMDD形式、省略時は当日）')
    parser.add_argument('--model', type=str, default='models/lightgbm_model.pkl', help='モデルパス')
    parser.add_argument('--mode', type=str, choices=['scheduled', 'immediate'], 
                        default='immediate', help='実行モード')
    parser.add_argument('--race_id', type=str, help='単一レース予測（即時）')
    
    args = parser.parse_args()
    
    # 日付を設定
    if args.date:
        date = args.date
    else:
        date = datetime.now().strftime('%Y%m%d')
    
    config = load_config()
    
    # モデルパス
    model_path = project_root / args.model
    if not model_path.exists():
        logger.error(f"モデルが見つかりません: {model_path}")
        logger.error("先に scripts/train_model.py を実行してください")
        return
    
    try:
        if args.race_id:
            # 単一レースの即時予測
            run_realtime_prediction(str(model_path), args.race_id, config)
        elif args.mode == 'scheduled':
            # スケジュールモード
            run_scheduled_mode(str(model_path), date, config)
        else:
            # 即時実行モード
            run_immediate_mode(str(model_path), date, config)
        
        logger.info("処理完了")
        
    except KeyboardInterrupt:
        logger.info("中断されました")
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == '__main__':
    main()
