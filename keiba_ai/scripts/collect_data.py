"""
データ収集実行スクリプト
過去データの収集から予測に必要な基礎データを構築

Usage:
    python scripts/collect_data.py --start_year 2020 --end_year 2024
    python scripts/collect_data.py --dates 20240101,20240102
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.src.utils import load_config, setup_logging, ensure_dir
from common.src.scraping import (
    CalendarScraper,
    RaceIdScraper,
    RaceResultScraper,
    HorseDataScraper,
    PedigreeScraper,
    LeadingScraper,
)
from common.src.create_raw_df import RawDataCreator

logger = setup_logging()


def collect_race_dates(start_year: int, end_year: int, config: dict) -> list:
    """開催日一覧を取得"""
    logger.info(f"開催日取得: {start_year}年〜{end_year}年")
    
    scraper = CalendarScraper(config)
    all_dates = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            dates = scraper.get_race_dates(year, month)
            all_dates.extend(dates)
    
    logger.info(f"合計 {len(all_dates)} 開催日を取得")
    return all_dates


def collect_race_ids(dates: list, config: dict) -> list:
    """レースID一覧を取得"""
    logger.info(f"レースID取得: {len(dates)} 開催日")
    
    scraper = RaceIdScraper(config, use_selenium=True)
    race_ids = scraper.get_race_ids_from_dates(dates)
    
    logger.info(f"合計 {len(race_ids)} レースIDを取得")
    return race_ids


def collect_race_results(race_ids: list, config: dict, skip: bool = True) -> dict:
    """レース結果HTMLを取得"""
    logger.info(f"レース結果取得: {len(race_ids)} レース")
    
    scraper = RaceResultScraper(config)
    results = scraper.get_race_results_batch(race_ids, skip=skip)
    
    return results


def collect_horse_data(race_ids: list, config: dict, skip: bool = True) -> dict:
    """馬データを取得"""
    # まず、レース結果から馬IDを抽出
    logger.info("馬ID抽出中...")
    
    creator = RawDataCreator(config)
    horse_ids = set()
    
    # レース結果HTMLから馬IDを抽出
    race_html_dir = project_root / config.get('paths', {}).get('html_race', 'data/html/race')
    
    for race_id in race_ids:
        html_path = race_html_dir / f"{race_id}.bin"
        if html_path.exists():
            df, _ = creator.race_parser.parse_race_html(str(html_path))
            if df is not None and 'horse_id' in df.columns:
                horse_ids.update(df['horse_id'].dropna().astype(str).tolist())
    
    logger.info(f"馬データ取得: {len(horse_ids)} 頭")
    
    scraper = HorseDataScraper(config)
    results = scraper.get_horses_batch(list(horse_ids), skip=skip)
    
    return results


def collect_pedigree_data(horse_ids: list, config: dict, skip: bool = True) -> dict:
    """血統データを取得"""
    logger.info(f"血統データ取得: {len(horse_ids)} 頭")
    
    scraper = PedigreeScraper(config)
    results = {}
    
    for horse_id in horse_ids:
        path = scraper.get_pedigree_html(horse_id, skip=skip)
        if path:
            results[horse_id] = path
    
    return results


def collect_leading_data(start_year: int, end_year: int, config: dict, skip: bool = True):
    """リーディングデータを取得"""
    logger.info(f"リーディングデータ取得: {start_year}年〜{end_year}年")
    
    scraper = LeadingScraper(config)
    
    for year in range(start_year, end_year + 1):
        scraper.get_jockey_leading(year, skip=skip)
        scraper.get_trainer_leading(year, skip=skip)
        scraper.get_sire_leading(year, skip=skip)


def create_raw_data(config: dict):
    """RawデータCSVを作成"""
    logger.info("Rawデータ作成...")
    
    creator = RawDataCreator(config)
    
    # レースデータ
    race_df = creator.create_race_raw_df()
    if not race_df.empty:
        creator.save_raw_df(race_df, 'race_results.csv')
    
    # 馬データ
    horse_df = creator.create_horse_raw_df()
    if not horse_df.empty:
        creator.save_raw_df(horse_df, 'horse_history.csv')


def main():
    parser = argparse.ArgumentParser(description='競馬データ収集スクリプト')
    parser.add_argument('--start_year', type=int, default=2020, help='収集開始年')
    parser.add_argument('--end_year', type=int, default=2024, help='収集終了年')
    parser.add_argument('--dates', type=str, help='収集日リスト（カンマ区切り、YYYYMMDD形式）')
    parser.add_argument('--skip', action='store_true', default=True, help='既存ファイルをスキップ')
    parser.add_argument('--no-skip', dest='skip', action='store_false', help='既存ファイルも再取得')
    parser.add_argument('--step', type=str, choices=['all', 'dates', 'ids', 'results', 'horses', 'pedigree', 'leading', 'raw'],
                        default='all', help='実行ステップ')
    
    args = parser.parse_args()
    
    config = load_config()
    
    # 日付リストを準備
    if args.dates:
        dates = args.dates.split(',')
    else:
        dates = None
    
    try:
        if args.step in ['all', 'dates']:
            dates = collect_race_dates(args.start_year, args.end_year, config)
            # 日付リストを保存
            dates_path = ensure_dir(project_root / 'data') / 'race_dates.txt'
            with open(dates_path, 'w') as f:
                f.write('\n'.join(dates))
        
        if dates is None:
            # 保存済みの日付リストを読み込み
            dates_path = project_root / 'data' / 'race_dates.txt'
            if dates_path.exists():
                with open(dates_path, 'r') as f:
                    dates = f.read().strip().split('\n')
            else:
                logger.error("日付リストがありません。--step dates を先に実行してください")
                return
        
        if args.step in ['all', 'ids']:
            race_ids = collect_race_ids(dates, config)
            # レースIDリストを保存
            ids_path = ensure_dir(project_root / 'data') / 'race_ids.txt'
            with open(ids_path, 'w') as f:
                f.write('\n'.join(race_ids))
        else:
            # 保存済みのレースIDリストを読み込み
            ids_path = project_root / 'data' / 'race_ids.txt'
            if ids_path.exists():
                with open(ids_path, 'r') as f:
                    race_ids = f.read().strip().split('\n')
            else:
                race_ids = []
        
        if args.step in ['all', 'results'] and race_ids:
            collect_race_results(race_ids, config, skip=args.skip)
        
        if args.step in ['all', 'horses'] and race_ids:
            collect_horse_data(race_ids, config, skip=args.skip)
        
        if args.step in ['all', 'pedigree']:
            # 馬IDリストを読み込み（馬データ取得後に作成される）
            horse_html_dir = project_root / config.get('paths', {}).get('html_horse', 'data/html/horse')
            horse_ids = [p.stem for p in horse_html_dir.glob('*.bin')]
            if horse_ids:
                collect_pedigree_data(horse_ids, config, skip=args.skip)
        
        if args.step in ['all', 'leading']:
            collect_leading_data(args.start_year, args.end_year, config, skip=args.skip)
        
        if args.step in ['all', 'raw']:
            create_raw_data(config)
        
        logger.info("データ収集完了")
        
    except KeyboardInterrupt:
        logger.info("中断されました")
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == '__main__':
    main()
