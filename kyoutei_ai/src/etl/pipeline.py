"""
ETL パイプライン
データの抽出・変換・ロードを統合管理
"""
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from loguru import logger
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, RAW_DATA_DIR, PROCESSED_DATA_DIR

from .parser import BangumiParser, ResultParser, RacerParser
from .database import db, DatabaseManager


class ETLPipeline:
    """ETLパイプラインクラス"""
    
    def __init__(self):
        self.bangumi_parser = BangumiParser()
        self.result_parser = ResultParser()
        self.racer_parser = RacerParser()
        self.db = db
    
    def run_full_pipeline(self):
        """フルETLパイプラインを実行"""
        logger.info("Starting full ETL pipeline...")
        
        # スキーマ初期化
        self.db.init_schema()
        
        # 番組表のパースとロード
        logger.info("Processing Bangumi data...")
        bangumi_df = self.bangumi_parser.parse_all_files()
        if not bangumi_df.empty:
            self.db.insert_dataframe('bangumi', bangumi_df, if_exists='replace')
            logger.info(f"Loaded {len(bangumi_df)} bangumi records")
        
        # 競争結果のパースとロード
        logger.info("Processing Race Result data...")
        result_df = self.result_parser.parse_all_files()
        if not result_df.empty:
            self.db.insert_dataframe('race_result', result_df, if_exists='replace')
            logger.info(f"Loaded {len(result_df)} result records")
        
        # 選手データのパースとロード
        logger.info("Processing Racer data...")
        racer_df = self.racer_parser.parse_all_files()
        if not racer_df.empty:
            self.db.insert_dataframe('racer_master', racer_df, if_exists='replace')
            logger.info(f"Loaded {len(racer_df)} racer records")
        
        # Parquetエクスポート
        logger.info("Exporting to Parquet...")
        self.db.export_to_parquet()
        
        logger.info("ETL pipeline completed!")
    
    def incremental_update(self, start_date: datetime, end_date: datetime = None):
        """増分更新"""
        end_date = end_date or datetime.now()
        
        logger.info(f"Incremental update: {start_date.date()} to {end_date.date()}")
        
        # 日付範囲のデータを処理
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            
            # 該当日のファイルを処理
            self._process_date(date_str)
            
            current_date += timedelta(days=1)
    
    def _process_date(self, date_str: str):
        """特定日のデータを処理"""
        # 実装は日付ベースのファイル検索に合わせて調整
        pass
    
    def validate_data(self) -> Dict[str, Any]:
        """データの整合性を検証"""
        conn = self.db.connect()
        
        validation_results = {}
        
        # レコード数チェック
        bangumi_count = conn.execute("SELECT COUNT(*) FROM bangumi").fetchone()[0]
        result_count = conn.execute("SELECT COUNT(*) FROM race_result").fetchone()[0]
        racer_count = conn.execute("SELECT COUNT(*) FROM racer_master").fetchone()[0]
        
        validation_results['record_counts'] = {
            'bangumi': bangumi_count,
            'race_result': result_count,
            'racer_master': racer_count
        }
        
        # 日付範囲チェック
        date_range = conn.execute("""
            SELECT MIN(race_date), MAX(race_date) FROM race_result
        """).fetchone()
        validation_results['date_range'] = {
            'min': date_range[0],
            'max': date_range[1]
        }
        
        # 番組表と結果の不整合チェック
        missing_results = conn.execute("""
            SELECT COUNT(DISTINCT race_date || place_code || race_number)
            FROM bangumi b
            WHERE NOT EXISTS (
                SELECT 1 FROM race_result r
                WHERE b.race_date = r.race_date
                AND b.place_code = r.place_code
                AND b.race_number = r.race_number
            )
        """).fetchone()[0]
        validation_results['missing_results'] = missing_results
        
        # NULL値チェック
        null_checks = conn.execute("""
            SELECT 
                SUM(CASE WHEN racer_id IS NULL THEN 1 ELSE 0 END) as null_racer_id,
                SUM(CASE WHEN win_rate IS NULL THEN 1 ELSE 0 END) as null_win_rate
            FROM bangumi
        """).fetchone()
        validation_results['null_values'] = {
            'racer_id': null_checks[0],
            'win_rate': null_checks[1]
        }
        
        logger.info(f"Validation results: {validation_results}")
        return validation_results
    
    def generate_summary_stats(self) -> pd.DataFrame:
        """サマリー統計を生成"""
        return self.db.query("""
            SELECT
                place_code,
                COUNT(DISTINCT race_date) as race_days,
                COUNT(*) as total_races,
                AVG(CASE WHEN first = 1 THEN 1.0 ELSE 0.0 END) as in_win_rate,
                AVG(CASE WHEN first = 1 OR second = 1 THEN 1.0 ELSE 0.0 END) as in_top2_rate,
                AVG(trifecta_payout) as avg_trifecta_payout,
                AVG(exacta_payout) as avg_exacta_payout
            FROM race_result
            GROUP BY place_code
            ORDER BY place_code
        """)


def main():
    """メイン実行"""
    pipeline = ETLPipeline()
    pipeline.run_full_pipeline()
    
    # バリデーション
    results = pipeline.validate_data()
    print(results)
    
    # サマリー統計
    summary = pipeline.generate_summary_stats()
    print(summary)


if __name__ == "__main__":
    main()
