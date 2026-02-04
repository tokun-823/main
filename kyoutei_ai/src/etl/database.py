"""
DuckDB データベース管理
高速な分析用DBでデータを管理
"""
import duckdb
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, DATA_DIR, PROCESSED_DATA_DIR


class DatabaseManager:
    """DuckDBデータベース管理クラス"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.db_path
        self.conn = None
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """データベースに接続"""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
            # メモリ設定
            self.conn.execute(f"SET memory_limit='{config.database.memory_limit}'")
            self.conn.execute(f"SET threads TO {config.database.threads}")
            logger.info(f"Connected to DuckDB: {self.db_path}")
        return self.conn
    
    def close(self):
        """接続を閉じる"""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("DuckDB connection closed")
    
    def execute(self, query: str, params: tuple = None) -> duckdb.DuckDBPyRelation:
        """SQLクエリを実行"""
        conn = self.connect()
        if params:
            return conn.execute(query, params)
        return conn.execute(query)
    
    def query(self, query: str) -> pd.DataFrame:
        """クエリを実行してDataFrameで返す"""
        return self.execute(query).fetchdf()
    
    def init_schema(self):
        """データベーススキーマを初期化"""
        conn = self.connect()
        
        # 番組表テーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bangumi (
                race_date VARCHAR,
                place_code VARCHAR,
                race_number INTEGER,
                race_grade VARCHAR,
                race_name VARCHAR,
                distance INTEGER,
                waku INTEGER,
                racer_id VARCHAR,
                racer_name VARCHAR,
                racer_class VARCHAR,
                branch VARCHAR,
                age INTEGER,
                weight FLOAT,
                win_rate FLOAT,
                two_rate FLOAT,
                three_rate FLOAT,
                motor_no VARCHAR,
                motor_win_rate FLOAT,
                motor_two_rate FLOAT,
                boat_no VARCHAR,
                boat_win_rate FLOAT,
                boat_two_rate FLOAT,
                PRIMARY KEY (race_date, place_code, race_number, waku)
            )
        """)
        
        # 競争結果テーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS race_result (
                race_date VARCHAR,
                place_code VARCHAR,
                race_number INTEGER,
                winning_pattern VARCHAR,
                first INTEGER,
                second INTEGER,
                third INTEGER,
                st_1 FLOAT,
                st_2 FLOAT,
                st_3 FLOAT,
                st_4 FLOAT,
                st_5 FLOAT,
                st_6 FLOAT,
                trifecta_combo VARCHAR,
                trifecta_payout INTEGER,
                trio_combo VARCHAR,
                trio_payout INTEGER,
                exacta_combo VARCHAR,
                exacta_payout INTEGER,
                quinella_combo VARCHAR,
                quinella_payout INTEGER,
                win_combo VARCHAR,
                win_payout INTEGER,
                PRIMARY KEY (race_date, place_code, race_number)
            )
        """)
        
        # 選手マスターテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS racer_master (
                racer_id VARCHAR PRIMARY KEY,
                racer_name VARCHAR,
                branch VARCHAR,
                racer_class VARCHAR,
                total_races INTEGER,
                first_place INTEGER,
                second_place INTEGER,
                third_place INTEGER,
                win_rate FLOAT,
                two_rate FLOAT,
                avg_start_timing FLOAT,
                flying_count INTEGER,
                late_start_count INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # オッズテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS odds (
                race_date VARCHAR,
                place_code VARCHAR,
                race_number INTEGER,
                ticket_type VARCHAR,
                combination VARCHAR,
                odds_value FLOAT,
                captured_at TIMESTAMP,
                PRIMARY KEY (race_date, place_code, race_number, ticket_type, combination)
            )
        """)
        
        # 直前情報テーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS before_info (
                race_date VARCHAR,
                place_code VARCHAR,
                race_number INTEGER,
                waku INTEGER,
                exhibition_time FLOAT,
                start_timing FLOAT,
                entry_course INTEGER,
                tilt FLOAT,
                parts_exchange VARCHAR,
                PRIMARY KEY (race_date, place_code, race_number, waku)
            )
        """)
        
        # 気象情報テーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weather (
                race_date VARCHAR,
                place_code VARCHAR,
                race_number INTEGER,
                weather_condition VARCHAR,
                wind_direction VARCHAR,
                wind_speed FLOAT,
                wave_height FLOAT,
                temperature FLOAT,
                water_temperature FLOAT,
                PRIMARY KEY (race_date, place_code, race_number)
            )
        """)
        
        # 特徴量テーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS features (
                race_date VARCHAR,
                place_code VARCHAR,
                race_number INTEGER,
                feature_json JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (race_date, place_code, race_number)
            )
        """)
        
        # 予測結果テーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                race_date VARCHAR,
                place_code VARCHAR,
                race_number INTEGER,
                combination VARCHAR,
                predicted_prob FLOAT,
                predicted_ev_q10 FLOAT,
                predicted_ev_q50 FLOAT,
                predicted_ev_q80 FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (race_date, place_code, race_number, combination)
            )
        """)
        
        # インデックス作成
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bangumi_date ON bangumi(race_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bangumi_racer ON bangumi(racer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_result_date ON race_result(race_date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_odds_date ON odds(race_date)")
        
        logger.info("Database schema initialized")
    
    def insert_dataframe(self, table_name: str, df: pd.DataFrame, if_exists: str = 'append'):
        """DataFrameをテーブルに挿入"""
        conn = self.connect()
        
        if if_exists == 'replace':
            conn.execute(f"DELETE FROM {table_name}")
        
        # 一時テーブル経由で挿入（高速）
        conn.register('df_temp', df)
        conn.execute(f"""
            INSERT OR REPLACE INTO {table_name}
            SELECT * FROM df_temp
        """)
        conn.unregister('df_temp')
        
        logger.info(f"Inserted {len(df)} rows into {table_name}")
    
    def get_race_data(
        self,
        race_date: str,
        place_code: str,
        race_number: int
    ) -> Dict[str, pd.DataFrame]:
        """1レースの全データを取得"""
        bangumi = self.query(f"""
            SELECT * FROM bangumi
            WHERE race_date = '{race_date}'
            AND place_code = '{place_code}'
            AND race_number = {race_number}
            ORDER BY waku
        """)
        
        result = self.query(f"""
            SELECT * FROM race_result
            WHERE race_date = '{race_date}'
            AND place_code = '{place_code}'
            AND race_number = {race_number}
        """)
        
        odds = self.query(f"""
            SELECT * FROM odds
            WHERE race_date = '{race_date}'
            AND place_code = '{place_code}'
            AND race_number = {race_number}
        """)
        
        before_info = self.query(f"""
            SELECT * FROM before_info
            WHERE race_date = '{race_date}'
            AND place_code = '{place_code}'
            AND race_number = {race_number}
        """)
        
        return {
            'bangumi': bangumi,
            'result': result,
            'odds': odds,
            'before_info': before_info
        }
    
    def get_racer_stats(self, racer_id: str) -> pd.DataFrame:
        """選手の過去成績を集計"""
        return self.query(f"""
            SELECT 
                b.racer_id,
                b.racer_name,
                b.place_code,
                b.waku,
                r.first,
                r.second,
                r.third,
                r.st_{b.waku} as start_timing,
                CASE WHEN r.first = b.waku THEN 1 ELSE 0 END as is_first,
                CASE WHEN r.second = b.waku THEN 1 ELSE 0 END as is_second,
                CASE WHEN r.first = b.waku OR r.second = b.waku THEN 1 ELSE 0 END as is_top2
            FROM bangumi b
            JOIN race_result r ON 
                b.race_date = r.race_date 
                AND b.place_code = r.place_code 
                AND b.race_number = r.race_number
            WHERE b.racer_id = '{racer_id}'
            ORDER BY b.race_date DESC
            LIMIT 100
        """)
    
    def get_place_stats(self, place_code: str) -> pd.DataFrame:
        """会場別統計を取得"""
        return self.query(f"""
            SELECT
                place_code,
                first as waku,
                COUNT(*) as win_count,
                COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() as win_rate
            FROM race_result
            WHERE place_code = '{place_code}'
            GROUP BY first
            ORDER BY first
        """)
    
    def export_to_parquet(self, output_dir: Path = None):
        """全テーブルをParquetファイルにエクスポート"""
        output_dir = output_dir or PROCESSED_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tables = ['bangumi', 'race_result', 'racer_master', 'odds', 'before_info', 'weather']
        
        for table in tables:
            try:
                df = self.query(f"SELECT * FROM {table}")
                output_path = output_dir / f"{table}.parquet"
                df.to_parquet(output_path, index=False)
                logger.info(f"Exported {table} to {output_path}")
            except Exception as e:
                logger.warning(f"Failed to export {table}: {e}")


# グローバルインスタンス
db = DatabaseManager()
