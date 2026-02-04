# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- 統合スクレイパー
=====================================
複数ソースを統合してデータを取得
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

import pandas as pd
from loguru import logger

from .base_scraper import BaseScraper
from .keirin_jp_scraper import KeirinJPScraper
from .netkeiba_scraper import NetkeibaKeirinScraper
from .oddspark_scraper import OddsparkScraper
from ..config import ScrapingConfig, RAW_DATA_DIR


class IntegratedScraper:
    """
    複数のスクレイパーを統合して、
    最も信頼性の高いデータを取得する
    
    戦略:
    1. 主ソース（keirin.jp）から取得を試みる
    2. 失敗時は代替ソース（netkeiba, oddspark）を使用
    3. データを統合・補完
    """
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        
        # スクレイパーインスタンス
        self._keirin_jp = None
        self._netkeiba = None
        self._oddspark = None
        
        # データキャッシュディレクトリ
        self.data_dir = RAW_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def keirin_jp(self) -> KeirinJPScraper:
        if self._keirin_jp is None:
            self._keirin_jp = KeirinJPScraper(self.config)
        return self._keirin_jp
    
    @property
    def netkeiba(self) -> NetkeibaKeirinScraper:
        if self._netkeiba is None:
            self._netkeiba = NetkeibaKeirinScraper(self.config)
        return self._netkeiba
    
    @property
    def oddspark(self) -> OddsparkScraper:
        if self._oddspark is None:
            self._oddspark = OddsparkScraper(self.config)
        return self._oddspark
    
    def fetch_race_list(self, date: str) -> List[Dict]:
        """
        指定日のレース一覧を取得（複数ソースから）
        
        Args:
            date: 日付（YYYY-MM-DD形式）
            
        Returns:
            List[Dict]: レース情報リスト
        """
        logger.info(f"Fetching race list for {date}")
        
        # 主ソース: keirin.jp
        races = self.keirin_jp.scrape_race_list(date)
        
        if not races:
            logger.warning("keirin.jp failed, trying netkeiba")
            races = self.netkeiba.scrape_race_list(date)
        
        if not races:
            logger.warning("netkeiba failed, trying oddspark")
            races = self.oddspark.scrape_race_list(date)
        
        if races:
            logger.info(f"Found {len(races)} races for {date}")
        else:
            logger.error(f"All sources failed for {date}")
        
        return races
    
    def fetch_race_data(self, race_id: str) -> Dict:
        """
        レースの詳細データを取得
        
        Args:
            race_id: レースID
            
        Returns:
            Dict: レース詳細データ（info, entries, result含む）
        """
        logger.info(f"Fetching race data for {race_id}")
        
        race_data = {
            "race_id": race_id,
            "info": {},
            "entries": [],
            "result": {},
        }
        
        # レース情報
        info = self.keirin_jp.scrape_race_info(race_id)
        if not info:
            info = self.netkeiba.scrape_race_info(race_id)
        race_data["info"] = info
        
        # 出走表
        entries = self.keirin_jp.scrape_entry_table(race_id)
        if not entries:
            entries = self.netkeiba.scrape_entry_table(race_id)
        if not entries:
            entries = self.oddspark.scrape_entry_table(race_id)
        race_data["entries"] = entries
        
        # 結果（過去レースの場合）
        result = self.keirin_jp.scrape_race_result(race_id)
        if not result.get("results"):
            result = self.netkeiba.scrape_race_result(race_id)
        race_data["result"] = result
        
        return race_data
    
    def fetch_entries_for_prediction(self, date: str) -> pd.DataFrame:
        """
        予測用の出走表データを取得してDataFrameで返す
        
        Args:
            date: 日付
            
        Returns:
            pd.DataFrame: 予測用データフレーム
        """
        races = self.fetch_race_list(date)
        all_entries = []
        
        for race in races:
            race_id = race["race_id"]
            
            # レース情報
            info = self.keirin_jp.scrape_race_info(race_id)
            
            # 出走表
            entries = self.keirin_jp.scrape_entry_table(race_id)
            if not entries:
                entries = self.netkeiba.scrape_entry_table(race_id)
            
            # 各エントリーにレース情報を追加
            for entry in entries:
                entry.update({
                    "date": date,
                    "venue_name": race.get("venue_name", ""),
                    "race_no": race.get("race_no", 0),
                    "grade": race.get("grade", "F1"),
                    "bank_type": info.get("bank_type", 400),
                    "is_girls": info.get("is_girls", False),
                    "is_challenge": info.get("is_challenge", False),
                })
                all_entries.append(entry)
            
            time.sleep(0.5)  # レート制限対策
        
        df = pd.DataFrame(all_entries)
        
        # 保存
        output_file = self.data_dir / f"entries_{date.replace('-', '')}.csv"
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info(f"Saved entries to {output_file}")
        
        return df
    
    def fetch_historical_data(
        self,
        start_date: str,
        end_date: str,
        save_interval: int = 10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        過去データを一括取得
        
        Args:
            start_date: 開始日
            end_date: 終了日
            save_interval: 中間保存間隔（日数）
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (出走表DF, 結果DF)
        """
        from ..utils import get_date_range
        
        dates = get_date_range(start_date, end_date)
        all_entries = []
        all_results = []
        
        for i, date in enumerate(dates):
            logger.info(f"Processing {date} ({i+1}/{len(dates)})")
            
            try:
                races = self.fetch_race_list(date)
                
                for race in races:
                    race_id = race["race_id"]
                    
                    try:
                        race_data = self.fetch_race_data(race_id)
                        
                        # 出走表
                        for entry in race_data["entries"]:
                            entry.update({
                                "date": date,
                                "venue_name": race.get("venue_name", ""),
                                "race_no": race.get("race_no", 0),
                                "grade": race.get("grade", "F1"),
                            })
                            all_entries.append(entry)
                        
                        # 結果
                        result = race_data["result"]
                        if result.get("results"):
                            result_entry = {
                                "race_id": race_id,
                                "date": date,
                            }
                            
                            # 着順を展開
                            for r in result["results"]:
                                rank = r["rank"]
                                car = r["car_number"]
                                result_entry[f"rank{rank}"] = car
                            
                            # 払戻
                            payouts = result.get("payouts", {})
                            for key, val in payouts.items():
                                if isinstance(val, dict):
                                    result_entry[f"{key}_payout"] = val.get("payout", 0)
                            
                            # 落車・失格フラグ
                            result_entry["has_crash"] = result.get("has_crash", False)
                            result_entry["has_disqualification"] = result.get("has_disqualification", False)
                            
                            all_results.append(result_entry)
                        
                        time.sleep(0.3)
                        
                    except Exception as e:
                        logger.error(f"Error processing race {race_id}: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error processing date {date}: {e}")
                continue
            
            # 中間保存
            if (i + 1) % save_interval == 0:
                self._save_intermediate(all_entries, all_results, i + 1)
        
        # 最終保存
        entries_df = pd.DataFrame(all_entries)
        results_df = pd.DataFrame(all_results)
        
        entries_file = self.data_dir / f"entries_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        results_file = self.data_dir / f"results_{start_date.replace('-', '')}_{end_date.replace('-', '')}.csv"
        
        entries_df.to_csv(entries_file, index=False, encoding="utf-8-sig")
        results_df.to_csv(results_file, index=False, encoding="utf-8-sig")
        
        logger.info(f"Saved {len(entries_df)} entries to {entries_file}")
        logger.info(f"Saved {len(results_df)} results to {results_file}")
        
        return entries_df, results_df
    
    def _save_intermediate(
        self,
        entries: List[Dict],
        results: List[Dict],
        count: int
    ) -> None:
        """中間データを保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        entries_file = self.data_dir / f"entries_intermediate_{timestamp}.csv"
        results_file = self.data_dir / f"results_intermediate_{timestamp}.csv"
        
        pd.DataFrame(entries).to_csv(entries_file, index=False, encoding="utf-8-sig")
        pd.DataFrame(results).to_csv(results_file, index=False, encoding="utf-8-sig")
        
        logger.info(f"Intermediate save at {count} dates: {entries_file}")
    
    def close(self) -> None:
        """リソースを解放"""
        if self._keirin_jp:
            self._keirin_jp.close()
        if self._netkeiba:
            self._netkeiba.close()
        if self._oddspark:
            self._oddspark.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
