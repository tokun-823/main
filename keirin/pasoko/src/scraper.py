# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」- データスクレイピングモジュール
Data Scraping Module for Keirin Prediction AI "Pasoko"
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import calendar
import time
import re
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import pickle

from config import (
    BASE_URL, SCHEDULE_URL, RACE_CARD_URL,
    START_YEAR, END_YEAR, SCRAPE_DELAY, REQUEST_TIMEOUT, MAX_RETRIES,
    DATA_DIR, RACE_IDS_FILE, RACE_INFO_FILE, RACE_CARD_FILE, RACE_RETURN_FILE
)


class KeirinScraper:
    """競輪データスクレイピングクラス"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _request_with_retry(self, url, retries=MAX_RETRIES):
        """リトライ機能付きリクエスト"""
        for i in range(retries):
            try:
                response = self.session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                return response
            except Exception as e:
                if i == retries - 1:
                    print(f"リクエスト失敗: {url}, エラー: {e}")
                    return None
                time.sleep(SCRAPE_DELAY * 2)
        return None
    
    def get_schedule_race_ids(self, year, month):
        """
        指定年月の開催スケジュールからレースIDを取得
        
        Args:
            year: 年 (2018-2025)
            month: 月 (1-12)
            
        Returns:
            list: レースIDのリスト
        """
        race_ids = []
        url = f"{SCHEDULE_URL}{year}{month:02d}/"
        
        response = self._request_with_retry(url)
        if response is None:
            return race_ids
            
        soup = BeautifulSoup(response.content, 'lxml')
        
        # 開催日のリンクを取得
        schedule_links = soup.find_all('a', href=re.compile(r'/gamboo/keirin-kaisai/\d+/'))
        
        for link in schedule_links:
            href = link.get('href', '')
            match = re.search(r'/gamboo/keirin-kaisai/(\d+)/', href)
            if match:
                base_id = match.group(1)
                race_ids.append(base_id)
                
        time.sleep(SCRAPE_DELAY)
        return list(set(race_ids))
    
    def generate_race_ids_for_period(self, start_year=START_YEAR, end_year=END_YEAR):
        """
        指定期間の全レースIDを生成
        
        Args:
            start_year: 開始年
            end_year: 終了年
            
        Returns:
            list: 全レースIDのリスト
        """
        all_race_ids = []
        
        # 年月をループ
        for year in tqdm(range(start_year, end_year + 1), desc="年度処理"):
            for month in tqdm(range(1, 13), desc=f"{year}年", leave=False):
                # 月末日を取得
                _, last_day = calendar.monthrange(year, month)
                
                # スケジュールからベースIDを取得
                base_ids = self.get_schedule_race_ids(year, month)
                
                # 各開催日のレースIDを生成
                for base_id in base_ids:
                    # 最大12レースまで
                    for race_num in range(1, 13):
                        race_id = f"{base_id}{race_num:02d}"
                        all_race_ids.append(race_id)
                        
        return all_race_ids
    
    def generate_race_ids_by_date_range(self, start_date, end_date):
        """
        日付範囲でレースIDを生成（月をまたぐ開催も処理）
        
        Args:
            start_date: 開始日 (datetime)
            end_date: 終了日 (datetime)
            
        Returns:
            list: レースIDのリスト
        """
        race_ids = []
        current_date = start_date
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            
            base_ids = self.get_schedule_race_ids(year, month)
            
            for base_id in base_ids:
                for race_num in range(1, 13):
                    race_id = f"{base_id}{race_num:02d}"
                    race_ids.append(race_id)
                    
            # 翌月へ
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)
                
        return list(set(race_ids))
    
    def scrape_race_data(self, race_id):
        """
        レースデータ（出走表・結果・払戻情報）をスクレイピング
        
        Args:
            race_id: レースID
            
        Returns:
            tuple: (race_info, race_card, race_return) or (None, None, None)
        """
        url = f"{RACE_CARD_URL}{race_id}/"
        
        response = self._request_with_retry(url)
        if response is None:
            return None, None, None
        
        try:
            soup = BeautifulSoup(response.content, 'lxml')
            
            # レース情報を取得
            race_info = self._extract_race_info(soup, race_id)
            if race_info is None:
                return None, None, None
            
            # 出走表を取得
            race_card = self._extract_race_card(soup, race_id)
            if race_card is None or race_card.empty:
                return None, None, None
            
            # 払戻情報を取得
            race_return = self._extract_race_return(soup, race_id)
            
            time.sleep(SCRAPE_DELAY)
            return race_info, race_card, race_return
            
        except (AttributeError, IndexError) as e:
            # 開催中止やレース中止の場合はスキップ
            print(f"スキップ: {race_id} - {e}")
            return None, None, None
    
    def _extract_race_info(self, soup, race_id):
        """レース個別情報を抽出"""
        try:
            race_info = {"race_id": race_id}
            
            # レースタイトル
            title_elem = soup.find('h2', class_='race-title') or soup.find('div', class_='race-name')
            if title_elem:
                race_info['race_name'] = title_elem.get_text(strip=True)
            else:
                race_info['race_name'] = ""
            
            # 競輪場名
            venue_elem = soup.find('div', class_='venue-name') or soup.find('span', class_='place')
            if venue_elem:
                race_info['venue'] = venue_elem.get_text(strip=True)
            else:
                race_info['venue'] = ""
            
            # グレード
            grade_elem = soup.find('span', class_='grade')
            if grade_elem:
                race_info['grade'] = grade_elem.get_text(strip=True)
            else:
                race_info['grade'] = ""
            
            # 開催日
            date_elem = soup.find('span', class_='race-date') or soup.find('div', class_='date')
            if date_elem:
                race_info['race_date'] = date_elem.get_text(strip=True)
            else:
                race_info['race_date'] = ""
                
            # レース番号
            race_num_elem = soup.find('span', class_='race-number')
            if race_num_elem:
                race_info['race_number'] = race_num_elem.get_text(strip=True)
            else:
                race_info['race_number'] = race_id[-2:]
            
            return race_info
            
        except Exception:
            return None
    
    def _extract_race_card(self, soup, race_id):
        """出走表を抽出"""
        try:
            # 出走表テーブルを探す
            tables = soup.find_all('table')
            
            for table in tables:
                # 出走表っぽいテーブルを探す
                headers = table.find_all('th')
                header_text = [h.get_text(strip=True) for h in headers]
                
                if any(keyword in str(header_text) for keyword in ['車番', '選手名', '競走得点', '府県']):
                    # テーブルをパース
                    rows = table.find_all('tr')
                    data = []
                    
                    for row in rows[1:]:  # ヘッダーをスキップ
                        cells = row.find_all(['td', 'th'])
                        row_data = [cell.get_text(strip=True) for cell in cells]
                        if row_data:
                            data.append(row_data)
                    
                    if data:
                        df = pd.DataFrame(data)
                        df['race_id'] = race_id
                        return df
            
            # 代替：class指定で探す
            race_table = soup.find('table', class_='race-card') or soup.find('div', class_='entry-table')
            if race_table:
                df = pd.read_html(str(race_table))[0]
                df['race_id'] = race_id
                return df
                
            return pd.DataFrame()
            
        except Exception as e:
            print(f"出走表抽出エラー: {race_id} - {e}")
            return pd.DataFrame()
    
    def _extract_race_return(self, soup, race_id):
        """払戻情報を抽出"""
        try:
            race_return = {"race_id": race_id}
            
            # 払戻テーブルを探す
            return_tables = soup.find_all('table', class_='payout') or soup.find_all('div', class_='result-payout')
            
            if return_tables:
                for table in return_tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 2:
                            bet_type = cells[0].get_text(strip=True)
                            payout = cells[-1].get_text(strip=True).replace(',', '').replace('円', '')
                            race_return[bet_type] = payout
            
            # 着順情報
            result_elem = soup.find('div', class_='race-result') or soup.find('table', class_='result')
            if result_elem:
                result_rows = result_elem.find_all('tr')
                for idx, row in enumerate(result_rows[:3], 1):
                    cells = row.find_all('td')
                    if cells:
                        race_return[f'{idx}着'] = cells[0].get_text(strip=True)
            
            return race_return
            
        except Exception:
            return {"race_id": race_id}
    
    def scrape_all_races(self, race_ids, save_interval=100):
        """
        全レースデータをスクレイピング
        
        Args:
            race_ids: レースIDのリスト
            save_interval: 中間保存間隔
            
        Returns:
            tuple: (all_race_info, all_race_card, all_race_return)
        """
        all_race_info = []
        all_race_card = []
        all_race_return = []
        
        for idx, race_id in enumerate(tqdm(race_ids, desc="レースデータ取得")):
            race_info, race_card, race_return = self.scrape_race_data(race_id)
            
            if race_info is not None:
                all_race_info.append(race_info)
            if race_card is not None and not race_card.empty:
                all_race_card.append(race_card)
            if race_return is not None:
                all_race_return.append(race_return)
            
            # 中間保存
            if (idx + 1) % save_interval == 0:
                self._save_intermediate(all_race_info, all_race_card, all_race_return)
                print(f"\n中間保存完了: {idx + 1}/{len(race_ids)}")
        
        return all_race_info, all_race_card, all_race_return
    
    def _save_intermediate(self, race_info, race_card, race_return):
        """中間データを保存"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(f"{DATA_DIR}/race_info_temp.pkl", 'wb') as f:
            pickle.dump(race_info, f)
        
        if race_card:
            df_card = pd.concat(race_card, ignore_index=True)
            with open(f"{DATA_DIR}/race_card_temp.pkl", 'wb') as f:
                pickle.dump(df_card, f)
        
        with open(f"{DATA_DIR}/race_return_temp.pkl", 'wb') as f:
            pickle.dump(race_return, f)
    
    def save_data(self, race_info, race_card, race_return):
        """スクレイピングデータを保存"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # レース情報
        df_info = pd.DataFrame(race_info)
        df_info.to_pickle(RACE_INFO_FILE)
        print(f"レース情報保存: {len(df_info)}件")
        
        # 出走表
        if race_card:
            df_card = pd.concat(race_card, ignore_index=True)
            df_card.to_pickle(RACE_CARD_FILE)
            print(f"出走表保存: {len(df_card)}件")
        
        # 払戻情報
        df_return = pd.DataFrame(race_return)
        df_return.to_pickle(RACE_RETURN_FILE)
        print(f"払戻情報保存: {len(df_return)}件")
    
    def load_data(self):
        """保存されたデータを読み込み"""
        df_info = pd.read_pickle(RACE_INFO_FILE) if os.path.exists(RACE_INFO_FILE) else pd.DataFrame()
        df_card = pd.read_pickle(RACE_CARD_FILE) if os.path.exists(RACE_CARD_FILE) else pd.DataFrame()
        df_return = pd.read_pickle(RACE_RETURN_FILE) if os.path.exists(RACE_RETURN_FILE) else pd.DataFrame()
        
        return df_info, df_card, df_return


def scrape_future_races(date_str):
    """
    未来のレース（今日以降）の出走表を取得
    
    Args:
        date_str: 日付文字列 (YYYYMMDD形式)
        
    Returns:
        tuple: (race_info_list, race_card_list)
    """
    scraper = KeirinScraper()
    
    # 日付からレースIDを生成
    year = int(date_str[:4])
    month = int(date_str[4:6])
    
    base_ids = scraper.get_schedule_race_ids(year, month)
    
    race_info_list = []
    race_card_list = []
    
    for base_id in tqdm(base_ids, desc="未来レース取得"):
        for race_num in range(1, 13):
            race_id = f"{base_id}{race_num:02d}"
            
            # 出走表のみ取得（結果はまだない）
            url = f"{RACE_CARD_URL}{race_id}/"
            try:
                response = scraper._request_with_retry(url)
                if response:
                    soup = BeautifulSoup(response.content, 'lxml')
                    race_info = scraper._extract_race_info(soup, race_id)
                    race_card = scraper._extract_race_card(soup, race_id)
                    
                    if race_info:
                        race_info_list.append(race_info)
                    if race_card is not None and not race_card.empty:
                        race_card_list.append(race_card)
            except:
                continue
                
            time.sleep(SCRAPE_DELAY)
    
    return race_info_list, race_card_list


if __name__ == "__main__":
    # テスト実行
    scraper = KeirinScraper()
    
    # 2024年1月のデータを試しに取得
    print("レースID生成テスト...")
    race_ids = scraper.get_schedule_race_ids(2024, 1)
    print(f"取得したベースID数: {len(race_ids)}")
    
    if race_ids:
        print(f"サンプルID: {race_ids[:5]}")
