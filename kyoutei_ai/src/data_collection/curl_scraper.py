"""
データ収集モジュール - curl-cffi を使用した高度なスクレイパー
Cloudflare等のWAF/Bot対策を回避
"""
import os
import time
import random
import asyncio
import hashlib
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, RAW_DATA_DIR, RACECOURSE_CODES


class CurlCffiScraper:
    """
    curl-cffi を使用したスクレイパー
    TLS Fingerprint を本物のブラウザに偽装してBot検出を回避
    """
    
    BASE_URL = "https://www.boatrace.jp"
    
    # curl-cffi でサポートされているブラウザ偽装
    BROWSER_IMPERSONATES = [
        "chrome110",
        "chrome107",
        "chrome104",
        "chrome101",
        "chrome99",
        "edge101",
        "edge99",
        "safari15_3",
        "safari15_5",
    ]
    
    def __init__(self):
        self.config = config.scraping
        self.cache_dir = RAW_DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.odds_dir = RAW_DATA_DIR / "odds"
        self.before_info_dir = RAW_DATA_DIR / "before_info"
        self.odds_dir.mkdir(parents=True, exist_ok=True)
        self.before_info_dir.mkdir(parents=True, exist_ok=True)
        
        self.request_count = 0
        self.session = None
    
    def _get_impersonate(self) -> str:
        """ランダムなブラウザ偽装を選択"""
        return random.choice(self.BROWSER_IMPERSONATES)
    
    def _get_cache_path(self, url: str) -> Path:
        """キャッシュファイルパスを生成"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"
    
    def _load_from_cache(self, url: str) -> Optional[str]:
        """キャッシュから読み込み"""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        return None
    
    def _save_to_cache(self, url: str, content: str):
        """キャッシュに保存"""
        cache_path = self._get_cache_path(url)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _smart_delay(self):
        """スマートな遅延処理"""
        if self.request_count % 100 == 0 and self.request_count > 0:
            # 100リクエストごとに長めの休憩
            delay = random.uniform(30, 60)
            logger.info(f"Long break: {delay:.1f}s (after {self.request_count} requests)")
        elif self.request_count % 20 == 0 and self.request_count > 0:
            # 20リクエストごとに中程度の休憩
            delay = random.uniform(10, 20)
        else:
            # 通常の遅延（人間らしいランダム性）
            base = self.config.request_interval
            delay = random.uniform(base * 0.8, base * 2.5)
        
        time.sleep(delay)
        self.request_count += 1
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=120))
    def fetch_page(self, url: str, use_cache: bool = True) -> Optional[str]:
        """ページを取得（curl-cffi使用）"""
        try:
            from curl_cffi import requests as curl_requests
        except ImportError:
            logger.error("curl-cffi not installed. Run: pip install curl-cffi")
            raise
        
        # キャッシュチェック
        if use_cache:
            cached = self._load_from_cache(url)
            if cached:
                logger.debug(f"Cache hit: {url}")
                return cached
        
        # 遅延
        self._smart_delay()
        
        try:
            impersonate = self._get_impersonate()
            
            response = curl_requests.get(
                url,
                impersonate=impersonate,
                timeout=self.config.request_timeout,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "ja,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                },
                allow_redirects=True,
            )
            
            if response.status_code == 200:
                content = response.text
                
                # ブロック検出
                if "アクセスが集中" in content or "しばらく時間" in content:
                    logger.warning(f"Blocked detected: {url}")
                    time.sleep(random.uniform(60, 120))
                    raise Exception("Blocked")
                
                # キャッシュ保存
                if use_cache:
                    self._save_to_cache(url, content)
                
                logger.debug(f"Fetched: {url} (impersonate: {impersonate})")
                return content
                
            elif response.status_code == 403:
                logger.warning(f"Forbidden (403): {url}")
                time.sleep(random.uniform(120, 180))
                raise Exception("Forbidden")
                
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429): {url}")
                time.sleep(random.uniform(180, 300))
                raise Exception("Rate limited")
                
            else:
                logger.warning(f"Status {response.status_code}: {url}")
                return None
                
        except Exception as e:
            logger.error(f"Fetch error: {url}, {e}")
            raise
    
    def parse_odds_3tan(self, html: str) -> Dict[str, float]:
        """3連単オッズをパース"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'lxml')
        odds_dict = {}
        
        # 全ての1着番号についてオッズを取得
        for first in range(1, 7):
            # 各1着固定のオッズテーブルを探す
            table = soup.find('table', {'data-pivot': str(first)})
            if not table:
                continue
            
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td', class_='oddsPoint')
                for cell in cells:
                    odds_text = cell.get_text(strip=True)
                    combo_id = cell.get('id', '')
                    
                    if odds_text and odds_text not in ['-', '欠場', '除外', '取消']:
                        try:
                            odds_value = float(odds_text.replace(',', ''))
                            if combo_id:
                                odds_dict[combo_id] = odds_value
                        except ValueError:
                            pass
        
        return odds_dict
    
    def parse_before_info(self, html: str) -> Dict[str, Any]:
        """直前情報をパース"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'lxml')
        info = {
            'exhibition_times': {},
            'start_timing': {},
            'entry_course': [],
            'weather': {},
            'tilt': {},
            'parts_exchange': [],
        }
        
        # 展示タイム
        tenji_rows = soup.select('.table1 tbody tr')
        for row in tenji_rows:
            waku_elem = row.select_one('.is-fs14')
            time_elem = row.select_one('.is-lineH2')
            if waku_elem and time_elem:
                waku = waku_elem.get_text(strip=True)
                time_text = time_elem.get_text(strip=True)
                try:
                    info['exhibition_times'][waku] = float(time_text)
                except ValueError:
                    pass
        
        # 気象情報
        weather_section = soup.select_one('.weather1')
        if weather_section:
            # 天候
            weather_label = weather_section.select_one('.weather1_bodyUnitLabelTitle')
            if weather_label:
                info['weather']['condition'] = weather_label.get_text(strip=True)
            
            # 風・波
            data_labels = weather_section.select('.weather1_bodyUnitLabelData')
            if len(data_labels) >= 1:
                info['weather']['wind'] = data_labels[0].get_text(strip=True)
            if len(data_labels) >= 2:
                info['weather']['wave'] = data_labels[1].get_text(strip=True)
        
        # スタート展示
        start_section = soup.select_one('.table1_boatImage1')
        if start_section:
            entries = start_section.select('.table1_boatImage1Number')
            for i, entry in enumerate(entries):
                num = entry.get_text(strip=True)
                info['entry_course'].append(num)
        
        # 部品交換
        parts_section = soup.select('.labelGroup1Unit')
        for part in parts_section:
            part_text = part.get_text(strip=True)
            if part_text:
                info['parts_exchange'].append(part_text)
        
        return info
    
    def parse_race_result(self, html: str) -> Dict[str, Any]:
        """レース結果をパース"""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'lxml')
        result = {
            'finish_order': [],
            'start_timing': {},
            'payoff_3tan': None,
            'payoff_3fuku': None,
            'payoff_2tan': None,
            'payoff_2fuku': None,
            'payoff_1tan': None,
            'decision_time': None,
        }
        
        # 着順
        result_table = soup.select_one('.is-w495')
        if result_table:
            rows = result_table.select('tbody tr')
            for row in rows:
                cells = row.select('td')
                if len(cells) >= 3:
                    order = cells[0].get_text(strip=True)
                    waku = cells[1].get_text(strip=True)
                    result['finish_order'].append({
                        'order': order,
                        'waku': waku
                    })
        
        # 払戻金テーブル
        payout_table = soup.select_one('.is-w243')
        if payout_table:
            rows = payout_table.select('tr')
            for row in rows:
                header = row.select_one('th')
                cells = row.select('td')
                if header and len(cells) >= 2:
                    ticket_type = header.get_text(strip=True)
                    combo = cells[0].get_text(strip=True)
                    payout = cells[1].get_text(strip=True)
                    
                    payout_data = {
                        'combination': combo,
                        'payout': payout.replace('¥', '').replace(',', '').strip()
                    }
                    
                    if '3連単' in ticket_type:
                        result['payoff_3tan'] = payout_data
                    elif '3連複' in ticket_type:
                        result['payoff_3fuku'] = payout_data
                    elif '2連単' in ticket_type:
                        result['payoff_2tan'] = payout_data
                    elif '2連複' in ticket_type:
                        result['payoff_2fuku'] = payout_data
                    elif '単勝' in ticket_type:
                        result['payoff_1tan'] = payout_data
        
        return result
    
    def scrape_race_data(
        self,
        race_date: str,
        place_code: str,
        race_number: int
    ) -> Dict[str, Any]:
        """1レースの全データを取得"""
        race_data = {}
        
        # オッズURL
        odds_3tan_url = (
            f"{self.BASE_URL}/owpc/pc/race/odds3t"
            f"?rno={race_number}&jcd={place_code}&hd={race_date}"
        )
        
        # 直前情報URL
        before_url = (
            f"{self.BASE_URL}/owpc/pc/race/beforeinfo"
            f"?rno={race_number}&jcd={place_code}&hd={race_date}"
        )
        
        # 結果URL
        result_url = (
            f"{self.BASE_URL}/owpc/pc/race/raceresult"
            f"?rno={race_number}&jcd={place_code}&hd={race_date}"
        )
        
        try:
            # オッズ取得
            odds_html = self.fetch_page(odds_3tan_url)
            if odds_html:
                race_data['odds_3tan'] = self.parse_odds_3tan(odds_html)
            
            # 直前情報取得
            before_html = self.fetch_page(before_url)
            if before_html:
                race_data['before_info'] = self.parse_before_info(before_html)
            
            # 結果取得
            result_html = self.fetch_page(result_url)
            if result_html:
                race_data['result'] = self.parse_race_result(result_html)
                
        except Exception as e:
            logger.error(f"Error scraping {place_code}-{race_number}R on {race_date}: {e}")
        
        return race_data
    
    def scrape_all_races_for_date(self, date: datetime) -> Dict[str, Any]:
        """指定日の全レースデータを取得"""
        date_str = date.strftime("%Y%m%d")
        all_data = {}
        
        logger.info(f"Scraping all races for {date_str}")
        
        for place_code, place_name in RACECOURSE_CODES.items():
            logger.info(f"  Processing {place_name}...")
            place_data = {}
            
            for race_num in range(1, 13):
                race_key = f"{place_code}_{race_num:02d}"
                
                try:
                    race_data = self.scrape_race_data(date_str, place_code, race_num)
                    if race_data:
                        place_data[race_key] = race_data
                        logger.debug(f"    {place_name} {race_num}R OK")
                    
                except Exception as e:
                    logger.error(f"    {place_name} {race_num}R Error: {e}")
            
            if place_data:
                all_data[place_code] = place_data
        
        return all_data
    
    def scrape_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ):
        """日付範囲のデータをスクレイピング"""
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            save_path = self.odds_dir / f"{date_str}.pkl"
            
            if save_path.exists():
                logger.info(f"Already exists: {date_str}")
                current_date += timedelta(days=1)
                continue
            
            try:
                data = self.scrape_all_races_for_date(current_date)
                
                if data:
                    with open(save_path, 'wb') as f:
                        pickle.dump(data, f)
                    logger.info(f"Saved: {date_str}")
                
            except Exception as e:
                logger.error(f"Error on {date_str}: {e}")
            
            current_date += timedelta(days=1)
            
            # 日付ごとに長めの休憩
            time.sleep(random.uniform(10, 30))


def main():
    """メイン実行"""
    scraper = CurlCffiScraper()
    
    # 2018年から2025年までスクレイピング
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # 未来の日付は除外
    if end_date > datetime.now():
        end_date = datetime.now() - timedelta(days=1)
    
    scraper.scrape_date_range(start_date, end_date)


if __name__ == "__main__":
    main()
