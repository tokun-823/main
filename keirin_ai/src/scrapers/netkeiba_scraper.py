# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- Netkeiba競輪スクレイパー
==============================================
keirin.netkeiba.com からデータを取得
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from loguru import logger

from .base_scraper import BaseScraper
from ..config import ScrapingConfig
from ..utils import safe_float, safe_int, get_bank_type_numeric


class NetkeibaKeirinScraper(BaseScraper):
    """
    netkeiba競輪からデータを取得するスクレイパー
    
    特徴:
    - 詳細な選手成績データ
    - 過去レースへのアクセスが容易
    - JavaScript動的読み込みあり
    """
    
    BASE_URL = "https://keirin.netkeiba.com"
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        super().__init__(config)
    
    def scrape_race_list(self, date: str) -> List[Dict]:
        """
        指定日のレース一覧を取得
        
        Args:
            date: 日付（YYYY-MM-DD形式）
            
        Returns:
            List[Dict]: レース情報のリスト
        """
        date_formatted = date.replace("-", "")
        url = f"{self.BASE_URL}/race/calendar/{date_formatted}/"
        
        # Seleniumを使用（動的コンテンツ対応）
        html = self.fetch(url, use_selenium=True, wait_selector=".RaceList")
        if not html:
            # フォールバック
            html = self.fetch(url, use_cloudscraper=True)
        
        if not html:
            return []
        
        soup = self.parse_html(html)
        races = []
        
        # レースリストを取得
        race_items = soup.find_all("li", class_="RaceList_Item")
        
        for item in race_items:
            link = item.find("a")
            if not link:
                continue
            
            href = link.get("href", "")
            race_id_match = re.search(r"/race/(\d+)/", href)
            if not race_id_match:
                continue
            
            race_id = race_id_match.group(1)
            
            race_info = {
                "race_id": race_id,
                "date": date,
                "url": urljoin(self.BASE_URL, href),
            }
            
            # 会場名
            venue_elem = item.find("span", class_="Venue")
            if venue_elem:
                race_info["venue_name"] = venue_elem.get_text(strip=True)
            
            # レース番号
            race_no_elem = item.find("span", class_="RaceNo")
            if race_no_elem:
                race_no_match = re.search(r"(\d+)R", race_no_elem.get_text())
                if race_no_match:
                    race_info["race_no"] = int(race_no_match.group(1))
            
            # グレード
            grade_elem = item.find("span", class_="Grade")
            if grade_elem:
                race_info["grade"] = grade_elem.get_text(strip=True)
            
            races.append(race_info)
        
        logger.info(f"Found {len(races)} races on {date} from netkeiba")
        return races
    
    def scrape_race_info(self, race_id: str) -> Dict:
        """
        レース情報を取得
        """
        url = f"{self.BASE_URL}/race/{race_id}/"
        
        html = self.fetch(url, use_selenium=True, wait_selector=".RaceData")
        if not html:
            html = self.fetch(url, use_cloudscraper=True)
        
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        race_info = {
            "race_id": race_id,
        }
        
        # レース名
        title_elem = soup.find("h1", class_="RaceName")
        if title_elem:
            race_info["race_name"] = title_elem.get_text(strip=True)
        
        # レースデータ
        data_elem = soup.find("div", class_="RaceData")
        if data_elem:
            text = data_elem.get_text()
            
            # 日付
            date_match = re.search(r"(\d{4})/(\d{2})/(\d{2})", text)
            if date_match:
                race_info["date"] = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
            
            # 会場
            venue_match = re.search(r"([^\s]+)競輪", text)
            if venue_match:
                race_info["venue_name"] = venue_match.group(1)
                race_info["bank_type"] = get_bank_type_numeric(race_info["venue_name"])
            
            # レース番号
            race_no_match = re.search(r"(\d+)R", text)
            if race_no_match:
                race_info["race_no"] = int(race_no_match.group(1))
            
            # 距離
            distance_match = re.search(r"(\d+)m", text)
            if distance_match:
                race_info["distance"] = int(distance_match.group(1))
        
        return race_info
    
    def scrape_entry_table(self, race_id: str) -> List[Dict]:
        """
        出走表を取得
        """
        url = f"{self.BASE_URL}/race/{race_id}/"
        
        html = self.fetch(url, use_selenium=True, wait_selector=".EntryTable")
        if not html:
            html = self.fetch(url, use_cloudscraper=True)
        
        if not html:
            return []
        
        soup = self.parse_html(html)
        entries = []
        
        # 出走表テーブルを探す
        table = soup.find("table", class_="EntryTable") or soup.find("table", class_="RaceTable")
        if not table:
            tables = soup.find_all("table")
            for t in tables:
                if t.find("th", string=re.compile("車番|選手")):
                    table = t
                    break
        
        if not table:
            return []
        
        rows = table.find_all("tr")
        
        for row in rows:
            if row.find("th"):  # ヘッダー行スキップ
                continue
            
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            
            entry = self._parse_entry_row_netkeiba(cells, race_id)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _parse_entry_row_netkeiba(self, cells: List[Tag], race_id: str) -> Optional[Dict]:
        """netkeiba形式の出走表行を解析"""
        try:
            entry = {"race_id": race_id}
            
            for cell in cells:
                text = cell.get_text(strip=True)
                class_names = cell.get("class", [])
                
                # 車番
                if "Waku" in str(class_names) or "CarNo" in str(class_names):
                    num_match = re.search(r"(\d+)", text)
                    if num_match:
                        entry["car_number"] = int(num_match.group(1))
                
                # 選手名
                link = cell.find("a")
                if link and "player" in link.get("href", ""):
                    entry["player_name"] = link.get_text(strip=True)
                    player_id_match = re.search(r"player/(\d+)", link.get("href", ""))
                    if player_id_match:
                        entry["player_id"] = player_id_match.group(1)
                
                # 競走得点
                score_match = re.search(r"(\d{2,3}\.\d{2})", text)
                if score_match:
                    score = float(score_match.group(1))
                    if 70 <= score <= 130:
                        entry["competition_score"] = score
                
                # 級班
                rank_match = re.search(r"(SS|S1|S2|A1|A2|A3|L1)", text)
                if rank_match:
                    entry["rank_class"] = rank_match.group(1)
                
                # 府県
                pref_match = re.search(r"(..府|..県|東京|大阪|京都)", text)
                if pref_match:
                    entry["prefecture"] = pref_match.group(1)
            
            if "car_number" in entry:
                return entry
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing netkeiba entry: {e}")
            return None
    
    def scrape_race_result(self, race_id: str) -> Dict:
        """
        レース結果を取得
        """
        url = f"{self.BASE_URL}/race/result/{race_id}/"
        
        html = self.fetch(url, use_selenium=True)
        if not html:
            html = self.fetch(url, use_cloudscraper=True)
        
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        result = {
            "race_id": race_id,
            "results": [],
            "payouts": {},
        }
        
        # 着順結果
        result_table = soup.find("table", class_="ResultTable")
        if result_table:
            for row in result_table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2:
                    rank_match = re.search(r"(\d+)", cells[0].get_text())
                    car_match = re.search(r"(\d+)", cells[1].get_text())
                    if rank_match and car_match:
                        result["results"].append({
                            "rank": int(rank_match.group(1)),
                            "car_number": int(car_match.group(1)),
                        })
        
        # 払戻金
        payout_div = soup.find("div", class_="Payout")
        if payout_div:
            text = payout_div.get_text()
            
            # 3連単
            match = re.search(r"3連単[^\d]*([\d,]+)円", text)
            if match:
                result["payouts"]["sanrentan"] = {
                    "payout": int(match.group(1).replace(",", ""))
                }
            
            # 3連複
            match = re.search(r"3連複[^\d]*([\d,]+)円", text)
            if match:
                result["payouts"]["sanrenpuku"] = {
                    "payout": int(match.group(1).replace(",", ""))
                }
        
        return result
    
    def scrape_player_stats(self, player_id: str) -> Dict:
        """
        選手の詳細成績を取得
        """
        url = f"{self.BASE_URL}/player/{player_id}/"
        
        html = self.fetch(url, use_cloudscraper=True)
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        player = {"player_id": player_id}
        
        # プロフィール
        profile = soup.find("div", class_="PlayerProfile")
        if profile:
            name_elem = profile.find("h1")
            if name_elem:
                player["name"] = name_elem.get_text(strip=True)
            
            text = profile.get_text()
            
            # 級班
            rank_match = re.search(r"(SS|S1|S2|A1|A2|A3|L1)", text)
            if rank_match:
                player["rank_class"] = rank_match.group(1)
            
            # 期別
            term_match = re.search(r"(\d+)期", text)
            if term_match:
                player["term"] = int(term_match.group(1))
        
        # 成績データ
        stats_table = soup.find("table", class_="StatsTable")
        if stats_table:
            for row in stats_table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    
                    if "勝率" in label:
                        player["win_rate"] = safe_float(value.replace("%", ""))
                    elif "連対率" in label:
                        player["second_rate"] = safe_float(value.replace("%", ""))
                    elif "3連対率" in label:
                        player["third_rate"] = safe_float(value.replace("%", ""))
                    elif "バック" in label:
                        player["back_count"] = safe_int(value)
        
        return player
