# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- Oddspark スクレイパー
==========================================
oddsparkからオッズ・レースデータを取得
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


class OddsparkScraper(BaseScraper):
    """
    Oddsparkからデータを取得するスクレイパー
    
    特徴:
    - オッズデータが充実
    - 予想情報あり
    """
    
    BASE_URL = "https://www.oddspark.com/keirin"
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        super().__init__(config)
    
    def scrape_race_list(self, date: str) -> List[Dict]:
        """
        指定日のレース一覧を取得
        """
        date_formatted = date.replace("-", "")
        url = f"{self.BASE_URL}/RaceList.do?kaisaiDate={date_formatted}"
        
        html = self.fetch(url, use_cloudscraper=True)
        if not html:
            return []
        
        soup = self.parse_html(html)
        races = []
        
        # 会場ごとのセクション
        venue_sections = soup.find_all("div", class_="section")
        
        for section in venue_sections:
            venue_header = section.find("h3") or section.find("div", class_="title")
            if not venue_header:
                continue
            
            venue_name = venue_header.get_text(strip=True)
            venue_name = re.sub(r"[\s\d]+R.*$", "", venue_name)
            
            # レースリンク
            links = section.find_all("a", href=re.compile(r"RaceOdds|RaceResult"))
            for link in links:
                href = link.get("href", "")
                
                # レースIDを抽出
                race_match = re.search(r"opTrackCd=(\d+).*?raceDy=(\d+).*?raceNo=(\d+)", href)
                if not race_match:
                    continue
                
                track_cd = race_match.group(1)
                race_date = race_match.group(2)
                race_no = int(race_match.group(3))
                
                race_id = f"{race_date}_{track_cd}_{race_no:02d}"
                
                races.append({
                    "race_id": race_id,
                    "date": date,
                    "venue_name": venue_name,
                    "race_no": race_no,
                    "url": urljoin(self.BASE_URL, href),
                    "bank_type": get_bank_type_numeric(venue_name),
                })
        
        logger.info(f"Found {len(races)} races on {date} from oddspark")
        return races
    
    def scrape_race_info(self, race_id: str) -> Dict:
        """レース情報を取得"""
        parts = race_id.split("_")
        if len(parts) != 3:
            return {}
        
        race_date, track_cd, race_no = parts
        
        url = f"{self.BASE_URL}/RaceList.do"
        params = {
            "opTrackCd": track_cd,
            "raceDy": race_date,
            "raceNo": race_no,
        }
        
        html = self.fetch(url, params, use_cloudscraper=True)
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        race_info = {
            "race_id": race_id,
            "date": f"{race_date[:4]}-{race_date[4:6]}-{race_date[6:8]}",
            "race_no": int(race_no),
        }
        
        # レース名
        title = soup.find("h2") or soup.find("div", class_="race_name")
        if title:
            race_info["race_name"] = title.get_text(strip=True)
        
        return race_info
    
    def scrape_entry_table(self, race_id: str) -> List[Dict]:
        """出走表を取得"""
        parts = race_id.split("_")
        if len(parts) != 3:
            return []
        
        race_date, track_cd, race_no = parts
        
        url = f"{self.BASE_URL}/RaceOdds.do"
        params = {
            "opTrackCd": track_cd,
            "raceDy": race_date,
            "raceNo": race_no,
        }
        
        html = self.fetch(url, params, use_cloudscraper=True)
        if not html:
            return []
        
        soup = self.parse_html(html)
        entries = []
        
        # 出走表
        table = soup.find("table", class_="tb")
        if not table:
            tables = soup.find_all("table")
            for t in tables:
                if t.find("th", string=re.compile("車番")):
                    table = t
                    break
        
        if not table:
            return []
        
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            
            entry = {"race_id": race_id}
            
            for cell in cells:
                text = cell.get_text(strip=True)
                
                # 車番
                if cell.get("class") and "waku" in str(cell.get("class")):
                    num_match = re.search(r"(\d+)", text)
                    if num_match:
                        entry["car_number"] = int(num_match.group(1))
                
                # 選手名
                link = cell.find("a")
                if link and "player" in str(link.get("href", "")).lower():
                    entry["player_name"] = link.get_text(strip=True)
                
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
            
            if "car_number" in entry:
                entries.append(entry)
        
        return entries
    
    def scrape_race_result(self, race_id: str) -> Dict:
        """レース結果を取得"""
        parts = race_id.split("_")
        if len(parts) != 3:
            return {}
        
        race_date, track_cd, race_no = parts
        
        url = f"{self.BASE_URL}/RaceResult.do"
        params = {
            "opTrackCd": track_cd,
            "raceDy": race_date,
            "raceNo": race_no,
        }
        
        html = self.fetch(url, params, use_cloudscraper=True)
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        result = {
            "race_id": race_id,
            "results": [],
            "payouts": {},
        }
        
        # 着順
        result_table = soup.find("table", class_="result")
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
        
        # 払戻
        payout_table = soup.find("table", class_="payout")
        if payout_table:
            text = payout_table.get_text()
            
            for bet_type, pattern in [
                ("sanrentan", r"3連単[^\d]*([\d,]+)円"),
                ("sanrenpuku", r"3連複[^\d]*([\d,]+)円"),
                ("nishatan", r"2車単[^\d]*([\d,]+)円"),
                ("nishafuku", r"2車複[^\d]*([\d,]+)円"),
            ]:
                match = re.search(pattern, text)
                if match:
                    result["payouts"][bet_type] = {
                        "payout": int(match.group(1).replace(",", ""))
                    }
        
        return result
    
    def scrape_odds(self, race_id: str) -> Dict:
        """
        オッズを取得
        """
        parts = race_id.split("_")
        if len(parts) != 3:
            return {}
        
        race_date, track_cd, race_no = parts
        
        url = f"{self.BASE_URL}/RaceOdds.do"
        params = {
            "opTrackCd": track_cd,
            "raceDy": race_date,
            "raceNo": race_no,
        }
        
        html = self.fetch(url, params, use_cloudscraper=True)
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        odds = {
            "race_id": race_id,
            "tansho": {},      # 単勝
            "fukusho": {},     # 複勝
            "nishafuku": {},   # 2車複
            "sanrenpuku": {},  # 3連複
        }
        
        # オッズテーブルを探索
        odds_tables = soup.find_all("table", class_="odds")
        
        for table in odds_tables:
            header = table.find("th")
            if not header:
                continue
            
            header_text = header.get_text(strip=True)
            
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                
                num_text = cells[0].get_text(strip=True)
                odds_text = cells[-1].get_text(strip=True)
                
                num_match = re.search(r"(\d+)", num_text)
                odds_match = re.search(r"([\d.]+)", odds_text)
                
                if num_match and odds_match:
                    num = num_match.group(1)
                    odds_val = float(odds_match.group(1))
                    
                    if "単勝" in header_text:
                        odds["tansho"][num] = odds_val
                    elif "複勝" in header_text:
                        odds["fukusho"][num] = odds_val
        
        return odds
