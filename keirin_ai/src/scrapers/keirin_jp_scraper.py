# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- keirin.jp スクレイパー
============================================
keirin.jp（競輪公式サイト）からデータを取得
"""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlencode

from bs4 import BeautifulSoup, Tag
from loguru import logger

from .base_scraper import BaseScraper
from ..config import ScrapingConfig, GradeType
from ..utils import safe_float, safe_int, get_bank_type_numeric


class KeirinJPScraper(BaseScraper):
    """
    keirin.jp からデータを取得するスクレイパー
    
    主な取得データ:
    - レース一覧
    - 出走表（選手情報、競走得点、バック回数等）
    - レース結果
    - オッズ情報
    """
    
    BASE_URL = "https://keirin.jp"
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        super().__init__(config)
        self.venue_codes = self._get_venue_codes()
    
    def _get_venue_codes(self) -> Dict[str, str]:
        """競輪場コードのマッピング"""
        return {
            "11": "函館", "12": "青森", "13": "いわき平",
            "21": "弥彦", "22": "前橋", "23": "取手", "24": "宇都宮", "25": "大宮",
            "26": "西武園", "27": "京王閣", "28": "立川",
            "31": "松戸", "32": "千葉", "33": "川崎", "34": "平塚",
            "35": "小田原", "36": "伊東温泉", "37": "静岡",
            "38": "名古屋", "39": "岐阜", "40": "大垣",
            "42": "豊橋", "43": "富山", "44": "松阪", "45": "四日市",
            "46": "福井",
            "51": "奈良", "52": "向日町", "53": "和歌山", "54": "岸和田",
            "61": "玉野", "62": "広島", "63": "防府",
            "71": "高松", "72": "小松島", "73": "高知", "74": "松山",
            "81": "小倉", "82": "久留米", "83": "武雄", "84": "佐世保",
            "85": "別府", "86": "熊本",
        }
    
    def _get_venue_name(self, code: str) -> str:
        """競輪場コードから名前を取得"""
        return self.venue_codes.get(code, "不明")
    
    def _get_venue_code(self, name: str) -> Optional[str]:
        """競輪場名からコードを取得"""
        for code, venue_name in self.venue_codes.items():
            if venue_name == name:
                return code
        return None
    
    def _build_race_url(self, date: str, venue_code: str, race_no: int) -> str:
        """レースページURLを構築"""
        # 日付をyyyymmdd形式に変換
        date_formatted = date.replace("-", "").replace("/", "")
        return f"{self.BASE_URL}/pc/dfw/dataplaza/guest/raceresult?KCD={venue_code}&KBI={date_formatted}&RNO={race_no:02d}"
    
    def _build_entry_url(self, date: str, venue_code: str, race_no: int) -> str:
        """出走表ページURLを構築"""
        date_formatted = date.replace("-", "").replace("/", "")
        return f"{self.BASE_URL}/pc/dfw/dataplaza/guest/racetable?KCD={venue_code}&KBI={date_formatted}&RNO={race_no:02d}"
    
    def scrape_race_calendar(self, year: int, month: int) -> List[Dict]:
        """
        指定年月の開催カレンダーを取得
        
        Returns:
            List[Dict]: 開催情報のリスト
        """
        url = f"{self.BASE_URL}/pc/dfw/dataplaza/guest/kaisaiinfo"
        params = {"YR": year, "MN": f"{month:02d}"}
        
        html = self.fetch(url, params, use_cloudscraper=True)
        if not html:
            return []
        
        soup = self.parse_html(html)
        races = []
        
        # カレンダーから開催情報を抽出
        calendar_table = soup.find("table", class_="tbl01")
        if calendar_table:
            for row in calendar_table.find_all("tr"):
                for cell in row.find_all("td"):
                    links = cell.find_all("a")
                    for link in links:
                        href = link.get("href", "")
                        # 開催情報を解析
                        match = re.search(r"KCD=(\d+)&KBI=(\d{8})", href)
                        if match:
                            venue_code = match.group(1)
                            date_str = match.group(2)
                            races.append({
                                "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                                "venue_code": venue_code,
                                "venue_name": self._get_venue_name(venue_code),
                            })
        
        return races
    
    def scrape_race_list(self, date: str) -> List[Dict]:
        """
        指定日のレース一覧を取得
        
        Args:
            date: 日付（YYYY-MM-DD形式）
            
        Returns:
            List[Dict]: レース情報のリスト
        """
        date_formatted = date.replace("-", "").replace("/", "")
        url = f"{self.BASE_URL}/pc/dfw/dataplaza/guest/raceindex"
        params = {"KBI": date_formatted}
        
        html = self.fetch(url, params, use_cloudscraper=True)
        if not html:
            logger.warning(f"Failed to fetch race list for {date}")
            return []
        
        soup = self.parse_html(html)
        races = []
        
        # 会場ごとのレース一覧を取得
        venue_sections = soup.find_all("div", class_="box02")
        for section in venue_sections:
            # 会場名を取得
            venue_header = section.find("h3") or section.find("div", class_="ttl01")
            if not venue_header:
                continue
            
            venue_text = venue_header.get_text(strip=True)
            venue_name = re.sub(r"[\s\d]+.*$", "", venue_text)
            venue_code = self._get_venue_code(venue_name)
            
            # グレード情報を取得
            grade = self._extract_grade(section)
            
            # レース一覧を取得
            race_links = section.find_all("a", href=re.compile(r"RNO=\d+"))
            for link in race_links:
                href = link.get("href", "")
                match = re.search(r"RNO=(\d+)", href)
                if match:
                    race_no = int(match.group(1))
                    race_id = f"{date_formatted}_{venue_code}_{race_no:02d}"
                    
                    races.append({
                        "race_id": race_id,
                        "date": date,
                        "venue_code": venue_code,
                        "venue_name": venue_name,
                        "race_no": race_no,
                        "grade": grade,
                        "url": urljoin(self.BASE_URL, href),
                    })
        
        logger.info(f"Found {len(races)} races on {date}")
        return races
    
    def _extract_grade(self, element: Tag) -> str:
        """要素からグレード情報を抽出"""
        text = element.get_text()
        if "GP" in text:
            return "GP"
        elif "GI" in text or "G1" in text:
            return "G1"
        elif "GII" in text or "G2" in text:
            return "G2"
        elif "GIII" in text or "G3" in text:
            return "G3"
        elif "FI" in text or "F1" in text:
            return "F1"
        elif "FII" in text or "F2" in text:
            return "F2"
        return "F1"
    
    def scrape_race_info(self, race_id: str) -> Dict:
        """
        レース基本情報を取得
        
        Args:
            race_id: レースID（日付_会場コード_レース番号）
            
        Returns:
            Dict: レース情報
        """
        parts = race_id.split("_")
        if len(parts) != 3:
            logger.error(f"Invalid race_id format: {race_id}")
            return {}
        
        date_str, venue_code, race_no_str = parts
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        race_no = int(race_no_str)
        
        url = self._build_entry_url(date, venue_code, race_no)
        html = self.fetch(url, use_cloudscraper=True)
        
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        race_info = {
            "race_id": race_id,
            "date": date,
            "venue_code": venue_code,
            "venue_name": self._get_venue_name(venue_code),
            "race_no": race_no,
            "bank_type": get_bank_type_numeric(self._get_venue_name(venue_code)),
        }
        
        # レース名・グレード
        race_title = soup.find("h2", class_="ttl01") or soup.find("div", class_="race_ttl")
        if race_title:
            title_text = race_title.get_text(strip=True)
            race_info["race_name"] = title_text
            race_info["grade"] = self._extract_grade(race_title)
        
        # 距離・発走時刻
        race_detail = soup.find("div", class_="race_info") or soup.find("p", class_="race_detail")
        if race_detail:
            detail_text = race_detail.get_text(strip=True)
            
            # 距離を抽出
            distance_match = re.search(r"(\d+)m", detail_text)
            if distance_match:
                race_info["distance"] = int(distance_match.group(1))
            
            # 発走時刻を抽出
            time_match = re.search(r"(\d{1,2}):(\d{2})", detail_text)
            if time_match:
                race_info["start_time"] = f"{time_match.group(1)}:{time_match.group(2)}"
        
        # ガールズ/チャレンジ判定
        page_text = soup.get_text()
        race_info["is_girls"] = "ガールズ" in page_text
        race_info["is_challenge"] = "チャレンジ" in page_text
        
        return race_info
    
    def scrape_entry_table(self, race_id: str) -> List[Dict]:
        """
        出走表を取得
        
        Args:
            race_id: レースID
            
        Returns:
            List[Dict]: 選手情報のリスト
        """
        parts = race_id.split("_")
        if len(parts) != 3:
            return []
        
        date_str, venue_code, race_no_str = parts
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        race_no = int(race_no_str)
        
        url = self._build_entry_url(date, venue_code, race_no)
        html = self.fetch(url, use_cloudscraper=True)
        
        if not html:
            return []
        
        soup = self.parse_html(html)
        entries = []
        
        # 出走表テーブルを探す
        table = soup.find("table", class_="race_table") or soup.find("table", id="entry_table")
        if not table:
            # 別のパターンを試す
            tables = soup.find_all("table")
            for t in tables:
                headers = t.find_all("th")
                header_text = " ".join([h.get_text() for h in headers])
                if "車番" in header_text or "選手名" in header_text:
                    table = t
                    break
        
        if not table:
            logger.warning(f"Entry table not found for {race_id}")
            return []
        
        rows = table.find_all("tr")[1:]  # ヘッダー行をスキップ
        
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) < 5:
                continue
            
            entry = self._parse_entry_row(cells, race_id)
            if entry:
                entries.append(entry)
        
        # ライン構成を解析して追加
        entries = self._add_line_info(soup, entries)
        
        logger.info(f"Found {len(entries)} entries for {race_id}")
        return entries
    
    def _parse_entry_row(self, cells: List[Tag], race_id: str) -> Optional[Dict]:
        """出走表の1行を解析"""
        try:
            entry = {
                "race_id": race_id,
            }
            
            # 車番
            car_num_cell = cells[0]
            car_num_text = car_num_cell.get_text(strip=True)
            car_num_match = re.search(r"(\d+)", car_num_text)
            if car_num_match:
                entry["car_number"] = int(car_num_match.group(1))
            else:
                return None
            
            # 枠番（車番と同じか、セルの背景色から判定）
            entry["waku"] = entry["car_number"]  # 7車立ての場合は同じ
            
            # 選手名
            for i, cell in enumerate(cells[1:4]):
                text = cell.get_text(strip=True)
                if text and not text.isdigit():
                    # リンクから選手IDを取得
                    link = cell.find("a")
                    if link:
                        href = link.get("href", "")
                        player_id_match = re.search(r"SCD=(\d+)", href)
                        if player_id_match:
                            entry["player_id"] = player_id_match.group(1)
                    entry["player_name"] = text
                    break
            
            # 府県・年齢・級班
            for cell in cells:
                text = cell.get_text(strip=True)
                
                # 級班
                rank_match = re.search(r"(SS|S1|S2|A1|A2|A3|L1)", text)
                if rank_match:
                    entry["rank_class"] = rank_match.group(1)
                
                # 年齢
                age_match = re.search(r"(\d{2})歳", text)
                if age_match:
                    entry["age"] = int(age_match.group(1))
            
            # 競走得点
            for cell in cells:
                text = cell.get_text(strip=True)
                score_match = re.search(r"(\d{2,3}\.\d{2})", text)
                if score_match:
                    score = float(score_match.group(1))
                    if 70 <= score <= 130:  # 妥当な範囲
                        entry["competition_score"] = score
                        break
            
            # バック回数、勝率等を抽出
            stats_text = " ".join([c.get_text() for c in cells])
            
            # バック回数
            back_match = re.search(r"B\s*(\d+)", stats_text)
            if back_match:
                entry["back_count"] = int(back_match.group(1))
            
            # 勝率
            win_match = re.search(r"勝率?\s*(\d+\.?\d*)", stats_text)
            if win_match:
                entry["win_rate"] = float(win_match.group(1))
            
            # 2連対率
            second_match = re.search(r"2連対率?\s*(\d+\.?\d*)", stats_text)
            if second_match:
                entry["second_rate"] = float(second_match.group(1))
            
            # 3連対率
            third_match = re.search(r"3連対率?\s*(\d+\.?\d*)", stats_text)
            if third_match:
                entry["third_rate"] = float(third_match.group(1))
            
            # ギア倍率
            gear_match = re.search(r"(\d\.\d{2})", stats_text)
            if gear_match:
                gear = float(gear_match.group(1))
                if 3.5 <= gear <= 4.5:  # 妥当な範囲
                    entry["gear_ratio"] = gear
            
            return entry
            
        except Exception as e:
            logger.debug(f"Error parsing entry row: {e}")
            return None
    
    def _add_line_info(self, soup: BeautifulSoup, entries: List[Dict]) -> List[Dict]:
        """ライン構成情報を追加"""
        # ライン情報を探す
        line_div = soup.find("div", class_="line_info") or soup.find("div", id="line_formation")
        
        if not line_div:
            # テキストからライン情報を抽出
            page_text = soup.get_text()
            line_match = re.search(r"並び予想[：:]?\s*([0-9/\-\s]+)", page_text)
            if line_match:
                line_str = line_match.group(1).strip()
            else:
                # デフォルト: 単騎扱い
                for entry in entries:
                    entry["line_position"] = "単騎"
                    entry["line_formation"] = "細切れ"
                return entries
        else:
            line_str = line_div.get_text(strip=True)
        
        # ライン構成を解析
        lines = self._parse_line_string(line_str)
        num_lines = len(lines)
        
        # 分戦タイプを判定
        if num_lines == 2:
            formation = "2分戦"
        elif num_lines == 3:
            formation = "3分戦"
        elif num_lines == 4:
            formation = "4分戦"
        else:
            formation = "細切れ"
        
        # 各選手にライン情報を追加
        for entry in entries:
            car_num = entry.get("car_number")
            entry["line_formation"] = formation
            entry["line_position"] = "単騎"
            
            for line in lines:
                if car_num in line:
                    idx = line.index(car_num)
                    if idx == 0:
                        entry["line_position"] = "先頭"
                    elif idx == 1:
                        entry["line_position"] = "番手"
                    else:
                        entry["line_position"] = "三番手以降"
                    entry["line_cars"] = line
                    break
        
        return entries
    
    def _parse_line_string(self, line_str: str) -> List[List[int]]:
        """ライン文字列を解析"""
        lines = []
        
        # 区切り文字で分割
        for sep in ["/", "-", "｜", "|", "・"]:
            if sep in line_str:
                parts = line_str.split(sep)
                for part in parts:
                    cars = [int(c) for c in re.findall(r"\d", part)]
                    if cars:
                        lines.append(cars)
                return lines
        
        # 区切りなしの場合
        cars = [int(c) for c in re.findall(r"\d", line_str)]
        if cars:
            lines.append(cars)
        
        return lines
    
    def scrape_race_result(self, race_id: str) -> Dict:
        """
        レース結果を取得
        
        Args:
            race_id: レースID
            
        Returns:
            Dict: レース結果
        """
        parts = race_id.split("_")
        if len(parts) != 3:
            return {}
        
        date_str, venue_code, race_no_str = parts
        date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        race_no = int(race_no_str)
        
        url = self._build_race_url(date, venue_code, race_no)
        html = self.fetch(url, use_cloudscraper=True)
        
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        result = {
            "race_id": race_id,
            "date": date,
            "venue_code": venue_code,
            "race_no": race_no,
            "results": [],
            "payouts": {},
        }
        
        # 着順結果を取得
        result_table = soup.find("table", class_="result_table") or soup.find("table", id="race_result")
        if result_table:
            rows = result_table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 3:
                    rank_text = cells[0].get_text(strip=True)
                    car_text = cells[1].get_text(strip=True)
                    
                    rank_match = re.search(r"(\d+)", rank_text)
                    car_match = re.search(r"(\d+)", car_text)
                    
                    if rank_match and car_match:
                        result["results"].append({
                            "rank": int(rank_match.group(1)),
                            "car_number": int(car_match.group(1)),
                        })
        
        # 払戻金を取得
        payout_table = soup.find("table", class_="payout_table") or soup.find("div", class_="payout")
        if payout_table:
            payout_text = payout_table.get_text()
            
            # 3連単
            sanrentan_match = re.search(r"3連単[^\d]*(\d+)[^\d]*(\d+)[^\d]*(\d+)[^\d]*([\d,]+)円", payout_text)
            if sanrentan_match:
                result["payouts"]["sanrentan"] = {
                    "numbers": [int(sanrentan_match.group(i)) for i in range(1, 4)],
                    "payout": int(sanrentan_match.group(4).replace(",", "")),
                }
            
            # 3連複
            sanrenpuku_match = re.search(r"3連複[^\d]*([\d,]+)円", payout_text)
            if sanrenpuku_match:
                result["payouts"]["sanrenpuku"] = {
                    "payout": int(sanrenpuku_match.group(1).replace(",", "")),
                }
            
            # 2車単
            nishatan_match = re.search(r"2車単[^\d]*([\d,]+)円", payout_text)
            if nishatan_match:
                result["payouts"]["nishatan"] = {
                    "payout": int(nishatan_match.group(1).replace(",", "")),
                }
            
            # 2車複
            nishafuku_match = re.search(r"2車複[^\d]*([\d,]+)円", payout_text)
            if nishafuku_match:
                result["payouts"]["nishafuku"] = {
                    "payout": int(nishafuku_match.group(1).replace(",", "")),
                }
        
        # 落車・失格フラグ
        page_text = soup.get_text()
        result["has_crash"] = "落車" in page_text or "落" in page_text
        result["has_disqualification"] = "失格" in page_text or "失" in page_text
        
        return result
    
    def scrape_player_info(self, player_id: str) -> Dict:
        """
        選手情報を取得
        
        Args:
            player_id: 選手ID
            
        Returns:
            Dict: 選手情報
        """
        url = f"{self.BASE_URL}/pc/dfw/dataplaza/guest/playerinfo"
        params = {"SCD": player_id}
        
        html = self.fetch(url, params, use_cloudscraper=True)
        if not html:
            return {}
        
        soup = self.parse_html(html)
        
        player = {
            "player_id": player_id,
        }
        
        # 基本情報を抽出
        info_table = soup.find("table", class_="player_info")
        if info_table:
            for row in info_table.find_all("tr"):
                cells = row.find_all(["th", "td"])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    
                    if "選手名" in label or "氏名" in label:
                        player["name"] = value
                    elif "登録地" in label or "府県" in label:
                        player["prefecture"] = value
                    elif "生年月日" in label:
                        player["birthday"] = value
                    elif "級班" in label:
                        player["rank_class"] = value
                    elif "期別" in label:
                        player["term"] = value
        
        return player
    
    def scrape_historical_data(
        self,
        start_date: str,
        end_date: str,
        venue_codes: Optional[List[str]] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        過去データを一括取得
        
        Args:
            start_date: 開始日（YYYY-MM-DD）
            end_date: 終了日（YYYY-MM-DD）
            venue_codes: 対象会場コード（Noneの場合は全会場）
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (レース情報リスト, 結果情報リスト)
        """
        from ..utils import get_date_range
        
        dates = get_date_range(start_date, end_date)
        all_races = []
        all_results = []
        
        for date in dates:
            logger.info(f"Fetching races for {date}")
            
            races = self.scrape_race_list(date)
            
            for race in races:
                if venue_codes and race["venue_code"] not in venue_codes:
                    continue
                
                race_id = race["race_id"]
                
                # 出走表を取得
                entries = self.scrape_entry_table(race_id)
                race["entries"] = entries
                
                # 結果を取得（過去データの場合）
                result = self.scrape_race_result(race_id)
                if result.get("results"):
                    all_results.append(result)
                
                all_races.append(race)
        
        logger.info(f"Total: {len(all_races)} races, {len(all_results)} results")
        return all_races, all_results
