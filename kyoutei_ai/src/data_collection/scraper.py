"""
データ収集モジュール - 高度なWebスクレイパー
スクレイピング制限対策を実装した堅牢なスクレイパー
"""
import os
import time
import random
import asyncio
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, RAW_DATA_DIR, RACECOURSE_CODES


@dataclass
class RaceInfo:
    """レース情報"""
    race_date: str
    place_code: str
    race_number: int
    race_name: str = ""


class AntiBlockingStrategy:
    """スクレイピング制限回避戦略"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session_cookies: Dict[str, str] = {}
        self.request_count = 0
        self.last_request_time = 0
        self.blocked_count = 0
        
        # プロキシプール
        self.proxies: List[str] = []
        self.current_proxy_index = 0
        
        # User-Agentプール
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        ]
        
    def get_random_user_agent(self) -> str:
        """ランダムなUser-Agentを取得"""
        return random.choice(self.user_agents)
    
    def get_headers(self, referer: str = None) -> Dict[str, str]:
        """ランダム化されたヘッダーを生成"""
        headers = {
            "User-Agent": self.get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "ja,en;q=0.9,en-US;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }
        if referer:
            headers["Referer"] = referer
        return headers
    
    def get_random_delay(self, base_delay: float = 2.0) -> float:
        """ランダムな遅延時間を生成（人間らしい動作をシミュレート）"""
        # 正規分布に近い遅延
        jitter = random.uniform(0.5, 2.0)
        return base_delay * jitter
    
    async def smart_delay(self):
        """スマートな遅延処理"""
        # リクエスト数に応じて遅延を調整
        if self.request_count % 50 == 0 and self.request_count > 0:
            # 50リクエストごとに長めの休憩
            delay = random.uniform(10, 30)
            logger.info(f"Taking a longer break: {delay:.1f}s (after {self.request_count} requests)")
        elif self.request_count % 10 == 0 and self.request_count > 0:
            # 10リクエストごとに短い休憩
            delay = random.uniform(3, 8)
        else:
            delay = self.get_random_delay(config.scraping.request_interval)
        
        await asyncio.sleep(delay)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_proxy(self) -> Optional[str]:
        """プロキシを取得（ローテーション）"""
        if not self.proxies:
            return None
        proxy = self.proxies[self.current_proxy_index % len(self.proxies)]
        self.current_proxy_index += 1
        return proxy


class BoatRaceScraper:
    """ボートレース公式サイトスクレイパー"""
    
    BASE_URL = "https://www.boatrace.jp"
    
    def __init__(self):
        self.config = config.scraping
        self.anti_blocking = AntiBlockingStrategy()
        self.cache_dir = RAW_DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # データ保存先
        self.odds_dir = RAW_DATA_DIR / "odds"
        self.before_info_dir = RAW_DATA_DIR / "before_info"
        self.odds_dir.mkdir(parents=True, exist_ok=True)
        self.before_info_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, url: str) -> Path:
        """キャッシュファイルパスを生成"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"
    
    async def _load_from_cache(self, url: str) -> Optional[str]:
        """キャッシュから読み込み"""
        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            async with aiofiles.open(cache_path, 'r', encoding='utf-8') as f:
                return await f.read()
        return None
    
    async def _save_to_cache(self, url: str, content: str):
        """キャッシュに保存"""
        cache_path = self._get_cache_path(url)
        async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
            await f.write(content)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _fetch_page(
        self,
        session: aiohttp.ClientSession,
        url: str,
        referer: str = None,
        use_cache: bool = True
    ) -> Optional[str]:
        """ページを取得（キャッシュ対応）"""
        
        # キャッシュチェック
        if use_cache:
            cached = await self._load_from_cache(url)
            if cached:
                logger.debug(f"Cache hit: {url}")
                return cached
        
        # スマート遅延
        await self.anti_blocking.smart_delay()
        
        try:
            headers = self.anti_blocking.get_headers(referer or self.BASE_URL)
            proxy = self.anti_blocking.get_proxy()
            
            async with session.get(
                url,
                headers=headers,
                proxy=proxy,
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
                ssl=False  # SSL検証をスキップ（必要に応じて）
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # ブロック検出
                    if "アクセスが集中しています" in content or "しばらく時間をおいて" in content:
                        logger.warning(f"Possible blocking detected: {url}")
                        self.anti_blocking.blocked_count += 1
                        # 長めの待機
                        await asyncio.sleep(random.uniform(30, 60))
                        raise aiohttp.ClientError("Blocked")
                    
                    # キャッシュ保存
                    if use_cache:
                        await self._save_to_cache(url, content)
                    
                    logger.debug(f"Fetched: {url}")
                    return content
                    
                elif response.status == 403:
                    logger.warning(f"Forbidden (403): {url}")
                    self.anti_blocking.blocked_count += 1
                    await asyncio.sleep(random.uniform(60, 120))
                    raise aiohttp.ClientError("Forbidden")
                    
                elif response.status == 429:
                    logger.warning(f"Too many requests (429): {url}")
                    self.anti_blocking.blocked_count += 1
                    await asyncio.sleep(random.uniform(120, 300))
                    raise aiohttp.ClientError("Rate limited")
                    
                else:
                    logger.warning(f"Unexpected status {response.status}: {url}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout: {url}")
            raise
        except Exception as e:
            logger.error(f"Fetch error: {url}, {e}")
            raise
    
    async def get_race_list(self, date: datetime) -> List[RaceInfo]:
        """指定日のレース一覧を取得"""
        date_str = date.strftime("%Y%m%d")
        url = f"{self.BASE_URL}/owpc/pc/race/index?hd={date_str}"
        
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url)
            if not html:
                return []
            
            soup = BeautifulSoup(html, 'lxml')
            races = []
            
            # 各レース場のレースを取得
            for place_code in RACECOURSE_CODES.keys():
                for race_num in range(1, 13):  # 最大12レース
                    races.append(RaceInfo(
                        race_date=date_str,
                        place_code=place_code,
                        race_number=race_num
                    ))
            
            return races
    
    async def get_odds_3tan(self, race: RaceInfo) -> Dict[str, float]:
        """3連単オッズを取得"""
        url = (
            f"{self.BASE_URL}/owpc/pc/race/odds3t"
            f"?rno={race.race_number}&jcd={race.place_code}&hd={race.race_date}"
        )
        
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url, use_cache=True)
            if not html:
                return {}
            
            soup = BeautifulSoup(html, 'lxml')
            odds_dict = {}
            
            # オッズテーブルを解析
            odds_tables = soup.find_all('table', class_='is-w495')
            
            for table in odds_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    for i, cell in enumerate(cells):
                        # オッズ値を抽出
                        odds_text = cell.get_text(strip=True)
                        if odds_text and odds_text != '-':
                            try:
                                odds_value = float(odds_text.replace(',', ''))
                                # 組み合わせを特定（セルのdata属性等から）
                                combo_attr = cell.get('data-id', '')
                                if combo_attr:
                                    odds_dict[combo_attr] = odds_value
                            except ValueError:
                                pass
            
            return odds_dict
    
    async def get_odds_2tan(self, race: RaceInfo) -> Dict[str, float]:
        """2連単オッズを取得"""
        url = (
            f"{self.BASE_URL}/owpc/pc/race/odds2tf"
            f"?rno={race.race_number}&jcd={race.place_code}&hd={race.race_date}"
        )
        
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url, use_cache=True)
            if not html:
                return {}
            
            soup = BeautifulSoup(html, 'lxml')
            odds_dict = {}
            
            # 2連単オッズテーブルを解析
            table = soup.find('table', class_='is-w495')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    for cell in cells:
                        odds_text = cell.get_text(strip=True)
                        if odds_text and odds_text != '-':
                            try:
                                odds_value = float(odds_text.replace(',', ''))
                                combo_attr = cell.get('data-id', '')
                                if combo_attr:
                                    odds_dict[combo_attr] = odds_value
                            except ValueError:
                                pass
            
            return odds_dict
    
    async def get_before_info(self, race: RaceInfo) -> Dict[str, Any]:
        """直前情報を取得（展示タイム、進入コース、気象など）"""
        url = (
            f"{self.BASE_URL}/owpc/pc/race/beforeinfo"
            f"?rno={race.race_number}&jcd={race.place_code}&hd={race.race_date}"
        )
        
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url, use_cache=True)
            if not html:
                return {}
            
            soup = BeautifulSoup(html, 'lxml')
            info = {
                'exhibition_times': {},  # 展示タイム
                'start_timing': {},      # スタートタイミング
                'entry_course': {},      # 進入コース
                'weather': {},           # 気象情報
                'parts_exchange': {},    # 部品交換
            }
            
            # 展示タイム
            tenji_table = soup.find('table', class_='is-w243')
            if tenji_table:
                rows = tenji_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        waku = cells[0].get_text(strip=True)
                        time_text = cells[1].get_text(strip=True)
                        try:
                            info['exhibition_times'][waku] = float(time_text)
                        except ValueError:
                            pass
            
            # 気象情報
            weather_area = soup.find('div', class_='weather1')
            if weather_area:
                # 天候
                weather_elem = weather_area.find('span', class_='weather1_bodyUnitLabelTitle')
                if weather_elem:
                    info['weather']['condition'] = weather_elem.get_text(strip=True)
                
                # 風速・風向
                wind_elem = weather_area.find('p', class_='weather1_bodyUnitLabelData')
                if wind_elem:
                    wind_text = wind_elem.get_text(strip=True)
                    info['weather']['wind'] = wind_text
                
                # 波高
                wave_elem = weather_area.find_all('p', class_='weather1_bodyUnitLabelData')
                if len(wave_elem) >= 2:
                    info['weather']['wave'] = wave_elem[1].get_text(strip=True)
            
            # スタート情報・進入コース
            start_table = soup.find('table', class_='is-p10-0')
            if start_table:
                rows = start_table.find_all('tr')
                for i, row in enumerate(rows):
                    cells = row.find_all('td')
                    if cells:
                        # 進入コース
                        course_elem = cells[0] if cells else None
                        if course_elem:
                            info['entry_course'][str(i+1)] = course_elem.get_text(strip=True)
            
            return info
    
    async def get_race_result(self, race: RaceInfo) -> Dict[str, Any]:
        """レース結果を取得"""
        url = (
            f"{self.BASE_URL}/owpc/pc/race/raceresult"
            f"?rno={race.race_number}&jcd={race.place_code}&hd={race.race_date}"
        )
        
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url, use_cache=True)
            if not html:
                return {}
            
            soup = BeautifulSoup(html, 'lxml')
            result = {
                'finish_order': [],  # 着順
                'start_timing': {},  # ST
                'payoff': {},        # 払戻金
            }
            
            # 着順テーブル
            result_table = soup.find('table', class_='is-w495')
            if result_table:
                rows = result_table.find_all('tr')
                for row in rows[1:]:  # ヘッダーをスキップ
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        order = cells[0].get_text(strip=True)
                        waku = cells[1].get_text(strip=True)
                        result['finish_order'].append({
                            'order': order,
                            'waku': waku
                        })
            
            # 払戻金
            payout_table = soup.find('table', class_='is-w243')
            if payout_table:
                rows = payout_table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        ticket_type = cells[0].get_text(strip=True)
                        combo = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                        payout = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                        result['payoff'][ticket_type] = {
                            'combination': combo,
                            'payout': payout
                        }
            
            return result
    
    async def get_racer_info(self, racer_id: str) -> Dict[str, Any]:
        """選手情報を取得"""
        url = f"{self.BASE_URL}/owpc/pc/data/racersearch/profile?toession={racer_id}"
        
        async with aiohttp.ClientSession() as session:
            html = await self._fetch_page(session, url, use_cache=True)
            if not html:
                return {}
            
            soup = BeautifulSoup(html, 'lxml')
            info = {}
            
            # 基本情報
            profile_table = soup.find('table', class_='is-w480')
            if profile_table:
                rows = profile_table.find_all('tr')
                for row in rows:
                    header = row.find('th')
                    data = row.find('td')
                    if header and data:
                        key = header.get_text(strip=True)
                        value = data.get_text(strip=True)
                        info[key] = value
            
            return info
    
    async def scrape_all_races_for_date(self, date: datetime) -> Dict[str, Any]:
        """指定日の全レースデータを取得"""
        date_str = date.strftime("%Y%m%d")
        all_data = {}
        
        logger.info(f"Scraping all races for {date_str}")
        
        for place_code, place_name in RACECOURSE_CODES.items():
            place_data = {}
            
            for race_num in range(1, 13):
                race = RaceInfo(
                    race_date=date_str,
                    place_code=place_code,
                    race_number=race_num
                )
                
                race_key = f"{place_code}_{race_num:02d}"
                
                try:
                    # 各種データを取得
                    odds_3tan = await self.get_odds_3tan(race)
                    odds_2tan = await self.get_odds_2tan(race)
                    before_info = await self.get_before_info(race)
                    result = await self.get_race_result(race)
                    
                    place_data[race_key] = {
                        'odds_3tan': odds_3tan,
                        'odds_2tan': odds_2tan,
                        'before_info': before_info,
                        'result': result,
                    }
                    
                    logger.debug(f"Scraped: {place_name} {race_num}R")
                    
                except Exception as e:
                    logger.error(f"Error scraping {place_name} {race_num}R: {e}")
                    continue
            
            all_data[place_code] = place_data
        
        return all_data
    
    async def scrape_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        save_interval: int = 1
    ):
        """日付範囲のデータをスクレイピング"""
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            save_path = self.odds_dir / f"{date_str}.pkl"
            
            if save_path.exists():
                logger.info(f"Already scraped: {date_str}")
                current_date += timedelta(days=1)
                continue
            
            try:
                data = await self.scrape_all_races_for_date(current_date)
                
                # 保存
                with open(save_path, 'wb') as f:
                    pickle.dump(data, f)
                
                logger.info(f"Saved: {date_str}")
                
            except Exception as e:
                logger.error(f"Error on {date_str}: {e}")
            
            current_date += timedelta(days=1)
            
            # 日付ごとに長めの休憩
            await asyncio.sleep(random.uniform(5, 15))


class PlaywrightScraper:
    """Playwright を使用した高度なスクレイパー（JavaScriptレンダリング対応）"""
    
    def __init__(self):
        self.config = config.scraping
        
    async def init_browser(self):
        """ブラウザを初期化"""
        try:
            from playwright.async_api import async_playwright
            
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-accelerated-2d-canvas',
                    '--disable-gpu',
                ]
            )
            self.context = await self.browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                locale='ja-JP',
                timezone_id='Asia/Tokyo',
            )
            
            # 追加のスクリプトでWebDriver検出を回避
            await self.context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['ja-JP', 'ja', 'en-US', 'en']
                });
                
                window.chrome = {
                    runtime: {}
                };
            """)
            
            logger.info("Playwright browser initialized")
            
        except ImportError:
            logger.error("Playwright not installed. Run: playwright install")
            raise
    
    async def close(self):
        """ブラウザを閉じる"""
        if hasattr(self, 'browser'):
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
    
    async def fetch_with_js(self, url: str, wait_selector: str = None) -> str:
        """JavaScriptをレンダリングしてページを取得"""
        page = await self.context.new_page()
        
        try:
            # ランダムな遅延
            await asyncio.sleep(random.uniform(1, 3))
            
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=10000)
            
            # 追加の待機（動的コンテンツ用）
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            content = await page.content()
            return content
            
        except Exception as e:
            logger.error(f"Playwright fetch error: {url}, {e}")
            return ""
        finally:
            await page.close()


async def main():
    """メイン実行"""
    scraper = BoatRaceScraper()
    
    # テスト: 今日のレースをスクレイピング
    today = datetime.now()
    data = await scraper.scrape_all_races_for_date(today)
    
    print(f"Scraped {len(data)} places")


if __name__ == "__main__":
    asyncio.run(main())
