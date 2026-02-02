"""
JRAリアルタイムオッズ取得モジュール
フェーズ4-20: リアルタイムオッズ取得機能（JRA公式サイト）

- Playwrightを使用したJRA公式サイトからのオッズ取得
- 全券種（単勝、複勝、枠連、馬連、ワイド、馬単、三連複、三連単）対応
"""

import re
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any

import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.src.utils import load_config, get_project_root, ensure_dir, setup_logging

logger = setup_logging()


class JRAOddsScraper:
    """JRA公式サイトからオッズを取得するクラス"""
    
    BASE_URL = "https://www.jra.go.jp"
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.realtime_config = self.config.get('realtime', {})
        self.project_root = get_project_root()
        self.paths = self.config.get('paths', {})
        self.html_dir = self.project_root / self.paths.get('html_jra_odds', 'data/html/jra_odds')
        ensure_dir(self.html_dir)
        
        self.browser = None
        self.page = None
    
    async def _init_browser(self):
        """Playwrightブラウザを初期化"""
        if self.browser is not None:
            return
        
        from playwright.async_api import async_playwright
        
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--disable-gpu',
            ]
        )
        
        self.context = await self.browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport={'width': 1920, 'height': 1080}
        )
        
        self.page = await self.context.new_page()
        
        # ナビゲーションのタイムアウト設定
        self.page.set_default_timeout(30000)
    
    async def _close_browser(self):
        """ブラウザを閉じる"""
        if self.browser:
            await self.browser.close()
            await self.playwright.stop()
            self.browser = None
            self.page = None
    
    async def get_race_list(self, date: str) -> List[Dict]:
        """指定日のレース一覧を取得
        
        Args:
            date: 日付（YYYYMMDD形式）
        
        Returns:
            レース情報リスト
        """
        await self._init_browser()
        
        # JRAのレース一覧ページ
        url = f"{self.BASE_URL}/keiba/calendar.html"
        
        try:
            await self.page.goto(url)
            await self.page.wait_for_load_state('networkidle')
            
            # レース情報を抽出
            races = []
            
            # ページ内容を取得してパース
            content = await self.page.content()
            
            # 日付リンクを探す
            race_links = await self.page.query_selector_all(f'a[href*="kaisai"]')
            
            for link in race_links:
                href = await link.get_attribute('href')
                text = await link.text_content()
                
                if date in str(href):
                    races.append({
                        'url': href,
                        'text': text
                    })
            
            return races
            
        except Exception as e:
            logger.error(f"レース一覧取得エラー: {e}")
            return []
    
    async def get_win_odds(self, race_url: str) -> pd.DataFrame:
        """単勝オッズを取得
        
        Args:
            race_url: レースページURL
        
        Returns:
            単勝オッズDataFrame
        """
        await self._init_browser()
        
        try:
            # オッズページに移動
            odds_url = race_url.replace('/race/', '/odds/') if '/race/' in race_url else race_url
            
            await self.page.goto(odds_url)
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)  # JS実行待ち
            
            # オッズテーブルを探す
            content = await self.page.content()
            
            # BeautifulSoupでパース
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # 単勝オッズテーブルを探す
            odds_data = []
            
            # JRAのオッズテーブル構造に合わせてパース
            odds_table = soup.find('table', class_='odds_table')
            if not odds_table:
                odds_table = soup.find('table', id='odds_tan')
            
            if odds_table:
                rows = odds_table.find_all('tr')
                for row in rows[1:]:  # ヘッダーをスキップ
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        horse_num = cells[0].get_text(strip=True)
                        odds = cells[-1].get_text(strip=True)
                        
                        try:
                            odds_data.append({
                                'horse_number': int(horse_num),
                                'win_odds': float(odds.replace(',', ''))
                            })
                        except:
                            continue
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            logger.error(f"単勝オッズ取得エラー: {e}")
            return pd.DataFrame()
    
    async def get_place_odds(self, race_url: str) -> pd.DataFrame:
        """複勝オッズを取得
        
        Args:
            race_url: レースページURL
        
        Returns:
            複勝オッズDataFrame（min-max）
        """
        await self._init_browser()
        
        try:
            odds_url = race_url.replace('/race/', '/odds/') if '/race/' in race_url else race_url
            
            await self.page.goto(odds_url + '?type=fuku')
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            content = await self.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            odds_data = []
            
            odds_table = soup.find('table', id='odds_fuku')
            if not odds_table:
                odds_table = soup.find('table', class_='odds_table')
            
            if odds_table:
                rows = odds_table.find_all('tr')
                for row in rows[1:]:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        horse_num = cells[0].get_text(strip=True)
                        odds_min = cells[-2].get_text(strip=True) if len(cells) >= 3 else cells[-1].get_text(strip=True)
                        odds_max = cells[-1].get_text(strip=True)
                        
                        try:
                            odds_data.append({
                                'horse_number': int(horse_num),
                                'place_odds_min': float(odds_min.replace(',', '').replace('-', '0')),
                                'place_odds_max': float(odds_max.replace(',', '').replace('-', '0'))
                            })
                        except:
                            continue
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            logger.error(f"複勝オッズ取得エラー: {e}")
            return pd.DataFrame()
    
    async def get_quinella_odds(self, race_url: str) -> pd.DataFrame:
        """馬連オッズを取得
        
        Args:
            race_url: レースページURL
        
        Returns:
            馬連オッズDataFrame
        """
        await self._init_browser()
        
        try:
            odds_url = race_url.replace('/race/', '/odds/') if '/race/' in race_url else race_url
            
            await self.page.goto(odds_url + '?type=umaren')
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            content = await self.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            odds_data = []
            
            # 馬連オッズテーブルを探す
            odds_tables = soup.find_all('table', class_='odds_table')
            
            for table in odds_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        combination = cells[0].get_text(strip=True)
                        odds = cells[-1].get_text(strip=True)
                        
                        # 組み合わせをパース（例: "1-2"）
                        match = re.match(r'(\d+)\s*[-−]\s*(\d+)', combination)
                        if match:
                            try:
                                odds_data.append({
                                    'horse_1': int(match.group(1)),
                                    'horse_2': int(match.group(2)),
                                    'quinella_odds': float(odds.replace(',', ''))
                                })
                            except:
                                continue
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            logger.error(f"馬連オッズ取得エラー: {e}")
            return pd.DataFrame()
    
    async def get_wide_odds(self, race_url: str) -> pd.DataFrame:
        """ワイドオッズを取得"""
        await self._init_browser()
        
        try:
            odds_url = race_url.replace('/race/', '/odds/') if '/race/' in race_url else race_url
            
            await self.page.goto(odds_url + '?type=wide')
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            content = await self.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            odds_data = []
            
            odds_tables = soup.find_all('table', class_='odds_table')
            
            for table in odds_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 3:
                        combination = cells[0].get_text(strip=True)
                        odds_min = cells[-2].get_text(strip=True)
                        odds_max = cells[-1].get_text(strip=True)
                        
                        match = re.match(r'(\d+)\s*[-−]\s*(\d+)', combination)
                        if match:
                            try:
                                odds_data.append({
                                    'horse_1': int(match.group(1)),
                                    'horse_2': int(match.group(2)),
                                    'wide_odds_min': float(odds_min.replace(',', '')),
                                    'wide_odds_max': float(odds_max.replace(',', ''))
                                })
                            except:
                                continue
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            logger.error(f"ワイドオッズ取得エラー: {e}")
            return pd.DataFrame()
    
    async def get_exacta_odds(self, race_url: str) -> pd.DataFrame:
        """馬単オッズを取得"""
        await self._init_browser()
        
        try:
            odds_url = race_url.replace('/race/', '/odds/') if '/race/' in race_url else race_url
            
            await self.page.goto(odds_url + '?type=umatan')
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            content = await self.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            odds_data = []
            
            odds_tables = soup.find_all('table', class_='odds_table')
            
            for table in odds_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        combination = cells[0].get_text(strip=True)
                        odds = cells[-1].get_text(strip=True)
                        
                        match = re.match(r'(\d+)\s*[→→]\s*(\d+)', combination)
                        if match:
                            try:
                                odds_data.append({
                                    'horse_1': int(match.group(1)),
                                    'horse_2': int(match.group(2)),
                                    'exacta_odds': float(odds.replace(',', ''))
                                })
                            except:
                                continue
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            logger.error(f"馬単オッズ取得エラー: {e}")
            return pd.DataFrame()
    
    async def get_trio_odds(self, race_url: str) -> pd.DataFrame:
        """三連複オッズを取得"""
        await self._init_browser()
        
        try:
            odds_url = race_url.replace('/race/', '/odds/') if '/race/' in race_url else race_url
            
            await self.page.goto(odds_url + '?type=sanrenpuku')
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            content = await self.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            odds_data = []
            
            odds_tables = soup.find_all('table', class_='odds_table')
            
            for table in odds_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        combination = cells[0].get_text(strip=True)
                        odds = cells[-1].get_text(strip=True)
                        
                        match = re.match(r'(\d+)\s*[-−]\s*(\d+)\s*[-−]\s*(\d+)', combination)
                        if match:
                            try:
                                odds_data.append({
                                    'horse_1': int(match.group(1)),
                                    'horse_2': int(match.group(2)),
                                    'horse_3': int(match.group(3)),
                                    'trio_odds': float(odds.replace(',', ''))
                                })
                            except:
                                continue
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            logger.error(f"三連複オッズ取得エラー: {e}")
            return pd.DataFrame()
    
    async def get_trifecta_odds(self, race_url: str) -> pd.DataFrame:
        """三連単オッズを取得"""
        await self._init_browser()
        
        try:
            odds_url = race_url.replace('/race/', '/odds/') if '/race/' in race_url else race_url
            
            await self.page.goto(odds_url + '?type=sanrentan')
            await self.page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            
            content = await self.page.content()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            odds_data = []
            
            odds_tables = soup.find_all('table', class_='odds_table')
            
            for table in odds_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        combination = cells[0].get_text(strip=True)
                        odds = cells[-1].get_text(strip=True)
                        
                        match = re.match(r'(\d+)\s*[→→]\s*(\d+)\s*[→→]\s*(\d+)', combination)
                        if match:
                            try:
                                odds_data.append({
                                    'horse_1': int(match.group(1)),
                                    'horse_2': int(match.group(2)),
                                    'horse_3': int(match.group(3)),
                                    'trifecta_odds': float(odds.replace(',', ''))
                                })
                            except:
                                continue
            
            return pd.DataFrame(odds_data)
            
        except Exception as e:
            logger.error(f"三連単オッズ取得エラー: {e}")
            return pd.DataFrame()
    
    async def get_all_odds(self, race_url: str, race_id: str) -> Dict[str, pd.DataFrame]:
        """全券種のオッズを取得
        
        Args:
            race_url: レースページURL
            race_id: レースID
        
        Returns:
            券種別オッズDataFrameの辞書
        """
        logger.info(f"オッズ取得開始: {race_id}")
        
        odds_data = {}
        
        try:
            # 単勝
            odds_data['win'] = await self.get_win_odds(race_url)
            await asyncio.sleep(1)
            
            # 複勝
            odds_data['place'] = await self.get_place_odds(race_url)
            await asyncio.sleep(1)
            
            # 馬連
            odds_data['quinella'] = await self.get_quinella_odds(race_url)
            await asyncio.sleep(1)
            
            # ワイド
            odds_data['wide'] = await self.get_wide_odds(race_url)
            await asyncio.sleep(1)
            
            # 馬単
            odds_data['exacta'] = await self.get_exacta_odds(race_url)
            await asyncio.sleep(1)
            
            # 三連複
            odds_data['trio'] = await self.get_trio_odds(race_url)
            await asyncio.sleep(1)
            
            # 三連単
            odds_data['trifecta'] = await self.get_trifecta_odds(race_url)
            
            # レースIDを追加
            for key in odds_data:
                if not odds_data[key].empty:
                    odds_data[key]['race_id'] = race_id
                    odds_data[key]['fetch_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"オッズ取得完了: {race_id}")
            
        except Exception as e:
            logger.error(f"オッズ取得エラー: {race_id} - {e}")
        
        return odds_data
    
    async def save_odds(self, odds_data: Dict[str, pd.DataFrame], race_id: str):
        """オッズデータを保存
        
        Args:
            odds_data: オッズデータ辞書
            race_id: レースID
        """
        save_dir = self.html_dir / race_id
        ensure_dir(save_dir)
        
        for bet_type, df in odds_data.items():
            if not df.empty:
                save_path = save_dir / f"{bet_type}.csv"
                df.to_csv(save_path, index=False, encoding='utf-8-sig')
                logger.debug(f"保存: {save_path}")


# 同期版のラッパー関数
def get_jra_odds(race_url: str, race_id: str, config: Optional[dict] = None) -> Dict[str, pd.DataFrame]:
    """JRAオッズを取得（同期版）
    
    Args:
        race_url: レースURL
        race_id: レースID
        config: 設定辞書
    
    Returns:
        オッズデータ辞書
    """
    scraper = JRAOddsScraper(config)
    
    async def _run():
        try:
            odds = await scraper.get_all_odds(race_url, race_id)
            await scraper.save_odds(odds, race_id)
            return odds
        finally:
            await scraper._close_browser()
    
    return asyncio.run(_run())
