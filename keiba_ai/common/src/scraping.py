"""
スクレイピング関数群
netkeiba、JRA公式サイトからのデータ取得を行う

- スクレイピング制限対策として堅牢な実装
- 中断・再開機能
- サーバー負荷対策（time.sleep）
- リトライ機能
"""

import os
import re
import time
import random
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from common.src.utils import (
    load_config, get_project_root, ensure_dir, 
    setup_logging, parse_race_id
)

# ロガー設定
logger = setup_logging()


class RobustScraper:
    """堅牢なスクレイピングクラス
    
    スクレイピング制限対策:
    - ランダム遅延
    - User-Agentローテーション
    - リトライ機能
    - セッション管理
    """
    
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]
    
    def __init__(self, config: Optional[dict] = None):
        """初期化
        
        Args:
            config: 設定辞書（Noneの場合はconfig.yamlから読み込み）
        """
        self.config = config or load_config()
        self.scraping_config = self.config.get('scraping', {})
        
        self.sleep_time = self.scraping_config.get('sleep_time', 1.5)
        self.max_retries = self.scraping_config.get('max_retries', 3)
        self.timeout = self.scraping_config.get('timeout', 30)
        self.skip_existing = self.scraping_config.get('skip_existing', True)
        
        # セッション初期化
        self.session = requests.Session()
        self._update_headers()
    
    def _update_headers(self):
        """ヘッダーを更新（User-Agentローテーション）"""
        self.session.headers.update({
            'User-Agent': random.choice(self.USER_AGENTS),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def _random_sleep(self, base: float = None):
        """ランダム遅延（サーバー負荷対策）"""
        base = base or self.sleep_time
        # 基本時間の80%〜150%でランダム
        sleep_time = base * (0.8 + random.random() * 0.7)
        time.sleep(sleep_time)
    
    def get_html(self, url: str, retry_count: int = 0) -> Optional[str]:
        """HTMLを取得（リトライ機能付き）
        
        Args:
            url: 取得するURL
            retry_count: 現在のリトライ回数
        
        Returns:
            HTML文字列（失敗時はNone）
        """
        try:
            # User-Agentをランダム更新
            if retry_count > 0:
                self._update_headers()
                self._random_sleep(self.sleep_time * (retry_count + 1))
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # エンコーディング処理
            response.encoding = response.apparent_encoding or 'utf-8'
            
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"リクエストエラー (試行 {retry_count + 1}/{self.max_retries}): {url} - {e}")
            
            if retry_count < self.max_retries - 1:
                return self.get_html(url, retry_count + 1)
            else:
                logger.error(f"最大リトライ回数に達しました: {url}")
                return None
    
    def get_html_binary(self, url: str, retry_count: int = 0) -> Optional[bytes]:
        """HTMLをバイナリで取得（リトライ機能付き）
        
        Args:
            url: 取得するURL
            retry_count: 現在のリトライ回数
        
        Returns:
            HTMLバイナリ（失敗時はNone）
        """
        try:
            if retry_count > 0:
                self._update_headers()
                self._random_sleep(self.sleep_time * (retry_count + 1))
            
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            return response.content
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"リクエストエラー (試行 {retry_count + 1}/{self.max_retries}): {url} - {e}")
            
            if retry_count < self.max_retries - 1:
                return self.get_html_binary(url, retry_count + 1)
            else:
                logger.error(f"最大リトライ回数に達しました: {url}")
                return None
    
    def save_html(self, content: bytes, file_path: str) -> bool:
        """HTMLをファイルに保存
        
        Args:
            content: HTMLバイナリ
            file_path: 保存先パス
        
        Returns:
            成功時True
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                f.write(content)
            return True
            
        except Exception as e:
            logger.error(f"ファイル保存エラー: {file_path} - {e}")
            return False


class NetkeibaScraperBase(RobustScraper):
    """netkeiba用スクレイピング基底クラス"""
    
    BASE_URL = "https://db.netkeiba.com"
    RACE_BASE_URL = "https://race.netkeiba.com"
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.paths = self.config.get('paths', {})
        self.project_root = get_project_root()


class CalendarScraper(NetkeibaScraperBase):
    """開催日一覧取得クラス（フェーズ1-1）"""
    
    def get_race_dates(self, year: int, month: int) -> List[str]:
        """指定年月の開催日リストを取得
        
        Args:
            year: 年（4桁）
            month: 月（1-12）
        
        Returns:
            開催日リスト（YYYYMMDD形式）
        """
        url = f"{self.RACE_BASE_URL}/top/calendar.html?year={year}&month={month}"
        
        html = self.get_html(url)
        if not html:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # カレンダー上のリンクから日付を抽出
        race_dates = []
        
        # カレンダーテーブルを探す
        calendar_table = soup.find('table', class_='Calendar_Table')
        if not calendar_table:
            # 代替: 日付リンクを直接検索
            date_links = soup.find_all('a', href=re.compile(r'/race/list/\d+'))
            for link in date_links:
                match = re.search(r'/race/list/(\d{8})', link.get('href', ''))
                if match:
                    race_dates.append(match.group(1))
        else:
            # カレンダーテーブルからリンクを抽出
            links = calendar_table.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                match = re.search(r'/race/list/(\d{8})', href)
                if match:
                    race_dates.append(match.group(1))
        
        # 重複除去してソート
        race_dates = sorted(list(set(race_dates)))
        
        logger.info(f"{year}年{month}月: {len(race_dates)}開催日を取得")
        self._random_sleep()
        
        return race_dates
    
    def get_race_dates_range(self, start_year: int, start_month: int, 
                              end_year: int, end_month: int) -> List[str]:
        """期間指定で開催日リストを取得
        
        Args:
            start_year: 開始年
            start_month: 開始月
            end_year: 終了年
            end_month: 終了月
        
        Returns:
            開催日リスト（YYYYMMDD形式）
        """
        all_dates = []
        
        current_year = start_year
        current_month = start_month
        
        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            dates = self.get_race_dates(current_year, current_month)
            all_dates.extend(dates)
            
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1
        
        return sorted(list(set(all_dates)))


class RaceIdScraper(NetkeibaScraperBase):
    """レースID一覧取得クラス（フェーズ1-2）
    
    Selenium/Playwright対応（JavaScript動的生成ページ用）
    """
    
    def __init__(self, config: Optional[dict] = None, use_selenium: bool = True):
        """初期化
        
        Args:
            config: 設定辞書
            use_selenium: Seleniumを使用するか（Falseの場合はrequestsで試行）
        """
        super().__init__(config)
        self.use_selenium = use_selenium
        self.driver = None
    
    def _init_selenium(self):
        """Seleniumドライバを初期化"""
        if self.driver is not None:
            return
        
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument(f'--user-agent={random.choice(self.USER_AGENTS)}')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        
        # WebDriverの検出を回避
        self.driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
    
    def _close_selenium(self):
        """Seleniumドライバを閉じる"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def get_race_ids_from_date(self, date: str) -> List[str]:
        """開催日からレースIDリストを取得
        
        Args:
            date: 開催日（YYYYMMDD形式）
        
        Returns:
            レースIDリスト（12桁）
        """
        url = f"{self.RACE_BASE_URL}/race/list/{date}/"
        
        race_ids = []
        
        if self.use_selenium:
            try:
                self._init_selenium()
                
                self.driver.get(url)
                time.sleep(2)  # JS実行待ち
                
                html = self.driver.page_source
                soup = BeautifulSoup(html, 'html.parser')
                
            except Exception as e:
                logger.error(f"Seleniumエラー: {e}")
                # フォールバック: requestsで試行
                html = self.get_html(url)
                if not html:
                    return []
                soup = BeautifulSoup(html, 'html.parser')
        else:
            html = self.get_html(url)
            if not html:
                return []
            soup = BeautifulSoup(html, 'html.parser')
        
        # レースIDを含むリンクを検索
        # パターン1: /race/レースID/
        links = soup.find_all('a', href=re.compile(r'/race/\d{12}'))
        for link in links:
            href = link.get('href', '')
            match = re.search(r'/race/(\d{12})', href)
            if match:
                race_ids.append(match.group(1))
        
        # パターン2: race_id=レースID
        links2 = soup.find_all('a', href=re.compile(r'race_id=\d{12}'))
        for link in links2:
            href = link.get('href', '')
            match = re.search(r'race_id=(\d{12})', href)
            if match:
                race_ids.append(match.group(1))
        
        # 重複除去
        race_ids = list(set(race_ids))
        
        logger.info(f"{date}: {len(race_ids)}レースを取得")
        self._random_sleep()
        
        return sorted(race_ids)
    
    def get_race_ids_from_dates(self, dates: List[str]) -> List[str]:
        """複数の開催日からレースIDリストを取得
        
        Args:
            dates: 開催日リスト（YYYYMMDD形式）
        
        Returns:
            レースIDリスト（12桁）
        """
        all_race_ids = []
        
        try:
            for date in tqdm(dates, desc="レースID取得"):
                race_ids = self.get_race_ids_from_date(date)
                all_race_ids.extend(race_ids)
        finally:
            self._close_selenium()
        
        return sorted(list(set(all_race_ids)))


class RaceResultScraper(NetkeibaScraperBase):
    """レース結果データ取得クラス（フェーズ1-3）"""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.html_dir = self.project_root / self.paths.get('html_race', 'data/html/race')
        ensure_dir(self.html_dir)
    
    def get_race_result_html(self, race_id: str, skip: bool = True) -> Optional[str]:
        """レース結果HTMLを取得・保存
        
        Args:
            race_id: 12桁のレースID
            skip: Trueの場合、既存ファイルをスキップ
        
        Returns:
            保存されたファイルパス（失敗時はNone）
        """
        file_path = self.html_dir / f"{race_id}.bin"
        
        # スキップ処理
        if skip and file_path.exists():
            logger.debug(f"スキップ（既存）: {race_id}")
            return str(file_path)
        
        url = f"{self.BASE_URL}/race/{race_id}/"
        
        content = self.get_html_binary(url)
        if content is None:
            return None
        
        if self.save_html(content, str(file_path)):
            logger.debug(f"保存完了: {race_id}")
            self._random_sleep()
            return str(file_path)
        
        return None
    
    def get_race_results_batch(self, race_ids: List[str], skip: bool = True) -> Dict[str, str]:
        """複数のレース結果HTMLをバッチ取得
        
        Args:
            race_ids: レースIDリスト
            skip: Trueの場合、既存ファイルをスキップ
        
        Returns:
            {race_id: file_path}の辞書
        """
        results = {}
        
        for race_id in tqdm(race_ids, desc="レース結果取得"):
            file_path = self.get_race_result_html(race_id, skip=skip)
            if file_path:
                results[race_id] = file_path
        
        logger.info(f"{len(results)}/{len(race_ids)}レースの結果を取得")
        return results


class HorseDataScraper(NetkeibaScraperBase):
    """馬の過去成績データ取得クラス（フェーズ1-4）"""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.html_dir = self.project_root / self.paths.get('html_horse', 'data/html/horse')
        ensure_dir(self.html_dir)
    
    def get_horse_html(self, horse_id: str, skip: bool = True, force_update: bool = False) -> Optional[str]:
        """馬の成績HTMLを取得・保存
        
        Args:
            horse_id: 馬ID
            skip: Trueの場合、既存ファイルをスキップ
            force_update: Trueの場合、既存ファイルを上書き更新
        
        Returns:
            保存されたファイルパス（失敗時はNone）
        """
        file_path = self.html_dir / f"{horse_id}.bin"
        
        # スキップ処理（force_updateが優先）
        if not force_update and skip and file_path.exists():
            logger.debug(f"スキップ（既存）: {horse_id}")
            return str(file_path)
        
        url = f"{self.BASE_URL}/horse/{horse_id}/"
        
        content = self.get_html_binary(url)
        if content is None:
            return None
        
        if self.save_html(content, str(file_path)):
            logger.debug(f"保存完了: {horse_id}")
            self._random_sleep()
            return str(file_path)
        
        return None
    
    def get_horses_batch(self, horse_ids: List[str], skip: bool = True, 
                         force_update: bool = False) -> Dict[str, str]:
        """複数の馬成績HTMLをバッチ取得
        
        Args:
            horse_ids: 馬IDリスト
            skip: Trueの場合、既存ファイルをスキップ
            force_update: Trueの場合、既存ファイルを上書き更新
        
        Returns:
            {horse_id: file_path}の辞書
        """
        results = {}
        
        for horse_id in tqdm(horse_ids, desc="馬データ取得"):
            file_path = self.get_horse_html(horse_id, skip=skip, force_update=force_update)
            if file_path:
                results[horse_id] = file_path
        
        logger.info(f"{len(results)}/{len(horse_ids)}頭のデータを取得")
        return results


class PedigreeScraper(NetkeibaScraperBase):
    """血統データ取得クラス（フェーズ3-16）"""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.html_dir = self.project_root / self.paths.get('html_pedigree', 'data/html/pedigree')
        ensure_dir(self.html_dir)
    
    def get_pedigree_html(self, horse_id: str, skip: bool = True) -> Optional[str]:
        """馬の血統HTMLを取得・保存
        
        Args:
            horse_id: 馬ID
            skip: Trueの場合、既存ファイルをスキップ
        
        Returns:
            保存されたファイルパス（失敗時はNone）
        """
        file_path = self.html_dir / f"{horse_id}.bin"
        
        if skip and file_path.exists():
            logger.debug(f"スキップ（既存）: {horse_id}")
            return str(file_path)
        
        url = f"{self.BASE_URL}/horse/ped/{horse_id}/"
        
        content = self.get_html_binary(url)
        if content is None:
            return None
        
        if self.save_html(content, str(file_path)):
            logger.debug(f"血統保存完了: {horse_id}")
            self._random_sleep()
            return str(file_path)
        
        return None
    
    def parse_pedigree(self, html_content: bytes) -> Dict[str, str]:
        """血統HTMLをパースして父・母父IDを抽出
        
        Args:
            html_content: HTMLバイナリ
        
        Returns:
            {'sire_id': 父ID, 'bms_id': 母父ID}
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        result = {'sire_id': None, 'bms_id': None}
        
        # 血統テーブルを探す
        pedigree_table = soup.find('table', class_='blood_table')
        if not pedigree_table:
            return result
        
        # 父（1世代目の最初）
        rows = pedigree_table.find_all('tr')
        if rows:
            # 父は通常最初のセル
            first_cell = rows[0].find('td')
            if first_cell:
                sire_link = first_cell.find('a', href=re.compile(r'/horse/\w+'))
                if sire_link:
                    match = re.search(r'/horse/(\w+)', sire_link.get('href', ''))
                    if match:
                        result['sire_id'] = match.group(1)
        
        # 母父（2世代目の母の父）
        # 通常、母父は5行目あたりにある
        if len(rows) >= 5:
            bms_cell = rows[4].find('td')
            if bms_cell:
                bms_link = bms_cell.find('a', href=re.compile(r'/horse/\w+'))
                if bms_link:
                    match = re.search(r'/horse/(\w+)', bms_link.get('href', ''))
                    if match:
                        result['bms_id'] = match.group(1)
        
        return result


class LeadingScraper(NetkeibaScraperBase):
    """リーディング情報取得クラス（フェーズ3-14）"""
    
    def __init__(self, config: Optional[dict] = None):
        super().__init__(config)
        self.html_dir = self.project_root / self.paths.get('html_leading', 'data/html/leading')
        ensure_dir(self.html_dir)
    
    def get_jockey_leading(self, year: int, skip: bool = True) -> Optional[str]:
        """騎手リーディングを取得
        
        Args:
            year: 年度
            skip: 既存ファイルをスキップ
        
        Returns:
            保存されたファイルパス
        """
        file_path = self.html_dir / f"jockey_{year}.bin"
        
        if skip and file_path.exists():
            return str(file_path)
        
        url = f"{self.BASE_URL}/jockey/leading/?year={year}"
        
        content = self.get_html_binary(url)
        if content and self.save_html(content, str(file_path)):
            self._random_sleep()
            return str(file_path)
        
        return None
    
    def get_trainer_leading(self, year: int, skip: bool = True) -> Optional[str]:
        """調教師リーディングを取得
        
        Args:
            year: 年度
            skip: 既存ファイルをスキップ
        
        Returns:
            保存されたファイルパス
        """
        file_path = self.html_dir / f"trainer_{year}.bin"
        
        if skip and file_path.exists():
            return str(file_path)
        
        url = f"{self.BASE_URL}/trainer/leading/?year={year}"
        
        content = self.get_html_binary(url)
        if content and self.save_html(content, str(file_path)):
            self._random_sleep()
            return str(file_path)
        
        return None
    
    def get_sire_leading(self, year: int, skip: bool = True) -> Optional[str]:
        """種牡馬リーディングを取得
        
        Args:
            year: 年度
            skip: 既存ファイルをスキップ
        
        Returns:
            保存されたファイルパス
        """
        file_path = self.html_dir / f"sire_{year}.bin"
        
        if skip and file_path.exists():
            return str(file_path)
        
        url = f"{self.BASE_URL}/horse/leading_sire/?year={year}"
        
        content = self.get_html_binary(url)
        if content and self.save_html(content, str(file_path)):
            self._random_sleep()
            return str(file_path)
        
        return None


class ShutsubaTableScraper(NetkeibaScraperBase):
    """出馬表取得クラス（フェーズ2-8）"""
    
    def get_shutsuba_table(self, race_id: str) -> Optional[str]:
        """出馬表HTMLを取得
        
        Args:
            race_id: 12桁のレースID
        
        Returns:
            HTML文字列（失敗時はNone）
        """
        url = f"{self.RACE_BASE_URL}/race/shutuba/{race_id}/"
        
        html = self.get_html(url)
        self._random_sleep()
        
        return html


# 簡易使用のためのファクトリ関数
def create_scraper(scraper_type: str, config: Optional[dict] = None) -> RobustScraper:
    """スクレイパーを作成
    
    Args:
        scraper_type: スクレイパーの種類
            - 'calendar': 開催日一覧
            - 'race_id': レースID
            - 'race_result': レース結果
            - 'horse': 馬データ
            - 'pedigree': 血統
            - 'leading': リーディング
            - 'shutsuba': 出馬表
        config: 設定辞書
    
    Returns:
        対応するスクレイパーインスタンス
    """
    scrapers = {
        'calendar': CalendarScraper,
        'race_id': RaceIdScraper,
        'race_result': RaceResultScraper,
        'horse': HorseDataScraper,
        'pedigree': PedigreeScraper,
        'leading': LeadingScraper,
        'shutsuba': ShutsubaTableScraper,
    }
    
    if scraper_type not in scrapers:
        raise ValueError(f"Unknown scraper type: {scraper_type}")
    
    return scrapers[scraper_type](config)
