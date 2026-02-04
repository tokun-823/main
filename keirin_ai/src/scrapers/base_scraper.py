# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- スクレイパー基底クラス
=============================================
スクレイピングの共通機能を提供
制限対策（User-Agent偽装、リトライ、プロキシ等）を実装
"""

import random
import time
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from loguru import logger

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium is not installed. Dynamic scraping will be limited.")

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

try:
    from fake_useragent import UserAgent
    FAKE_UA_AVAILABLE = True
except ImportError:
    FAKE_UA_AVAILABLE = False

from .config import ScrapingConfig, RAW_DATA_DIR


class BaseScraper(ABC):
    """
    スクレイパー基底クラス
    
    特徴:
    - User-Agent自動ローテーション
    - リトライ機構（指数バックオフ）
    - Cloudflare対策（cloudscraper使用）
    - Selenium対応（動的ページ用）
    - キャッシュ機能
    - プロキシ対応
    """
    
    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.session = self._create_session()
        self.driver = None
        self.cache_dir = RAW_DATA_DIR / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # User-Agent管理
        if FAKE_UA_AVAILABLE:
            try:
                self.ua = UserAgent()
            except Exception:
                self.ua = None
        else:
            self.ua = None
        
        # リクエストカウンター（レート制限対策）
        self.request_count = 0
        self.last_request_time = None
        
    def _create_session(self) -> requests.Session:
        """リトライ機能付きセッションを作成"""
        session = requests.Session()
        
        # リトライ設定
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,  # 指数バックオフ
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_user_agent(self) -> str:
        """User-Agentをランダムに取得"""
        if self.ua:
            try:
                return self.ua.random
            except Exception:
                pass
        return random.choice(self.config.user_agents)
    
    def _get_headers(self, referer: Optional[str] = None) -> Dict[str, str]:
        """リクエストヘッダーを生成"""
        headers = {
            "User-Agent": self._get_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        if referer:
            headers["Referer"] = referer
        return headers
    
    def _wait_between_requests(self) -> None:
        """リクエスト間隔を確保（レート制限対策）"""
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.config.request_interval:
                sleep_time = self.config.request_interval - elapsed
                # ランダム性を追加
                sleep_time += random.uniform(0.1, 0.5)
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # 一定リクエスト数ごとに長めの休憩
        if self.request_count % 10 == 0:
            time.sleep(random.uniform(2.0, 5.0))
    
    def _get_cache_path(self, url: str, params: Optional[Dict] = None) -> Path:
        """キャッシュファイルパスを生成"""
        cache_str = url + (json.dumps(params, sort_keys=True) if params else "")
        cache_key = hashlib.md5(cache_str.encode()).hexdigest()
        return self.cache_dir / f"{cache_key}.html"
    
    def _load_from_cache(self, cache_path: Path, max_age_hours: int = 24) -> Optional[str]:
        """キャッシュからHTMLを読み込み"""
        if cache_path.exists():
            # キャッシュの有効期限チェック
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            if age_hours < max_age_hours:
                with open(cache_path, "r", encoding="utf-8") as f:
                    logger.debug(f"Cache hit: {cache_path}")
                    return f.read()
        return None
    
    def _save_to_cache(self, cache_path: Path, content: str) -> None:
        """HTMLをキャッシュに保存"""
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Saved to cache: {cache_path}")
    
    def fetch_with_requests(
        self,
        url: str,
        params: Optional[Dict] = None,
        method: str = "GET",
        data: Optional[Dict] = None,
        use_cache: bool = True,
        cache_max_age: int = 24,
    ) -> Optional[str]:
        """requestsを使用してページを取得"""
        # キャッシュチェック
        if use_cache:
            cache_path = self._get_cache_path(url, params)
            cached = self._load_from_cache(cache_path, cache_max_age)
            if cached:
                return cached
        
        self._wait_between_requests()
        
        try:
            headers = self._get_headers()
            
            if method.upper() == "GET":
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.request_timeout,
                )
            else:
                response = self.session.post(
                    url,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=self.config.request_timeout,
                )
            
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"
            content = response.text
            
            # キャッシュ保存
            if use_cache:
                self._save_to_cache(cache_path, content)
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {url} - {e}")
            return None
    
    def fetch_with_cloudscraper(
        self,
        url: str,
        params: Optional[Dict] = None,
        use_cache: bool = True,
        cache_max_age: int = 24,
    ) -> Optional[str]:
        """cloudscraperを使用してページを取得（Cloudflare対策）"""
        if not CLOUDSCRAPER_AVAILABLE:
            logger.warning("cloudscraper not available, falling back to requests")
            return self.fetch_with_requests(url, params, use_cache=use_cache)
        
        # キャッシュチェック
        if use_cache:
            cache_path = self._get_cache_path(url, params)
            cached = self._load_from_cache(cache_path, cache_max_age)
            if cached:
                return cached
        
        self._wait_between_requests()
        
        try:
            scraper = cloudscraper.create_scraper(
                browser={
                    "browser": "chrome",
                    "platform": "windows",
                    "mobile": False,
                }
            )
            
            response = scraper.get(
                url,
                params=params,
                timeout=self.config.request_timeout,
            )
            response.raise_for_status()
            content = response.text
            
            # キャッシュ保存
            if use_cache:
                self._save_to_cache(cache_path, content)
            
            return content
            
        except Exception as e:
            logger.error(f"Cloudscraper failed: {url} - {e}")
            return None
    
    def _init_selenium(self) -> None:
        """Seleniumドライバーを初期化"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is not installed")
        
        if self.driver is not None:
            return
        
        options = ChromeOptions()
        
        if self.config.headless:
            options.add_argument("--headless=new")
        
        # 検出回避オプション
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-agent={self._get_user_agent()}")
        
        # 自動化検出を回避
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        
        service = ChromeService(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        
        # 追加の検出回避
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['ja-JP', 'ja', 'en-US', 'en']
                });
                """
            }
        )
    
    def fetch_with_selenium(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        use_cache: bool = True,
        cache_max_age: int = 24,
    ) -> Optional[str]:
        """Seleniumを使用してページを取得（JavaScript対応）"""
        # キャッシュチェック
        if use_cache:
            cache_path = self._get_cache_path(url)
            cached = self._load_from_cache(cache_path, cache_max_age)
            if cached:
                return cached
        
        self._wait_between_requests()
        
        try:
            self._init_selenium()
            
            self.driver.get(url)
            
            # 特定の要素が表示されるまで待機
            if wait_selector:
                WebDriverWait(self.driver, self.config.selenium_wait_time).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
                )
            else:
                time.sleep(2)  # デフォルト待機
            
            # 追加のスクロール（遅延読み込み対応）
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
            
            content = self.driver.page_source
            
            # キャッシュ保存
            if use_cache:
                self._save_to_cache(cache_path, content)
            
            return content
            
        except Exception as e:
            logger.error(f"Selenium failed: {url} - {e}")
            return None
    
    def fetch(
        self,
        url: str,
        params: Optional[Dict] = None,
        use_selenium: bool = False,
        use_cloudscraper: bool = True,
        wait_selector: Optional[str] = None,
        use_cache: bool = True,
    ) -> Optional[str]:
        """
        ページを取得（自動的に最適な方法を選択）
        
        優先順位:
        1. キャッシュ
        2. cloudscraper（Cloudflare対策）
        3. Selenium（動的ページ）
        4. 通常のrequests
        """
        if use_selenium and SELENIUM_AVAILABLE:
            return self.fetch_with_selenium(url, wait_selector, use_cache)
        
        if use_cloudscraper and CLOUDSCRAPER_AVAILABLE:
            result = self.fetch_with_cloudscraper(url, params, use_cache)
            if result:
                return result
        
        # フォールバック: 通常のrequests
        return self.fetch_with_requests(url, params, use_cache=use_cache)
    
    def parse_html(self, html: str) -> BeautifulSoup:
        """HTMLをBeautifulSoupでパース"""
        return BeautifulSoup(html, "lxml")
    
    def close(self) -> None:
        """リソースを解放"""
        if self.driver:
            self.driver.quit()
            self.driver = None
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @abstractmethod
    def scrape_race_list(self, date: str) -> List[Dict]:
        """指定日のレース一覧を取得"""
        pass
    
    @abstractmethod
    def scrape_race_info(self, race_id: str) -> Dict:
        """レース情報を取得"""
        pass
    
    @abstractmethod
    def scrape_entry_table(self, race_id: str) -> List[Dict]:
        """出走表を取得"""
        pass
    
    @abstractmethod
    def scrape_race_result(self, race_id: str) -> Dict:
        """レース結果を取得"""
        pass
