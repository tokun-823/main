# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- スクレイパーパッケージ
"""

from .base_scraper import BaseScraper
from .keirin_jp_scraper import KeirinJPScraper
from .netkeiba_scraper import NetkeibaKeirinScraper
from .oddspark_scraper import OddsparkScraper
from .integrated_scraper import IntegratedScraper

__all__ = [
    "BaseScraper",
    "KeirinJPScraper",
    "NetkeibaKeirinScraper",
    "OddsparkScraper",
    "IntegratedScraper",
]
