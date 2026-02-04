"""
データ収集モジュール - 公式LZHファイルダウンローダー
ボートレース公式サイトからLZH圧縮ファイルをダウンロード・解凍
"""
import os
import time
import struct
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, RAW_DATA_DIR


class LZHDownloader:
    """公式LZHファイルダウンローダー"""
    
    # 公式データURLテンプレート
    BANGUMI_URL_TEMPLATE = "https://www1.mbrace.or.jp/od2/B/{year:04d}/{month:02d}/b{year:04d}{month:02d}.lzh"
    RESULT_URL_TEMPLATE = "https://www1.mbrace.or.jp/od2/K/{year:04d}/{month:02d}/k{year:04d}{month:02d}.lzh"
    FAN_URL_TEMPLATE = "https://www1.mbrace.or.jp/od2/F/fan{year:02d}{term:02d}.lzh"
    
    def __init__(self):
        self.config = config.scraping
        self.download_dir = RAW_DATA_DIR / "lzh"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # 解凍先
        self.extract_dir = RAW_DATA_DIR / "extracted"
        self.extract_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_headers(self) -> dict:
        """リクエストヘッダーを生成"""
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _download_file(self, session: aiohttp.ClientSession, url: str, save_path: Path) -> bool:
        """ファイルをダウンロード"""
        try:
            async with session.get(url, headers=self._get_headers(), timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    content = await response.read()
                    async with aiofiles.open(save_path, 'wb') as f:
                        await f.write(content)
                    logger.info(f"Downloaded: {save_path.name}")
                    return True
                elif response.status == 404:
                    logger.warning(f"File not found: {url}")
                    return False
                else:
                    logger.error(f"Download failed: {url}, status: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Download error: {url}, error: {e}")
            raise
    
    async def download_bangumi(self, year: int, month: int) -> Optional[Path]:
        """番組表データをダウンロード"""
        url = self.BANGUMI_URL_TEMPLATE.format(year=year, month=month)
        save_path = self.download_dir / f"b{year:04d}{month:02d}.lzh"
        
        if save_path.exists():
            logger.info(f"Already exists: {save_path.name}")
            return save_path
        
        async with aiohttp.ClientSession() as session:
            if await self._download_file(session, url, save_path):
                await asyncio.sleep(self.config.request_interval)
                return save_path
        return None
    
    async def download_result(self, year: int, month: int) -> Optional[Path]:
        """競争結果データをダウンロード"""
        url = self.RESULT_URL_TEMPLATE.format(year=year, month=month)
        save_path = self.download_dir / f"k{year:04d}{month:02d}.lzh"
        
        if save_path.exists():
            logger.info(f"Already exists: {save_path.name}")
            return save_path
        
        async with aiohttp.ClientSession() as session:
            if await self._download_file(session, url, save_path):
                await asyncio.sleep(self.config.request_interval)
                return save_path
        return None
    
    async def download_fan_data(self, year: int, term: int) -> Optional[Path]:
        """選手期別成績データをダウンロード（半年ごと）"""
        # 年の下2桁を使用
        year_short = year % 100
        url = self.FAN_URL_TEMPLATE.format(year=year_short, term=term)
        save_path = self.download_dir / f"fan{year_short:02d}{term:02d}.lzh"
        
        if save_path.exists():
            logger.info(f"Already exists: {save_path.name}")
            return save_path
        
        async with aiohttp.ClientSession() as session:
            if await self._download_file(session, url, save_path):
                await asyncio.sleep(self.config.request_interval)
                return save_path
        return None
    
    def extract_lzh(self, lzh_path: Path) -> Path:
        """LZHファイルを解凍"""
        extract_path = self.extract_dir / lzh_path.stem
        extract_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # unlzhライブラリを使用
            import unlzh
            
            with open(lzh_path, 'rb') as f:
                data = f.read()
            
            # LZHヘッダーを解析して解凍
            pos = 0
            while pos < len(data):
                if data[pos] == 0:
                    break
                
                header_size = data[pos]
                if header_size == 0:
                    break
                
                checksum = data[pos + 1]
                method = data[pos + 2:pos + 7].decode('ascii', errors='ignore')
                
                compressed_size = struct.unpack('<I', data[pos + 7:pos + 11])[0]
                original_size = struct.unpack('<I', data[pos + 11:pos + 15])[0]
                
                # ファイル名を取得
                name_length = data[pos + 21]
                filename = data[pos + 22:pos + 22 + name_length].decode('cp932', errors='ignore')
                
                # データ開始位置
                data_start = pos + 2 + header_size
                compressed_data = data[data_start:data_start + compressed_size]
                
                # 解凍
                try:
                    if method == '-lh0-':
                        # 無圧縮
                        decompressed = compressed_data
                    else:
                        decompressed = unlzh.unlzh(compressed_data)
                    
                    output_path = extract_path / filename
                    with open(output_path, 'wb') as out_f:
                        out_f.write(decompressed)
                    logger.debug(f"Extracted: {filename}")
                except Exception as e:
                    logger.warning(f"Failed to extract {filename}: {e}")
                
                pos = data_start + compressed_size
            
            logger.info(f"Extracted LZH: {lzh_path.name} -> {extract_path}")
            return extract_path
            
        except ImportError:
            # unlzhがない場合は7zを試す
            logger.warning("unlzh not available, trying py7zr...")
            try:
                import subprocess
                result = subprocess.run(
                    ['7z', 'x', str(lzh_path), f'-o{extract_path}', '-y'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    logger.info(f"Extracted with 7z: {lzh_path.name}")
                    return extract_path
                else:
                    logger.error(f"7z extraction failed: {result.stderr}")
            except FileNotFoundError:
                logger.error("Neither unlzh nor 7z available for LZH extraction")
        except Exception as e:
            logger.error(f"LZH extraction failed: {e}")
        
        return extract_path
    
    async def download_all_data(self, start_year: int = None, end_year: int = None):
        """指定期間の全データをダウンロード"""
        start_year = start_year or self.config.start_year
        end_year = end_year or self.config.end_year
        
        logger.info(f"Downloading data from {start_year} to {end_year}")
        
        # 番組表と競争結果
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # 未来の月はスキップ
                if year == datetime.now().year and month > datetime.now().month:
                    break
                
                logger.info(f"Downloading {year}/{month:02d}...")
                
                bangumi_path = await self.download_bangumi(year, month)
                if bangumi_path:
                    self.extract_lzh(bangumi_path)
                
                result_path = await self.download_result(year, month)
                if result_path:
                    self.extract_lzh(result_path)
        
        # 選手期別成績（半年ごと）
        for year in range(start_year, end_year + 1):
            for term in [1, 2]:  # 前期・後期
                fan_path = await self.download_fan_data(year, term)
                if fan_path:
                    self.extract_lzh(fan_path)
        
        logger.info("Download completed!")


async def main():
    """メイン実行"""
    downloader = LZHDownloader()
    await downloader.download_all_data()


if __name__ == "__main__":
    asyncio.run(main())
