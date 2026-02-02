"""
HTMLからRawデータフレームを作成する処理
フェーズ1-5: データ前処理・整形機能

保存されたHTMLからデータを抽出し、
機械学習に使えるDataFrame形式（CSV）に変換する
"""

import re
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

from common.src.utils import (
    load_config, get_project_root, ensure_dir, 
    setup_logging, parse_race_id, PLACE_CODE_MAP
)

warnings.filterwarnings('ignore')
logger = setup_logging()


class RaceResultParser:
    """レース結果HTMLをパースしてDataFrameに変換"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
        self.paths = self.config.get('paths', {})
    
    def parse_race_html(self, html_path: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """レース結果HTMLをパース
        
        Args:
            html_path: HTMLファイルパス
        
        Returns:
            (レース結果DataFrame, レース情報辞書)
        """
        try:
            with open(html_path, 'rb') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # レースIDをファイル名から取得
            race_id = Path(html_path).stem
            
            # レース情報を抽出
            race_info = self._extract_race_info(soup, race_id)
            
            # 結果テーブルを抽出
            result_df = self._extract_result_table(soup, race_id)
            
            if result_df is not None and race_info:
                # レース情報をDataFrameに追加
                for key, value in race_info.items():
                    result_df[key] = value
            
            return result_df, race_info
            
        except Exception as e:
            logger.error(f"パースエラー: {html_path} - {e}")
            return None, None
    
    def _extract_race_info(self, soup: BeautifulSoup, race_id: str) -> Dict[str, Any]:
        """レース情報を抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            race_id: レースID
        
        Returns:
            レース情報辞書
        """
        info = {
            'race_id': race_id,
            'year': race_id[0:4],
            'place_code': race_id[4:6],
            'kai': race_id[6:8],
            'day': race_id[8:10],
            'race_num': race_id[10:12],
        }
        
        # 競馬場名
        info['place'] = PLACE_CODE_MAP.get(info['place_code'], '不明')
        
        # レース名
        race_name_elem = soup.find('h1', class_='RaceName')
        if race_name_elem:
            info['race_name'] = race_name_elem.get_text(strip=True)
        else:
            # 代替: diary_snap_cut内のタイトル
            title_elem = soup.find('title')
            if title_elem:
                info['race_name'] = title_elem.get_text(strip=True).split('|')[0].strip()
        
        # レース詳細情報（距離、芝/ダート、天候など）
        race_data_elem = soup.find('div', class_='RaceData01')
        if race_data_elem:
            race_data_text = race_data_elem.get_text()
            
            # 距離
            distance_match = re.search(r'(\d+)m', race_data_text)
            if distance_match:
                info['distance'] = int(distance_match.group(1))
            
            # 芝/ダート/障害
            if '芝' in race_data_text:
                info['race_type'] = '芝'
            elif 'ダ' in race_data_text or 'ダート' in race_data_text:
                info['race_type'] = 'ダート'
            elif '障' in race_data_text:
                info['race_type'] = '障害'
            
            # 回り（右/左/直線）
            if '右' in race_data_text:
                info['course_direction'] = '右'
            elif '左' in race_data_text:
                info['course_direction'] = '左'
            elif '直' in race_data_text:
                info['course_direction'] = '直線'
            
            # 天候
            weather_match = re.search(r'天候:(\S+)', race_data_text)
            if weather_match:
                info['weather'] = weather_match.group(1)
            
            # 馬場状態
            condition_match = re.search(r'(芝|ダ):(\S+)', race_data_text)
            if condition_match:
                info['track_condition'] = condition_match.group(2)
        
        # レース日付
        race_data02_elem = soup.find('div', class_='RaceData02')
        if race_data02_elem:
            date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', race_data02_elem.get_text())
            if date_match:
                info['date'] = f"{date_match.group(1)}{int(date_match.group(2)):02d}{int(date_match.group(3)):02d}"
        
        # レースクラス
        info['race_class'] = self._extract_race_class(soup)
        
        return info
    
    def _extract_race_class(self, soup: BeautifulSoup) -> str:
        """レースクラスを抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
        
        Returns:
            レースクラス文字列
        """
        class_mapping = self.config.get('race_class', {}).get('mapping', {})
        
        # クラス情報を探す
        race_data01 = soup.find('div', class_='RaceData01')
        race_data02 = soup.find('div', class_='RaceData02')
        
        text = ''
        if race_data01:
            text += race_data01.get_text()
        if race_data02:
            text += race_data02.get_text()
        
        # G1, G2, G3
        if 'G1' in text or 'Ｇ１' in text or 'GⅠ' in text:
            return 'G1'
        if 'G2' in text or 'Ｇ２' in text or 'GⅡ' in text:
            return 'G2'
        if 'G3' in text or 'Ｇ３' in text or 'GⅢ' in text:
            return 'G3'
        
        # リステッド
        if 'L' in text or 'リステッド' in text:
            return 'L'
        
        # オープン
        if 'オープン' in text or 'OP' in text:
            return 'OP'
        
        # 条件戦
        if '3勝クラス' in text or '３勝クラス' in text or '1600万下' in text:
            return '3勝クラス'
        if '2勝クラス' in text or '２勝クラス' in text or '1000万下' in text:
            return '2勝クラス'
        if '1勝クラス' in text or '１勝クラス' in text or '500万下' in text:
            return '1勝クラス'
        
        # 新馬・未勝利
        if '新馬' in text:
            return '新馬'
        if '未勝利' in text:
            return '未勝利'
        
        return '不明'
    
    def _extract_result_table(self, soup: BeautifulSoup, race_id: str) -> Optional[pd.DataFrame]:
        """レース結果テーブルを抽出
        
        Args:
            soup: BeautifulSoupオブジェクト
            race_id: レースID
        
        Returns:
            結果DataFrame
        """
        # 結果テーブルを探す
        result_table = soup.find('table', class_='RaceTable01')
        
        if not result_table:
            # 代替: id="All_Result_Table"
            result_table = soup.find('table', id='All_Result_Table')
        
        if not result_table:
            logger.warning(f"結果テーブルが見つかりません: {race_id}")
            return None
        
        # pandasでテーブル読み込み
        try:
            # HTMLからDataFrameを作成
            html_str = str(result_table)
            dfs = pd.read_html(html_str, flavor='lxml')
            
            if not dfs:
                return None
            
            df = dfs[0]
            
            # 馬IDを抽出
            horse_ids = self._extract_horse_ids(result_table)
            if horse_ids:
                df['horse_id'] = horse_ids[:len(df)]
            
            # 騎手IDを抽出
            jockey_ids = self._extract_jockey_ids(result_table)
            if jockey_ids:
                df['jockey_id'] = jockey_ids[:len(df)]
            
            # 調教師IDを抽出
            trainer_ids = self._extract_trainer_ids(result_table)
            if trainer_ids:
                df['trainer_id'] = trainer_ids[:len(df)]
            
            # データクリーニング
            df = self._clean_result_df(df)
            
            return df
            
        except Exception as e:
            logger.error(f"テーブル抽出エラー: {race_id} - {e}")
            return None
    
    def _extract_horse_ids(self, table) -> List[str]:
        """テーブルから馬IDを抽出"""
        horse_ids = []
        rows = table.find_all('tr')
        
        for row in rows[1:]:  # ヘッダーをスキップ
            horse_link = row.find('a', href=re.compile(r'/horse/\d+'))
            if horse_link:
                match = re.search(r'/horse/(\d+)', horse_link.get('href', ''))
                if match:
                    horse_ids.append(match.group(1))
                else:
                    horse_ids.append(None)
            else:
                horse_ids.append(None)
        
        return horse_ids
    
    def _extract_jockey_ids(self, table) -> List[str]:
        """テーブルから騎手IDを抽出"""
        jockey_ids = []
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            jockey_link = row.find('a', href=re.compile(r'/jockey/\d+'))
            if jockey_link:
                match = re.search(r'/jockey/(\d+)', jockey_link.get('href', ''))
                if match:
                    jockey_ids.append(match.group(1))
                else:
                    jockey_ids.append(None)
            else:
                jockey_ids.append(None)
        
        return jockey_ids
    
    def _extract_trainer_ids(self, table) -> List[str]:
        """テーブルから調教師IDを抽出"""
        trainer_ids = []
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            trainer_link = row.find('a', href=re.compile(r'/trainer/\d+'))
            if trainer_link:
                match = re.search(r'/trainer/(\d+)', trainer_link.get('href', ''))
                if match:
                    trainer_ids.append(match.group(1))
                else:
                    trainer_ids.append(None)
            else:
                trainer_ids.append(None)
        
        return trainer_ids
    
    def _clean_result_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """結果DataFrameをクリーニング
        
        Args:
            df: 生のDataFrame
        
        Returns:
            クリーニング済みDataFrame
        """
        # カラム名の正規化
        df.columns = df.columns.str.strip()
        
        # よくあるカラム名のマッピング
        column_mapping = {
            '着順': 'rank',
            '着 順': 'rank',
            '枠番': 'frame_number',
            '枠 番': 'frame_number',
            '馬番': 'horse_number',
            '馬 番': 'horse_number',
            '馬名': 'horse_name',
            '馬 名': 'horse_name',
            '性齢': 'sex_age',
            '性 齢': 'sex_age',
            '斤量': 'weight_carried',
            '斤 量': 'weight_carried',
            '騎手': 'jockey',
            '騎 手': 'jockey',
            'タイム': 'time',
            '着差': 'margin',
            '単勝': 'win_odds',
            '人気': 'popularity',
            '人 気': 'popularity',
            '馬体重': 'horse_weight',
            '馬 体重': 'horse_weight',
            '調教師': 'trainer',
            '調 教師': 'trainer',
            '上り': 'last_3f',
            '上がり': 'last_3f',
            '賞金': 'prize',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 着順の処理
        if 'rank' in df.columns:
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
            # 取消、除外、中止などを除外
            df = df[df['rank'].notna()]
            df['rank'] = df['rank'].astype(int)
        
        # 枠番・馬番の処理
        for col in ['frame_number', 'horse_number']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
        
        # 性齢の分離
        if 'sex_age' in df.columns:
            df['sex'] = df['sex_age'].str[0]
            df['age'] = pd.to_numeric(df['sex_age'].str[1:], errors='coerce').astype('Int64')
            
            # 性別をエンコード
            sex_mapping = {'牡': 0, '牝': 1, 'セ': 2}
            df['sex_encoded'] = df['sex'].map(sex_mapping)
        
        # 斤量の処理
        if 'weight_carried' in df.columns:
            df['weight_carried'] = pd.to_numeric(df['weight_carried'], errors='coerce')
        
        # 馬体重の処理（体重と増減値を分離）
        if 'horse_weight' in df.columns:
            df['horse_weight_value'] = df['horse_weight'].apply(self._extract_weight)
            df['horse_weight_diff'] = df['horse_weight'].apply(self._extract_weight_diff)
        
        # タイムの処理
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(self._time_to_seconds)
        
        # オッズの処理
        if 'win_odds' in df.columns:
            df['win_odds'] = pd.to_numeric(df['win_odds'], errors='coerce')
        
        # 人気の処理
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').astype('Int64')
        
        # 上がり3Fの処理
        if 'last_3f' in df.columns:
            df['last_3f'] = pd.to_numeric(df['last_3f'], errors='coerce')
        
        # 賞金の処理
        if 'prize' in df.columns:
            df['prize'] = df['prize'].apply(self._parse_prize)
        
        return df
    
    @staticmethod
    def _extract_weight(weight_str) -> Optional[float]:
        """馬体重文字列から体重を抽出"""
        if pd.isna(weight_str):
            return None
        
        match = re.match(r'(\d+)', str(weight_str))
        if match:
            return float(match.group(1))
        return None
    
    @staticmethod
    def _extract_weight_diff(weight_str) -> Optional[float]:
        """馬体重文字列から増減値を抽出"""
        if pd.isna(weight_str):
            return None
        
        match = re.search(r'\(([+-]?\d+)\)', str(weight_str))
        if match:
            return float(match.group(1))
        return None
    
    @staticmethod
    def _time_to_seconds(time_str) -> Optional[float]:
        """タイム文字列を秒に変換"""
        if pd.isna(time_str):
            return None
        
        time_str = str(time_str).strip()
        
        # 分:秒.ミリ秒 形式
        match = re.match(r'(\d+):(\d+)\.(\d+)', time_str)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            milliseconds = int(match.group(3))
            return minutes * 60 + seconds + milliseconds / 10
        
        # 秒.ミリ秒 形式
        match = re.match(r'(\d+)\.(\d+)', time_str)
        if match:
            seconds = int(match.group(1))
            milliseconds = int(match.group(2))
            return seconds + milliseconds / 10
        
        return None
    
    @staticmethod
    def _parse_prize(prize_str) -> Optional[float]:
        """賞金文字列をパース"""
        if pd.isna(prize_str):
            return 0.0
        
        prize_str = str(prize_str).replace(',', '').replace('万', '0000').replace('円', '')
        
        try:
            return float(re.sub(r'[^\d.]', '', prize_str))
        except:
            return 0.0


class HorseDataParser:
    """馬の過去成績HTMLをパースしてDataFrameに変換"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
    
    def parse_horse_html(self, html_path: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """馬の成績HTMLをパース
        
        Args:
            html_path: HTMLファイルパス
        
        Returns:
            (成績DataFrame, 馬情報辞書)
        """
        try:
            with open(html_path, 'rb') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 馬IDをファイル名から取得
            horse_id = Path(html_path).stem
            
            # 馬情報を抽出
            horse_info = self._extract_horse_info(soup, horse_id)
            
            # 成績テーブルを抽出
            result_df = self._extract_result_table(soup, horse_id)
            
            if result_df is not None and horse_info:
                for key, value in horse_info.items():
                    result_df[key] = value
            
            return result_df, horse_info
            
        except Exception as e:
            logger.error(f"パースエラー: {html_path} - {e}")
            return None, None
    
    def _extract_horse_info(self, soup: BeautifulSoup, horse_id: str) -> Dict[str, Any]:
        """馬情報を抽出"""
        info = {'horse_id': horse_id}
        
        # 馬名
        horse_name_elem = soup.find('h1', class_='horse_title')
        if horse_name_elem:
            info['horse_name'] = horse_name_elem.get_text(strip=True)
        
        # プロフィールテーブルから情報抽出
        profile_table = soup.find('table', class_='db_prof_table')
        if profile_table:
            rows = profile_table.find_all('tr')
            for row in rows:
                th = row.find('th')
                td = row.find('td')
                if th and td:
                    key = th.get_text(strip=True)
                    value = td.get_text(strip=True)
                    
                    if '生年月日' in key:
                        info['birthday'] = value
                    elif '調教師' in key:
                        info['trainer'] = value
                    elif '馬主' in key:
                        info['owner'] = value
                    elif '生産者' in key:
                        info['breeder'] = value
        
        return info
    
    def _extract_result_table(self, soup: BeautifulSoup, horse_id: str) -> Optional[pd.DataFrame]:
        """馬の成績テーブルを抽出"""
        result_table = soup.find('table', class_='db_h_race_results')
        
        if not result_table:
            logger.warning(f"成績テーブルが見つかりません: {horse_id}")
            return None
        
        try:
            html_str = str(result_table)
            dfs = pd.read_html(html_str, flavor='lxml')
            
            if not dfs:
                return None
            
            df = dfs[0]
            
            # レースIDを抽出
            race_ids = self._extract_race_ids(result_table)
            if race_ids:
                df['race_id'] = race_ids[:len(df)]
            
            # データクリーニング
            df = self._clean_horse_result_df(df)
            
            return df
            
        except Exception as e:
            logger.error(f"テーブル抽出エラー: {horse_id} - {e}")
            return None
    
    def _extract_race_ids(self, table) -> List[str]:
        """テーブルからレースIDを抽出"""
        race_ids = []
        rows = table.find_all('tr')
        
        for row in rows[1:]:  # ヘッダーをスキップ
            race_link = row.find('a', href=re.compile(r'/race/\d+'))
            if race_link:
                match = re.search(r'/race/(\d+)', race_link.get('href', ''))
                if match:
                    race_ids.append(match.group(1))
                else:
                    race_ids.append(None)
            else:
                race_ids.append(None)
        
        return race_ids
    
    def _clean_horse_result_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """馬成績DataFrameをクリーニング"""
        # カラム名の正規化
        df.columns = df.columns.str.strip()
        
        column_mapping = {
            '日付': 'date',
            '開催': 'place',
            '天気': 'weather',
            'R': 'race_num',
            'レース名': 'race_name',
            '映像': 'video',
            '頭数': 'horse_count',
            '枠番': 'frame_number',
            '馬番': 'horse_number',
            'オッズ': 'odds',
            '人気': 'popularity',
            '着順': 'rank',
            '騎手': 'jockey',
            '斤量': 'weight_carried',
            '距離': 'distance_str',
            '馬場': 'track_condition',
            'タイム': 'time',
            '着差': 'margin',
            'ペース': 'pace',
            '上り': 'last_3f',
            '馬体重': 'horse_weight',
            '賞金': 'prize',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 着順の処理
        if 'rank' in df.columns:
            df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
        
        # 距離の処理
        if 'distance_str' in df.columns:
            df['race_type'] = df['distance_str'].str[0]
            df['distance'] = pd.to_numeric(
                df['distance_str'].str[1:].str.replace(',', ''), 
                errors='coerce'
            )
        
        # タイムの処理
        if 'time' in df.columns:
            df['time_seconds'] = df['time'].apply(RaceResultParser._time_to_seconds)
        
        # 馬体重の処理
        if 'horse_weight' in df.columns:
            df['horse_weight_value'] = df['horse_weight'].apply(RaceResultParser._extract_weight)
            df['horse_weight_diff'] = df['horse_weight'].apply(RaceResultParser._extract_weight_diff)
        
        # 日付の処理
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df


class PayoutParser:
    """払い戻しテーブルをパースしてDataFrameに変換（フェーズ2-9）"""
    
    def parse_payout_table(self, soup: BeautifulSoup) -> Optional[pd.DataFrame]:
        """払い戻しテーブルを抽出・整形
        
        Args:
            soup: BeautifulSoupオブジェクト
        
        Returns:
            払い戻しDataFrame
        """
        payout_table = soup.find('table', class_='pay_table_01')
        
        if not payout_table:
            return None
        
        records = []
        rows = payout_table.find_all('tr')
        
        for row in rows:
            th = row.find('th')
            tds = row.find_all('td')
            
            if th and len(tds) >= 2:
                bet_type = th.get_text(strip=True)
                
                # 馬番
                horse_nums_td = tds[0]
                horse_nums = [span.get_text(strip=True) for span in horse_nums_td.find_all('span')]
                if not horse_nums:
                    horse_nums = [horse_nums_td.get_text(strip=True)]
                
                # 払い戻し金額
                payout_td = tds[1]
                payouts = [span.get_text(strip=True).replace(',', '') for span in payout_td.find_all('span')]
                if not payouts:
                    payouts = [payout_td.get_text(strip=True).replace(',', '')]
                
                # 人気
                popularity_list = []
                if len(tds) >= 3:
                    pop_td = tds[2]
                    popularity_list = [span.get_text(strip=True) for span in pop_td.find_all('span')]
                    if not popularity_list:
                        popularity_list = [pop_td.get_text(strip=True)]
                
                # 複数的中がある場合は行を分割
                for i, horse_num in enumerate(horse_nums):
                    record = {
                        'bet_type': bet_type,
                        'horse_numbers': horse_num,
                        'payout': payouts[i] if i < len(payouts) else payouts[0],
                        'popularity': popularity_list[i] if i < len(popularity_list) else None,
                    }
                    records.append(record)
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        
        # 払い戻し金額を数値化
        df['payout'] = pd.to_numeric(df['payout'], errors='coerce')
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
        
        return df


class RawDataCreator:
    """Rawデータ作成の統合クラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
        self.paths = self.config.get('paths', {})
        
        self.race_parser = RaceResultParser(config)
        self.horse_parser = HorseDataParser(config)
        self.payout_parser = PayoutParser()
    
    def create_race_raw_df(self, html_dir: Optional[str] = None) -> pd.DataFrame:
        """全レースのRawデータを作成
        
        Args:
            html_dir: HTMLファイルディレクトリ（Noneの場合はconfig参照）
        
        Returns:
            レース結果DataFrame
        """
        if html_dir is None:
            html_dir = self.project_root / self.paths.get('html_race', 'data/html/race')
        else:
            html_dir = Path(html_dir)
        
        all_dfs = []
        html_files = list(html_dir.glob('*.bin'))
        
        logger.info(f"レースHTML: {len(html_files)}ファイルを処理")
        
        for html_path in tqdm(html_files, desc="レースデータ処理"):
            df, _ = self.race_parser.parse_race_html(str(html_path))
            if df is not None and len(df) > 0:
                all_dfs.append(df)
        
        if not all_dfs:
            logger.warning("有効なレースデータがありません")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"レースデータ: {len(combined_df)}行を作成")
        
        return combined_df
    
    def create_horse_raw_df(self, html_dir: Optional[str] = None) -> pd.DataFrame:
        """全馬のRawデータを作成
        
        Args:
            html_dir: HTMLファイルディレクトリ（Noneの場合はconfig参照）
        
        Returns:
            馬成績DataFrame
        """
        if html_dir is None:
            html_dir = self.project_root / self.paths.get('html_horse', 'data/html/horse')
        else:
            html_dir = Path(html_dir)
        
        all_dfs = []
        html_files = list(html_dir.glob('*.bin'))
        
        logger.info(f"馬HTML: {len(html_files)}ファイルを処理")
        
        for html_path in tqdm(html_files, desc="馬データ処理"):
            df, _ = self.horse_parser.parse_horse_html(str(html_path))
            if df is not None and len(df) > 0:
                all_dfs.append(df)
        
        if not all_dfs:
            logger.warning("有効な馬データがありません")
            return pd.DataFrame()
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"馬データ: {len(combined_df)}行を作成")
        
        return combined_df
    
    def save_raw_df(self, df: pd.DataFrame, filename: str) -> str:
        """RawデータをCSVで保存
        
        Args:
            df: 保存するDataFrame
            filename: ファイル名
        
        Returns:
            保存パス
        """
        save_dir = ensure_dir(self.project_root / self.paths.get('raw_df', 'data/raw_df'))
        save_path = save_dir / filename
        
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        logger.info(f"保存完了: {save_path}")
        
        return str(save_path)
    
    def update_raw_df(self, existing_path: str, new_df: pd.DataFrame, 
                      key_column: str = 'race_id') -> pd.DataFrame:
        """既存RawデータにNewデータを追加（重複排除）
        
        Args:
            existing_path: 既存CSVのパス
            new_df: 追加するDataFrame
            key_column: 重複チェックのキーカラム
        
        Returns:
            更新後のDataFrame
        """
        existing_df = pd.read_csv(existing_path)
        
        # 重複排除して結合
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=[key_column], keep='last')
        
        # 保存
        combined_df.to_csv(existing_path, index=False, encoding='utf-8-sig')
        logger.info(f"更新完了: {existing_path} ({len(combined_df)}行)")
        
        return combined_df
