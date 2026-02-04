"""
データパーサー - 公式LZHファイルの固定長データをパース
番組表・競争結果・選手期別成績のパース処理
"""
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from loguru import logger
import jaconv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import RAW_DATA_DIR, RACECOURSE_CODES


def zen_to_han(text: str) -> str:
    """全角を半角に変換"""
    return jaconv.z2h(text, kana=False, digit=True, ascii=True)


def han_to_zen(text: str) -> str:
    """半角を全角に変換"""
    return jaconv.h2z(text, kana=True, digit=False, ascii=False)


def safe_float(value: str, default: float = 0.0) -> float:
    """安全なfloat変換"""
    try:
        clean = zen_to_han(value.strip())
        clean = clean.replace(' ', '').replace('　', '')
        if not clean or clean == '-':
            return default
        return float(clean)
    except (ValueError, TypeError):
        return default


def safe_int(value: str, default: int = 0) -> int:
    """安全なint変換"""
    try:
        clean = zen_to_han(value.strip())
        clean = clean.replace(' ', '').replace('　', '')
        if not clean or clean == '-':
            return default
        return int(float(clean))
    except (ValueError, TypeError):
        return default


@dataclass
class BangumiRecord:
    """番組表レコード"""
    race_date: str = ""
    place_code: str = ""
    race_number: int = 0
    race_grade: str = ""
    race_name: str = ""
    distance: int = 0
    
    # 各枠の選手情報
    racer_ids: List[str] = field(default_factory=list)
    racer_names: List[str] = field(default_factory=list)
    racer_classes: List[str] = field(default_factory=list)  # A1, A2, B1, B2
    racer_branches: List[str] = field(default_factory=list)  # 支部
    racer_ages: List[int] = field(default_factory=list)
    racer_weights: List[float] = field(default_factory=list)
    
    # 各枠の成績情報
    win_rates: List[float] = field(default_factory=list)  # 勝率
    two_rate: List[float] = field(default_factory=list)  # 2連対率
    three_rate: List[float] = field(default_factory=list)  # 3連対率
    
    # モーター・ボート
    motor_numbers: List[str] = field(default_factory=list)
    motor_win_rates: List[float] = field(default_factory=list)
    motor_two_rates: List[float] = field(default_factory=list)
    boat_numbers: List[str] = field(default_factory=list)
    boat_win_rates: List[float] = field(default_factory=list)
    boat_two_rates: List[float] = field(default_factory=list)


@dataclass
class ResultRecord:
    """競争結果レコード"""
    race_date: str = ""
    place_code: str = ""
    race_number: int = 0
    
    # 着順
    finish_order: List[int] = field(default_factory=list)  # [1着の枠番, 2着の枠番, ...]
    
    # 決まり手
    winning_pattern: str = ""  # 逃げ、差し、まくり等
    
    # スタートタイミング
    start_timings: List[float] = field(default_factory=list)  # 各枠のST
    
    # 払戻金
    trifecta_combo: str = ""  # 3連単組み合わせ
    trifecta_payout: int = 0
    trio_combo: str = ""  # 3連複組み合わせ
    trio_payout: int = 0
    exacta_combo: str = ""  # 2連単組み合わせ
    exacta_payout: int = 0
    quinella_combo: str = ""  # 2連複組み合わせ
    quinella_payout: int = 0
    win_combo: str = ""  # 単勝
    win_payout: int = 0


@dataclass
class RacerRecord:
    """選手期別成績レコード"""
    racer_id: str = ""
    racer_name: str = ""
    branch: str = ""
    birth_date: str = ""
    racer_class: str = ""
    
    # 全国成績
    total_races: int = 0
    first_place: int = 0
    second_place: int = 0
    third_place: int = 0
    win_rate: float = 0.0
    two_rate: float = 0.0
    
    # 当地成績（各場ごと）
    local_stats: Dict[str, Dict] = field(default_factory=dict)
    
    # 平均ST
    avg_start_timing: float = 0.0
    
    # フライング・出遅れ
    flying_count: int = 0
    late_start_count: int = 0


class BangumiParser:
    """番組表パーサー（固定長データ）"""
    
    def __init__(self):
        self.extract_dir = RAW_DATA_DIR / "extracted"
    
    def parse_file(self, file_path: Path) -> List[BangumiRecord]:
        """番組表ファイルをパース"""
        records = []
        
        try:
            # Shift_JIS (CP932) でエンコードされている
            with open(file_path, 'r', encoding='cp932', errors='replace') as f:
                lines = f.readlines()
            
            # ヘッダー行を解析してレース情報を取得
            current_record = None
            current_waku = 0
            
            for line in lines:
                line = line.rstrip('\n\r')
                if not line:
                    continue
                
                # レコード種別を判定（先頭バイトで判別）
                record_type = line[0:2] if len(line) >= 2 else ""
                
                if record_type == "BG":
                    # 番組基本情報
                    if current_record:
                        records.append(current_record)
                    
                    current_record = BangumiRecord()
                    current_record = self._parse_bg_line(line, current_record)
                    current_waku = 0
                    
                elif record_type == "BD" and current_record:
                    # 選手詳細情報（枠番ごと）
                    current_waku += 1
                    current_record = self._parse_bd_line(line, current_record, current_waku)
            
            # 最後のレコード
            if current_record:
                records.append(current_record)
                
        except Exception as e:
            logger.error(f"Parse error: {file_path}, {e}")
        
        return records
    
    def _parse_bg_line(self, line: str, record: BangumiRecord) -> BangumiRecord:
        """番組基本情報行をパース"""
        try:
            # 固定長フォーマット（バイト位置は仕様により調整が必要）
            record.race_date = zen_to_han(line[2:10].strip())
            record.place_code = zen_to_han(line[10:12].strip())
            record.race_number = safe_int(line[12:14])
            record.race_grade = line[14:16].strip()
            record.race_name = line[16:36].strip()
            record.distance = safe_int(line[36:40])
        except Exception as e:
            logger.warning(f"BG parse error: {e}")
        
        return record
    
    def _parse_bd_line(self, line: str, record: BangumiRecord, waku: int) -> BangumiRecord:
        """選手詳細行をパース"""
        try:
            # 固定長フォーマット
            racer_id = zen_to_han(line[2:6].strip())
            racer_name = line[6:14].strip()
            racer_class = line[14:16].strip()
            branch = line[16:18].strip()
            age = safe_int(line[18:20])
            weight = safe_float(line[20:24])
            
            win_rate = safe_float(line[24:28])
            two_rate = safe_float(line[28:32])
            three_rate = safe_float(line[32:36])
            
            motor_no = zen_to_han(line[36:38].strip())
            motor_win = safe_float(line[38:42])
            motor_two = safe_float(line[42:46])
            
            boat_no = zen_to_han(line[46:48].strip())
            boat_win = safe_float(line[48:52])
            boat_two = safe_float(line[52:56])
            
            # レコードに追加
            record.racer_ids.append(racer_id)
            record.racer_names.append(racer_name)
            record.racer_classes.append(racer_class)
            record.racer_branches.append(branch)
            record.racer_ages.append(age)
            record.racer_weights.append(weight)
            record.win_rates.append(win_rate)
            record.two_rate.append(two_rate)
            record.three_rate.append(three_rate)
            record.motor_numbers.append(motor_no)
            record.motor_win_rates.append(motor_win)
            record.motor_two_rates.append(motor_two)
            record.boat_numbers.append(boat_no)
            record.boat_win_rates.append(boat_win)
            record.boat_two_rates.append(boat_two)
            
        except Exception as e:
            logger.warning(f"BD parse error: {e}")
        
        return record
    
    def parse_all_files(self) -> pd.DataFrame:
        """全番組表ファイルをパースしてDataFrameに変換"""
        all_records = []
        
        for dir_path in self.extract_dir.glob("b*"):
            if dir_path.is_dir():
                for file_path in dir_path.glob("*.txt"):
                    records = self.parse_file(file_path)
                    all_records.extend(records)
        
        # DataFrameに変換
        return self._records_to_dataframe(all_records)
    
    def _records_to_dataframe(self, records: List[BangumiRecord]) -> pd.DataFrame:
        """レコードリストをDataFrameに変換"""
        rows = []
        
        for rec in records:
            for waku in range(6):
                if waku >= len(rec.racer_ids):
                    continue
                
                row = {
                    'race_date': rec.race_date,
                    'place_code': rec.place_code,
                    'race_number': rec.race_number,
                    'race_grade': rec.race_grade,
                    'race_name': rec.race_name,
                    'distance': rec.distance,
                    'waku': waku + 1,
                    'racer_id': rec.racer_ids[waku] if waku < len(rec.racer_ids) else '',
                    'racer_name': rec.racer_names[waku] if waku < len(rec.racer_names) else '',
                    'racer_class': rec.racer_classes[waku] if waku < len(rec.racer_classes) else '',
                    'branch': rec.racer_branches[waku] if waku < len(rec.racer_branches) else '',
                    'age': rec.racer_ages[waku] if waku < len(rec.racer_ages) else 0,
                    'weight': rec.racer_weights[waku] if waku < len(rec.racer_weights) else 0,
                    'win_rate': rec.win_rates[waku] if waku < len(rec.win_rates) else 0,
                    'two_rate': rec.two_rate[waku] if waku < len(rec.two_rate) else 0,
                    'three_rate': rec.three_rate[waku] if waku < len(rec.three_rate) else 0,
                    'motor_no': rec.motor_numbers[waku] if waku < len(rec.motor_numbers) else '',
                    'motor_win_rate': rec.motor_win_rates[waku] if waku < len(rec.motor_win_rates) else 0,
                    'motor_two_rate': rec.motor_two_rates[waku] if waku < len(rec.motor_two_rates) else 0,
                    'boat_no': rec.boat_numbers[waku] if waku < len(rec.boat_numbers) else '',
                    'boat_win_rate': rec.boat_win_rates[waku] if waku < len(rec.boat_win_rates) else 0,
                    'boat_two_rate': rec.boat_two_rates[waku] if waku < len(rec.boat_two_rates) else 0,
                }
                rows.append(row)
        
        return pd.DataFrame(rows)


class ResultParser:
    """競争結果パーサー（固定長データ）"""
    
    def __init__(self):
        self.extract_dir = RAW_DATA_DIR / "extracted"
    
    def parse_file(self, file_path: Path) -> List[ResultRecord]:
        """競争結果ファイルをパース"""
        records = []
        
        try:
            with open(file_path, 'r', encoding='cp932', errors='replace') as f:
                lines = f.readlines()
            
            current_record = None
            
            for line in lines:
                line = line.rstrip('\n\r')
                if not line:
                    continue
                
                record_type = line[0:2] if len(line) >= 2 else ""
                
                if record_type == "KG":
                    # 競争結果基本情報
                    if current_record:
                        records.append(current_record)
                    
                    current_record = ResultRecord()
                    current_record = self._parse_kg_line(line, current_record)
                    
                elif record_type == "KD" and current_record:
                    # 選手結果詳細
                    current_record = self._parse_kd_line(line, current_record)
                    
                elif record_type == "KP" and current_record:
                    # 払戻金情報
                    current_record = self._parse_kp_line(line, current_record)
            
            if current_record:
                records.append(current_record)
                
        except Exception as e:
            logger.error(f"Parse error: {file_path}, {e}")
        
        return records
    
    def _parse_kg_line(self, line: str, record: ResultRecord) -> ResultRecord:
        """競争結果基本行をパース"""
        try:
            record.race_date = zen_to_han(line[2:10].strip())
            record.place_code = zen_to_han(line[10:12].strip())
            record.race_number = safe_int(line[12:14])
            record.winning_pattern = line[14:20].strip()
        except Exception as e:
            logger.warning(f"KG parse error: {e}")
        return record
    
    def _parse_kd_line(self, line: str, record: ResultRecord) -> ResultRecord:
        """選手結果詳細行をパース"""
        try:
            waku = safe_int(line[2:4])
            order = safe_int(line[4:6])
            st = safe_float(line[6:12])
            
            # 着順を記録
            while len(record.finish_order) < order:
                record.finish_order.append(0)
            if order > 0 and order <= 6:
                record.finish_order[order - 1] = waku
            
            # STを記録
            while len(record.start_timings) < waku:
                record.start_timings.append(0.0)
            if waku > 0:
                record.start_timings[waku - 1] = st
                
        except Exception as e:
            logger.warning(f"KD parse error: {e}")
        return record
    
    def _parse_kp_line(self, line: str, record: ResultRecord) -> ResultRecord:
        """払戻金行をパース"""
        try:
            ticket_type = line[2:4].strip()
            combo = zen_to_han(line[4:16].strip())
            payout = safe_int(line[16:28])
            
            if ticket_type == "3T":
                record.trifecta_combo = combo
                record.trifecta_payout = payout
            elif ticket_type == "3F":
                record.trio_combo = combo
                record.trio_payout = payout
            elif ticket_type == "2T":
                record.exacta_combo = combo
                record.exacta_payout = payout
            elif ticket_type == "2F":
                record.quinella_combo = combo
                record.quinella_payout = payout
            elif ticket_type == "1T":
                record.win_combo = combo
                record.win_payout = payout
                
        except Exception as e:
            logger.warning(f"KP parse error: {e}")
        return record
    
    def parse_all_files(self) -> pd.DataFrame:
        """全競争結果ファイルをパースしてDataFrameに変換"""
        all_records = []
        
        for dir_path in self.extract_dir.glob("k*"):
            if dir_path.is_dir():
                for file_path in dir_path.glob("*.txt"):
                    records = self.parse_file(file_path)
                    all_records.extend(records)
        
        return self._records_to_dataframe(all_records)
    
    def _records_to_dataframe(self, records: List[ResultRecord]) -> pd.DataFrame:
        """レコードリストをDataFrameに変換"""
        rows = []
        
        for rec in records:
            row = {
                'race_date': rec.race_date,
                'place_code': rec.place_code,
                'race_number': rec.race_number,
                'winning_pattern': rec.winning_pattern,
                'first': rec.finish_order[0] if len(rec.finish_order) > 0 else 0,
                'second': rec.finish_order[1] if len(rec.finish_order) > 1 else 0,
                'third': rec.finish_order[2] if len(rec.finish_order) > 2 else 0,
                'st_1': rec.start_timings[0] if len(rec.start_timings) > 0 else 0,
                'st_2': rec.start_timings[1] if len(rec.start_timings) > 1 else 0,
                'st_3': rec.start_timings[2] if len(rec.start_timings) > 2 else 0,
                'st_4': rec.start_timings[3] if len(rec.start_timings) > 3 else 0,
                'st_5': rec.start_timings[4] if len(rec.start_timings) > 4 else 0,
                'st_6': rec.start_timings[5] if len(rec.start_timings) > 5 else 0,
                'trifecta_combo': rec.trifecta_combo,
                'trifecta_payout': rec.trifecta_payout,
                'trio_combo': rec.trio_combo,
                'trio_payout': rec.trio_payout,
                'exacta_combo': rec.exacta_combo,
                'exacta_payout': rec.exacta_payout,
                'quinella_combo': rec.quinella_combo,
                'quinella_payout': rec.quinella_payout,
                'win_combo': rec.win_combo,
                'win_payout': rec.win_payout,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


class RacerParser:
    """選手期別成績パーサー（固定長バイトデータ）"""
    
    def __init__(self):
        self.extract_dir = RAW_DATA_DIR / "extracted"
    
    def parse_file(self, file_path: Path) -> List[RacerRecord]:
        """選手データファイルをパース（バイト指定）"""
        records = []
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # 1レコードのバイト長（仕様による）
            record_length = 200  # 実際の仕様に合わせて調整
            
            pos = 0
            while pos + record_length <= len(data):
                record_data = data[pos:pos + record_length]
                record = self._parse_record(record_data)
                if record and record.racer_id:
                    records.append(record)
                pos += record_length
                
        except Exception as e:
            logger.error(f"Parse error: {file_path}, {e}")
        
        return records
    
    def _parse_record(self, data: bytes) -> Optional[RacerRecord]:
        """1レコードをパース"""
        try:
            record = RacerRecord()
            
            # バイト位置指定でデータを抽出（仕様に合わせて調整）
            record.racer_id = data[0:4].decode('cp932', errors='replace').strip()
            record.racer_name = data[4:12].decode('cp932', errors='replace').strip()
            record.branch = data[12:14].decode('cp932', errors='replace').strip()
            record.racer_class = data[14:16].decode('cp932', errors='replace').strip()
            
            # 成績データ
            record.total_races = safe_int(data[16:20].decode('cp932', errors='replace'))
            record.first_place = safe_int(data[20:24].decode('cp932', errors='replace'))
            record.second_place = safe_int(data[24:28].decode('cp932', errors='replace'))
            record.third_place = safe_int(data[28:32].decode('cp932', errors='replace'))
            
            # 勝率等
            if record.total_races > 0:
                record.win_rate = safe_float(data[32:36].decode('cp932', errors='replace'))
                record.two_rate = safe_float(data[36:40].decode('cp932', errors='replace'))
            
            # 平均ST
            record.avg_start_timing = safe_float(data[40:44].decode('cp932', errors='replace'))
            
            # F・L
            record.flying_count = safe_int(data[44:46].decode('cp932', errors='replace'))
            record.late_start_count = safe_int(data[46:48].decode('cp932', errors='replace'))
            
            return record
            
        except Exception as e:
            logger.warning(f"Record parse error: {e}")
            return None
    
    def parse_all_files(self) -> pd.DataFrame:
        """全選手ファイルをパース"""
        all_records = []
        
        for dir_path in self.extract_dir.glob("fan*"):
            if dir_path.is_dir():
                for file_path in dir_path.glob("*.txt"):
                    records = self.parse_file(file_path)
                    all_records.extend(records)
        
        return self._records_to_dataframe(all_records)
    
    def _records_to_dataframe(self, records: List[RacerRecord]) -> pd.DataFrame:
        """レコードをDataFrameに変換"""
        rows = []
        
        for rec in records:
            row = {
                'racer_id': rec.racer_id,
                'racer_name': rec.racer_name,
                'branch': rec.branch,
                'racer_class': rec.racer_class,
                'total_races': rec.total_races,
                'first_place': rec.first_place,
                'second_place': rec.second_place,
                'third_place': rec.third_place,
                'win_rate': rec.win_rate,
                'two_rate': rec.two_rate,
                'avg_start_timing': rec.avg_start_timing,
                'flying_count': rec.flying_count,
                'late_start_count': rec.late_start_count,
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
