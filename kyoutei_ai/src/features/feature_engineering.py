"""
特徴量エンジニアリング
機械学習モデル用の特徴量を生成
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    config, RACECOURSE_CODES, 
    TRIFECTA_COMBINATIONS, TRIFECTA_TO_INDEX
)
from src.etl import db


@dataclass
class RaceFeatures:
    """1レース分の特徴量"""
    race_id: str = ""  # race_date + place_code + race_number
    
    # 基本情報
    place_code: str = ""
    race_number: int = 0
    distance: int = 0
    race_grade: str = ""
    
    # 各枠の特徴量 (6枠分)
    waku_features: List[Dict] = field(default_factory=list)
    
    # レース全体の特徴量
    race_level_features: Dict = field(default_factory=dict)
    
    # ラベル（教師データ）
    trifecta_result: Optional[Tuple[int, int, int]] = None
    trifecta_payout: int = 0


class FeatureEngineer:
    """特徴量生成クラス"""
    
    def __init__(self):
        self.db = db
    
    def create_race_features(
        self,
        race_date: str,
        place_code: str,
        race_number: int
    ) -> Optional[RaceFeatures]:
        """1レースの特徴量を生成"""
        
        # データ取得
        race_data = self.db.get_race_data(race_date, place_code, race_number)
        
        bangumi_df = race_data['bangumi']
        result_df = race_data['result']
        odds_df = race_data['odds']
        before_info_df = race_data['before_info']
        
        if bangumi_df.empty:
            return None
        
        features = RaceFeatures()
        features.race_id = f"{race_date}_{place_code}_{race_number:02d}"
        features.place_code = place_code
        features.race_number = race_number
        
        if not bangumi_df.empty:
            features.distance = bangumi_df.iloc[0]['distance']
            features.race_grade = bangumi_df.iloc[0]['race_grade']
        
        # 各枠の特徴量
        for waku in range(1, 7):
            waku_data = bangumi_df[bangumi_df['waku'] == waku]
            if waku_data.empty:
                continue
            
            row = waku_data.iloc[0]
            
            # 基本特徴量
            waku_feat = {
                'waku': waku,
                'racer_id': row.get('racer_id', ''),
                
                # 選手成績
                'win_rate': row.get('win_rate', 0),
                'two_rate': row.get('two_rate', 0),
                'three_rate': row.get('three_rate', 0),
                
                # 選手属性
                'racer_class': self._encode_class(row.get('racer_class', '')),
                'age': row.get('age', 0),
                'weight': row.get('weight', 0),
                'branch': self._encode_branch(row.get('branch', '')),
                
                # モーター成績
                'motor_win_rate': row.get('motor_win_rate', 0),
                'motor_two_rate': row.get('motor_two_rate', 0),
                
                # ボート成績
                'boat_win_rate': row.get('boat_win_rate', 0),
                'boat_two_rate': row.get('boat_two_rate', 0),
            }
            
            # 過去成績から追加特徴量
            racer_id = row.get('racer_id', '')
            if racer_id:
                past_stats = self._get_racer_past_stats(racer_id, race_date, place_code)
                waku_feat.update(past_stats)
            
            # 直前情報
            if not before_info_df.empty:
                before_row = before_info_df[before_info_df['waku'] == waku]
                if not before_row.empty:
                    br = before_row.iloc[0]
                    waku_feat['exhibition_time'] = br.get('exhibition_time', 0)
                    waku_feat['entry_course'] = br.get('entry_course', waku)
                    waku_feat['tilt'] = br.get('tilt', 0)
            
            features.waku_features.append(waku_feat)
        
        # レース全体の特徴量
        features.race_level_features = self._create_race_level_features(
            bangumi_df, place_code, race_number
        )
        
        # 結果（ラベル）
        if not result_df.empty:
            res = result_df.iloc[0]
            first = int(res.get('first', 0))
            second = int(res.get('second', 0))
            third = int(res.get('third', 0))
            
            if first > 0 and second > 0 and third > 0:
                features.trifecta_result = (first, second, third)
                features.trifecta_payout = int(res.get('trifecta_payout', 0))
        
        return features
    
    def _encode_class(self, racer_class: str) -> int:
        """選手クラスを数値エンコード"""
        class_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        return class_map.get(racer_class.strip(), 0)
    
    def _encode_branch(self, branch: str) -> int:
        """支部を数値エンコード"""
        branch_list = [
            '群馬', '埼玉', '東京', '静岡', '愛知', '三重',
            '福井', '滋賀', '大阪', '兵庫', '岡山', '広島',
            '山口', '徳島', '香川', '愛媛', '福岡', '佐賀', '長崎', '熊本', '大分'
        ]
        try:
            return branch_list.index(branch.strip()) + 1
        except ValueError:
            return 0
    
    def _get_racer_past_stats(
        self,
        racer_id: str,
        race_date: str,
        place_code: str
    ) -> Dict[str, float]:
        """選手の過去成績から特徴量を生成"""
        
        stats = {
            'recent_win_rate': 0,
            'recent_two_rate': 0,
            'recent_avg_st': 0,
            'local_win_rate': 0,
            'local_two_rate': 0,
            'flying_count': 0,
            'late_start_count': 0,
            'course_1_rate': 0,
            'course_2_rate': 0,
            'course_3_rate': 0,
            'course_4_rate': 0,
            'course_5_rate': 0,
            'course_6_rate': 0,
        }
        
        try:
            # 直近30走の成績
            recent_df = self.db.query(f"""
                SELECT
                    b.waku,
                    r.first,
                    r.second,
                    r.st_{'{'}b.waku{'}'} as st,
                    bi.entry_course
                FROM bangumi b
                JOIN race_result r ON 
                    b.race_date = r.race_date 
                    AND b.place_code = r.place_code 
                    AND b.race_number = r.race_number
                LEFT JOIN before_info bi ON
                    b.race_date = bi.race_date 
                    AND b.place_code = bi.place_code 
                    AND b.race_number = bi.race_number
                    AND b.waku = bi.waku
                WHERE b.racer_id = '{racer_id}'
                AND b.race_date < '{race_date}'
                ORDER BY b.race_date DESC
                LIMIT 30
            """)
            
            if not recent_df.empty:
                total = len(recent_df)
                wins = sum(recent_df['first'] == recent_df['waku'])
                top2 = sum((recent_df['first'] == recent_df['waku']) | 
                          (recent_df['second'] == recent_df['waku']))
                
                stats['recent_win_rate'] = wins / total if total > 0 else 0
                stats['recent_two_rate'] = top2 / total if total > 0 else 0
                
                # 平均ST
                valid_st = recent_df['st'].dropna()
                if len(valid_st) > 0:
                    stats['recent_avg_st'] = valid_st.mean()
                
                # 進入コース別成績
                for course in range(1, 7):
                    course_df = recent_df[recent_df['entry_course'] == course]
                    if len(course_df) > 0:
                        course_wins = sum(course_df['first'] == course_df['waku'])
                        stats[f'course_{course}_rate'] = course_wins / len(course_df)
            
            # 当地成績
            local_df = self.db.query(f"""
                SELECT
                    b.waku,
                    r.first,
                    r.second
                FROM bangumi b
                JOIN race_result r ON 
                    b.race_date = r.race_date 
                    AND b.place_code = r.place_code 
                    AND b.race_number = r.race_number
                WHERE b.racer_id = '{racer_id}'
                AND b.place_code = '{place_code}'
                AND b.race_date < '{race_date}'
                ORDER BY b.race_date DESC
                LIMIT 20
            """)
            
            if not local_df.empty:
                total = len(local_df)
                wins = sum(local_df['first'] == local_df['waku'])
                top2 = sum((local_df['first'] == local_df['waku']) | 
                          (local_df['second'] == local_df['waku']))
                
                stats['local_win_rate'] = wins / total if total > 0 else 0
                stats['local_two_rate'] = top2 / total if total > 0 else 0
            
            # F・L回数
            racer_master = self.db.query(f"""
                SELECT flying_count, late_start_count
                FROM racer_master
                WHERE racer_id = '{racer_id}'
            """)
            
            if not racer_master.empty:
                stats['flying_count'] = racer_master.iloc[0].get('flying_count', 0)
                stats['late_start_count'] = racer_master.iloc[0].get('late_start_count', 0)
                
        except Exception as e:
            logger.warning(f"Error getting racer stats: {racer_id}, {e}")
        
        return stats
    
    def _create_race_level_features(
        self,
        bangumi_df: pd.DataFrame,
        place_code: str,
        race_number: int
    ) -> Dict[str, float]:
        """レース全体の特徴量を生成"""
        
        features = {
            'place_code_encoded': int(place_code),
            'race_number': race_number,
            'is_morning': 1 if race_number <= 4 else 0,
            'is_evening': 1 if race_number >= 10 else 0,
            
            # 選手レベルの統計
            'avg_win_rate': bangumi_df['win_rate'].mean() if not bangumi_df.empty else 0,
            'std_win_rate': bangumi_df['win_rate'].std() if not bangumi_df.empty else 0,
            'max_win_rate': bangumi_df['win_rate'].max() if not bangumi_df.empty else 0,
            'min_win_rate': bangumi_df['win_rate'].min() if not bangumi_df.empty else 0,
            
            # A1選手数
            'a1_count': sum(bangumi_df['racer_class'] == 'A1') if not bangumi_df.empty else 0,
            'a2_count': sum(bangumi_df['racer_class'] == 'A2') if not bangumi_df.empty else 0,
            'b1_count': sum(bangumi_df['racer_class'] == 'B1') if not bangumi_df.empty else 0,
            'b2_count': sum(bangumi_df['racer_class'] == 'B2') if not bangumi_df.empty else 0,
        }
        
        # 1号艇の相対的強さ
        if not bangumi_df.empty:
            waku1_rate = bangumi_df[bangumi_df['waku'] == 1]['win_rate'].values
            if len(waku1_rate) > 0:
                features['waku1_relative_strength'] = waku1_rate[0] / features['avg_win_rate'] if features['avg_win_rate'] > 0 else 1
            else:
                features['waku1_relative_strength'] = 1
        
        # 会場特性（過去データから）
        try:
            place_stats = self.db.query(f"""
                SELECT
                    AVG(CASE WHEN first = 1 THEN 1.0 ELSE 0.0 END) as in_win_rate,
                    AVG(CASE WHEN first = 1 OR second = 1 THEN 1.0 ELSE 0.0 END) as in_top2_rate
                FROM race_result
                WHERE place_code = '{place_code}'
            """)
            
            if not place_stats.empty:
                features['place_in_win_rate'] = place_stats.iloc[0]['in_win_rate']
                features['place_in_top2_rate'] = place_stats.iloc[0]['in_top2_rate']
        except:
            features['place_in_win_rate'] = 0.5
            features['place_in_top2_rate'] = 0.7
        
        return features
    
    def features_to_array(self, features: RaceFeatures) -> Tuple[np.ndarray, Optional[int]]:
        """特徴量を配列に変換"""
        
        # 特徴量の順序を固定
        feature_cols = [
            # 枠ごとの特徴量
            'win_rate', 'two_rate', 'three_rate',
            'racer_class', 'age', 'weight', 'branch',
            'motor_win_rate', 'motor_two_rate',
            'boat_win_rate', 'boat_two_rate',
            'recent_win_rate', 'recent_two_rate', 'recent_avg_st',
            'local_win_rate', 'local_two_rate',
            'flying_count', 'late_start_count',
            'course_1_rate', 'course_2_rate', 'course_3_rate',
            'course_4_rate', 'course_5_rate', 'course_6_rate',
            'exhibition_time', 'entry_course', 'tilt'
        ]
        
        # レース全体の特徴量
        race_cols = [
            'place_code_encoded', 'race_number',
            'is_morning', 'is_evening',
            'avg_win_rate', 'std_win_rate', 'max_win_rate', 'min_win_rate',
            'a1_count', 'a2_count', 'b1_count', 'b2_count',
            'waku1_relative_strength',
            'place_in_win_rate', 'place_in_top2_rate'
        ]
        
        # 配列構築
        X = []
        
        # 枠ごとの特徴量（6枠分）
        for waku in range(6):
            if waku < len(features.waku_features):
                waku_feat = features.waku_features[waku]
                for col in feature_cols:
                    X.append(waku_feat.get(col, 0))
            else:
                # 欠損枠は0埋め
                X.extend([0] * len(feature_cols))
        
        # レース全体の特徴量
        for col in race_cols:
            X.append(features.race_level_features.get(col, 0))
        
        X = np.array(X, dtype=np.float32)
        
        # ラベル
        label = None
        if features.trifecta_result:
            label = TRIFECTA_TO_INDEX.get(features.trifecta_result)
        
        return X, label
    
    def create_dataset(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """期間のデータセットを作成"""
        
        # レース一覧を取得
        races = self.db.query(f"""
            SELECT DISTINCT race_date, place_code, race_number
            FROM race_result
            WHERE race_date >= '{start_date}'
            AND race_date <= '{end_date}'
            ORDER BY race_date, place_code, race_number
        """)
        
        X_list = []
        y_list = []
        race_ids = []
        
        logger.info(f"Creating dataset for {len(races)} races...")
        
        for _, row in races.iterrows():
            features = self.create_race_features(
                row['race_date'],
                row['place_code'],
                row['race_number']
            )
            
            if features and features.trifecta_result:
                X, y = self.features_to_array(features)
                if y is not None:
                    X_list.append(X)
                    y_list.append(y)
                    race_ids.append(features.race_id)
        
        X = np.vstack(X_list) if X_list else np.array([])
        y = np.array(y_list)
        
        logger.info(f"Created dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y, race_ids


def get_feature_names() -> List[str]:
    """特徴量名のリストを取得"""
    feature_cols = [
        'win_rate', 'two_rate', 'three_rate',
        'racer_class', 'age', 'weight', 'branch',
        'motor_win_rate', 'motor_two_rate',
        'boat_win_rate', 'boat_two_rate',
        'recent_win_rate', 'recent_two_rate', 'recent_avg_st',
        'local_win_rate', 'local_two_rate',
        'flying_count', 'late_start_count',
        'course_1_rate', 'course_2_rate', 'course_3_rate',
        'course_4_rate', 'course_5_rate', 'course_6_rate',
        'exhibition_time', 'entry_course', 'tilt'
    ]
    
    race_cols = [
        'place_code_encoded', 'race_number',
        'is_morning', 'is_evening',
        'avg_win_rate', 'std_win_rate', 'max_win_rate', 'min_win_rate',
        'a1_count', 'a2_count', 'b1_count', 'b2_count',
        'waku1_relative_strength',
        'place_in_win_rate', 'place_in_top2_rate'
    ]
    
    names = []
    for waku in range(1, 7):
        for col in feature_cols:
            names.append(f"waku{waku}_{col}")
    
    names.extend(race_cols)
    
    return names
