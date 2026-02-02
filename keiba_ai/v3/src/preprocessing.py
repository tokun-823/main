"""
データ前処理モジュール
フェーズ1-5, 1-6: データ前処理・整形、リーク防止

- 数値変換
- カテゴリ変換
- 過去成績の集計（リーク防止）
"""

import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.src.utils import load_config, get_project_root, ensure_dir, setup_logging

logger = setup_logging()


class DataPreprocessor:
    """データ前処理クラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
        self.paths = self.config.get('paths', {})
    
    def preprocess_race_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """レースデータの前処理
        
        Args:
            df: 生のレースDataFrame
        
        Returns:
            前処理済みDataFrame
        """
        df = df.copy()
        
        # 1. 欠損値処理
        df = self._handle_missing_values(df)
        
        # 2. 数値変換
        df = self._convert_numeric_columns(df)
        
        # 3. カテゴリ変換
        df = self._encode_categorical(df)
        
        # 4. 日付処理
        df = self._process_dates(df)
        
        # 5. 追加特徴量
        df = self._add_basic_features(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        # 着順がNaNの行を除外（取消、除外など）
        if 'rank' in df.columns:
            df = df[df['rank'].notna()].copy()
        
        # 賞金は0で埋める
        if 'prize' in df.columns:
            df['prize'] = df['prize'].fillna(0)
        
        return df
    
    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値カラムの変換"""
        numeric_columns = [
            'rank', 'frame_number', 'horse_number', 'weight_carried',
            'horse_weight_value', 'horse_weight_diff', 'win_odds',
            'popularity', 'last_3f', 'time_seconds', 'distance', 'prize'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ変数のエンコーディング"""
        # 性別エンコーディング
        if 'sex' in df.columns and 'sex_encoded' not in df.columns:
            sex_mapping = {'牡': 0, '牝': 1, 'セ': 2}
            df['sex_encoded'] = df['sex'].map(sex_mapping)
        
        # レースタイプエンコーディング
        if 'race_type' in df.columns:
            race_type_mapping = {'芝': 0, 'ダート': 1, '障害': 2}
            df['race_type_encoded'] = df['race_type'].map(race_type_mapping)
        
        # コース方向エンコーディング
        if 'course_direction' in df.columns:
            direction_mapping = {'右': 0, '左': 1, '直線': 2}
            df['course_direction_encoded'] = df['course_direction'].map(direction_mapping)
        
        # 馬場状態エンコーディング
        if 'track_condition' in df.columns:
            condition_mapping = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
            df['track_condition_encoded'] = df['track_condition'].map(condition_mapping)
        
        # レースクラスエンコーディング
        if 'race_class' in df.columns:
            class_mapping = {
                'G1': 10, 'G2': 9, 'G3': 8, 'L': 7, 'OP': 6,
                '3勝クラス': 5, '2勝クラス': 4, '1勝クラス': 3,
                '未勝利': 2, '新馬': 1, '不明': 0
            }
            df['race_class_encoded'] = df['race_class'].map(class_mapping)
        
        # 競馬場エンコーディング
        if 'place_code' in df.columns:
            df['place_encoded'] = pd.to_numeric(df['place_code'], errors='coerce')
        
        return df
    
    def _process_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """日付の処理"""
        if 'date' in df.columns:
            # 日付をdatetimeに変換
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            
            # 年月日を分離
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek  # 0=月曜
        
        return df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """基本的な追加特徴量"""
        # 1着フラグ（目的変数の基本形）
        if 'rank' in df.columns:
            df['is_win'] = (df['rank'] == 1).astype(int)
            df['is_place'] = (df['rank'] <= 3).astype(int)  # 複勝圏内
        
        # 斤量/体重比
        if 'weight_carried' in df.columns and 'horse_weight_value' in df.columns:
            df['weight_ratio'] = df['weight_carried'] / df['horse_weight_value']
        
        return df


class HistoricalDataAggregator:
    """過去成績の集計クラス（リーク防止対応）
    フェーズ1-6: 過去成績の集計（リーク防止）
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
    
    def aggregate_historical_stats(
        self, 
        race_df: pd.DataFrame, 
        horse_history_df: pd.DataFrame,
        target_race_id: str = None,
        target_date: str = None
    ) -> pd.DataFrame:
        """過去成績を集計（リーク防止）
        
        予測対象のレース日より「過去」の成績のみを参照
        
        Args:
            race_df: レースデータ（予測対象）
            horse_history_df: 馬の全成績データ
            target_race_id: 対象レースID
            target_date: 対象日付（YYYYMMDD形式）
        
        Returns:
            過去成績が追加されたDataFrame
        """
        df = race_df.copy()
        
        # 馬ごとに過去成績を集計
        stats_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="過去成績集計"):
            horse_id = row.get('horse_id')
            race_date = row.get('date')
            
            if pd.isna(horse_id) or pd.isna(race_date):
                stats_list.append(self._empty_stats())
                continue
            
            # 当該レース以前のデータのみ抽出
            past_data = horse_history_df[
                (horse_history_df['horse_id'] == horse_id) &
                (horse_history_df['date'] < race_date)
            ]
            
            if len(past_data) == 0:
                stats_list.append(self._empty_stats())
                continue
            
            stats = self._calculate_stats(past_data)
            stats_list.append(stats)
        
        # 統計量をDataFrameに追加
        stats_df = pd.DataFrame(stats_list)
        df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)
        
        return df
    
    def _empty_stats(self) -> Dict:
        """空の統計量辞書を返す"""
        return {
            'hist_race_count': 0,
            'hist_win_count': 0,
            'hist_place_count': 0,
            'hist_win_rate': np.nan,
            'hist_place_rate': np.nan,
            'hist_rank_mean': np.nan,
            'hist_rank_min': np.nan,
            'hist_prize_sum': 0,
            'hist_prize_mean': np.nan,
            'hist_time_mean': np.nan,
            'hist_last_3f_mean': np.nan,
        }
    
    def _calculate_stats(self, past_data: pd.DataFrame) -> Dict:
        """過去成績から統計量を計算"""
        stats = {}
        
        # レース数
        stats['hist_race_count'] = len(past_data)
        
        # 勝利数・連対数
        if 'rank' in past_data.columns:
            stats['hist_win_count'] = (past_data['rank'] == 1).sum()
            stats['hist_place_count'] = (past_data['rank'] <= 3).sum()
            
            # 勝率・複勝率
            stats['hist_win_rate'] = stats['hist_win_count'] / len(past_data)
            stats['hist_place_rate'] = stats['hist_place_count'] / len(past_data)
            
            # 着順統計
            stats['hist_rank_mean'] = past_data['rank'].mean()
            stats['hist_rank_min'] = past_data['rank'].min()
        else:
            stats['hist_win_count'] = 0
            stats['hist_place_count'] = 0
            stats['hist_win_rate'] = np.nan
            stats['hist_place_rate'] = np.nan
            stats['hist_rank_mean'] = np.nan
            stats['hist_rank_min'] = np.nan
        
        # 賞金
        if 'prize' in past_data.columns:
            stats['hist_prize_sum'] = past_data['prize'].sum()
            stats['hist_prize_mean'] = past_data['prize'].mean()
        else:
            stats['hist_prize_sum'] = 0
            stats['hist_prize_mean'] = np.nan
        
        # タイム
        if 'time_seconds' in past_data.columns:
            stats['hist_time_mean'] = past_data['time_seconds'].mean()
        else:
            stats['hist_time_mean'] = np.nan
        
        # 上がり3F
        if 'last_3f' in past_data.columns:
            stats['hist_last_3f_mean'] = past_data['last_3f'].mean()
        else:
            stats['hist_last_3f_mean'] = np.nan
        
        return stats
    
    def aggregate_conditional_stats(
        self,
        race_df: pd.DataFrame,
        horse_history_df: pd.DataFrame,
        condition_type: str
    ) -> pd.DataFrame:
        """条件別の過去成績を集計
        
        Args:
            race_df: レースデータ
            horse_history_df: 馬の全成績
            condition_type: 条件タイプ（'distance', 'place', 'race_type'）
        
        Returns:
            条件別成績が追加されたDataFrame
        """
        df = race_df.copy()
        prefix = f'hist_{condition_type}_'
        
        stats_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"条件別集計({condition_type})"):
            horse_id = row.get('horse_id')
            race_date = row.get('date')
            
            if pd.isna(horse_id) or pd.isna(race_date):
                stats_list.append({f'{prefix}count': 0})
                continue
            
            # 過去データのみ
            past_data = horse_history_df[
                (horse_history_df['horse_id'] == horse_id) &
                (horse_history_df['date'] < race_date)
            ]
            
            if len(past_data) == 0:
                stats_list.append({f'{prefix}count': 0})
                continue
            
            # 条件でフィルタ
            if condition_type == 'distance':
                target_distance = row.get('distance')
                if pd.notna(target_distance):
                    # 同距離（±200m許容）
                    filtered = past_data[
                        (past_data['distance'] >= target_distance - 200) &
                        (past_data['distance'] <= target_distance + 200)
                    ]
                else:
                    filtered = pd.DataFrame()
            
            elif condition_type == 'place':
                target_place = row.get('place_code')
                if pd.notna(target_place):
                    filtered = past_data[past_data['place_code'] == target_place]
                else:
                    filtered = pd.DataFrame()
            
            elif condition_type == 'race_type':
                target_type = row.get('race_type')
                if pd.notna(target_type):
                    filtered = past_data[past_data['race_type'] == target_type]
                else:
                    filtered = pd.DataFrame()
            else:
                filtered = pd.DataFrame()
            
            if len(filtered) == 0:
                stats_list.append({f'{prefix}count': 0})
                continue
            
            # 統計量計算
            stats = {
                f'{prefix}count': len(filtered),
                f'{prefix}win_rate': (filtered['rank'] == 1).mean() if 'rank' in filtered else np.nan,
                f'{prefix}place_rate': (filtered['rank'] <= 3).mean() if 'rank' in filtered else np.nan,
                f'{prefix}rank_mean': filtered['rank'].mean() if 'rank' in filtered else np.nan,
            }
            stats_list.append(stats)
        
        stats_df = pd.DataFrame(stats_list)
        df = pd.concat([df.reset_index(drop=True), stats_df], axis=1)
        
        return df


class TargetVariableCreator:
    """目的変数作成クラス
    フェーズ3-18: 目的変数の最適化
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.target_config = self.config.get('model', {}).get('target', {})
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """最適化された目的変数を作成
        
        仕様: 「1着」および「1着とタイム差なし（0.0秒差）の2着」も正例として扱う
        
        Args:
            df: レースデータ
        
        Returns:
            目的変数が追加されたDataFrame
        """
        df = df.copy()
        
        include_close_second = self.target_config.get('include_close_second', True)
        time_diff_threshold = self.target_config.get('time_diff_threshold', 0.0)
        
        # 基本の目的変数（1着のみ）
        df['target_win'] = (df['rank'] == 1).astype(int)
        
        if include_close_second:
            # 拡張目的変数（惜敗馬を含む）
            df['target_extended'] = df['target_win'].copy()
            
            # 各レースごとに処理
            for race_id in df['race_id'].unique():
                race_mask = df['race_id'] == race_id
                race_data = df[race_mask]
                
                # 1着のタイム
                winner_time = race_data[race_data['rank'] == 1]['time_seconds'].values
                if len(winner_time) == 0:
                    continue
                
                winner_time = winner_time[0]
                
                # タイム差が閾値以下の2着馬を正例に
                second_place_mask = (
                    (df['race_id'] == race_id) &
                    (df['rank'] == 2) &
                    (df['time_seconds'] - winner_time <= time_diff_threshold)
                )
                df.loc[second_place_mask, 'target_extended'] = 1
        else:
            df['target_extended'] = df['target_win']
        
        return df


def preprocess_pipeline(
    race_raw_path: str,
    horse_raw_path: str,
    output_path: str,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """前処理パイプライン
    
    Args:
        race_raw_path: レース生データのパス
        horse_raw_path: 馬成績生データのパス
        output_path: 出力パス
        config: 設定辞書
    
    Returns:
        前処理済みDataFrame
    """
    config = config or load_config()
    
    # データ読み込み
    logger.info("データ読み込み...")
    race_df = pd.read_csv(race_raw_path)
    horse_df = pd.read_csv(horse_raw_path)
    
    # 前処理
    logger.info("前処理...")
    preprocessor = DataPreprocessor(config)
    df = preprocessor.preprocess_race_df(race_df)
    
    # 過去成績集計
    logger.info("過去成績集計...")
    aggregator = HistoricalDataAggregator(config)
    df = aggregator.aggregate_historical_stats(df, horse_df)
    
    # 条件別成績（設定で有効な場合）
    features_config = config.get('features', {}).get('conditional_aggregations', {})
    
    if features_config.get('distance', False):
        df = aggregator.aggregate_conditional_stats(df, horse_df, 'distance')
    
    if features_config.get('place', False):
        df = aggregator.aggregate_conditional_stats(df, horse_df, 'place')
    
    if features_config.get('race_type', False):
        df = aggregator.aggregate_conditional_stats(df, horse_df, 'race_type')
    
    # 目的変数作成
    logger.info("目的変数作成...")
    target_creator = TargetVariableCreator(config)
    df = target_creator.create_target(df)
    
    # 保存
    output_dir = ensure_dir(Path(output_path).parent)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"保存完了: {output_path}")
    
    return df
