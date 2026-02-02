"""
特徴量エンジニアリングモジュール
フェーズ3: 特徴量エンジニアリングと精度向上

- 相対値（標準化）特徴量
- リーディング情報
- 血統データ
- 季節性・交互作用特徴量
- 脚質データ
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


class FeatureCreator:
    """特徴量作成クラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
        self.features_config = self.config.get('features', {})
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を作成
        
        Args:
            df: 前処理済みDataFrame
        
        Returns:
            特徴量追加済みDataFrame
        """
        df = df.copy()
        
        # 1. 相対値特徴量
        logger.info("相対値特徴量作成...")
        df = self.create_relative_features(df)
        
        # 2. 季節性特徴量
        logger.info("季節性特徴量作成...")
        df = self.create_seasonality_features(df)
        
        # 3. 交互作用特徴量
        logger.info("交互作用特徴量作成...")
        df = self.create_interaction_features(df)
        
        # 4. 出走間隔特徴量
        logger.info("出走間隔特徴量作成...")
        df = self.create_interval_features(df)
        
        # 5. 枠順特徴量
        logger.info("枠順特徴量作成...")
        df = self.create_frame_features(df)
        
        return df
    
    def create_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """相対値（標準化）特徴量の作成
        フェーズ3-12: 相対値特徴量の導入
        
        計算式: (値 - レース内平均) / レース内標準偏差
        
        Args:
            df: DataFrame
        
        Returns:
            相対値特徴量追加済みDataFrame
        """
        df = df.copy()
        
        # 相対値化する対象カラム
        target_columns = [
            'win_odds', 'horse_weight_value', 'weight_carried',
            'hist_rank_mean', 'hist_prize_mean', 'hist_win_rate',
            'hist_place_rate', 'hist_last_3f_mean'
        ]
        
        for col in target_columns:
            if col not in df.columns:
                continue
            
            # レースごとに標準化
            relative_col = f'{col}_relative'
            
            df[relative_col] = df.groupby('race_id')[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        
        # 過去成績の相対値（全体での標準化も追加）
        hist_stats_cols = [c for c in df.columns if c.startswith('hist_') and not c.endswith('_relative')]
        
        for col in hist_stats_cols:
            if col in target_columns:
                continue
            if df[col].std() > 0:
                df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def create_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """季節性特徴量の作成
        フェーズ3-17: 季節性特徴量
        
        日付を円周上の値（sin/cos）に変換し、12月と1月の連続性を保つ
        
        Args:
            df: DataFrame
        
        Returns:
            季節性特徴量追加済みDataFrame
        """
        df = df.copy()
        
        if 'month' not in df.columns:
            if 'date' in df.columns:
                df['month'] = pd.to_datetime(df['date']).dt.month
            else:
                return df
        
        # 月をsin/cosに変換（円周上の位置）
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 日をsin/cosに変換
        if 'day' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # 季節カテゴリ（1-4）
        season_mapping = {
            12: 4, 1: 4, 2: 4,  # 冬
            3: 1, 4: 1, 5: 1,   # 春
            6: 2, 7: 2, 8: 2,   # 夏
            9: 3, 10: 3, 11: 3  # 秋
        }
        df['season'] = df['month'].map(season_mapping)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """交互作用特徴量の作成
        フェーズ3-17: 交互作用特徴量
        
        「夏は牝馬が強い」などの傾向を学習させる
        
        Args:
            df: DataFrame
        
        Returns:
            交互作用特徴量追加済みDataFrame
        """
        df = df.copy()
        
        # 季節 × 性別（夏牝傾向）
        if 'month_cos' in df.columns and 'sex_encoded' in df.columns:
            df['season_sex_interaction'] = df['month_cos'] * df['sex_encoded']
        
        # 季節 × 距離
        if 'month_sin' in df.columns and 'distance' in df.columns:
            df['season_distance_interaction'] = df['month_sin'] * (df['distance'] / 1000)
        
        # 枠番 × コース（内枠有利/外枠有利）
        if 'frame_number' in df.columns:
            # ダートは外枠有利の傾向
            if 'race_type_encoded' in df.columns:
                df['frame_course_interaction'] = df['frame_number'] * df['race_type_encoded']
            
            # 距離による枠順影響（短距離は内枠有利）
            if 'distance' in df.columns:
                df['frame_distance_interaction'] = df['frame_number'] * (df['distance'] / 1000)
        
        # 人気 × オッズ（穴馬検出）
        if 'popularity' in df.columns and 'win_odds' in df.columns:
            df['popularity_odds_ratio'] = df['win_odds'] / (df['popularity'] + 1)
        
        # 体重 × 距離（重い馬は長距離不利）
        if 'horse_weight_value' in df.columns and 'distance' in df.columns:
            df['weight_distance_interaction'] = (df['horse_weight_value'] / 500) * (df['distance'] / 1000)
        
        # 斤量 × 体重（斤量負担率）
        if 'weight_carried' in df.columns and 'horse_weight_value' in df.columns:
            df['weight_burden_rate'] = df['weight_carried'] / df['horse_weight_value']
        
        return df
    
    def create_interval_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """出走間隔特徴量の作成
        フェーズ3-17: 出走間隔
        
        前走からの経過日数を計算し、中3~4週のピーク等を学習させる
        
        Args:
            df: DataFrame
        
        Returns:
            出走間隔特徴量追加済みDataFrame
        """
        df = df.copy()
        
        # 出走間隔は前処理で計算済みの場合もある
        if 'days_since_last_race' not in df.columns:
            # ここでは単純にNaNを設定（別途計算が必要）
            df['days_since_last_race'] = np.nan
        
        # 出走間隔のカテゴリ化
        def categorize_interval(days):
            if pd.isna(days):
                return 0  # 初出走または不明
            elif days <= 14:
                return 1  # 連闘〜中1週
            elif days <= 28:
                return 2  # 中2〜3週（標準）
            elif days <= 56:
                return 3  # 中4〜7週
            elif days <= 180:
                return 4  # 中8週〜半年
            else:
                return 5  # 休み明け
        
        df['interval_category'] = df['days_since_last_race'].apply(categorize_interval)
        
        # 中3〜4週フラグ（ベストローテーション）
        df['is_best_rotation'] = (
            (df['days_since_last_race'] >= 21) & 
            (df['days_since_last_race'] <= 35)
        ).astype(int)
        
        # 出走間隔の対数（非線形効果）
        df['interval_log'] = np.log1p(df['days_since_last_race'].fillna(0))
        
        return df
    
    def create_frame_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """枠順特徴量の作成
        フェーズ3-17: 枠順傾向
        
        直線コースやダートにおける枠順の有利不利を反映
        
        Args:
            df: DataFrame
        
        Returns:
            枠順特徴量追加済みDataFrame
        """
        df = df.copy()
        
        if 'frame_number' not in df.columns:
            return df
        
        # 内枠/外枠フラグ
        df['is_inner_frame'] = (df['frame_number'] <= 3).astype(int)
        df['is_outer_frame'] = (df['frame_number'] >= 6).astype(int)
        
        # 馬番の相対位置（出走頭数に対する比率）
        if 'horse_number' in df.columns:
            # レースごとの頭数を計算
            race_horse_counts = df.groupby('race_id')['horse_number'].transform('max')
            df['horse_number_ratio'] = df['horse_number'] / race_horse_counts
        
        # 枠番の2乗（非線形効果）
        df['frame_squared'] = df['frame_number'] ** 2
        
        return df


class LeadingFeatureCreator:
    """リーディング情報特徴量作成クラス
    フェーズ3-14: リーディング情報の追加
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
    
    def add_leading_features(
        self, 
        df: pd.DataFrame,
        jockey_leading_df: Optional[pd.DataFrame] = None,
        trainer_leading_df: Optional[pd.DataFrame] = None,
        sire_leading_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """リーディング情報を追加
        
        リーク防止: 予測対象年の「前年」のリーディングデータを使用
        
        Args:
            df: 対象DataFrame
            jockey_leading_df: 騎手リーディングデータ
            trainer_leading_df: 調教師リーディングデータ
            sire_leading_df: 種牡馬リーディングデータ
        
        Returns:
            リーディング情報追加済みDataFrame
        """
        df = df.copy()
        
        # 年を取得
        if 'year' not in df.columns:
            if 'date' in df.columns:
                df['year'] = pd.to_datetime(df['date']).dt.year
            else:
                return df
        
        # 騎手リーディング
        if jockey_leading_df is not None:
            df = self._merge_leading(
                df, jockey_leading_df, 
                id_col='jockey_id', 
                prefix='jockey_leading'
            )
        
        # 調教師リーディング
        if trainer_leading_df is not None:
            df = self._merge_leading(
                df, trainer_leading_df,
                id_col='trainer_id',
                prefix='trainer_leading'
            )
        
        # 種牡馬リーディング
        if sire_leading_df is not None:
            df = self._merge_leading(
                df, sire_leading_df,
                id_col='sire_id',
                prefix='sire_leading'
            )
        
        return df
    
    def _merge_leading(
        self, 
        df: pd.DataFrame, 
        leading_df: pd.DataFrame,
        id_col: str,
        prefix: str
    ) -> pd.DataFrame:
        """リーディングデータをマージ（前年データを使用）"""
        if id_col not in df.columns:
            return df
        
        # 前年のリーディングを使用（リーク防止）
        df['leading_year'] = df['year'] - 1
        
        # リーディングデータのカラムをリネーム
        leading_cols = [c for c in leading_df.columns if c not in [id_col, 'year']]
        rename_dict = {c: f'{prefix}_{c}' for c in leading_cols}
        leading_df = leading_df.rename(columns=rename_dict)
        
        # マージ
        df = df.merge(
            leading_df,
            left_on=[id_col, 'leading_year'],
            right_on=[id_col, 'year'],
            how='left',
            suffixes=('', '_leading')
        )
        
        df = df.drop(columns=['leading_year'], errors='ignore')
        
        return df


class PedigreeFeatureCreator:
    """血統特徴量作成クラス
    フェーズ3-16: 血統データの活用
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
    
    def add_pedigree_features(
        self,
        df: pd.DataFrame,
        pedigree_df: pd.DataFrame,
        sire_stats_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """血統特徴量を追加
        
        Args:
            df: 対象DataFrame
            pedigree_df: 血統データ（horse_id, sire_id, bms_id）
            sire_stats_df: 種牡馬成績データ
        
        Returns:
            血統特徴量追加済みDataFrame
        """
        df = df.copy()
        
        # 血統情報をマージ
        if 'horse_id' in df.columns and 'horse_id' in pedigree_df.columns:
            pedigree_cols = ['horse_id', 'sire_id', 'bms_id']
            pedigree_subset = pedigree_df[
                [c for c in pedigree_cols if c in pedigree_df.columns]
            ].drop_duplicates()
            
            df = df.merge(pedigree_subset, on='horse_id', how='left')
        
        # 種牡馬成績をマージ
        if sire_stats_df is not None and 'sire_id' in df.columns:
            # 芝/ダート別成績
            sire_cols = [c for c in sire_stats_df.columns if c != 'sire_id']
            rename_dict = {c: f'sire_{c}' for c in sire_cols}
            sire_stats_renamed = sire_stats_df.rename(columns=rename_dict)
            
            df = df.merge(sire_stats_renamed, on='sire_id', how='left')
        
        return df


class RunningStyleFeatureCreator:
    """脚質特徴量作成クラス
    フェーズ4-24: 脚質データの活用
    """
    
    RUNNING_STYLES = ['逃げ', '先行', '差し', '追込']
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
    
    def add_running_style_features(
        self,
        df: pd.DataFrame,
        running_style_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """脚質特徴量を追加
        
        Args:
            df: 対象DataFrame
            running_style_df: 脚質データ（horse_id, running_style）
        
        Returns:
            脚質特徴量追加済みDataFrame
        """
        df = df.copy()
        
        if running_style_df is not None and 'horse_id' in df.columns:
            df = df.merge(
                running_style_df[['horse_id', 'running_style']],
                on='horse_id',
                how='left'
            )
        
        # 脚質のエンコーディング
        if 'running_style' in df.columns:
            style_mapping = {style: i for i, style in enumerate(self.RUNNING_STYLES)}
            df['running_style_encoded'] = df['running_style'].map(style_mapping)
        
        # 脚質と距離の交互作用
        if 'running_style_encoded' in df.columns and 'distance' in df.columns:
            # 長距離は差し/追込有利
            df['style_distance_interaction'] = (
                df['running_style_encoded'] * (df['distance'] / 1000)
            )
        
        # 脚質とコースの交互作用
        if 'running_style_encoded' in df.columns and 'course_direction_encoded' in df.columns:
            df['style_course_interaction'] = (
                df['running_style_encoded'] * df['course_direction_encoded']
            )
        
        return df
    
    def estimate_running_style(self, horse_history_df: pd.DataFrame) -> pd.DataFrame:
        """過去成績から脚質を推定
        
        Args:
            horse_history_df: 馬の過去成績
        
        Returns:
            脚質推定結果（horse_id, running_style）
        """
        results = []
        
        for horse_id in horse_history_df['horse_id'].unique():
            horse_data = horse_history_df[horse_history_df['horse_id'] == horse_id]
            
            # コーナー通過順位などから推定（データがある場合）
            # ここでは簡易的に着順と上がり3Fから推定
            if len(horse_data) == 0:
                continue
            
            avg_last_3f_rank = np.nan
            if 'last_3f' in horse_data.columns:
                # 上がり3Fが速い = 差し/追込傾向
                avg_last_3f = horse_data['last_3f'].mean()
                if pd.notna(avg_last_3f):
                    if avg_last_3f < 34:
                        style = '追込'
                    elif avg_last_3f < 35:
                        style = '差し'
                    elif avg_last_3f < 36:
                        style = '先行'
                    else:
                        style = '逃げ'
                else:
                    style = '先行'  # デフォルト
            else:
                style = '先行'
            
            results.append({
                'horse_id': horse_id,
                'running_style': style
            })
        
        return pd.DataFrame(results)


def create_features_pipeline(
    preprocessed_path: str,
    output_path: str,
    leading_data_paths: Optional[Dict[str, str]] = None,
    pedigree_path: Optional[str] = None,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """特徴量作成パイプライン
    
    Args:
        preprocessed_path: 前処理済みデータのパス
        output_path: 出力パス
        leading_data_paths: リーディングデータのパス辞書
        pedigree_path: 血統データのパス
        config: 設定辞書
    
    Returns:
        特徴量追加済みDataFrame
    """
    config = config or load_config()
    
    # データ読み込み
    logger.info("データ読み込み...")
    df = pd.read_csv(preprocessed_path)
    
    # 基本特徴量作成
    feature_creator = FeatureCreator(config)
    df = feature_creator.create_all_features(df)
    
    # リーディング情報
    if leading_data_paths:
        leading_creator = LeadingFeatureCreator(config)
        
        jockey_df = None
        trainer_df = None
        sire_df = None
        
        if 'jockey' in leading_data_paths:
            jockey_df = pd.read_csv(leading_data_paths['jockey'])
        if 'trainer' in leading_data_paths:
            trainer_df = pd.read_csv(leading_data_paths['trainer'])
        if 'sire' in leading_data_paths:
            sire_df = pd.read_csv(leading_data_paths['sire'])
        
        df = leading_creator.add_leading_features(
            df, jockey_df, trainer_df, sire_df
        )
    
    # 血統情報
    if pedigree_path:
        pedigree_creator = PedigreeFeatureCreator(config)
        pedigree_df = pd.read_csv(pedigree_path)
        df = pedigree_creator.add_pedigree_features(df, pedigree_df)
    
    # 保存
    output_dir = ensure_dir(Path(output_path).parent)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"保存完了: {output_path}")
    
    return df
