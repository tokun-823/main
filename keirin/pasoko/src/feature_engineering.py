# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」- 特徴量エンジニアリングモジュール
Feature Engineering Module for Keirin Prediction AI "Pasoko"
"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import pickle
import os

from config import (
    BANK_333, BANK_400, BANK_500,
    EXCLUDE_STATUSES, EXCLUDE_RACE_TYPES,
    DATA_DIR, PROCESSED_DATA_FILE
)


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""
    
    def __init__(self):
        self.bank_mapping = self._create_bank_mapping()
        
    def _create_bank_mapping(self):
        """バンク周長マッピングを作成"""
        mapping = {}
        for venue in BANK_333:
            mapping[venue] = 333
        for venue in BANK_400:
            mapping[venue] = 400
        for venue in BANK_500:
            mapping[venue] = 500
        return mapping
    
    def get_bank_length(self, venue_name):
        """
        競輪場名からバンク周長を取得
        
        Args:
            venue_name: 競輪場名
            
        Returns:
            int: バンク周長 (333, 400, 500)
        """
        for key in self.bank_mapping:
            if key in venue_name:
                return self.bank_mapping[key]
        return 400  # デフォルト
    
    def create_bank_features(self, df, venue_column='venue'):
        """
        バンク特徴量を追加
        
        Args:
            df: DataFrame
            venue_column: 競輪場名のカラム名
            
        Returns:
            DataFrame: バンク特徴量を追加したDataFrame
        """
        df = df.copy()
        
        # バンク周長を取得
        df['bank_length'] = df[venue_column].apply(self.get_bank_length)
        
        # ワンホットエンコーディング
        df['bank_333'] = (df['bank_length'] == 333).astype(int)
        df['bank_400'] = (df['bank_length'] == 400).astype(int)
        df['bank_500'] = (df['bank_length'] == 500).astype(int)
        
        return df
    
    def calculate_relative_stats(self, df, race_id_column='race_id'):
        """
        レース内での相対的ステータス（差分）を算出
        
        Args:
            df: DataFrame（選手データ）
            race_id_column: レースIDのカラム名
            
        Returns:
            DataFrame: 相対ステータスを追加したDataFrame
        """
        df = df.copy()
        
        # 数値カラムを特定
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # バック回数と逃げの特徴量
        back_cols = [c for c in df.columns if 'B' in c or 'バック' in c]
        escape_cols = [c for c in df.columns if '逃' in c]
        other_cols = [c for c in df.columns if '捲' in c or '差' in c or 'マ' in c]
        
        # レースごとにグループ化して処理
        result_dfs = []
        
        for race_id, group in tqdm(df.groupby(race_id_column), desc="相対ステータス計算"):
            group = group.copy()
            
            # バック回数と逃げ: 3種類の数値を算出
            for col in back_cols + escape_cols:
                if col in group.columns:
                    try:
                        values = pd.to_numeric(group[col], errors='coerce').fillna(0)
                        max_val = values.max()
                        
                        # 最大値との差分
                        group[f'{col}_sa'] = max_val - values
                        
                        # 上位2名の差
                        sorted_vals = values.sort_values(ascending=False)
                        if len(sorted_vals) >= 2:
                            top2_diff = sorted_vals.iloc[0] - sorted_vals.iloc[1]
                        else:
                            top2_diff = 0
                        group[f'{col}_sa2'] = top2_diff
                        
                        # 10回以上の選手数
                        group[f'{col}_suu'] = (values >= 10).sum()
                        
                    except Exception:
                        continue
            
            # その他の項目: 最大値との差分のみ
            for col in other_cols:
                if col in group.columns:
                    try:
                        values = pd.to_numeric(group[col], errors='coerce').fillna(0)
                        max_val = values.max()
                        group[f'{col}_sa'] = max_val - values
                    except Exception:
                        continue
            
            result_dfs.append(group)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def extract_line_composition(self, df, race_id_column='race_id'):
        """
        ライン構成を抽出
        
        Args:
            df: DataFrame
            race_id_column: レースIDのカラム名
            
        Returns:
            DataFrame: ライン構成特徴量を追加したDataFrame
        """
        df = df.copy()
        
        # 初期化
        df['car_count'] = 0  # 車立
        df['line_count'] = 0  # ライン数
        df['line_composition'] = ''  # ライン構成
        df['line_head'] = 0  # ライン先頭の車番
        df['position_in_line'] = 0  # 番手
        
        result_dfs = []
        
        for race_id, group in tqdm(df.groupby(race_id_column), desc="ライン構成抽出"):
            group = group.copy()
            n_players = len(group)
            
            # 車立
            group['car_count'] = n_players
            
            # ライン構成の推定（出走表のデータから）
            # 実際のデータではライン情報がある場合はそれを使用
            if 'line_info' in group.columns:
                line_info = group['line_info'].iloc[0]
                lines = self._parse_line_info(line_info)
            else:
                # ライン構成がない場合はデフォルト推定
                lines = self._estimate_lines(group)
            
            # ライン数
            group['line_count'] = len(lines)
            
            # ライン構成文字列
            line_str = '-'.join([str(len(l)) for l in lines])
            group['line_composition'] = line_str
            
            # 各選手のライン情報を設定
            for line_idx, line in enumerate(lines):
                for pos_idx, player_num in enumerate(line):
                    mask = group['車番'] == player_num if '車番' in group.columns else pd.Series([False] * len(group))
                    if mask.any():
                        group.loc[mask, 'line_head'] = line[0]
                        group.loc[mask, 'position_in_line'] = pos_idx + 1
            
            # 単騎の選手は番手1に修正
            group.loc[group['position_in_line'] == 0, 'position_in_line'] = 1
            
            # 番手は最大5まで
            group['position_in_line'] = group['position_in_line'].clip(upper=5)
            
            result_dfs.append(group)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _parse_line_info(self, line_info):
        """ライン情報文字列をパース"""
        if not line_info or pd.isna(line_info):
            return [[1], [2], [3], [4], [5], [6], [7]]
        
        lines = []
        # "3-2-2" のような形式をパース
        parts = str(line_info).split('-')
        current_num = 1
        
        for part in parts:
            try:
                count = int(part)
                line = list(range(current_num, current_num + count))
                lines.append(line)
                current_num += count
            except ValueError:
                continue
        
        return lines if lines else [[1], [2], [3], [4], [5], [6], [7]]
    
    def _estimate_lines(self, group):
        """ライン構成を推定（デフォルト）"""
        n = len(group)
        if n == 7:
            return [[1, 2], [3, 4], [5, 6, 7]]  # 典型的な3分戦
        elif n == 9:
            return [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        else:
            # 単騎として扱う
            return [[i] for i in range(1, n + 1)]
    
    def calculate_line_strength(self, df, race_id_column='race_id'):
        """
        ライン強度を算出（独自指標）
        
        ライン先頭の選手に対し、バック回数が多い順に記号（a, b, c...）を付与し、
        それに番手の数字を結合して「a1」「b2」「c3」のような強度フラグを生成
        
        Args:
            df: DataFrame
            race_id_column: レースIDのカラム名
            
        Returns:
            DataFrame: ライン強度を追加したDataFrame
        """
        df = df.copy()
        df['line_strength'] = ''
        df['line_strength_code'] = ''
        
        result_dfs = []
        
        for race_id, group in tqdm(df.groupby(race_id_column), desc="ライン強度計算"):
            group = group.copy()
            
            # バック回数と得点でソート用の列を作成
            back_col = None
            for col in group.columns:
                if 'B' in col or 'バック' in col:
                    back_col = col
                    break
            
            score_col = None
            for col in group.columns:
                if '得点' in col or 'score' in col.lower():
                    score_col = col
                    break
            
            if back_col and 'position_in_line' in group.columns:
                # 番手（昇順）、バック回数（降順）、得点（昇順）で並べ替え
                group['_back_numeric'] = pd.to_numeric(group[back_col], errors='coerce').fillna(0)
                group['_score_numeric'] = pd.to_numeric(group[score_col], errors='coerce').fillna(0) if score_col else 0
                
                # 得点順位を計算
                group['_score_rank'] = group['_score_numeric'].rank(ascending=False)
                
                # ライン先頭の選手を抽出
                line_heads = group[group['position_in_line'] == 1].copy()
                
                if len(line_heads) > 0:
                    # バック回数降順でソート
                    line_heads = line_heads.sort_values('_back_numeric', ascending=False)
                    
                    # 記号を付与
                    symbols = 'abcdefghij'
                    for idx, (row_idx, row) in enumerate(line_heads.iterrows()):
                        if idx < len(symbols):
                            symbol = symbols[idx]
                            group.loc[row_idx, 'line_strength_code'] = f"{symbol}1"
                
                # 番手選手の強度コードを設定
                for line_head in group['line_head'].unique():
                    if line_head > 0:
                        head_mask = (group['車番'] == line_head) if '車番' in group.columns else pd.Series([False] * len(group))
                        if head_mask.any():
                            head_code = group.loc[head_mask, 'line_strength_code'].iloc[0]
                            if head_code:
                                symbol = head_code[0]
                                line_mask = group['line_head'] == line_head
                                for idx, row in group[line_mask].iterrows():
                                    pos = row['position_in_line']
                                    group.loc[idx, 'line_strength_code'] = f"{symbol}{int(pos)}"
                
                # 一時列を削除
                group = group.drop(columns=['_back_numeric', '_score_numeric', '_score_rank'], errors='ignore')
            
            result_dfs.append(group)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def create_flag4(self, df, race_id_column='race_id'):
        """
        フラグ4を作成
        
        ライン先頭の選手において、「競走得点」と「バック回数」が
        両方とも出走メンバー中で1位である場合に「1」、それ以外は「0」
        
        Args:
            df: DataFrame
            race_id_column: レースIDのカラム名
            
        Returns:
            DataFrame: フラグ4を追加したDataFrame
        """
        df = df.copy()
        df['flag4'] = 0
        
        result_dfs = []
        
        for race_id, group in tqdm(df.groupby(race_id_column), desc="フラグ4計算"):
            group = group.copy()
            
            # 得点とバック回数のカラムを特定
            score_col = None
            back_col = None
            
            for col in group.columns:
                if '得点' in col or 'score' in col.lower():
                    score_col = col
                if 'B' in col or 'バック' in col:
                    back_col = col
            
            if score_col and back_col and 'position_in_line' in group.columns:
                # 数値化
                group['_score'] = pd.to_numeric(group[score_col], errors='coerce').fillna(0)
                group['_back'] = pd.to_numeric(group[back_col], errors='coerce').fillna(0)
                
                # ライン先頭の選手
                line_heads = group[group['position_in_line'] == 1]
                
                if len(line_heads) > 0:
                    # 最大値を取得
                    max_score = group['_score'].max()
                    max_back = group['_back'].max()
                    
                    # 両方1位の選手にフラグを立てる
                    for idx, row in line_heads.iterrows():
                        if row['_score'] == max_score and row['_back'] == max_back:
                            group.loc[idx, 'flag4'] = 1
                
                group = group.drop(columns=['_score', '_back'], errors='ignore')
            
            result_dfs.append(group)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def create_target_variable(self, df, result_column='着順'):
        """
        目的変数を作成（3着以内: 1, それ以外: 0）
        
        Args:
            df: DataFrame
            result_column: 着順のカラム名
            
        Returns:
            DataFrame: 目的変数を追加したDataFrame
        """
        df = df.copy()
        
        if result_column in df.columns:
            # 着順を数値化
            df['rank'] = pd.to_numeric(df[result_column], errors='coerce')
            
            # 3着以内かどうか
            df['target'] = (df['rank'] <= 3).astype(int)
            
            # 欠損値は0
            df['target'] = df['target'].fillna(0).astype(int)
        else:
            df['target'] = 0
        
        return df
    
    def filter_data(self, df, race_name_column='race_name', status_column='着順'):
        """
        学習データのフィルタリング
        
        - 落車・失格のレースを除外
        - 新人戦を除外
        
        Args:
            df: DataFrame
            race_name_column: レース名のカラム名
            status_column: 着順/ステータスのカラム名
            
        Returns:
            DataFrame: フィルタリング済みDataFrame
        """
        df = df.copy()
        original_len = len(df)
        
        # 除外ステータスの選手を含むレースを除外
        if status_column in df.columns:
            df['_has_exclude'] = df[status_column].astype(str).apply(
                lambda x: any(status in x for status in EXCLUDE_STATUSES)
            )
            
            # レースIDごとに除外フラグを集計
            if 'race_id' in df.columns:
                race_exclude = df.groupby('race_id')['_has_exclude'].any()
                exclude_races = race_exclude[race_exclude].index
                df = df[~df['race_id'].isin(exclude_races)]
            
            df = df.drop(columns=['_has_exclude'], errors='ignore')
        
        # 新人戦を除外
        if race_name_column in df.columns:
            df = df[~df[race_name_column].str.contains('|'.join(EXCLUDE_RACE_TYPES), na=False)]
        
        print(f"データフィルタリング: {original_len} → {len(df)} 件")
        
        return df
    
    def categorize_race(self, df, race_name_column='race_name', grade_column='grade'):
        """
        レースカテゴリを分類
        
        - 7車立て / 9車立て / イレギュラー
        - ガールズケイリン / チャレンジ
        - G3 / G1
        
        Args:
            df: DataFrame
            race_name_column: レース名のカラム名
            grade_column: グレードのカラム名
            
        Returns:
            DataFrame: カテゴリを追加したDataFrame
        """
        df = df.copy()
        df['category'] = 'general'
        
        # 車立てでカテゴリ分け
        if 'car_count' in df.columns:
            df.loc[df['car_count'] == 7, 'category'] = '7car'
            df.loc[df['car_count'] == 9, 'category'] = '9car'
            df.loc[df['car_count'].isin([5, 6, 8]), 'category'] = 'irregular'
        
        # レース名でカテゴリ上書き
        if race_name_column in df.columns:
            df.loc[df[race_name_column].str.contains('ガールズ', na=False), 'category'] = 'girls'
            df.loc[df[race_name_column].str.contains('チャレンジ', na=False), 'category'] = 'challenge'
        
        # グレードでカテゴリ上書き
        if grade_column in df.columns:
            df.loc[df[grade_column].str.contains('G3|記念', na=False), 'category'] = 'g3'
            df.loc[df[grade_column].str.contains('G1|GI', na=False), 'category'] = 'g1'
        
        return df
    
    def process_all(self, df_card, df_info, df_return):
        """
        全ての特徴量エンジニアリングを実行
        
        Args:
            df_card: 出走表DataFrame
            df_info: レース情報DataFrame
            df_return: 払戻情報DataFrame
            
        Returns:
            DataFrame: 全特徴量を含むDataFrame
        """
        print("特徴量エンジニアリング開始...")
        
        # データ結合
        df = df_card.copy()
        
        if 'race_id' in df.columns and 'race_id' in df_info.columns:
            df = df.merge(df_info, on='race_id', how='left', suffixes=('', '_info'))
        
        if 'race_id' in df.columns and 'race_id' in df_return.columns:
            df = df.merge(df_return, on='race_id', how='left', suffixes=('', '_return'))
        
        # 各特徴量の追加
        print("1. バンク特徴量追加...")
        if 'venue' in df.columns:
            df = self.create_bank_features(df)
        
        print("2. ライン構成抽出...")
        df = self.extract_line_composition(df)
        
        print("3. 相対ステータス計算...")
        df = self.calculate_relative_stats(df)
        
        print("4. ライン強度計算...")
        df = self.calculate_line_strength(df)
        
        print("5. フラグ4作成...")
        df = self.create_flag4(df)
        
        print("6. カテゴリ分類...")
        df = self.categorize_race(df)
        
        print("7. 目的変数作成...")
        df = self.create_target_variable(df)
        
        print("8. データフィルタリング...")
        df = self.filter_data(df)
        
        print(f"特徴量エンジニアリング完了: {len(df)} 件")
        
        return df
    
    def save_processed_data(self, df):
        """処理済みデータを保存"""
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_pickle(PROCESSED_DATA_FILE)
        print(f"処理済みデータ保存: {PROCESSED_DATA_FILE}")
    
    def load_processed_data(self):
        """処理済みデータを読み込み"""
        if os.path.exists(PROCESSED_DATA_FILE):
            return pd.read_pickle(PROCESSED_DATA_FILE)
        return pd.DataFrame()
    
    def get_feature_columns(self, df):
        """
        学習に使用する特徴量カラムを取得
        
        注意: オッズ・人気は完全に排除
        
        Args:
            df: DataFrame
            
        Returns:
            list: 特徴量カラム名のリスト
        """
        # 除外するカラム
        exclude_patterns = [
            'race_id', 'target', 'rank', 'category',
            'オッズ', '人気', 'odds', 'popularity',  # 完全排除
            '選手名', '名前', 'name',
            '_info', '_return', '_temp'
        ]
        
        feature_cols = []
        
        for col in df.columns:
            # 除外パターンに該当するかチェック
            if any(pattern in col.lower() for pattern in exclude_patterns):
                continue
            
            # 数値カラムのみ
            if df[col].dtype in [np.int64, np.float64, np.int32, np.float32, int, float]:
                feature_cols.append(col)
        
        return feature_cols


if __name__ == "__main__":
    # テスト
    engineer = FeatureEngineer()
    
    # バンク周長テスト
    print("バンク周長テスト:")
    print(f"前橋: {engineer.get_bank_length('前橋')}")
    print(f"立川: {engineer.get_bank_length('立川')}")
    print(f"高知: {engineer.get_bank_length('高知')}")
