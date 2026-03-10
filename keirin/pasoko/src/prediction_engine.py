# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」- 予測エンジン・独自指標算出モジュール
Prediction Engine & Custom Indicators Module for Keirin Prediction AI "Pasoko"
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from config import OUTPUT_DIR


class PredictionEngine:
    """
    予測エンジン
    
    独自指標（A率、KS値、CT値）を算出し、
    レースの傾向を判定する
    """
    
    def __init__(self):
        self.results = None
        
    def calculate_indicators(self, df, race_id_column='race_id'):
        """
        独自指標を算出
        
        Args:
            df: ランク付け済みの予測結果DataFrame
            race_id_column: レースIDカラム
            
        Returns:
            DataFrame: 指標を追加したDataFrame
        """
        df = df.copy()
        
        # 初期化
        df['a_rate'] = 0.0
        df['b_rate'] = 0.0
        df['c_rate'] = 0.0
        df['ks_value'] = 0.0
        df['ct_value'] = 0.0
        
        result_dfs = []
        
        for race_id, group in tqdm(df.groupby(race_id_column), desc="指標計算"):
            group = group.copy()
            
            # 各ランクの確率を取得
            a_row = group[group['rank_label'] == 'A']
            b_row = group[group['rank_label'] == 'B']
            c_row = group[group['rank_label'] == 'C']
            
            a_rate = a_row['pred_proba'].values[0] if len(a_row) > 0 else 0
            b_rate = b_row['pred_proba'].values[0] if len(b_row) > 0 else 0
            c_rate = c_row['pred_proba'].values[0] if len(c_row) > 0 else 0
            
            # ラベルが0の選手は確率をマイナスとして扱う
            a_label = a_row['pred_label'].values[0] if len(a_row) > 0 else 0
            b_label = b_row['pred_label'].values[0] if len(b_row) > 0 else 0
            c_label = c_row['pred_label'].values[0] if len(c_row) > 0 else 0
            
            a_signed = a_rate if a_label == 1 else -a_rate
            b_signed = b_rate if b_label == 1 else -b_rate
            c_signed = c_rate if c_label == 1 else -c_rate
            
            # KS値（B率 - C率）
            ks_value = b_rate - c_rate
            
            # CT値（カラータイマー値）
            # A, B, C全てがプラスでないと高くならない
            if a_label == 1 and b_label == 1 and c_label == 1:
                ct_value = (a_rate + b_rate + c_rate) / 3
            elif c_label == 0:
                # C率がマイナスの場合は大幅に下がる
                ct_value = max(0, (a_rate + b_rate - c_rate) / 3 - 0.2)
            else:
                ct_value = (a_rate + b_rate + c_signed) / 3
            
            # グループ全体に指標を設定
            group['a_rate'] = a_rate
            group['b_rate'] = b_rate
            group['c_rate'] = c_rate
            group['ks_value'] = ks_value
            group['ct_value'] = ct_value
            
            result_dfs.append(group)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def get_top3_prediction(self, df, race_id_column='race_id'):
        """
        レースごとの上位3名（A, B, C）を取得
        
        Args:
            df: 予測結果DataFrame
            race_id_column: レースIDカラム
            
        Returns:
            DataFrame: 上位3名サマリー
        """
        summaries = []
        
        for race_id, group in df.groupby(race_id_column):
            # 上位3名を抽出
            top3 = group[group['rank_label'].isin(['A', 'B', 'C'])]
            
            summary = {
                'race_id': race_id,
                'A_車番': top3[top3['rank_label'] == 'A']['車番'].values[0] if len(top3[top3['rank_label'] == 'A']) > 0 else '',
                'B_車番': top3[top3['rank_label'] == 'B']['車番'].values[0] if len(top3[top3['rank_label'] == 'B']) > 0 else '',
                'C_車番': top3[top3['rank_label'] == 'C']['車番'].values[0] if len(top3[top3['rank_label'] == 'C']) > 0 else '',
                'A率': group['a_rate'].iloc[0],
                'B率': group['b_rate'].iloc[0],
                'C率': group['c_rate'].iloc[0],
                'KS値': group['ks_value'].iloc[0],
                'CT値': group['ct_value'].iloc[0]
            }
            
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def interpret_indicators(self, a_rate, ks_value, ct_value):
        """
        指標の解釈
        
        Args:
            a_rate: A率
            ks_value: KS値
            ct_value: CT値
            
        Returns:
            dict: 解釈結果
        """
        interpretation = {
            'confidence': '',
            'pattern': '',
            'description': ''
        }
        
        # A率の解釈
        if a_rate >= 0.9:
            interpretation['confidence'] = '超高信頼'
            interpretation['description'] = 'Aの1着率約75%、3着内率約95%'
        elif a_rate >= 0.8:
            interpretation['confidence'] = '高信頼'
        elif a_rate >= 0.6:
            interpretation['confidence'] = '中程度'
        else:
            interpretation['confidence'] = '低信頼'
        
        # KS値の解釈
        if ks_value >= 1.3:
            interpretation['pattern'] += 'AとBが突出 '
        elif ks_value >= 1.0:
            interpretation['pattern'] += 'ABが優勢 '
        
        # CT値の解釈
        if ct_value >= 0.8:
            interpretation['pattern'] += '本命サイド(ABC決着濃厚)'
        elif ct_value >= 0.5:
            interpretation['pattern'] += '中間(やや波乱含み)'
        else:
            interpretation['pattern'] += '波乱含み(荒れる可能性大)'
        
        return interpretation
    
    def analyze_race(self, df, race_id):
        """
        特定レースの詳細分析
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            
        Returns:
            dict: 分析結果
        """
        race_df = df[df['race_id'] == race_id]
        
        if len(race_df) == 0:
            return None
        
        # 各選手の情報
        players = []
        for _, row in race_df.sort_values('pred_score', ascending=False).iterrows():
            player = {
                'rank': row['rank_label'],
                'number': row.get('車番', ''),
                'name': row.get('選手名', ''),
                'proba': row['pred_proba'],
                'score': row['pred_score'],
                'label': '○' if row['pred_label'] == 1 else '×'
            }
            players.append(player)
        
        # 指標
        a_rate = race_df['a_rate'].iloc[0]
        ks_value = race_df['ks_value'].iloc[0]
        ct_value = race_df['ct_value'].iloc[0]
        
        interpretation = self.interpret_indicators(a_rate, ks_value, ct_value)
        
        return {
            'race_id': race_id,
            'players': players,
            'indicators': {
                'a_rate': a_rate,
                'ks_value': ks_value,
                'ct_value': ct_value
            },
            'interpretation': interpretation
        }
    
    def print_race_analysis(self, analysis):
        """分析結果を表示"""
        if analysis is None:
            print("レースが見つかりません")
            return
        
        print(f"\n===== レース分析: {analysis['race_id']} =====")
        print("\n【予測順位】")
        for p in analysis['players']:
            print(f"  {p['rank']}: {p['number']}番 {p['name']} "
                  f"[確率: {p['proba']:.3f}, スコア: {p['score']:.3f}, 3着内: {p['label']}]")
        
        print("\n【独自指標】")
        ind = analysis['indicators']
        print(f"  A率: {ind['a_rate']:.3f}")
        print(f"  KS値(B-C): {ind['ks_value']:.3f}")
        print(f"  CT値: {ind['ct_value']:.3f}")
        
        print("\n【解釈】")
        interp = analysis['interpretation']
        print(f"  信頼度: {interp['confidence']}")
        print(f"  パターン: {interp['pattern']}")
        if interp['description']:
            print(f"  補足: {interp['description']}")
    
    def save_predictions(self, df, filename='predictions.csv'):
        """予測結果を保存"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = f"{OUTPUT_DIR}/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"予測結果保存: {filepath}")
    
    def save_summary(self, summary_df, filename='summary.csv'):
        """サマリーを保存"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = f"{OUTPUT_DIR}/{filename}"
        summary_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"サマリー保存: {filepath}")


class HistoricalAnalyzer:
    """
    過去データの分析クラス
    
    A率と実際の結果の相関を分析し、
    指標の有効性を検証する
    """
    
    def __init__(self):
        pass
    
    def analyze_a_rate_accuracy(self, df):
        """
        A率と実際の1着率・3着内率の関係を分析
        
        Args:
            df: 予測結果と実際の結果を含むDataFrame
            
        Returns:
            DataFrame: A率別の統計
        """
        if 'target' not in df.columns:
            print("実際の結果データがありません")
            return None
        
        # A評価の選手のみ抽出
        a_players = df[df['rank_label'] == 'A'].copy()
        
        # A率でビニング
        bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        labels = ['0-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        a_players['a_rate_bin'] = pd.cut(a_players['a_rate'], bins=bins, labels=labels)
        
        # 統計計算
        stats = a_players.groupby('a_rate_bin').agg({
            'target': ['count', 'sum', 'mean']
        }).round(3)
        
        stats.columns = ['レース数', '3着内回数', '3着内率']
        
        # 1着率も計算（もしランクデータがあれば）
        if 'rank' in a_players.columns:
            first_rate = a_players.groupby('a_rate_bin').apply(
                lambda x: (x['rank'] == 1).sum() / len(x)
            )
            stats['1着率'] = first_rate
        
        print("\n===== A率別の実績 =====")
        print(stats)
        
        return stats
    
    def analyze_ct_value_pattern(self, df):
        """
        CT値とレース結果パターンの分析
        
        Args:
            df: 予測結果と実際の結果を含むDataFrame
            
        Returns:
            DataFrame: CT値別の統計
        """
        # レースごとにABC的中判定
        race_results = []
        
        for race_id, group in df.groupby('race_id'):
            top3 = group[group['rank_label'].isin(['A', 'B', 'C'])]
            
            if len(top3) < 3:
                continue
            
            # ABC全員が3着以内か
            abc_hit = top3['target'].sum() == 3
            
            ct_value = group['ct_value'].iloc[0]
            
            race_results.append({
                'race_id': race_id,
                'ct_value': ct_value,
                'abc_hit': abc_hit
            })
        
        results_df = pd.DataFrame(race_results)
        
        # CT値でビニング
        bins = [0, 0.3, 0.5, 0.6, 0.75, 0.8, 1.0]
        labels = ['0-30%', '30-50%', '50-60%', '60-75%', '75-80%', '80-100%']
        results_df['ct_bin'] = pd.cut(results_df['ct_value'], bins=bins, labels=labels)
        
        stats = results_df.groupby('ct_bin').agg({
            'abc_hit': ['count', 'sum', 'mean']
        }).round(3)
        
        stats.columns = ['レース数', 'ABC的中数', 'ABC的中率']
        
        print("\n===== CT値別のABC的中率 =====")
        print(stats)
        
        return stats


if __name__ == "__main__":
    # テスト
    print("予測エンジンテスト")
    
    # ダミーデータ作成
    np.random.seed(42)
    
    test_data = []
    for race in range(5):
        race_id = f'race_{race}'
        for player in range(7):
            proba = np.random.uniform(0.2, 0.9)
            label = 1 if proba > 0.5 else 0
            score = proba if label == 1 else -proba
            
            test_data.append({
                'race_id': race_id,
                '車番': player + 1,
                '選手名': f'選手{player + 1}',
                'pred_proba': proba,
                'pred_label': label,
                'pred_score': score,
                'rank_label': ['A', 'B', 'C', 'D', 'E', 'F', 'G'][player],
                'target': np.random.randint(0, 2)
            })
    
    df = pd.DataFrame(test_data)
    
    engine = PredictionEngine()
    
    # 指標計算
    df = engine.calculate_indicators(df)
    
    # サマリー
    summary = engine.get_top3_prediction(df)
    print("\n予測サマリー:")
    print(summary)
    
    # レース分析
    analysis = engine.analyze_race(df, 'race_0')
    engine.print_race_analysis(analysis)
