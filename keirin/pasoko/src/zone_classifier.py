# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」- ゾーン分類・車券戦略モジュール
Zone Classification & Betting Strategy Module for Keirin Prediction AI "Pasoko"
"""

import pandas as pd
import numpy as np
from itertools import combinations, permutations
from tqdm import tqdm

from config import ZONE_THRESHOLDS, TREASURE_CONDITION_A, TREASURE_CONDITION_B


class ZoneClassifier:
    """
    ゾーン分類クラス
    
    指標を用いてレースを4つのゾーンに分類し、
    それぞれに適した車券戦略を提案する
    """
    
    def __init__(self):
        self.zone_strategies = {
            'gachi': {
                'name': 'ガチゾーン',
                'description': 'A率が非常に高く、CT値も高い。ABC決着が濃厚',
                'strategy': '3連複1点（ABC）',
                'bet_type': 'trifecta_narrow'
            },
            'blue': {
                'name': 'ブルーゾーン',
                'description': '本命サイド。安定した展開が期待できる',
                'strategy': '3連複2点（ABC, ABD）',
                'bet_type': 'trifecta_standard'
            },
            'twilight': {
                'name': 'トワイライトゾーン',
                'description': '中間域。やや波乱含み',
                'strategy': '3連複3点（ABC, ABD, ACD）',
                'bet_type': 'trifecta_spread'
            },
            'red': {
                'name': 'レッドゾーン',
                'description': '大穴狙い。荒れる可能性大',
                'strategy': 'A軸全流し or efg絡み',
                'bet_type': 'exacta_spread'
            }
        }
    
    def classify_zone(self, a_rate, ct_value, ks_value=None):
        """
        レースをゾーン分類
        
        Args:
            a_rate: A率
            ct_value: CT値
            ks_value: KS値（オプション）
            
        Returns:
            str: ゾーン名
        """
        thresholds = ZONE_THRESHOLDS
        
        # ガチゾーン
        if a_rate >= thresholds['gachi']['a_rate_min'] and ct_value >= thresholds['gachi']['ct_min']:
            return 'gachi'
        
        # ブルーゾーン
        if ct_value >= thresholds['blue']['ct_min']:
            return 'blue'
        
        # レッドゾーン
        if ct_value < thresholds['red']['ct_max']:
            return 'red'
        
        # トワイライトゾーン（デフォルト）
        return 'twilight'
    
    def classify_races(self, df, race_id_column='race_id'):
        """
        全レースをゾーン分類
        
        Args:
            df: 予測結果DataFrame
            race_id_column: レースIDカラム
            
        Returns:
            DataFrame: ゾーンを追加したDataFrame
        """
        df = df.copy()
        df['zone'] = ''
        df['zone_name'] = ''
        df['strategy'] = ''
        
        for race_id, group in tqdm(df.groupby(race_id_column), desc="ゾーン分類"):
            a_rate = group['a_rate'].iloc[0]
            ct_value = group['ct_value'].iloc[0]
            ks_value = group['ks_value'].iloc[0]
            
            zone = self.classify_zone(a_rate, ct_value, ks_value)
            zone_info = self.zone_strategies[zone]
            
            df.loc[df[race_id_column] == race_id, 'zone'] = zone
            df.loc[df[race_id_column] == race_id, 'zone_name'] = zone_info['name']
            df.loc[df[race_id_column] == race_id, 'strategy'] = zone_info['strategy']
        
        return df
    
    def get_zone_summary(self, df, race_id_column='race_id'):
        """
        ゾーン別のサマリーを取得
        
        Args:
            df: ゾーン分類済みDataFrame
            race_id_column: レースIDカラム
            
        Returns:
            DataFrame: ゾーン別サマリー
        """
        # レースごとに集計
        race_zones = df.groupby(race_id_column).first()[['zone', 'zone_name', 'a_rate', 'ct_value', 'ks_value', 'strategy']]
        
        summary = race_zones.groupby('zone').agg({
            'a_rate': ['count', 'mean'],
            'ct_value': 'mean',
            'ks_value': 'mean'
        }).round(3)
        
        summary.columns = ['レース数', '平均A率', '平均CT値', '平均KS値']
        
        return summary
    
    def get_zone_strategy(self, zone):
        """ゾーンに対応する戦略を取得"""
        return self.zone_strategies.get(zone, self.zone_strategies['twilight'])


class TreasureHunter:
    """
    宝探し（金箱）ロジック
    
    高回収率が期待できる特定パターンを検出
    """
    
    def __init__(self):
        pass
    
    def find_condition_a(self, df, race_id_column='race_id'):
        """
        条件A（超鉄板）を検出
        
        A率 0.9以上 かつ CT値 0.8以上 かつ KS値 1.0以上
        月に数回しか出現しないが、3連単（AB-CD 2点）で回収率約300%
        
        Args:
            df: 予測結果DataFrame
            race_id_column: レースIDカラム
            
        Returns:
            list: 条件Aに該当するレースID
        """
        condition_a_races = []
        
        for race_id, group in df.groupby(race_id_column):
            a_rate = group['a_rate'].iloc[0]
            ct_value = group['ct_value'].iloc[0]
            ks_value = group['ks_value'].iloc[0]
            
            cond = TREASURE_CONDITION_A
            
            if (a_rate >= cond['a_rate_min'] and 
                ct_value >= cond['ct_min'] and 
                ks_value >= cond['ks_min']):
                
                condition_a_races.append({
                    'race_id': race_id,
                    'a_rate': a_rate,
                    'ct_value': ct_value,
                    'ks_value': ks_value,
                    'type': 'condition_a',
                    'strategy': '3連単 AB-CD 2点',
                    'expected_roi': '約300%'
                })
        
        return condition_a_races
    
    def find_condition_b(self, df, race_id_column='race_id'):
        """
        条件B（本命からの紐荒れ）を検出
        
        A率 0.9以上 かつ CT値 0.5未満
        Aは来るが他が荒れる
        
        Args:
            df: 予測結果DataFrame
            race_id_column: レースIDカラム
            
        Returns:
            list: 条件Bに該当するレースID
        """
        condition_b_races = []
        
        for race_id, group in df.groupby(race_id_column):
            a_rate = group['a_rate'].iloc[0]
            ct_value = group['ct_value'].iloc[0]
            
            cond = TREASURE_CONDITION_B
            
            if a_rate >= cond['a_rate_min'] and ct_value < cond['ct_max']:
                condition_b_races.append({
                    'race_id': race_id,
                    'a_rate': a_rate,
                    'ct_value': ct_value,
                    'type': 'condition_b',
                    'strategy': 'A軸 DEF流し',
                    'expected_roi': '高配当期待'
                })
        
        return condition_b_races
    
    def find_all_treasures(self, df):
        """全ての宝探しパターンを検出"""
        treasures = []
        
        print("\n===== 宝探し（金箱）検出 =====")
        
        # 条件A
        cond_a = self.find_condition_a(df)
        if cond_a:
            print(f"\n条件A（超鉄板）: {len(cond_a)} 件")
            for t in cond_a:
                print(f"  {t['race_id']}: A率={t['a_rate']:.3f}, CT={t['ct_value']:.3f}, KS={t['ks_value']:.3f}")
                print(f"    → {t['strategy']} （期待回収率: {t['expected_roi']}）")
        
        # 条件B
        cond_b = self.find_condition_b(df)
        if cond_b:
            print(f"\n条件B（紐荒れ）: {len(cond_b)} 件")
            for t in cond_b:
                print(f"  {t['race_id']}: A率={t['a_rate']:.3f}, CT={t['ct_value']:.3f}")
                print(f"    → {t['strategy']} （期待: {t['expected_roi']}）")
        
        treasures.extend(cond_a)
        treasures.extend(cond_b)
        
        return treasures


class BettingGenerator:
    """
    車券生成クラス
    
    ゾーンごとに適切な車券組み合わせを生成
    """
    
    def __init__(self):
        pass
    
    def generate_bets(self, df, race_id, zone=None):
        """
        レースの車券を生成
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            zone: ゾーン（Noneの場合は自動判定）
            
        Returns:
            list: 車券リスト
        """
        race_df = df[df['race_id'] == race_id].sort_values('pred_score', ascending=False)
        
        if len(race_df) == 0:
            return []
        
        if zone is None:
            zone = race_df['zone'].iloc[0]
        
        # ランク別に車番を取得
        ranks = {}
        for _, row in race_df.iterrows():
            ranks[row['rank_label']] = row.get('車番', row.name)
        
        bets = []
        
        if zone == 'gachi':
            # ガチゾーン: 3連複1点
            bets.append({
                'type': '3連複',
                'numbers': sorted([ranks['A'], ranks['B'], ranks['C']]),
                'description': 'ABC'
            })
            
        elif zone == 'blue':
            # ブルーゾーン: 3連複2点
            bets.append({
                'type': '3連複',
                'numbers': sorted([ranks['A'], ranks['B'], ranks['C']]),
                'description': 'ABC'
            })
            bets.append({
                'type': '3連複',
                'numbers': sorted([ranks['A'], ranks['B'], ranks['D']]),
                'description': 'ABD'
            })
            
        elif zone == 'twilight':
            # トワイライトゾーン: 3連複3点
            bets.append({
                'type': '3連複',
                'numbers': sorted([ranks['A'], ranks['B'], ranks['C']]),
                'description': 'ABC'
            })
            bets.append({
                'type': '3連複',
                'numbers': sorted([ranks['A'], ranks['B'], ranks['D']]),
                'description': 'ABD'
            })
            bets.append({
                'type': '3連複',
                'numbers': sorted([ranks['A'], ranks['C'], ranks['D']]),
                'description': 'ACD'
            })
            
        elif zone == 'red':
            # レッドゾーン: 大穴狙い
            # A軸全流し or efg絡み
            
            # オプション1: A軸全流し（2車単）
            all_numbers = [ranks[r] for r in ranks.keys() if r in 'BCDEFG']
            for num in all_numbers:
                bets.append({
                    'type': '2車単',
                    'numbers': [ranks['A'], num],
                    'description': f'A→{num}'
                })
            
            # オプション2: efg絡み
            if all(r in ranks for r in ['E', 'F', 'G']):
                efg = [ranks['E'], ranks['F'], ranks['G']]
                for combo in combinations(efg, 2):
                    bets.append({
                        'type': '2車複',
                        'numbers': sorted(list(combo)),
                        'description': 'efg絡み'
                    })
        
        return bets
    
    def generate_exacta_bets(self, df, race_id, pattern='trifecta'):
        """
        3連単/3連複の車券を生成
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            pattern: 'trifecta'/'exacta'
            
        Returns:
            list: 車券リスト
        """
        race_df = df[df['race_id'] == race_id].sort_values('pred_score', ascending=False)
        
        if len(race_df) == 0:
            return []
        
        # 上位の車番を取得
        top_numbers = race_df.head(5)['車番'].tolist() if '車番' in race_df.columns else list(range(1, 6))
        
        bets = []
        
        if pattern == 'trifecta':
            # 3連複: 順序を問わない組み合わせ
            for combo in combinations(top_numbers[:4], 3):
                bets.append({
                    'type': '3連複',
                    'numbers': sorted(combo),
                    'description': '-'.join(map(str, sorted(combo)))
                })
        
        elif pattern == 'exacta':
            # 3連単: 順序を考慮
            for perm in permutations(top_numbers[:3], 3):
                bets.append({
                    'type': '3連単',
                    'numbers': list(perm),
                    'description': '→'.join(map(str, perm))
                })
        
        return bets
    
    def print_bets(self, bets):
        """車券リストを表示"""
        print("\n【推奨車券】")
        for i, bet in enumerate(bets, 1):
            numbers_str = '-'.join(map(str, bet['numbers'])) if bet['type'] != '2車単' else '→'.join(map(str, bet['numbers']))
            print(f"  {i}. {bet['type']}: {numbers_str} ({bet['description']})")


class HosoNoKata:
    """
    細の型（特定パターンの手動補正）
    
    G1やG2の初日において、細切れ戦で
    2車ラインの先頭に「4, 6, 8（ヨーロッパ）」が複数いる場合、
    AI予測を無視して独自の車券術を適用
    """
    
    def __init__(self):
        pass
    
    def detect_hoso_pattern(self, df, race_id):
        """
        細の型パターンを検出
        
        Args:
            df: 予測結果DataFrame
            race_id: レースID
            
        Returns:
            dict or None: 検出結果
        """
        race_df = df[df['race_id'] == race_id]
        
        if len(race_df) == 0:
            return None
        
        # ライン構成を確認
        line_composition = race_df['line_composition'].iloc[0] if 'line_composition' in race_df.columns else ''
        
        # 細切れ戦（3-2-2-2など）かチェック
        if not line_composition:
            return None
        
        line_counts = [int(x) for x in line_composition.split('-') if x.isdigit()]
        
        # 2車ラインが3つ以上ある細切れ戦
        two_car_lines = sum(1 for x in line_counts if x == 2)
        
        if two_car_lines < 3:
            return None
        
        # グレードを確認
        grade = race_df['grade'].iloc[0] if 'grade' in race_df.columns else ''
        
        if 'G1' in grade or 'G2' in grade or 'GI' in grade or 'GII' in grade:
            # 2車ライン先頭に4, 6, 8がいるか確認
            europe_numbers = [4, 6, 8]
            
            line_heads = race_df[race_df['position_in_line'] == 1]
            
            if '車番' in line_heads.columns:
                head_numbers = line_heads['車番'].tolist()
                europe_heads = [n for n in head_numbers if n in europe_numbers]
                
                if len(europe_heads) >= 2:
                    return {
                        'race_id': race_id,
                        'pattern': 'hoso',
                        'europe_heads': europe_heads,
                        'line_composition': line_composition,
                        'strategy': f'1着: {europe_heads}から, 2着: 他の2車ライン先頭, 3着: 3車ラインの番手'
                    }
        
        return None
    
    def find_hoso_races(self, df):
        """全レースから細の型パターンを検出"""
        hoso_races = []
        
        for race_id in df['race_id'].unique():
            result = self.detect_hoso_pattern(df, race_id)
            if result:
                hoso_races.append(result)
        
        if hoso_races:
            print(f"\n===== 細の型検出: {len(hoso_races)} 件 =====")
            for h in hoso_races:
                print(f"\n{h['race_id']}:")
                print(f"  ライン構成: {h['line_composition']}")
                print(f"  ヨーロッパ車番: {h['europe_heads']}")
                print(f"  戦略: {h['strategy']}")
        
        return hoso_races


if __name__ == "__main__":
    # テスト
    print("ゾーン分類テスト")
    
    classifier = ZoneClassifier()
    
    # テストケース
    test_cases = [
        {'a_rate': 0.95, 'ct_value': 0.85, 'expected': 'gachi'},
        {'a_rate': 0.80, 'ct_value': 0.78, 'expected': 'blue'},
        {'a_rate': 0.70, 'ct_value': 0.55, 'expected': 'twilight'},
        {'a_rate': 0.60, 'ct_value': 0.40, 'expected': 'red'},
    ]
    
    for tc in test_cases:
        zone = classifier.classify_zone(tc['a_rate'], tc['ct_value'])
        status = "✓" if zone == tc['expected'] else "✗"
        print(f"A率={tc['a_rate']:.2f}, CT={tc['ct_value']:.2f} → {zone} {status}")
    
    # 車券生成テスト
    generator = BettingGenerator()
    print("\n車券生成テスト:")
    
    # ダミーデータ
    test_df = pd.DataFrame({
        'race_id': ['test'] * 7,
        'rank_label': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        '車番': [5, 3, 7, 1, 2, 4, 6],
        'zone': ['blue'] * 7,
        'pred_score': [0.9, 0.7, 0.5, 0.3, -0.1, -0.2, -0.3]
    })
    
    bets = generator.generate_bets(test_df, 'test', 'blue')
    generator.print_bets(bets)
