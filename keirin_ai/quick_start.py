# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- クイックスタートスクリプト
================================================
簡単に予測を実行するためのサンプルスクリプト
"""

import sys
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config, RaceCategory
from src.scrapers import IntegratedScraper
from src.preprocessing import DataPreprocessor
from src.model import KeirinPredictor, ModelManager
from src.indicators import IndicatorCalculator
from src.output import OutputGenerator
from src.ikasama_dice import IkasamaDice, MiddleHoleStrategy


def demo_with_sample_data():
    """
    サンプルデータを使用したデモ
    実際のスクレイピングなしで動作確認
    """
    import pandas as pd
    import numpy as np
    
    print("=" * 60)
    print("競輪予測AI「パソ子」デモ")
    print("=" * 60)
    
    # サンプルデータを作成
    np.random.seed(42)
    
    sample_entries = []
    for race_no in range(1, 4):
        race_id = f"20250204_28_{race_no:02d}"  # 立川
        
        for car_num in range(1, 10):
            entry = {
                "race_id": race_id,
                "date": "2025-02-04",
                "venue_name": "立川",
                "race_no": race_no,
                "car_number": car_num,
                "waku": car_num,
                "player_name": f"選手{car_num}",
                "competition_score": np.random.uniform(90, 115),
                "back_count": np.random.randint(0, 20),
                "win_rate": np.random.uniform(10, 35),
                "second_rate": np.random.uniform(20, 50),
                "third_rate": np.random.uniform(30, 60),
                "rank_class": np.random.choice(["SS", "S1", "S2", "A1", "A2"]),
                "line_position": np.random.choice(["先頭", "番手", "三番手以降"]),
                "line_formation": np.random.choice(["3分戦", "4分戦"]),
                "age": np.random.randint(25, 45),
                "gear_ratio": np.random.uniform(3.7, 4.0),
                "is_girls": False,
                "is_challenge": False,
                "grade": "F1",
            }
            sample_entries.append(entry)
    
    df = pd.DataFrame(sample_entries)
    
    print(f"\nサンプルデータ: {len(df)} エントリー")
    print("-" * 40)
    
    # 前処理
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.preprocess(df, is_training=False)
    
    print(f"前処理完了: {len(processed_df)} 行")
    
    # 疑似予測（モデルがないため、競走得点ベースで予測）
    print("\n疑似予測を実行中...")
    
    for race_id in processed_df["race_id"].unique():
        race_df = processed_df[processed_df["race_id"] == race_id].copy()
        
        # 競走得点をベースにした疑似確率
        scores = race_df["competition_score"].values
        min_score, max_score = scores.min(), scores.max()
        race_df["pred_proba"] = (scores - min_score) / (max_score - min_score) * 0.5 + 0.3
        
        # ランキング
        race_df = race_df.sort_values("pred_proba", ascending=False).reset_index(drop=True)
        race_df["rank_order"] = range(1, len(race_df) + 1)
        symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        race_df["rank_symbol"] = symbols[:len(race_df)]
        
        # 指標計算
        calculator = IndicatorCalculator()
        indicators = calculator.calculate_indicators(race_df)
        zone = calculator.classify_zone(indicators["a_rate"], indicators["ct_value"])
        
        # 結果表示
        race_no = race_df["race_no"].iloc[0]
        venue = race_df["venue_name"].iloc[0]
        
        print(f"\n{'=' * 50}")
        print(f"{venue} {race_no}R - 【{zone.value}】")
        print(f"{'=' * 50}")
        print(f"A率: {indicators['a_rate']:.1%}  CT値: {indicators['ct_value']:.1f}  KS値: {indicators['ks_value']:.3f}")
        print("-" * 50)
        print("予想ランキング:")
        
        for _, row in race_df.head(5).iterrows():
            print(f"  {row['rank_symbol']}: {int(row['car_number'])}番 "
                  f"({row['pred_proba']:.1%}) 得点:{row['competition_score']:.2f}")
        
        # 推奨買い目
        bets = calculator.generate_recommended_bets(race_df, zone)
        print("\n推奨買い目:")
        for bet in bets[:2]:
            print(f"  {bet}")
    
    # イカサマサイコロデモ
    print("\n" + "=" * 60)
    print("イカサマサイコロ デモ")
    print("=" * 60)
    
    # 最後のレースでサイコロを振る
    dice = IkasamaDice()
    car_numbers = list(range(1, 10))
    probas = np.random.uniform(0.3, 0.8, 9).tolist()
    
    # 確率を正規化
    total = sum(probas)
    probas = [p / total * 3 for p in probas]  # 合計が3になるように（3着以内なので）
    
    recommended = dice.get_recommended_bets(car_numbers, probas, top_n=5, num_simulations=100)
    
    print("\nサイコロ推奨買い目（出現頻度順）:")
    for bet, count, prob in recommended:
        print(f"  3連複 {bet} - {prob:.1%} ({count}/100回)")
    
    # 中穴戦略
    strategy = MiddleHoleStrategy()
    
    # ダミーの race_df を作成
    dummy_df = pd.DataFrame({
        "car_number": list(range(1, 10)),
        "pred_proba": probas,
        "rank_order": list(range(1, 10)),
    })
    
    middle_bets = strategy.generate_middle_hole_bets(dummy_df)
    print("\n中穴狙い買い目:")
    for bet in middle_bets[:3]:
        print(f"  3連複 {bet}")
    
    print("\n" + "=" * 60)
    print("デモ完了！")
    print("=" * 60)
    print("\n実際の予測を行うには:")
    print("  python main.py predict --date 2025-02-04")
    print("\n過去データを取得するには:")
    print("  python main.py scrape --start 2024-01-01 --end 2024-12-31")


if __name__ == "__main__":
    demo_with_sample_data()
