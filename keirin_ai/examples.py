"""
使用例・サンプルスクリプト
様々な使い方のサンプルを提供
"""

from main import PasukoAI
from prediction import IkasamaDice
import pandas as pd


def example1_full_pipeline():
    """例1: フルパイプライン実行"""
    print("\n" + "="*70)
    print(" 例1: サンプルデータでフルパイプライン実行")
    print("="*70)
    
    pasuko = PasukoAI()
    predictions = pasuko.run_full_pipeline(
        data_source='sample',
        is_excel=False,
        save_models=True
    )
    
    print(f"\n予測完了: {len(predictions)}レース")
    return predictions


def example2_ikasama_dice(predictions):
    """例2: イカサマサイコロ投票"""
    print("\n" + "="*70)
    print(" 例2: イカサマサイコロ投票")
    print("="*70)
    
    # 最初のレースで試行
    race_id = list(predictions.keys())[0]
    pred = predictions[race_id]
    
    print(f"\nレースID: {race_id}")
    print(f"ゾーン: {pred.zone}")
    
    # イカサマサイコロ
    dice = IkasamaDice(weight_power=2.0)
    
    # 5パターン生成
    bets = dice.generate_bet(pred.race_data, num_bets=5)
    print("\n生成された買い目（5パターン）:")
    for i, bet in enumerate(bets, 1):
        print(f"  {i}. {bet[0]}-{bet[1]}-{bet[2]}")
    
    # 1000回試行して頻度分析
    df_bets = dice.generate_multiple_bets(pred.race_data, num_trials=1000)
    print("\n買い目出現頻度（トップ10）:")
    print(df_bets.head(10).to_string(index=False))


def example3_zone_analysis(predictions):
    """例3: ゾーン別分析"""
    print("\n" + "="*70)
    print(" 例3: ゾーン別分析")
    print("="*70)
    
    from config import ZONE_CONFIG
    
    zone_data = {zone: [] for zone in ZONE_CONFIG.keys()}
    
    for pred in predictions.values():
        zone_data[pred.zone].append({
            'race_id': pred.race_id,
            'a_rate': pred.a_rate,
            'ct_value': pred.ct_value,
            'ks_value': pred.ks_value
        })
    
    for zone, data in zone_data.items():
        if len(data) == 0:
            continue
        
        zone_name = ZONE_CONFIG[zone]['name']
        print(f"\n【{zone_name}】")
        print(f"  レース数: {len(data)}")
        
        df_zone = pd.DataFrame(data)
        print(f"  平均A率: {df_zone['a_rate'].mean():.4f}")
        print(f"  平均CT値: {df_zone['ct_value'].mean():.2f}")
        print(f"  平均KS値: {df_zone['ks_value'].mean():.4f}")
        
        # 推奨買い目
        print(f"  推奨: {ZONE_CONFIG[zone]['recommendation']}")


def example4_high_confidence_races(predictions):
    """例4: 高確信度レースの抽出"""
    print("\n" + "="*70)
    print(" 例4: 高確信度レース（A率0.7以上）")
    print("="*70)
    
    high_confidence = []
    for pred in predictions.values():
        if pred.a_rate >= 0.7:
            high_confidence.append({
                'race_id': pred.race_id,
                'a_rate': pred.a_rate,
                'ct_value': pred.ct_value,
                'zone': pred.zone,
                'top3': f"{pred.get_top_players(3)['car_number'].tolist()}"
            })
    
    if len(high_confidence) > 0:
        df_high = pd.DataFrame(high_confidence)
        df_high = df_high.sort_values('a_rate', ascending=False)
        print(f"\n該当レース: {len(df_high)}レース")
        print(df_high.to_string(index=False))
    else:
        print("\n該当レースなし")


def example5_predict_only():
    """例5: 学習済みモデルで予測のみ"""
    print("\n" + "="*70)
    print(" 例5: 学習済みモデルで予測のみ実行")
    print("="*70)
    
    pasuko = PasukoAI()
    
    # 学習済みモデルを読み込んで予測
    try:
        predictions = pasuko.predict_only(
            data_source='sample',
            is_excel=False,
            model_dir='./models'
        )
        print(f"\n予測完了: {len(predictions)}レース")
        return predictions
    except Exception as e:
        print(f"エラー: {e}")
        return None


def example6_custom_config():
    """例6: カスタム設定でAI実行"""
    print("\n" + "="*70)
    print(" 例6: カスタム設定でAI実行")
    print("="*70)
    
    # 設定をカスタマイズ
    custom_config = {
        'start_year': 2020,
        'end_year': 2023,
        'exclude_accidents': True
    }
    
    pasuko = PasukoAI(config_override=custom_config)
    
    print("カスタム設定:")
    for key, value in custom_config.items():
        print(f"  {key}: {value}")
    
    # ※実際の実行は省略（サンプルデータのため年は影響しない）


def main():
    """サンプル実行"""
    
    print("\n" + "="*70)
    print(" 競輪予測AI「パソ子」使用例デモ")
    print("="*70)
    
    # 例1: フルパイプライン
    predictions = example1_full_pipeline()
    
    # 例2: イカサマサイコロ
    example2_ikasama_dice(predictions)
    
    # 例3: ゾーン別分析
    example3_zone_analysis(predictions)
    
    # 例4: 高確信度レース
    example4_high_confidence_races(predictions)
    
    # 例5: 予測のみ（コメントアウト - 既に実行済みモデル使用）
    # example5_predict_only()
    
    # 例6: カスタム設定
    example6_custom_config()
    
    print("\n" + "="*70)
    print(" デモ完了!")
    print("="*70)


if __name__ == '__main__':
    main()
