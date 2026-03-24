"""
クイックスタートガイド
競輪予測AI「パソ子」の基本的な使い方
"""

# =====================================================================
# 1. 基本的な使い方
# =====================================================================

from main import PasukoAI

# AIインスタンス作成
pasuko = PasukoAI()

# サンプルデータでフルパイプライン実行
predictions = pasuko.run_full_pipeline(
    data_source='sample',
    is_excel=False,
    save_models=True
)

print(f"予測完了: {len(predictions)}レース")


# =====================================================================
# 2. Excelファイルから学習
# =====================================================================

# 実際のレースデータを使用する場合
predictions = pasuko.run_full_pipeline(
    data_source='race_data.xlsx',  # あなたのExcelファイル
    is_excel=True,
    save_models=True
)


# =====================================================================
# 3. 学習済みモデルで予測のみ
# =====================================================================

# 既に学習済みのモデルがある場合
pasuko = PasukoAI()
predictions = pasuko.predict_only(
    data_source='new_race_data.xlsx',
    is_excel=True,
    model_dir='./models'
)


# =====================================================================
# 4. 予測結果の確認
# =====================================================================

# 特定のレースの結果を確認
race_id = list(predictions.keys())[0]
pred = predictions[race_id]

print(f"レースID: {race_id}")
print(f"A率: {pred.a_rate:.4f}")
print(f"CT値: {pred.ct_value:.2f}")
print(f"KS値: {pred.ks_value:.4f}")
print(f"ゾーン: {pred.zone}")
print(f"推奨: {pred.get_recommendation()}")

# 上位3名の選手
top3 = pred.get_top_players(3)
print("\n上位3名:")
print(top3[['car_number', 'player_name', 'probability', 'rank']])


# =====================================================================
# 5. イカサマサイコロ投票
# =====================================================================

from prediction import IkasamaDice

# イカサマサイコロインスタンス作成
dice = IkasamaDice(weight_power=2.0)

# 買い目を5パターン生成
bets = dice.generate_bet(pred.race_data, num_bets=5)
print("\n買い目:")
for i, bet in enumerate(bets, 1):
    print(f"{i}. {bet[0]}-{bet[1]}-{bet[2]}")

# 頻度分析（1000回試行）
df_bets = dice.generate_multiple_bets(pred.race_data, num_trials=1000)
print("\n出現頻度トップ10:")
print(df_bets.head(10))


# =====================================================================
# 6. 高確信度レースの抽出
# =====================================================================

# A率が0.7以上のレースを抽出
high_confidence_races = []
for race_id, pred in predictions.items():
    if pred.a_rate >= 0.7:
        high_confidence_races.append({
            'race_id': race_id,
            'a_rate': pred.a_rate,
            'zone': pred.zone,
            'recommendation': pred.get_recommendation()
        })

import pandas as pd
df_high = pd.DataFrame(high_confidence_races)
print(df_high.sort_values('a_rate', ascending=False))


# =====================================================================
# 7. ゾーン別のレース数を集計
# =====================================================================

from config import ZONE_CONFIG

zone_counts = {}
for pred in predictions.values():
    zone = pred.zone
    zone_name = ZONE_CONFIG[zone]['name']
    zone_counts[zone_name] = zone_counts.get(zone_name, 0) + 1

print("\nゾーン別レース数:")
for zone_name, count in zone_counts.items():
    print(f"{zone_name}: {count}レース")


# =====================================================================
# 8. カスタム設定
# =====================================================================

# 設定をカスタマイズ
custom_config = {
    'start_year': 2020,
    'end_year': 2023,
    'exclude_accidents': True,
}

pasuko = PasukoAI(config_override=custom_config)


# =====================================================================
# 9. 出力ファイルの場所
# =====================================================================

# 以下のファイルが ./output/ に生成されます:
# - pasuko_predictions_summary.xlsx  : サマリー
# - pasuko_predictions_detail.xlsx   : 詳細
# - pasuko_statistics.xlsx           : 統計情報
# - pasuko_zone_distribution.png     : グラフ

# 学習済みモデルは ./models/ に保存されます


# =====================================================================
# 10. データ形式（Excelファイル）
# =====================================================================

"""
必要な列:
- race_id: レースID
- race_date: レース日
- track_name: 競輪場名
- race_number: レース番号
- car_number: 車番
- player_name: 選手名
- race_score: 競走得点
- back_count: バック回数
- finish_position: 着順（学習時のみ必須）
- bank_length: バンク周長（オプション）
- grade: グレード（オプション）
- is_line_leader: ライン先頭フラグ（オプション）
- line_formation: ライン構成（オプション）

サンプル:
race_id,race_date,track_name,race_number,car_number,player_name,race_score,back_count,finish_position
R001,2022-12-15,平塚,1,1,選手A,65.5,2,3
R001,2022-12-15,平塚,1,2,選手B,62.3,1,1
...
"""
