# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」- メイン実行スクリプト
Main Execution Script for Keirin Prediction AI "Pasoko"

使用方法:
    1. データ収集: python main.py --mode scrape
    2. モデル学習: python main.py --mode train
    3. 予測実行:   python main.py --mode predict --date YYYYMMDD
    4. 全実行:     python main.py --mode all
"""

import argparse
import os
import sys
from datetime import datetime
from tqdm import tqdm

# モジュールパスを追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, MODELS_DIR, OUTPUT_DIR,
    START_YEAR, END_YEAR,
    RACE_INFO_FILE, RACE_CARD_FILE, RACE_RETURN_FILE, PROCESSED_DATA_FILE
)


def setup_directories():
    """ディレクトリ作成"""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("ディレクトリ準備完了")


def run_scraping(start_year=START_YEAR, end_year=END_YEAR):
    """
    データスクレイピング実行
    
    2018年〜2025年の全レースデータを取得
    """
    from scraper import KeirinScraper
    
    print(f"\n===== データスクレイピング開始 =====")
    print(f"期間: {start_year}年 〜 {end_year}年")
    
    scraper = KeirinScraper()
    
    # レースID生成
    print("\n1. レースID生成中...")
    race_ids = scraper.generate_race_ids_for_period(start_year, end_year)
    print(f"   生成されたレースID数: {len(race_ids)}")
    
    # データ取得
    print("\n2. レースデータ取得中...")
    print("   ※1秒間隔でアクセスするため時間がかかります")
    
    race_info, race_card, race_return = scraper.scrape_all_races(race_ids)
    
    # 保存
    print("\n3. データ保存中...")
    scraper.save_data(race_info, race_card, race_return)
    
    print("\n===== スクレイピング完了 =====")
    return True


def run_feature_engineering():
    """
    特徴量エンジニアリング実行
    """
    from feature_engineering import FeatureEngineer
    import pandas as pd
    
    print(f"\n===== 特徴量エンジニアリング開始 =====")
    
    engineer = FeatureEngineer()
    
    # データ読み込み
    print("1. データ読み込み中...")
    if not os.path.exists(RACE_CARD_FILE):
        print("エラー: 出走表データがありません。先にスクレイピングを実行してください。")
        return False
    
    df_card = pd.read_pickle(RACE_CARD_FILE)
    df_info = pd.read_pickle(RACE_INFO_FILE) if os.path.exists(RACE_INFO_FILE) else pd.DataFrame()
    df_return = pd.read_pickle(RACE_RETURN_FILE) if os.path.exists(RACE_RETURN_FILE) else pd.DataFrame()
    
    print(f"   出走表: {len(df_card)} 件")
    print(f"   レース情報: {len(df_info)} 件")
    print(f"   払戻情報: {len(df_return)} 件")
    
    # 特徴量生成
    print("\n2. 特徴量生成中...")
    df_processed = engineer.process_all(df_card, df_info, df_return)
    
    # 保存
    print("\n3. 処理済みデータ保存中...")
    engineer.save_processed_data(df_processed)
    
    print("\n===== 特徴量エンジニアリング完了 =====")
    return True


def run_training(model_type='random_forest'):
    """
    モデル学習実行
    
    カテゴリ別に複数モデルを学習
    """
    from model import ModelManager
    from feature_engineering import FeatureEngineer
    import pandas as pd
    
    print(f"\n===== モデル学習開始 =====")
    print(f"モデルタイプ: {model_type}")
    
    # データ読み込み
    print("1. 処理済みデータ読み込み中...")
    if not os.path.exists(PROCESSED_DATA_FILE):
        print("エラー: 処理済みデータがありません。先に特徴量エンジニアリングを実行してください。")
        return False
    
    df = pd.read_pickle(PROCESSED_DATA_FILE)
    print(f"   データ件数: {len(df)}")
    
    # 特徴量カラム取得
    engineer = FeatureEngineer()
    feature_columns = engineer.get_feature_columns(df)
    print(f"   特徴量数: {len(feature_columns)}")
    
    # モデル学習
    print("\n2. モデル学習中...")
    manager = ModelManager(model_type=model_type)
    metrics = manager.train_all_models(df, feature_columns)
    
    # 結果サマリー
    print("\n===== 学習結果サマリー =====")
    for category, metric in metrics.items():
        print(f"\n{category}:")
        print(f"  精度: {metric['accuracy']:.4f}")
        print(f"  AUC: {metric['roc_auc']:.4f}")
        print(f"  F1: {metric['f1']:.4f}")
    
    print("\n===== モデル学習完了 =====")
    return True


def run_prediction(date_str=None):
    """
    予測実行
    
    未来のレース（今日以降）の予測
    """
    from scraper import scrape_future_races
    from feature_engineering import FeatureEngineer
    from model import ModelManager
    from prediction_engine import PredictionEngine
    from zone_classifier import ZoneClassifier, TreasureHunter, BettingGenerator
    from ikasama_dice import IkasamaDice
    import pandas as pd
    
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    
    print(f"\n===== 予測実行 ({date_str}) =====")
    
    # 学習済みモデル読み込み
    print("1. モデル読み込み中...")
    manager = ModelManager()
    manager.load_all_models()
    
    if not manager.models:
        print("エラー: 学習済みモデルがありません。先に学習を実行してください。")
        return False
    
    print(f"   読み込んだモデル: {list(manager.models.keys())}")
    
    # 未来レースデータ取得
    print("\n2. 出走表取得中...")
    race_info_list, race_card_list = scrape_future_races(date_str)
    
    if not race_card_list:
        print("対象レースがありません")
        return False
    
    df_card = pd.concat(race_card_list, ignore_index=True)
    df_info = pd.DataFrame(race_info_list)
    
    print(f"   取得レース数: {len(df_info)}")
    
    # 特徴量生成
    print("\n3. 特徴量生成中...")
    engineer = FeatureEngineer()
    df_processed = engineer.process_all(df_card, df_info, pd.DataFrame())
    
    # 予測実行
    print("\n4. 予測実行中...")
    df_predicted = manager.predict(df_processed)
    
    # 指標計算
    print("\n5. 指標計算中...")
    engine = PredictionEngine()
    df_predicted = engine.calculate_indicators(df_predicted)
    
    # ゾーン分類
    print("\n6. ゾーン分類中...")
    classifier = ZoneClassifier()
    df_predicted = classifier.classify_races(df_predicted)
    
    # 宝探し
    print("\n7. 宝探しパターン検出中...")
    hunter = TreasureHunter()
    treasures = hunter.find_all_treasures(df_predicted)
    
    # 車券生成
    print("\n8. 車券生成中...")
    generator = BettingGenerator()
    dice = IkasamaDice()
    
    # 結果出力
    print("\n===== 予測結果 =====")
    
    summary = engine.get_top3_prediction(df_predicted)
    
    for _, row in summary.iterrows():
        race_id = row['race_id']
        
        # ゾーン取得
        race_zone = df_predicted[df_predicted['race_id'] == race_id]['zone'].iloc[0]
        zone_name = df_predicted[df_predicted['race_id'] == race_id]['zone_name'].iloc[0]
        
        print(f"\n----- {race_id} [{zone_name}] -----")
        print(f"予想: {row['A_車番']}-{row['B_車番']}-{row['C_車番']}")
        print(f"A率: {row['A率']:.3f}, CT値: {row['CT値']:.3f}, KS値: {row['KS値']:.3f}")
        
        # 戦略
        strategy = df_predicted[df_predicted['race_id'] == race_id]['strategy'].iloc[0]
        print(f"戦略: {strategy}")
        
        # イカサマサイコロ車券
        ct_value = row['CT値']
        ikasama_bets = dice.generate_bets_by_zone(df_predicted, race_id, ct_value)
        if ikasama_bets:
            print(f"サイコロ車券: {ikasama_bets[:3]}")
    
    # 保存
    print("\n9. 結果保存中...")
    engine.save_predictions(df_predicted, f'predictions_{date_str}.csv')
    engine.save_summary(summary, f'summary_{date_str}.csv')
    
    print("\n===== 予測完了 =====")
    return True


def run_all():
    """
    全処理実行
    """
    print("\n" + "=" * 60)
    print("競輪予想AI「パソ子」 全処理実行")
    print("=" * 60)
    
    # 1. ディレクトリ準備
    setup_directories()
    
    # 2. スクレイピング
    if not run_scraping():
        print("スクレイピングに失敗しました")
        return False
    
    # 3. 特徴量エンジニアリング
    if not run_feature_engineering():
        print("特徴量エンジニアリングに失敗しました")
        return False
    
    # 4. モデル学習
    if not run_training():
        print("モデル学習に失敗しました")
        return False
    
    # 5. 予測（今日の日付）
    if not run_prediction():
        print("予測に失敗しました")
        return False
    
    print("\n" + "=" * 60)
    print("全処理完了")
    print("=" * 60)
    return True


def demo_mode():
    """
    デモモード（ダミーデータで動作確認）
    """
    import numpy as np
    import pandas as pd
    from model import KeirinModel
    from prediction_engine import PredictionEngine
    from zone_classifier import ZoneClassifier, TreasureHunter, BettingGenerator
    from ikasama_dice import IkasamaDice
    
    print("\n" + "=" * 60)
    print("競輪予想AI「パソ子」 デモモード")
    print("=" * 60)
    
    # ダミーデータ作成
    np.random.seed(42)
    n_races = 10
    n_players = 7
    
    data = []
    for race_num in range(n_races):
        race_id = f'demo_race_{race_num + 1}'
        
        for player_num in range(n_players):
            # ランダムな特徴量
            score = np.random.uniform(80, 120)
            back = np.random.randint(0, 30)
            
            data.append({
                'race_id': race_id,
                '車番': player_num + 1,
                '選手名': f'選手{player_num + 1}',
                '競走得点': score,
                'バック': back,
                'position_in_line': (player_num % 3) + 1,
                'line_head': (player_num // 3) + 1,
                'car_count': 7,
                'line_count': 3,
                'bank_333': 1,
                'bank_400': 0,
                'bank_500': 0,
                'category': '7car',
                'target': int(np.random.random() < 0.4)  # 約40%が3着以内
            })
    
    df = pd.DataFrame(data)
    
    # モデル学習
    print("\n1. モデル学習...")
    feature_cols = ['競走得点', 'バック', 'position_in_line', 'line_head', 
                   'car_count', 'line_count', 'bank_333', 'bank_400', 'bank_500']
    
    model = KeirinModel(category='7car')
    model.train(df, feature_cols)
    
    # 予測
    print("\n2. 予測実行...")
    df_predicted = model.predict(df)
    df_predicted = model.rank_predictions(df_predicted)
    
    # 指標計算
    print("\n3. 指標計算...")
    engine = PredictionEngine()
    df_predicted = engine.calculate_indicators(df_predicted)
    
    # ゾーン分類
    print("\n4. ゾーン分類...")
    classifier = ZoneClassifier()
    df_predicted = classifier.classify_races(df_predicted)
    
    # 宝探し
    print("\n5. 宝探し...")
    hunter = TreasureHunter()
    treasures = hunter.find_all_treasures(df_predicted)
    
    # 結果表示
    print("\n" + "=" * 60)
    print("デモ予測結果")
    print("=" * 60)
    
    summary = engine.get_top3_prediction(df_predicted)
    print("\n予測サマリー:")
    print(summary.to_string())
    
    # ゾーン別統計
    print("\nゾーン別統計:")
    zone_summary = classifier.get_zone_summary(df_predicted)
    print(zone_summary.to_string())
    
    # イカサマサイコロデモ
    print("\n" + "=" * 60)
    print("イカサマサイコロ デモ")
    print("=" * 60)
    
    dice = IkasamaDice(power=2.0)
    
    # 1レース目で車券生成
    race_id = df_predicted['race_id'].iloc[0]
    print(f"\n対象レース: {race_id}")
    
    # 1000回シミュレーション
    sim_result = dice.simulation(df_predicted, race_id, n_simulations=1000)
    dice.print_simulation_result(sim_result)
    
    print("\n" + "=" * 60)
    print("デモ完了")
    print("=" * 60)


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description='競輪予想AI「パソ子」',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py --mode demo       # デモモード（ダミーデータで動作確認）
  python main.py --mode scrape     # データ収集
  python main.py --mode feature    # 特徴量エンジニアリング
  python main.py --mode train      # モデル学習
  python main.py --mode predict    # 本日のレース予測
  python main.py --mode predict --date 20240315  # 指定日のレース予測
  python main.py --mode all        # 全処理実行
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['demo', 'scrape', 'feature', 'train', 'predict', 'all'],
        default='demo',
        help='実行モード'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='予測対象日（YYYYMMDD形式）'
    )
    
    parser.add_argument(
        '--model-type',
        choices=['random_forest', 'gradient_boosting', 'logistic'],
        default='random_forest',
        help='モデルタイプ'
    )
    
    args = parser.parse_args()
    
    # ディレクトリ準備
    setup_directories()
    
    # モード別処理
    if args.mode == 'demo':
        demo_mode()
    elif args.mode == 'scrape':
        run_scraping()
    elif args.mode == 'feature':
        run_feature_engineering()
    elif args.mode == 'train':
        run_training(model_type=args.model_type)
    elif args.mode == 'predict':
        run_prediction(date_str=args.date)
    elif args.mode == 'all':
        run_all()


if __name__ == "__main__":
    main()
