# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- メインアプリケーション
============================================
全機能を統合したコマンドラインインターフェース

使用方法:
    # データ取得
    python main.py scrape --date 2025-02-04
    python main.py scrape --start 2024-01-01 --end 2024-12-31
    
    # モデル学習
    python main.py train --category 9car
    python main.py train --all
    
    # 予測実行
    python main.py predict --date 2025-02-04
    python main.py predict --date 2025-02-04 --output excel
    
    # イカサマサイコロ
    python main.py dice --date 2025-02-04 --race 1
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    config, RaceCategory, ZoneType, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, OUTPUT_DIR
)
from src.utils import setup_logger, get_date_range
from src.scrapers import IntegratedScraper
from src.preprocessing import DataPreprocessor
from src.model import KeirinPredictor, ModelManager
from src.indicators import IndicatorCalculator, RacePrediction
from src.output import OutputGenerator, BettingFormatter
from src.ikasama_dice import IkasamaDice, MiddleHoleStrategy, DiceVoting


class PasokoAI:
    """
    競輪予測AI「パソ子」メインクラス
    
    全機能を統合し、簡単なAPIを提供
    """
    
    def __init__(self):
        setup_logger("pasoko")
        self.scraper = None
        self.preprocessor = DataPreprocessor()
        self.model_manager = ModelManager()
        self.calculator = IndicatorCalculator()
        self.output_gen = OutputGenerator()
        self.dice = IkasamaDice()
    
    # ============================================================
    # データ取得
    # ============================================================
    def scrape_date(self, date: str) -> pd.DataFrame:
        """
        指定日のデータを取得
        
        Args:
            date: 日付（YYYY-MM-DD）
            
        Returns:
            pd.DataFrame: 出走表データ
        """
        logger.info(f"Scraping data for {date}")
        
        if self.scraper is None:
            self.scraper = IntegratedScraper()
        
        try:
            df = self.scraper.fetch_entries_for_prediction(date)
            logger.info(f"Scraped {len(df)} entries")
            return df
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            raise
    
    def scrape_historical(
        self,
        start_date: str,
        end_date: str,
    ) -> tuple:
        """
        過去データを一括取得
        
        Args:
            start_date: 開始日
            end_date: 終了日
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (出走表DF, 結果DF)
        """
        logger.info(f"Scraping historical data from {start_date} to {end_date}")
        
        if self.scraper is None:
            self.scraper = IntegratedScraper()
        
        try:
            entries_df, results_df = self.scraper.fetch_historical_data(
                start_date, end_date
            )
            return entries_df, results_df
        except Exception as e:
            logger.error(f"Historical scraping failed: {e}")
            raise
    
    # ============================================================
    # モデル学習
    # ============================================================
    def train_model(
        self,
        entries_path: str,
        results_path: str,
        category: Optional[RaceCategory] = None,
    ) -> dict:
        """
        モデルを学習
        
        Args:
            entries_path: 出走表CSVパス
            results_path: 結果CSVパス
            category: 学習対象カテゴリ（Noneで全カテゴリ）
            
        Returns:
            dict: 評価指標
        """
        logger.info("Starting model training")
        
        # データ読み込み
        entries_df = pd.read_csv(entries_path, encoding="utf-8-sig")
        results_df = pd.read_csv(results_path, encoding="utf-8-sig")
        
        # 前処理
        processed_df = self.preprocessor.preprocess(
            entries_df, results_df, is_training=True
        )
        
        # 学習
        if category:
            # 単一カテゴリ
            predictor = KeirinPredictor(category)
            category_df = self.preprocessor.filter_by_category(processed_df, category)
            metrics = predictor.train(category_df)
            predictor.save()
            return {category.value: metrics}
        else:
            # 全カテゴリ
            results = self.model_manager.train_all(processed_df)
            return results
    
    def load_models(self) -> None:
        """学習済みモデルを読み込み"""
        logger.info("Loading trained models")
        self.model_manager.load_all()
    
    # ============================================================
    # 予測実行
    # ============================================================
    def predict(
        self,
        date: str,
        scrape: bool = True,
        output_format: str = "excel",
    ) -> List[RacePrediction]:
        """
        予測を実行
        
        Args:
            date: 予測対象日
            scrape: データを新規取得するか
            output_format: 出力形式（excel, csv, text）
            
        Returns:
            List[RacePrediction]: 予測結果リスト
        """
        logger.info(f"Running prediction for {date}")
        
        # モデル読み込み
        if not self.model_manager.models:
            self.load_models()
        
        # データ取得
        if scrape:
            entries_df = self.scrape_date(date)
        else:
            # 既存データを読み込み
            entries_file = RAW_DATA_DIR / f"entries_{date.replace('-', '')}.csv"
            if not entries_file.exists():
                raise FileNotFoundError(f"Data file not found: {entries_file}")
            entries_df = pd.read_csv(entries_file, encoding="utf-8-sig")
        
        # 前処理
        processed_df = self.preprocessor.preprocess(entries_df, is_training=False)
        
        # 全レースの予測を実行
        all_predictions = []
        
        for race_id in processed_df["race_id"].unique():
            race_df = processed_df[processed_df["race_id"] == race_id].copy()
            
            # カテゴリ検出
            category = self.model_manager._detect_category(race_df)
            
            # 予測
            if category in self.model_manager.models:
                predictor = self.model_manager.models[category]
                pred_df = predictor.predict_race(race_df)
            else:
                # フォールバック
                pred_df = race_df.copy()
                pred_df["pred_proba"] = 0.5
                pred_df["rank_order"] = range(1, len(pred_df) + 1)
                pred_df["rank_symbol"] = ["A", "B", "C", "D", "E", "F", "G", "H", "I"][:len(pred_df)]
            
            # レース情報
            race_info = {
                "race_id": race_id,
                "date": date,
                "venue_name": race_df["venue_name"].iloc[0] if "venue_name" in race_df.columns else "",
                "race_no": race_df["race_no"].iloc[0] if "race_no" in race_df.columns else 0,
            }
            
            # 指標算出・ゾーン分類
            prediction = self.calculator.process_race(pred_df, race_info)
            all_predictions.append(prediction)
        
        logger.info(f"Completed {len(all_predictions)} race predictions")
        
        # 出力
        self._output_predictions(all_predictions, date, output_format)
        
        return all_predictions
    
    def _output_predictions(
        self,
        predictions: List[RacePrediction],
        date: str,
        output_format: str,
    ) -> None:
        """予測結果を出力"""
        filename = f"prediction_{date.replace('-', '')}"
        
        if output_format == "excel":
            self.output_gen.export_predictions(predictions, filename, "excel")
        elif output_format == "csv":
            self.output_gen.export_predictions(predictions, filename, "csv")
        elif output_format == "text":
            report = self.output_gen.generate_daily_report(predictions, date)
            self.output_gen.save_report(report, f"{filename}.txt")
        
        # 常にレポートも生成
        report = self.output_gen.generate_daily_report(predictions, date)
        print(report)
    
    # ============================================================
    # イカサマサイコロ
    # ============================================================
    def roll_dice(
        self,
        date: str,
        race_no: int,
        venue_name: Optional[str] = None,
        num_simulations: int = 100,
    ) -> dict:
        """
        イカサマサイコロを振る
        
        Args:
            date: 日付
            race_no: レース番号
            venue_name: 会場名（省略時は最初の会場）
            num_simulations: シミュレーション回数
            
        Returns:
            dict: 抽選結果
        """
        logger.info(f"Rolling dice for {date} R{race_no}")
        
        # 予測データを取得
        predictions = self.predict(date, scrape=True, output_format="csv")
        
        # 対象レースを特定
        target_pred = None
        for pred in predictions:
            if pred.race_no == race_no:
                if venue_name is None or pred.venue_name == venue_name:
                    target_pred = pred
                    break
        
        if target_pred is None:
            raise ValueError(f"Race not found: {date} R{race_no}")
        
        race_df = target_pred.rankings
        
        # サイコロ抽選
        recommended = self.dice.get_recommended_bets(
            race_df["car_number"].tolist(),
            race_df["pred_proba"].tolist(),
            top_n=10,
            num_simulations=num_simulations,
        )
        
        # 中穴戦略
        strategy = MiddleHoleStrategy(self.dice)
        middle_bets = strategy.generate_middle_hole_bets(race_df)
        upset_bets = strategy.generate_upset_bets(race_df)
        
        result = {
            "race_info": {
                "date": date,
                "venue": target_pred.venue_name,
                "race_no": race_no,
                "zone": target_pred.zone.value,
            },
            "dice_recommended": [
                {"bet": bet, "count": count, "prob": f"{prob:.1%}"}
                for bet, count, prob in recommended
            ],
            "middle_hole_bets": middle_bets,
            "upset_bets": upset_bets,
        }
        
        # 結果を表示
        print("\n" + "=" * 50)
        print(f"【イカサマサイコロ結果】")
        print(f"{target_pred.venue_name} {race_no}R ({target_pred.zone.value})")
        print("=" * 50)
        print("\n▼ サイコロ推奨（出現頻度順）")
        for item in result["dice_recommended"][:5]:
            print(f"  3連複 {item['bet']} - {item['prob']} ({item['count']}/{num_simulations}回)")
        
        print("\n▼ 中穴狙い買い目")
        for bet in middle_bets[:3]:
            print(f"  3連複 {bet}")
        
        print("\n▼ 大穴狙い買い目")
        for bet in upset_bets[:3]:
            print(f"  3連複 {bet}")
        
        print("=" * 50)
        
        return result
    
    # ============================================================
    # ゾーン分析
    # ============================================================
    def analyze_zones(self, date: str) -> dict:
        """
        ゾーン分析を実行
        
        Args:
            date: 日付
            
        Returns:
            dict: ゾーン別レース数
        """
        predictions = self.predict(date, scrape=True, output_format="text")
        
        zone_counts = {}
        for zone in ZoneType:
            zone_counts[zone.value] = len([
                p for p in predictions if p.zone == zone
            ])
        
        # ガチゾーンのレースをハイライト
        gachi_races = [p for p in predictions if p.zone == ZoneType.GACHI]
        
        print("\n" + "=" * 50)
        print(f"【{date} ゾーン分析】")
        print("=" * 50)
        
        for zone, count in zone_counts.items():
            print(f"  {zone}: {count}レース")
        
        if gachi_races:
            print("\n▼ ガチゾーン（狙い目）")
            for race in gachi_races:
                print(f"  {race.venue_name} {race.race_no}R - A率:{race.a_rate:.1%} CT:{race.ct_value:.1f}")
        
        return zone_counts
    
    def close(self) -> None:
        """リソースを解放"""
        if self.scraper:
            self.scraper.close()


def main():
    """コマンドラインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="競輪予測AI「パソ子」",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 今日の予測
    python main.py predict --date 2025-02-04
    
    # 過去データの取得
    python main.py scrape --start 2024-01-01 --end 2024-12-31
    
    # モデル学習
    python main.py train --entries data/entries.csv --results data/results.csv
    
    # イカサマサイコロ
    python main.py dice --date 2025-02-04 --race 5
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="コマンド")
    
    # scrape コマンド
    scrape_parser = subparsers.add_parser("scrape", help="データ取得")
    scrape_parser.add_argument("--date", help="取得日（YYYY-MM-DD）")
    scrape_parser.add_argument("--start", help="開始日")
    scrape_parser.add_argument("--end", help="終了日")
    
    # train コマンド
    train_parser = subparsers.add_parser("train", help="モデル学習")
    train_parser.add_argument("--entries", required=True, help="出走表CSVパス")
    train_parser.add_argument("--results", required=True, help="結果CSVパス")
    train_parser.add_argument("--category", help="カテゴリ（7car, 9car, girls等）")
    
    # predict コマンド
    predict_parser = subparsers.add_parser("predict", help="予測実行")
    predict_parser.add_argument("--date", required=True, help="予測日（YYYY-MM-DD）")
    predict_parser.add_argument("--output", default="excel", choices=["excel", "csv", "text"], help="出力形式")
    predict_parser.add_argument("--no-scrape", action="store_true", help="既存データを使用")
    
    # dice コマンド
    dice_parser = subparsers.add_parser("dice", help="イカサマサイコロ")
    dice_parser.add_argument("--date", required=True, help="日付")
    dice_parser.add_argument("--race", type=int, required=True, help="レース番号")
    dice_parser.add_argument("--venue", help="会場名")
    dice_parser.add_argument("--simulations", type=int, default=100, help="シミュレーション回数")
    
    # zones コマンド
    zones_parser = subparsers.add_parser("zones", help="ゾーン分析")
    zones_parser.add_argument("--date", required=True, help="日付")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # AIインスタンス作成
    ai = PasokoAI()
    
    try:
        if args.command == "scrape":
            if args.date:
                ai.scrape_date(args.date)
            elif args.start and args.end:
                ai.scrape_historical(args.start, args.end)
            else:
                print("--date または --start/--end を指定してください")
        
        elif args.command == "train":
            category = None
            if args.category:
                category_map = {
                    "7car": RaceCategory.SEVEN_CAR,
                    "9car": RaceCategory.NINE_CAR,
                    "girls": RaceCategory.GIRLS,
                    "challenge": RaceCategory.CHALLENGE,
                    "g3": RaceCategory.G3_SPECIAL,
                    "g1": RaceCategory.G1_SPECIAL,
                }
                category = category_map.get(args.category)
            
            metrics = ai.train_model(args.entries, args.results, category)
            print("\n学習完了:")
            for cat, m in metrics.items():
                print(f"  {cat}: AUC={m.get('auc', 0):.4f}")
        
        elif args.command == "predict":
            ai.predict(args.date, scrape=not args.no_scrape, output_format=args.output)
        
        elif args.command == "dice":
            ai.roll_dice(args.date, args.race, args.venue, args.simulations)
        
        elif args.command == "zones":
            ai.analyze_zones(args.date)
    
    finally:
        ai.close()


if __name__ == "__main__":
    main()
