"""
競輪予測AI「パソ子」メインスクリプト v2.1
全モジュールを統合し、スクレイピングからデータ読み込み、予測、出力までを実行

【実行モード】
  demo    : サンプルデータで動作確認（デフォルト）
  scrape  : スクレイピング → 学習 → 予測 → 出力
  train   : 既存Excelから学習 → 予測 → 出力
  predict : 学習済みモデルで予測のみ

【コマンドライン例】
  python main.py                          # demoモード
  python main.py --mode scrape            # 2018-2025年 全期間スクレイプ→学習→予測
  python main.py --mode scrape --start-year 2022 --end-year 2023
  python main.py --mode train  --input ./scraped_data/training_data_2024.xlsx
  python main.py --mode predict --input ./scraped_data/training_data_2024.xlsx
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
import sys
import glob
import argparse
import logging

# 各モジュールのインポート
from config import *

# ロギング設定
def _setup_logging():
    cfg = globals().get('LOGGING_CONFIG', {'level': 'INFO', 'log_file': './pasuko.log', 'encoding': 'utf-8'})
    handlers = [logging.StreamHandler(sys.stdout)]
    if cfg.get('log_file'):
        handlers.append(logging.FileHandler(cfg['log_file'], encoding=cfg.get('encoding', 'utf-8')))
    logging.basicConfig(
        level=getattr(logging, cfg.get('level', 'INFO'), logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
    )

_setup_logging()
logger = logging.getLogger(__name__)
from data_processing import (
    KeirinDataCollector,
    KeirinDataPreprocessor,
    FeatureEngineering,
    create_sample_data
)
from model import KeirinModel, MultiModelManager
from prediction import RacePrediction, BatchPredictor, IkasamaDice
from output_generator import ExcelOutputGenerator, VisualizationGenerator


# =====================================================================
# scraper.py の遅延インポート（インストールされていない環境でもmain.pyが起動できる）
# =====================================================================
def _import_scraper():
    """scraper.py の KeirinDataPipeline を安全にインポート"""
    try:
        from scraper import KeirinDataPipeline
        return KeirinDataPipeline
    except ImportError as e:
        logger.error(f"scraper.py のインポートに失敗しました: {e}")
        logger.error("pip install requests beautifulsoup4 lxml を実行してください")
        return None


class PasukoAI:
    """競輪予測AI「パソ子」メインクラス v2.1"""
    
    def __init__(self, config_override: Optional[Dict] = None):
        """
        Args:
            config_override: 設定の上書き辞書
        """
        print("=" * 70)
        print(" 競輪予測AI「パソ子」初期化中...")
        print("=" * 70)
        
        # 設定の読み込み
        self.config = self._load_config(config_override)
        
        # 各コンポーネントの初期化
        self.data_collector = KeirinDataCollector(
            start_year=self.config['start_year'],
            end_year=self.config['end_year']
        )
        self.preprocessor = KeirinDataPreprocessor(
            exclude_accidents=self.config['exclude_accidents']
        )
        self.feature_engineering = FeatureEngineering()
        self.model_manager = MultiModelManager(
            model_params=MODEL_CONFIG['hyperparameters']
        )
        self.output_generator = ExcelOutputGenerator(
            output_dir=OUTPUT_CONFIG['output_dir']
        )
        self.visualizer = VisualizationGenerator(
            output_dir=OUTPUT_CONFIG['output_dir']
        )
        
        print("初期化完了!\n")
    
    def _load_config(self, override: Optional[Dict] = None) -> Dict:
        """設定の読み込み"""
        config = DATA_CONFIG.copy()
        if override:
            config.update(override)
        return config

    # ==================================================================
    # ★ スクレイピング統合メソッド
    # ==================================================================

    def run_scraping(
        self,
        start_year: int = None,
        end_year:   int = None,
        reset_progress: bool = False,
    ) -> List[str]:
        """
        scraper.py を呼び出して指定年範囲（デフォルト 2018〜2025年）の
        全レースデータを収集し、年単位のExcelとして保存する。

        Args:
            start_year:     開始年（None の場合 SCRAPER_CONFIG['start_year'] を使用）
            end_year:       終了年（None の場合 SCRAPER_CONFIG['end_year']   を使用）
            reset_progress: True = 進捗をリセットして最初から取得

        Returns:
            保存された Excel ファイルパスのリスト
        """
        # SCRAPER_CONFIG を参照（config.py で定義）
        sc = globals().get('SCRAPER_CONFIG', {})
        _start = start_year if start_year is not None else sc.get('start_year', 2018)
        _end   = end_year   if end_year   is not None else sc.get('end_year',   2025)
        _dir   = sc.get('output_dir',        './scraped_data')
        _ivl   = sc.get('request_interval',  1.5)
        _rsm   = sc.get('resume',            True)

        print("\n" + "=" * 70)
        print(f" [スクレイピング] {_start}年 〜 {_end}年  出力先: {_dir}")
        print("=" * 70)

        KeirinDataPipeline = _import_scraper()
        if KeirinDataPipeline is None:
            raise ImportError("scraper.py が読み込めません。requirements.txt を確認してください。")

        pipeline = KeirinDataPipeline(
            output_dir=_dir,
            request_interval=_ivl,
            resume=_rsm,
        )

        saved_paths = pipeline.run_full_history(
            start_year=_start,
            end_year=_end,
            reset_progress=reset_progress,
        )

        print(f"\nスクレイピング完了: {len(saved_paths)} ファイル保存")
        for p in saved_paths:
            print(f"  {p}")

        return saved_paths

    def load_scraped_excels(
        self,
        excel_pattern: str = None,
    ) -> pd.DataFrame:
        """
        スクレイピング済み Excel（training_data_*.xlsx）を
        まとめて読み込み、1つのDataFrameに結合して返す。

        Args:
            excel_pattern: glob パターン（None の場合 SCRAPER_CONFIG['output_dir'] を使用）

        Returns:
            結合済み DataFrame
        """
        sc = globals().get('SCRAPER_CONFIG', {})
        if excel_pattern is None:
            output_dir   = sc.get('output_dir', './scraped_data')
            excel_pattern = os.path.join(output_dir, 'training_data_*.xlsx')

        files = sorted(glob.glob(excel_pattern))
        if not files:
            raise FileNotFoundError(
                f"Excelファイルが見つかりません: {excel_pattern}\n"
                "先に run_scraping() を実行してください。"
            )

        print(f"\n読み込み対象: {len(files)} ファイル")
        dfs = []
        for f in files:
            try:
                df_tmp = pd.read_excel(f, sheet_name=0)
                # 日本語カラム名 → 英語カラム名に変換（KeirinExcelWriter.COLUMN_LABELS の逆）
                try:
                    from scraper import KeirinExcelWriter
                    inv = {v: k for k, v in KeirinExcelWriter.COLUMN_LABELS.items()}
                    df_tmp = df_tmp.rename(columns=inv)
                except Exception:
                    pass
                dfs.append(df_tmp)
                print(f"  {os.path.basename(f)}: {len(df_tmp):,}行")
            except Exception as e:
                print(f"  [スキップ] {f}: {e}")

        if not dfs:
            raise ValueError("有効なExcelファイルが読み込めませんでした")

        df_all = pd.concat(dfs, ignore_index=True)
        print(f"\n結合完了: {len(df_all):,}行 / {df_all['race_id'].nunique():,}レース")
        return df_all
    
    def prepare_training_data(self, data_source: str, 
                            is_excel: bool = True) -> pd.DataFrame:
        """
        学習用データの準備
        
        Args:
            data_source: データソース（Excelファイルパスまたはサンプルデータ）
            is_excel: Excelファイルから読み込むかどうか
            
        Returns:
            前処理済みDataFrame
        """
        print("\n" + "=" * 70)
        print(" Step 1: データ準備")
        print("=" * 70)
        
        # データ読み込み
        if is_excel:
            print(f"データ読み込み: {data_source}")
            df = self.data_collector.load_from_excel(data_source)
        else:
            print("サンプルデータ生成中...")
            df = create_sample_data(num_races=200, num_racers_per_race=9)
        
        print(f"読み込み完了: {len(df)}行, {len(df['race_id'].unique())}レース")
        
        # データクリーニング
        print("\nデータクリーニング...")
        df = self.preprocessor.clean_data(df)
        
        # バンク分類
        if 'bank_length' in df.columns:
            df['bank_category'] = df['bank_length'].apply(
                self.preprocessor.categorize_bank
            )
        
        # レースカテゴリ分類
        if 'is_girls' in df.columns:
            df['race_category'] = df.apply(
                lambda x: self.preprocessor.categorize_race_type(
                    num_racers=len(df[df['race_id'] == x['race_id']]),
                    grade=x.get('grade', 'F1'),
                    is_girls=x.get('is_girls', False),
                    is_challenge=x.get('is_challenge', False)
                ), axis=1
            )
        else:
            df['race_category'] = 'RACE_9'
        
        # 特徴量エンジニアリング
        print("\n特徴量生成...")
        df = self.feature_engineering.create_target_variable(df)
        df = self.feature_engineering.create_ranking_features(df)
        
        if 'is_line_leader' in df.columns:
            df = self.feature_engineering.create_flag4(df)
        else:
            df['flag4'] = 0
        
        print(f"準備完了: {len(df)}行")
        print(f"カテゴリ別レース数:")
        print(df.groupby('race_category')['race_id'].nunique())
        
        return df
    
    def train_models(self, df: pd.DataFrame, 
                    feature_columns: Optional[list] = None):
        """
        全モデルの学習
        
        Args:
            df: 学習用DataFrame
            feature_columns: 使用する特徴量のリスト
        """
        print("\n" + "=" * 70)
        print(" Step 2: モデル学習")
        print("=" * 70)
        
        # デフォルト特徴量
        if feature_columns is None:
            feature_columns = [
                'car_number', 'race_score', 'back_count',
                'score_rank', 'back_rank', 'flag4'
            ]
            # 存在する列のみ使用
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        print(f"\n使用特徴量: {feature_columns}")
        
        # カテゴリごとにデータを分割
        category_data = {}
        for category in df['race_category'].unique():
            df_category = df[df['race_category'] == category]
            
            if len(df_category) < 100:
                print(f"警告: {category}のデータが少ないためスキップ ({len(df_category)}行)")
                continue
            
            X = df_category[feature_columns]
            y = df_category['target']
            
            category_data[category] = (X, y)
        
        # モデル作成
        self.model_manager.create_models(list(category_data.keys()))
        
        # 学習実行
        print("\n学習開始...")
        results = self.model_manager.train_all_models(category_data)
        
        # 結果サマリー
        print("\n" + "=" * 70)
        print(" 学習結果サマリー")
        print("=" * 70)
        for category, metrics in results.items():
            print(f"\n{category}:")
            print(f"  検証精度: {metrics['val_accuracy']:.4f}")
            print(f"  検証AUC: {metrics['val_auc']:.4f}")
        
        return results
    
    def save_models(self, save_dir: str = './models'):
        """モデルの保存"""
        print("\n" + "=" * 70)
        print(" モデル保存")
        print("=" * 70)
        
        saved_paths = self.model_manager.save_all_models(save_dir)
        
        print("\n保存完了:")
        for category, path in saved_paths.items():
            print(f"  {category}: {path}")
        
        return saved_paths
    
    def load_models(self, model_dir: str = './models'):
        """モデルの読み込み"""
        print("\n" + "=" * 70)
        print(" モデル読み込み")
        print("=" * 70)
        
        self.model_manager.load_all_models(model_dir)
        print("読み込み完了!")
    
    def predict_races(self, df: pd.DataFrame) -> Dict[str, RacePrediction]:
        """
        レース予測の実行
        
        Args:
            df: 予測対象のDataFrame
            
        Returns:
            {race_id: RacePrediction} の辞書
        """
        print("\n" + "=" * 70)
        print(" Step 3: レース予測")
        print("=" * 70)
        
        predictor = BatchPredictor(self.model_manager)
        predictions = predictor.predict_races(df)
        
        print(f"\n予測完了: {len(predictions)}レース")
        
        # ゾーン別集計
        zone_counts = {}
        for pred in predictions.values():
            zone = pred.zone
            zone_counts[zone] = zone_counts.get(zone, 0) + 1
        
        print("\nゾーン別レース数:")
        for zone, count in zone_counts.items():
            zone_name = ZONE_CONFIG[zone]['name']
            print(f"  {zone_name}: {count}レース")
        
        return predictions
    
    def generate_outputs(self, predictions: Dict[str, RacePrediction]):
        """
        予測結果の出力
        
        Args:
            predictions: {race_id: RacePrediction} の辞書
        """
        print("\n" + "=" * 70)
        print(" Step 4: 結果出力")
        print("=" * 70)
        
        # Excel出力
        print("\nExcelファイル生成中...")
        self.output_generator.create_summary_excel(
            predictions,
            filename='pasuko_predictions_summary.xlsx'
        )
        self.output_generator.create_detailed_excel(
            predictions,
            filename='pasuko_predictions_detail.xlsx'
        )
        self.output_generator.create_statistics_excel(
            predictions,
            filename='pasuko_statistics.xlsx'
        )
        
        # グラフ生成
        print("\nグラフ生成中...")
        self.visualizer.create_zone_distribution_chart(
            predictions,
            filename='pasuko_zone_distribution.png'
        )
        
        print(f"\n出力先: {OUTPUT_CONFIG['output_dir']}")
    
    def run_full_pipeline(self, data_source: str, 
                         is_excel: bool = False,
                         save_models: bool = True):
        """
        フルパイプライン実行（学習から予測、出力まで）
        
        Args:
            data_source: データソース
            is_excel: Excelファイルかどうか
            save_models: モデルを保存するか
        """
        print("\n" + "=" * 70)
        print(" 競輪予測AI「パソ子」フルパイプライン実行")
        print("=" * 70)
        
        # 1. データ準備
        df = self.prepare_training_data(data_source, is_excel)
        
        # 2. モデル学習
        self.train_models(df)
        
        # 3. モデル保存
        if save_models:
            self.save_models()
        
        # 4. 予測実行（学習データの一部で検証）
        predictions = self.predict_races(df)
        
        # 5. 出力生成
        self.generate_outputs(predictions)
        
        print("\n" + "=" * 70)
        print(" パイプライン完了!")
        print("=" * 70)
        
        return predictions

    # ==================================================================
    # ★ スクレイピング統合フルパイプライン
    # ==================================================================

    def run_full_pipeline_with_scraping(
        self,
        start_year: int = None,
        end_year:   int = None,
        reset_progress: bool = False,
        save_models: bool = True,
    ):
        """
        ★ スクレイピング → 学習 → 予測 → 出力 の完全自動パイプライン

        1. scraper.py でデータを収集（2018〜2025年、年単位Excel保存）
        2. 保存したExcelをすべて読み込んで結合
        3. 前処理・特徴量エンジニアリング
        4. カテゴリ別モデル学習
        5. 予測・ゾーン分類
        6. Excel・グラフ出力

        Args:
            start_year:     取得開始年（None = SCRAPER_CONFIG の値）
            end_year:       取得終了年（None = SCRAPER_CONFIG の値）
            reset_progress: 進捗をリセットして最初から取得
            save_models:    モデルファイルを保存するか

        Returns:
            {race_id: RacePrediction} の辞書
        """
        print("\n" + "=" * 70)
        print(" 競輪予測AI「パソ子」スクレイピング統合パイプライン")
        print("=" * 70)

        # ---- Step 1: スクレイピング ----
        saved_paths = self.run_scraping(
            start_year=start_year,
            end_year=end_year,
            reset_progress=reset_progress,
        )

        if not saved_paths:
            print("[警告] スクレイピングデータが取得できませんでした")
            print("  ネットワーク接続・サイト構造を確認してください")
            return {}

        # ---- Step 2: Excel読み込み（全年分を結合） ----
        df = self.load_scraped_excels()

        # ---- Step 3〜6: 以降は通常パイプラインと共通 ----
        df_processed = self._preprocess_df(df)

        self.train_models(df_processed)

        if save_models:
            self.save_models()

        predictions = self.predict_races(df_processed)
        self.generate_outputs(predictions)

        print("\n" + "=" * 70)
        print(" スクレイピング統合パイプライン 完了!")
        print("=" * 70)
        return predictions

    def run_pipeline_from_excel(
        self,
        excel_pattern: str = None,
        save_models: bool = True,
    ):
        """
        ★ 既存Excelから学習 → 予測 → 出力
        （スクレイピング済みのExcelが手元にある場合に使用）

        Args:
            excel_pattern: glob パターン（例: './scraped_data/training_data_*.xlsx'）
            save_models:   モデルファイルを保存するか

        Returns:
            {race_id: RacePrediction} の辞書
        """
        print("\n" + "=" * 70)
        print(" 競輪予測AI「パソ子」Excel読み込みパイプライン")
        print("=" * 70)

        df = self.load_scraped_excels(excel_pattern)
        df_processed = self._preprocess_df(df)

        self.train_models(df_processed)

        if save_models:
            self.save_models()

        predictions = self.predict_races(df_processed)
        self.generate_outputs(predictions)

        print("\n" + "=" * 70)
        print(" Excel読み込みパイプライン 完了!")
        print("=" * 70)
        return predictions

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame の前処理共通処理"""
        print("\nデータクリーニング中...")
        df = self.preprocessor.clean_data(df)
        if 'bank_length' in df.columns:
            df['bank_category'] = df['bank_length'].apply(
                self.preprocessor.categorize_bank
            )
        print("\n特徴量エンジニアリング中...")
        df = self.feature_engineering.create_features(df)
        return df
    
    def predict_only(self, data_source: str, 
                    is_excel: bool = False,
                    model_dir: str = './models'):
        """
        予測のみ実行（学習済みモデルを使用）
        
        Args:
            data_source: データソース
            is_excel: Excelファイルかどうか
            model_dir: モデルディレクトリ
        """
        print("\n" + "=" * 70)
        print(" 競輪予測AI「パソ子」予測実行")
        print("=" * 70)
        
        # 1. モデル読み込み
        self.load_models(model_dir)
        
        # 2. データ準備
        df = self.prepare_training_data(data_source, is_excel)
        
        # 3. 予測実行
        predictions = self.predict_races(df)
        
        # 4. 出力生成
        self.generate_outputs(predictions)
        
        print("\n" + "=" * 70)
        print(" 予測完了!")
        print("=" * 70)
        
        return predictions


# =====================================================================
# コマンドライン インターフェース
# =====================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="競輪予測AI「パソ子」",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
実行モード一覧:
  demo    サンプルデータで動作確認（デフォルト）
  scrape  スクレイピング→学習→予測→出力（2018〜2025年 全期間）
  train   既存Excelから学習→予測→出力
  predict 学習済みモデルで予測のみ

使用例:
  python main.py
  python main.py --mode scrape
  python main.py --mode scrape --start-year 2022 --end-year 2023
  python main.py --mode scrape --reset
  python main.py --mode train  --input ./scraped_data/training_data_*.xlsx
  python main.py --mode predict --input ./scraped_data/training_data_2024.xlsx
        """,
    )
    p.add_argument(
        "--mode",
        choices=["demo", "scrape", "train", "predict"],
        default=globals().get("PIPELINE_MODE", {}).get("mode", "demo"),
        help="実行モード（デフォルト: demo）",
    )
    p.add_argument(
        "--start-year", type=int,
        default=globals().get("SCRAPER_CONFIG", {}).get("start_year", 2018),
        help="スクレイピング開始年（scrape モード用、デフォルト: 2018）",
    )
    p.add_argument(
        "--end-year", type=int,
        default=globals().get("SCRAPER_CONFIG", {}).get("end_year", 2025),
        help="スクレイピング終了年（scrape モード用、デフォルト: 2025）",
    )
    p.add_argument(
        "--input", type=str, default=None,
        help="入力Excelパス or globパターン（train/predict モード用）",
    )
    p.add_argument(
        "--model-dir", type=str, default="./models",
        help="モデルディレクトリ（デフォルト: ./models）",
    )
    p.add_argument(
        "--reset", action="store_true",
        help="スクレイピング進捗をリセットして最初から取得",
    )
    p.add_argument(
        "--no-save-models", action="store_true",
        help="モデルを保存しない",
    )
    return p


def main():
    """メイン実行関数"""
    parser  = _build_parser()
    args    = parser.parse_args()
    pasuko  = PasukoAI()
    save_m  = not args.no_save_models

    # ----------------------------------------------------------------
    if args.mode == "demo":
        print("\n【デモモード: サンプルデータで実行】")
        predictions = pasuko.run_full_pipeline(
            data_source='sample',
            is_excel=False,
            save_models=save_m,
        )

    # ----------------------------------------------------------------
    elif args.mode == "scrape":
        print(f"\n【スクレイピングモード: {args.start_year}〜{args.end_year}年】")
        predictions = pasuko.run_full_pipeline_with_scraping(
            start_year=args.start_year,
            end_year=args.end_year,
            reset_progress=args.reset,
            save_models=save_m,
        )

    # ----------------------------------------------------------------
    elif args.mode == "train":
        print("\n【学習モード: 既存Excelから学習→予測】")
        predictions = pasuko.run_pipeline_from_excel(
            excel_pattern=args.input,
            save_models=save_m,
        )

    # ----------------------------------------------------------------
    elif args.mode == "predict":
        print("\n【予測モード: 学習済みモデルで予測のみ】")
        input_src = args.input or globals().get('PIPELINE_MODE', {}).get('input_excel')
        predictions = pasuko.predict_only(
            data_source=input_src,
            is_excel=bool(input_src),
            model_dir=args.model_dir,
        )

    else:
        parser.print_help()
        return

    # ---- 結果サマリ表示（最初の5レース） ----
    if predictions:
        print("\n" + "=" * 70)
        print(" 予測結果サンプル（最初の5レース）")
        print("=" * 70)
        for i, (race_id, pred) in enumerate(list(predictions.items())[:5], 1):
            print(f"\n【レース{i}: {race_id}】")
            print(f"  A率: {pred.a_rate:.4f}")
            print(f"  CT値: {pred.ct_value:.2f}")
            print(f"  KS値: {pred.ks_value:.4f}")
            print(f"  ゾーン: {ZONE_CONFIG[pred.zone]['name']}")
            print(f"  推奨: {pred.get_recommendation()}")
            top3 = pred.get_top_players(3)
            print(f"  予想: {top3['car_number'].tolist()}")


if __name__ == '__main__':
    main()
