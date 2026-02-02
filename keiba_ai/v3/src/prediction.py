"""
本番予測・スケジュール実行モジュール
フェーズ2-8: 予測母集団作成機能（本番運用用）
フェーズ4-21: スケジュール実行機能
"""

import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any

import pandas as pd
import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.src.utils import load_config, get_project_root, ensure_dir, setup_logging
from common.src.scraping import ShutsubaTableScraper, RaceIdScraper

logger = setup_logging()


class PredictionDataCreator:
    """予測用データ作成クラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.project_root = get_project_root()
        self.shutsuba_scraper = ShutsubaTableScraper(config)
    
    def create_prediction_data(
        self, 
        race_id: str,
        feature_columns: List[str],
        model_path: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """予測用データを作成
        
        Args:
            race_id: レースID
            feature_columns: 特徴量カラムリスト
            model_path: モデルパス
        
        Returns:
            (予測用DataFrame, 不足特徴量リスト)
        """
        # 出馬表を取得
        html = self.shutsuba_scraper.get_shutsuba_table(race_id)
        
        if html is None:
            logger.error(f"出馬表取得失敗: {race_id}")
            return pd.DataFrame(), []
        
        # HTMLをパース
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # 出馬表テーブルを抽出
        shutsuba_table = soup.find('table', class_='Shutuba_Table')
        
        if not shutsuba_table:
            logger.error(f"出馬表テーブルが見つかりません: {race_id}")
            return pd.DataFrame(), []
        
        try:
            dfs = pd.read_html(str(shutsuba_table), flavor='lxml')
            if not dfs:
                return pd.DataFrame(), []
            
            df = dfs[0]
            
            # カラム名を正規化
            df.columns = df.columns.str.strip()
            
            # 馬IDを抽出
            horse_ids = self._extract_horse_ids(shutsuba_table)
            if horse_ids:
                df['horse_id'] = horse_ids[:len(df)]
            
            # 騎手IDを抽出
            jockey_ids = self._extract_jockey_ids(shutsuba_table)
            if jockey_ids:
                df['jockey_id'] = jockey_ids[:len(df)]
            
            # レースIDを追加
            df['race_id'] = race_id
            
            # 必要な特徴量を確認
            missing_features = [c for c in feature_columns if c not in df.columns]
            
            return df, missing_features
            
        except Exception as e:
            logger.error(f"出馬表パースエラー: {race_id} - {e}")
            return pd.DataFrame(), []
    
    def _extract_horse_ids(self, table) -> List[str]:
        """テーブルから馬IDを抽出"""
        import re
        horse_ids = []
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            horse_link = row.find('a', href=re.compile(r'/horse/\d+'))
            if horse_link:
                match = re.search(r'/horse/(\d+)', horse_link.get('href', ''))
                if match:
                    horse_ids.append(match.group(1))
                else:
                    horse_ids.append(None)
            else:
                horse_ids.append(None)
        
        return horse_ids
    
    def _extract_jockey_ids(self, table) -> List[str]:
        """テーブルから騎手IDを抽出"""
        import re
        jockey_ids = []
        rows = table.find_all('tr')
        
        for row in rows[1:]:
            jockey_link = row.find('a', href=re.compile(r'/jockey/\d+'))
            if jockey_link:
                match = re.search(r'/jockey/(\d+)', jockey_link.get('href', ''))
                if match:
                    jockey_ids.append(match.group(1))
                else:
                    jockey_ids.append(None)
            else:
                jockey_ids.append(None)
        
        return jockey_ids


class RacePredictor:
    """レース予測クラス"""
    
    def __init__(self, model_path: str, config: Optional[dict] = None):
        """初期化
        
        Args:
            model_path: 学習済みモデルのパス
            config: 設定辞書
        """
        self.config = config or load_config()
        self.model_path = model_path
        
        # モデル読み込み
        with open(model_path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.feature_importance = save_data.get('feature_importance')
        
        # 特徴量カラムを復元
        if self.feature_importance is not None:
            self.feature_columns = self.feature_importance['feature'].tolist()
        else:
            self.feature_columns = []
        
        self.data_creator = PredictionDataCreator(config)
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """予測を実行
        
        Args:
            df: 予測対象データ
        
        Returns:
            予測結果付きDataFrame
        """
        df = df.copy()
        
        # 特徴量を準備
        X = df[self.feature_columns].fillna(-999)
        
        # 予測
        pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        df['pred_proba'] = pred_proba
        df['pred_rank'] = df.groupby('race_id')['pred_proba'].rank(ascending=False, method='min')
        
        return df
    
    def predict_race(self, race_id: str) -> pd.DataFrame:
        """単一レースを予測
        
        Args:
            race_id: レースID
        
        Returns:
            予測結果DataFrame
        """
        # 予測用データ作成
        df, missing = self.data_creator.create_prediction_data(
            race_id, self.feature_columns, self.model_path
        )
        
        if df.empty:
            logger.error(f"予測データ作成失敗: {race_id}")
            return pd.DataFrame()
        
        if missing:
            logger.warning(f"不足特徴量: {missing}")
        
        # 予測実行
        result = self.predict(df)
        
        return result
    
    def get_recommended_bets(
        self, 
        prediction_df: pd.DataFrame,
        top_n: int = 3,
        min_probability: float = None
    ) -> pd.DataFrame:
        """推奨買い目を取得
        
        Args:
            prediction_df: 予測結果DataFrame
            top_n: 上位頭数
            min_probability: 最小確率閾値
        
        Returns:
            推奨買い目DataFrame
        """
        min_probability = min_probability or self.config.get('simulation', {}).get('threshold', {}).get('min_probability', 0.01)
        
        # 足切り
        filtered = prediction_df[prediction_df['pred_proba'] >= min_probability]
        
        # 上位N頭を選択
        recommendations = filtered.groupby('race_id').apply(
            lambda x: x.nlargest(top_n, 'pred_proba')
        ).reset_index(drop=True)
        
        return recommendations


class RaceScheduler:
    """レーススケジュール管理クラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.realtime_config = self.config.get('realtime', {})
        self.scheduler = BlockingScheduler()
    
    def get_timetable(self, date: str) -> List[Dict]:
        """当日のレース発走時刻を取得
        
        Args:
            date: 日付（YYYYMMDD形式）
        
        Returns:
            レーススケジュールリスト
        """
        # 出馬表からレース時刻を取得
        race_id_scraper = RaceIdScraper(self.config)
        race_ids = race_id_scraper.get_race_ids_from_date(date)
        
        timetable = []
        for race_id in race_ids:
            # 発走時刻はレースページから取得する必要がある
            # ここでは仮のデータを返す
            timetable.append({
                'race_id': race_id,
                'start_time': None,  # 実際には取得する
            })
        
        return timetable
    
    def schedule_predictions(
        self, 
        timetable: List[Dict],
        predictor: RacePredictor,
        minutes_before: float = None
    ):
        """予測タスクをスケジュール
        
        Args:
            timetable: レーススケジュール
            predictor: 予測器
            minutes_before: 発走何分前に実行するか
        """
        minutes_before = minutes_before or self.realtime_config.get('prediction_minutes_before', 4.5)
        
        for race_info in timetable:
            race_id = race_info['race_id']
            start_time = race_info.get('start_time')
            
            if start_time is None:
                continue
            
            # 予測実行時刻を計算
            exec_time = start_time - timedelta(minutes=minutes_before)
            
            if exec_time > datetime.now():
                self.scheduler.add_job(
                    self._run_prediction,
                    'date',
                    run_date=exec_time,
                    args=[predictor, race_id],
                    id=f'predict_{race_id}'
                )
                
                logger.info(f"スケジュール登録: {race_id} @ {exec_time}")
    
    def _run_prediction(self, predictor: RacePredictor, race_id: str):
        """予測を実行（スケジューラから呼び出される）"""
        logger.info(f"予測開始: {race_id}")
        
        try:
            result = predictor.predict_race(race_id)
            
            if not result.empty:
                # 推奨買い目を出力
                recommendations = predictor.get_recommended_bets(result)
                
                logger.info(f"予測完了: {race_id}")
                logger.info(f"推奨:")
                for _, row in recommendations.iterrows():
                    logger.info(f"  馬番{row['horse_number']}: {row['pred_proba']:.3f}")
                
                # 結果を保存
                self._save_prediction(race_id, result, recommendations)
                
        except Exception as e:
            logger.error(f"予測エラー: {race_id} - {e}")
    
    def _save_prediction(
        self, 
        race_id: str, 
        result: pd.DataFrame,
        recommendations: pd.DataFrame
    ):
        """予測結果を保存"""
        project_root = get_project_root()
        results_dir = ensure_dir(project_root / 'results' / 'predictions')
        
        # 結果を保存
        result_path = results_dir / f'{race_id}_prediction.csv'
        result.to_csv(result_path, index=False, encoding='utf-8-sig')
        
        rec_path = results_dir / f'{race_id}_recommendations.csv'
        recommendations.to_csv(rec_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"保存完了: {result_path}")
    
    def start(self):
        """スケジューラを開始"""
        logger.info("スケジューラ開始")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("スケジューラ停止")


def run_prediction_pipeline(
    model_path: str,
    race_id: str,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """予測パイプライン
    
    Args:
        model_path: モデルパス
        race_id: レースID
        config: 設定辞書
    
    Returns:
        予測結果DataFrame
    """
    config = config or load_config()
    
    predictor = RacePredictor(model_path, config)
    result = predictor.predict_race(race_id)
    
    if result.empty:
        logger.error("予測失敗")
        return pd.DataFrame()
    
    # 推奨買い目
    recommendations = predictor.get_recommended_bets(result)
    
    logger.info("予測結果:")
    for _, row in recommendations.iterrows():
        logger.info(f"  馬番{row.get('horse_number', '?')}: {row.get('pred_proba', 0):.3f}")
    
    return result


def run_realtime_mode(
    model_path: str,
    date: str,
    config: Optional[dict] = None
):
    """リアルタイム運用モード
    
    Args:
        model_path: モデルパス
        date: 対象日（YYYYMMDD形式）
        config: 設定辞書
    """
    config = config or load_config()
    
    predictor = RacePredictor(model_path, config)
    scheduler = RaceScheduler(config)
    
    # タイムテーブル取得
    timetable = scheduler.get_timetable(date)
    
    # スケジュール登録
    scheduler.schedule_predictions(timetable, predictor)
    
    # スケジューラ開始
    scheduler.start()
