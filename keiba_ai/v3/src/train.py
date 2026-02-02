"""
LightGBM モデル学習モジュール
フェーズ2-7: 機械学習モデル学習機能
フェーズ3-13: ハイパーパラメータ調整

- 時系列データを考慮したデータ分割
- LightGBMによる2値分類モデル
- Early Stopping
- モデル保存
"""

import pickle
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from common.src.utils import load_config, get_project_root, ensure_dir, setup_logging

logger = setup_logging()


class DataSplitter:
    """時系列データ分割クラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.split_config = self.config.get('model', {}).get('data_split', {})
    
    def split_by_date(
        self, 
        df: pd.DataFrame,
        date_column: str = 'date'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """日付ベースでデータを分割
        
        時系列データを考慮し、未来のデータがリークしないよう分割
        
        Args:
            df: 特徴量DataFrame
            date_column: 日付カラム名
        
        Returns:
            (train_df, valid_df, test_df)
        """
        df = df.copy()
        
        # 日付を変換
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column])
        
        # 分割日付を取得
        train_end = pd.to_datetime(self.split_config.get('train_end', '2023-12-31'))
        valid_start = pd.to_datetime(self.split_config.get('valid_start', '2024-01-01'))
        valid_end = pd.to_datetime(self.split_config.get('valid_end', '2024-06-30'))
        test_start = pd.to_datetime(self.split_config.get('test_start', '2024-07-01'))
        
        # 分割
        train_df = df[df[date_column] <= train_end]
        valid_df = df[(df[date_column] >= valid_start) & (df[date_column] <= valid_end)]
        test_df = df[df[date_column] >= test_start]
        
        logger.info(f"学習データ: {len(train_df)}行 (〜{train_end.strftime('%Y-%m-%d')})")
        logger.info(f"検証データ: {len(valid_df)}行 ({valid_start.strftime('%Y-%m-%d')}〜{valid_end.strftime('%Y-%m-%d')})")
        logger.info(f"テストデータ: {len(test_df)}行 ({test_start.strftime('%Y-%m-%d')}〜)")
        
        return train_df, valid_df, test_df


class LightGBMTrainer:
    """LightGBMモデル学習クラス"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.model_config = self.config.get('model', {})
        self.lgb_params = self.model_config.get('lgb_params', {})
        self.training_config = self.model_config.get('training', {})
        
        self.model = None
        self.feature_importance = None
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """特徴量カラムを取得
        
        Args:
            df: DataFrame
        
        Returns:
            特徴量カラムリスト
        """
        # 除外するカラム
        exclude_cols = [
            # ID系
            'race_id', 'horse_id', 'jockey_id', 'trainer_id', 'sire_id', 'bms_id',
            # 日付系
            'date', 'year', 'month', 'day',
            # 目的変数
            'rank', 'is_win', 'is_place', 'target_win', 'target_extended',
            # 結果系（リーク）
            'time', 'time_seconds', 'margin', 'prize', 'last_3f',
            # テキスト系
            'horse_name', 'jockey', 'trainer', 'race_name', 'sex', 'sex_age',
            'place', 'weather', 'track_condition', 'race_type', 'course_direction',
            'race_class', 'horse_weight', 'running_style'
        ]
        
        # 数値型カラムのみ選択
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 除外カラムを除く
        feature_cols = [c for c in numeric_cols if c not in exclude_cols]
        
        return feature_cols
    
    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        target_column: str = 'target_extended',
        feature_columns: Optional[List[str]] = None
    ) -> lgb.Booster:
        """モデルを学習
        
        Args:
            train_df: 学習データ
            valid_df: 検証データ
            target_column: 目的変数カラム
            feature_columns: 特徴量カラムリスト
        
        Returns:
            学習済みLightGBMモデル
        """
        # 特徴量カラムを決定
        if feature_columns is None:
            feature_columns = self.get_feature_columns(train_df)
        
        logger.info(f"特徴量数: {len(feature_columns)}")
        
        # データセット作成
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        X_valid = valid_df[feature_columns]
        y_valid = valid_df[target_column]
        
        # 欠損値を処理
        X_train = X_train.fillna(-999)
        X_valid = X_valid.fillna(-999)
        
        # LightGBMデータセット
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)
        
        # パラメータ
        params = self.lgb_params.copy()
        
        # 学習
        logger.info("モデル学習開始...")
        
        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=self.training_config.get('num_boost_round', 10000),
            valid_sets=[lgb_train, lgb_valid],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.training_config.get('early_stopping_rounds', 100)
                ),
                lgb.log_evaluation(
                    period=self.training_config.get('verbose_eval', 100)
                )
            ]
        )
        
        # 特徴量重要度
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info(f"学習完了: Best iteration = {self.model.best_iteration}")
        
        return self.model
    
    def predict(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """予測を実行
        
        Args:
            df: 予測対象データ
            feature_columns: 特徴量カラムリスト
        
        Returns:
            予測確率
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        if feature_columns is None:
            feature_columns = self.get_feature_columns(df)
        
        X = df[feature_columns].fillna(-999)
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def evaluate(
        self, 
        df: pd.DataFrame, 
        target_column: str = 'target_extended',
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """モデルを評価
        
        Args:
            df: 評価データ
            target_column: 目的変数カラム
            feature_columns: 特徴量カラムリスト
        
        Returns:
            評価指標の辞書
        """
        if feature_columns is None:
            feature_columns = self.get_feature_columns(df)
        
        y_true = df[target_column]
        y_pred_proba = self.predict(df, feature_columns)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'logloss': log_loss(y_true, y_pred_proba),
        }
        
        logger.info("評価結果:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """モデルを保存
        
        Args:
            path: 保存先パス
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # モデルと関連情報を保存
        save_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'config': self.config,
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"モデル保存完了: {save_path}")
    
    def load_model(self, path: str) -> lgb.Booster:
        """モデルを読み込み
        
        Args:
            path: モデルファイルパス
        
        Returns:
            LightGBMモデル
        """
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        self.model = save_data['model']
        self.feature_importance = save_data.get('feature_importance')
        
        logger.info(f"モデル読み込み完了: {path}")
        
        return self.model
    
    def get_top_features(self, n: int = 20) -> pd.DataFrame:
        """重要度上位の特徴量を取得
        
        Args:
            n: 取得数
        
        Returns:
            特徴量重要度DataFrame
        """
        if self.feature_importance is None:
            raise ValueError("特徴量重要度がありません")
        
        return self.feature_importance.head(n)


class CalibrationEvaluator:
    """キャリブレーション評価クラス
    フェーズ3-19: キャリブレーション
    """
    
    @staticmethod
    def plot_calibration_curve(
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
        save_path: Optional[str] = None
    ):
        """キャリブレーションプロットを作成
        
        Args:
            y_true: 正解ラベル
            y_pred_proba: 予測確率
            n_bins: ビン数
            save_path: 保存先パス
        """
        import matplotlib.pyplot as plt
        
        # キャリブレーション曲線を計算
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        
        # プロット
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        ax.plot(prob_pred, prob_true, 's-', label='Model')
        
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title('Calibration Plot')
        ax.legend(loc='lower right')
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"キャリブレーションプロット保存: {save_path}")
        
        plt.close()
        
        return prob_true, prob_pred
    
    @staticmethod
    def calculate_brier_score(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Brier Scoreを計算
        
        Args:
            y_true: 正解ラベル
            y_pred_proba: 予測確率
        
        Returns:
            Brier Score
        """
        return np.mean((y_pred_proba - y_true) ** 2)


def train_pipeline(
    feature_data_path: str,
    model_output_path: str,
    config: Optional[dict] = None
) -> Dict[str, Any]:
    """学習パイプライン
    
    Args:
        feature_data_path: 特徴量データのパス
        model_output_path: モデル出力パス
        config: 設定辞書
    
    Returns:
        学習結果の辞書
    """
    config = config or load_config()
    
    # データ読み込み
    logger.info("データ読み込み...")
    df = pd.read_csv(feature_data_path)
    
    # 日付を変換
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # データ分割
    logger.info("データ分割...")
    splitter = DataSplitter(config)
    train_df, valid_df, test_df = splitter.split_by_date(df)
    
    # モデル学習
    logger.info("モデル学習...")
    trainer = LightGBMTrainer(config)
    feature_columns = trainer.get_feature_columns(train_df)
    
    model = trainer.train(
        train_df, valid_df,
        target_column='target_extended',
        feature_columns=feature_columns
    )
    
    # 評価
    logger.info("モデル評価...")
    train_metrics = trainer.evaluate(train_df, feature_columns=feature_columns)
    valid_metrics = trainer.evaluate(valid_df, feature_columns=feature_columns)
    test_metrics = trainer.evaluate(test_df, feature_columns=feature_columns)
    
    # モデル保存
    trainer.save_model(model_output_path)
    
    # 特徴量重要度を出力
    top_features = trainer.get_top_features(20)
    logger.info("特徴量重要度 Top 20:")
    for _, row in top_features.iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")
    
    # キャリブレーション評価
    logger.info("キャリブレーション評価...")
    test_pred = trainer.predict(test_df, feature_columns)
    brier = CalibrationEvaluator.calculate_brier_score(
        test_df['target_extended'].values, test_pred
    )
    logger.info(f"Brier Score: {brier:.4f}")
    
    # 結果を返す
    results = {
        'train_metrics': train_metrics,
        'valid_metrics': valid_metrics,
        'test_metrics': test_metrics,
        'brier_score': brier,
        'feature_importance': trainer.feature_importance,
        'feature_columns': feature_columns,
    }
    
    return results
