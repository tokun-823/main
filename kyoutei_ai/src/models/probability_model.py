"""
確率推定モデル
LightGBMによる3連単的中確率の予測
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pickle
import json
from datetime import datetime
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, accuracy_score
import optuna

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    config, MODEL_DIR,
    TRIFECTA_COMBINATIONS, INDEX_TO_TRIFECTA
)
from src.features import FeatureEngineer, get_feature_names


class ProbabilityModel:
    """確率推定モデル"""
    
    def __init__(self):
        self.model = None
        self.feature_names = get_feature_names()
        self.model_path = Path(config.model.probability_model_path)
        self.num_classes = 120  # 3連単の組み合わせ数
        
        # LightGBMパラメータ
        self.params = {
            "objective": "multiclass",
            "num_class": self.num_classes,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        num_boost_round: int = None,
        early_stopping_rounds: int = None
    ):
        """モデルを学習"""
        
        num_boost_round = num_boost_round or config.model.num_boost_round
        early_stopping_rounds = early_stopping_rounds or config.model.early_stopping_rounds
        
        # データセット作成
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=self.feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # 学習
        logger.info("Training probability model...")
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=100)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """確率を予測"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """最も確率の高いクラスを予測"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_top_k(self, X: np.ndarray, k: int = 10) -> List[List[Tuple[Tuple[int, int, int], float]]]:
        """上位k個の予測を返す"""
        proba = self.predict_proba(X)
        
        results = []
        for i in range(len(proba)):
            top_indices = np.argsort(proba[i])[::-1][:k]
            predictions = []
            for idx in top_indices:
                combo = INDEX_TO_TRIFECTA[idx]
                prob = proba[i][idx]
                predictions.append((combo, prob))
            results.append(predictions)
        
        return results
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """モデルを評価"""
        
        proba = self.predict_proba(X_test)
        pred = np.argmax(proba, axis=1)
        
        # 各種メトリクス
        accuracy = accuracy_score(y_test, pred)
        logloss = log_loss(y_test, proba)
        
        # Top-k精度
        top_3_acc = self._top_k_accuracy(proba, y_test, k=3)
        top_5_acc = self._top_k_accuracy(proba, y_test, k=5)
        top_10_acc = self._top_k_accuracy(proba, y_test, k=10)
        
        metrics = {
            'accuracy': accuracy,
            'log_loss': logloss,
            'top_3_accuracy': top_3_acc,
            'top_5_accuracy': top_5_acc,
            'top_10_accuracy': top_10_acc
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def _top_k_accuracy(
        self,
        proba: np.ndarray,
        y_true: np.ndarray,
        k: int
    ) -> float:
        """Top-k精度を計算"""
        top_k_pred = np.argsort(proba, axis=1)[:, -k:]
        correct = sum(y_true[i] in top_k_pred[i] for i in range(len(y_true)))
        return correct / len(y_true)
    
    def calibration_analysis(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """キャリブレーション分析"""
        
        proba = self.predict_proba(X_test)
        
        # 勝利確率（最高確率）のキャリブレーション
        max_proba = np.max(proba, axis=1)
        correct = (np.argmax(proba, axis=1) == y_test).astype(int)
        
        prob_true, prob_pred = calibration_curve(correct, max_proba, n_bins=n_bins)
        
        return {
            'prob_true': prob_true,
            'prob_pred': prob_pred,
            'mean_predicted_proba': max_proba.mean(),
            'actual_accuracy': correct.mean()
        }
    
    def save(self, path: Path = None):
        """モデルを保存"""
        path = path or self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(path))
        
        # メタデータも保存
        meta_path = path.with_suffix('.meta.json')
        meta = {
            'feature_names': self.feature_names,
            'params': self.params,
            'best_iteration': self.model.best_iteration,
            'created_at': datetime.now().isoformat()
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path = None):
        """モデルを読み込み"""
        path = path or self.model_path
        
        self.model = lgb.Booster(model_file=str(path))
        
        # メタデータ読み込み
        meta_path = path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.feature_names = meta.get('feature_names', self.feature_names)
            self.params = meta.get('params', self.params)
        
        logger.info(f"Model loaded from {path}")
        return self
    
    def feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """特徴量重要度を取得"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False)
        
        return df


class ProbabilityModelOptimizer:
    """ハイパーパラメータ最適化"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self.feature_names = get_feature_names()
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optunaの目的関数"""
        
        params = {
            "objective": "multiclass",
            "num_class": 120,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42
        }
        
        # 交差検証
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)  # ログ抑制
            ]
        )
        
        proba = model.predict(X_val, num_iteration=model.best_iteration)
        loss = log_loss(y_val, proba)
        
        return loss
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """最適化を実行"""
        
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        return study.best_trial.params


def train_probability_model(
    start_date: str = "20180101",
    end_date: str = "20241231",
    optimize: bool = False
) -> ProbabilityModel:
    """確率モデルを学習"""
    
    # 特徴量生成
    fe = FeatureEngineer()
    X, y, race_ids = fe.create_dataset(start_date, end_date)
    
    if len(X) == 0:
        logger.error("No data for training")
        return None
    
    # 訓練/検証/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.model.train_test_split_ratio, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config.model.validation_ratio, random_state=42
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # ハイパーパラメータ最適化
    if optimize:
        optimizer = ProbabilityModelOptimizer(X_train, y_train)
        best_params = optimizer.optimize(n_trials=50)
    else:
        best_params = {}
    
    # モデル学習
    model = ProbabilityModel()
    if best_params:
        model.params.update(best_params)
    
    model.train(X_train, y_train, X_val, y_val)
    
    # 評価
    metrics = model.evaluate(X_test, y_test)
    
    # キャリブレーション分析
    calib = model.calibration_analysis(X_test, y_test)
    logger.info(f"Calibration - Predicted: {calib['mean_predicted_proba']:.4f}, Actual: {calib['actual_accuracy']:.4f}")
    
    # 特徴量重要度
    importance = model.feature_importance()
    logger.info(f"Top 10 features:\n{importance.head(10)}")
    
    # 保存
    model.save()
    
    return model


if __name__ == "__main__":
    model = train_probability_model(optimize=False)
