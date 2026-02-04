"""
期待値予測モデル
Quantile Regressionによる期待値の分位点予測
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import config, MODEL_DIR, INDEX_TO_TRIFECTA
from src.features import FeatureEngineer, get_feature_names


class ExpectedValueModel:
    """
    期待値予測モデル
    Quantile Regressionで期待値の分位点（10%, 50%, 80%）を予測
    """
    
    def __init__(self):
        self.models = {}  # quantile -> model
        self.feature_names = get_feature_names()
        self.quantiles = config.model.quantiles  # [0.1, 0.5, 0.8]
        self.model_dir = MODEL_DIR / "ev_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_lgb_params(self, quantile: float) -> Dict:
        """LightGBMパラメータを取得"""
        return {
            "objective": "quantile",
            "alpha": quantile,  # 分位点
            "metric": "quantile",
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
        y_train: np.ndarray,  # 実際の期待値（確率 × オッズ）
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 100
    ):
        """各分位点のモデルを学習"""
        
        for quantile in self.quantiles:
            logger.info(f"Training quantile model (q={quantile})...")
            
            params = self._get_lgb_params(quantile)
            
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
            
            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=100)
            ]
            
            self.models[quantile] = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            
            logger.info(f"q={quantile} training completed. Best iteration: {self.models[quantile].best_iteration}")
        
        return self
    
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """各分位点の期待値を予測"""
        predictions = {}
        
        for quantile, model in self.models.items():
            predictions[quantile] = model.predict(X, num_iteration=model.best_iteration)
        
        return predictions
    
    def predict_ev_bounds(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """期待値の下限・中央・上限を予測"""
        preds = self.predict(X)
        
        q_low = min(self.quantiles)
        q_mid = 0.5 if 0.5 in self.quantiles else sorted(self.quantiles)[len(self.quantiles)//2]
        q_high = max(self.quantiles)
        
        ev_low = preds.get(q_low, np.zeros(len(X)))
        ev_mid = preds.get(q_mid, np.zeros(len(X)))
        ev_high = preds.get(q_high, np.zeros(len(X)))
        
        return ev_low, ev_mid, ev_high
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """モデルを評価"""
        
        metrics = {}
        predictions = self.predict(X_test)
        
        for quantile, pred in predictions.items():
            # Pinball Loss
            errors = y_test - pred
            pinball_loss = np.mean(
                np.where(errors >= 0, quantile * errors, (quantile - 1) * errors)
            )
            
            # Coverage（実際の値が予測範囲内に収まる割合）
            coverage = np.mean(y_test <= pred)
            
            # MAE
            mae = np.mean(np.abs(errors))
            
            metrics[f'q{quantile}'] = {
                'pinball_loss': pinball_loss,
                'coverage': coverage,
                'mae': mae
            }
            
            logger.info(f"q={quantile}: Pinball={pinball_loss:.4f}, Coverage={coverage:.4f}, MAE={mae:.4f}")
        
        return metrics
    
    def save(self, path: Path = None):
        """モデルを保存"""
        path = path or self.model_dir
        
        for quantile, model in self.models.items():
            model_path = path / f"ev_model_q{int(quantile*100)}.txt"
            model.save_model(str(model_path))
        
        # メタデータ
        meta_path = path / "ev_model.meta.json"
        meta = {
            'quantiles': self.quantiles,
            'feature_names': self.feature_names,
            'created_at': datetime.now().isoformat()
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"EV models saved to {path}")
    
    def load(self, path: Path = None):
        """モデルを読み込み"""
        path = path or self.model_dir
        
        # メタデータ読み込み
        meta_path = path / "ev_model.meta.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            self.quantiles = meta.get('quantiles', self.quantiles)
            self.feature_names = meta.get('feature_names', self.feature_names)
        
        # 各分位点モデル読み込み
        for quantile in self.quantiles:
            model_path = path / f"ev_model_q{int(quantile*100)}.txt"
            if model_path.exists():
                self.models[quantile] = lgb.Booster(model_file=str(model_path))
        
        logger.info(f"EV models loaded from {path}")
        return self


class EVFeatureBuilder:
    """期待値予測用の特徴量ビルダー"""
    
    def __init__(self):
        self.prob_model = None  # 確率モデル
    
    def set_probability_model(self, prob_model):
        """確率モデルを設定"""
        self.prob_model = prob_model
    
    def build_features(
        self,
        base_features: np.ndarray,  # 基本特徴量
        prelim_odds: np.ndarray,    # 締切前オッズ（約3分前）
        predicted_probs: np.ndarray = None  # 予測確率
    ) -> np.ndarray:
        """期待値予測用の特徴量を構築"""
        
        features_list = []
        
        for i in range(len(base_features)):
            feat = list(base_features[i])
            
            # オッズ情報
            if i < len(prelim_odds):
                feat.extend([
                    prelim_odds[i],  # 締切前オッズ
                    np.log1p(prelim_odds[i]),  # 対数オッズ
                ])
            else:
                feat.extend([0, 0])
            
            # 予測確率
            if predicted_probs is not None and i < len(predicted_probs):
                feat.extend([
                    predicted_probs[i],
                    predicted_probs[i] * prelim_odds[i] if i < len(prelim_odds) else 0,  # 粗期待値
                ])
            else:
                feat.extend([0, 0])
            
            features_list.append(feat)
        
        return np.array(features_list, dtype=np.float32)
    
    def build_training_data(
        self,
        base_features: np.ndarray,
        prelim_odds: np.ndarray,
        actual_odds: np.ndarray,  # 確定オッズ
        actual_probs: np.ndarray,  # 実際の確率（結果から逆算）
        hit_flags: np.ndarray  # 的中フラグ
    ) -> Tuple[np.ndarray, np.ndarray]:
        """学習データを構築"""
        
        # 予測確率を取得
        if self.prob_model:
            predicted_probs = self.prob_model.predict_proba(base_features)
            # 最も確率の高い組み合わせの確率
            max_probs = np.max(predicted_probs, axis=1)
        else:
            max_probs = np.zeros(len(base_features))
        
        # 特徴量構築
        X = self.build_features(base_features, prelim_odds, max_probs)
        
        # 期待値（ラベル）
        # 実際の確率 × 確定オッズ
        # ただし、的中した場合は確定オッズ、外れた場合は0として計算
        y = np.where(hit_flags, actual_odds, 0)
        
        return X, y


class CombinedPredictor:
    """確率モデルと期待値モデルを組み合わせた予測器"""
    
    def __init__(self, prob_model=None, ev_model=None):
        self.prob_model = prob_model
        self.ev_model = ev_model
    
    def predict(
        self,
        X: np.ndarray,
        prelim_odds: Dict[Tuple[int, int, int], float] = None
    ) -> List[Dict]:
        """
        確率と期待値を予測
        
        Returns:
            List[Dict]: 各組み合わせの予測結果
                - combination: (1着, 2着, 3着)
                - probability: 的中確率
                - ev_low: 期待値下限 (10%ile)
                - ev_mid: 期待値中央 (50%ile)
                - ev_high: 期待値上限 (80%ile)
                - prelim_odds: 締切前オッズ
        """
        
        results = []
        
        # 確率予測
        if self.prob_model:
            probs = self.prob_model.predict_proba(X)
        else:
            probs = np.ones((len(X), 120)) / 120
        
        # 各組み合わせについて
        for combo_idx in range(120):
            combo = INDEX_TO_TRIFECTA[combo_idx]
            prob = probs[0][combo_idx] if len(probs) > 0 else 1/120
            
            # オッズ取得
            odds = prelim_odds.get(combo, 0) if prelim_odds else 0
            
            # 期待値予測
            if self.ev_model and odds > 0:
                # 期待値モデル用の特徴量を構築
                ev_features = np.concatenate([X[0], [odds, np.log1p(odds), prob, prob * odds]])
                ev_features = ev_features.reshape(1, -1)
                
                ev_low, ev_mid, ev_high = self.ev_model.predict_ev_bounds(ev_features)
                ev_low = ev_low[0]
                ev_mid = ev_mid[0]
                ev_high = ev_high[0]
            else:
                # 単純な期待値計算
                ev_low = ev_mid = ev_high = prob * odds if odds > 0 else 0
            
            results.append({
                'combination': combo,
                'probability': prob,
                'ev_low': ev_low,
                'ev_mid': ev_mid,
                'ev_high': ev_high,
                'prelim_odds': odds
            })
        
        # 期待値でソート
        results.sort(key=lambda x: x['ev_mid'], reverse=True)
        
        return results
    
    def filter_positive_ev(
        self,
        predictions: List[Dict],
        min_ev: float = 1.0,
        use_conservative: bool = True
    ) -> List[Dict]:
        """
        期待値がプラスの買い目をフィルタリング
        
        Args:
            min_ev: 最低期待値閾値
            use_conservative: True の場合は ev_low、False の場合は ev_mid を使用
        """
        
        ev_key = 'ev_low' if use_conservative else 'ev_mid'
        
        return [p for p in predictions if p[ev_key] >= min_ev]


def train_ev_model(
    start_date: str = "20180101",
    end_date: str = "20241231"
) -> ExpectedValueModel:
    """期待値モデルを学習"""
    
    # 注: 実際の実装では、オッズデータが必要
    # ここではダミーデータで構造を示す
    
    logger.info("Training EV model requires odds data")
    logger.info("Please ensure odds data is collected via scraper first")
    
    # モデル初期化
    model = ExpectedValueModel()
    
    # 学習データがある場合
    # X, y = prepare_ev_training_data(start_date, end_date)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    # model.train(X_train, y_train, X_val, y_val)
    # model.evaluate(X_test, y_test)
    # model.save()
    
    return model


if __name__ == "__main__":
    model = train_ev_model()
