# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- 機械学習モデル
====================================
二値分類モデルの学習・推論機能
カテゴリ別モデル管理
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not installed")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from .config import MODEL_DIR, ModelConfig, RaceCategory


class KeirinPredictor:
    """
    競輪予測モデルクラス
    
    特徴:
    - 二値分類（3着以内/外）
    - カテゴリ別モデル管理
    - アンサンブル対応
    - 確率スコア出力
    """
    
    def __init__(
        self,
        category: RaceCategory,
        config: Optional[ModelConfig] = None,
        model_type: str = "lightgbm"
    ):
        """
        Args:
            category: レースカテゴリ
            config: モデル設定
            model_type: モデルタイプ（lightgbm, xgboost, catboost）
        """
        self.category = category
        self.config = config or ModelConfig()
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
        # モデル保存ディレクトリ
        self.model_dir = MODEL_DIR / category.value
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_model(self):
        """モデルインスタンスを作成"""
        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return lgb.LGBMClassifier(**self.config.lgbm_params)
        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                objective="binary:logistic",
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric="auc",
            )
        elif self.model_type == "catboost" and CATBOOST_AVAILABLE:
            return CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                random_state=42,
                verbose=False,
            )
        else:
            # フォールバック: sklearn
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            )
    
    def train(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target_column: str = "target",
        validation_split: float = 0.2,
        early_stopping: bool = True,
    ) -> Dict[str, float]:
        """
        モデルを学習
        
        Args:
            df: 学習データ
            feature_columns: 特徴量列リスト
            target_column: 目的変数列名
            validation_split: 検証データ割合
            early_stopping: 早期停止の有無
            
        Returns:
            Dict[str, float]: 評価指標
        """
        logger.info(f"Training model for {self.category.value}")
        
        # 特徴量列の決定
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = self.config.feature_columns
        
        # 存在する特徴量のみ使用
        available_features = [col for col in self.feature_columns if col in df.columns]
        self.feature_columns = available_features
        
        if len(self.feature_columns) == 0:
            raise ValueError("No valid feature columns found")
        
        logger.info(f"Using {len(self.feature_columns)} features")
        
        # データ準備
        X = df[self.feature_columns].copy()
        y = df[target_column].values
        
        # 欠損値処理
        X = X.fillna(X.median())
        
        # 訓練/検証分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=42
        )
        
        # スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # モデル作成・学習
        self.model = self._create_model()
        
        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE and early_stopping:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=self.config.lgbm_params.get("early_stopping_rounds", 50),
                        verbose=False
                    )
                ],
            )
        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE and early_stopping:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        
        # 評価
        metrics = self._evaluate(X_val_scaled, y_val)
        
        logger.info(f"Training completed. AUC: {metrics['auc']:.4f}")
        return metrics
    
    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """モデルを評価"""
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc": roc_auc_score(y, y_prob),
        }
        
        return metrics
    
    def cross_validate(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
    ) -> Dict[str, float]:
        """
        交差検証を実行
        
        Args:
            df: データ
            n_folds: 分割数
            
        Returns:
            Dict[str, float]: 平均評価指標
        """
        logger.info(f"Running {n_folds}-fold cross validation")
        
        X = df[self.feature_columns].copy().fillna(0)
        y = df["target"].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        all_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self._create_model()
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_prob)
            all_metrics.append({"fold": fold + 1, "auc": auc})
            
            logger.info(f"Fold {fold + 1}: AUC = {auc:.4f}")
        
        mean_auc = np.mean([m["auc"] for m in all_metrics])
        logger.info(f"Mean AUC: {mean_auc:.4f}")
        
        return {"mean_auc": mean_auc, "folds": all_metrics}
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        予測を実行
        
        Args:
            df: 予測対象データ
            
        Returns:
            pd.DataFrame: 予測結果付きデータ
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call train() first.")
        
        result_df = df.copy()
        
        # 特徴量準備
        X = df[self.feature_columns].copy().fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # 予測
        labels = self.model.predict(X_scaled)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        
        # 結果を追加
        result_df["pred_label"] = labels
        result_df["pred_proba"] = probas
        
        # マイナス処理（0判定はスコアをマイナスに）
        result_df["adjusted_score"] = np.where(
            labels == 0,
            -probas,  # 0判定はマイナス符号
            probas
        )
        
        return result_df
    
    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        1レース分の予測を実行し、ランキングを付与
        
        Args:
            race_df: 1レース分のデータ
            
        Returns:
            pd.DataFrame: ランキング付き予測結果
        """
        result_df = self.predict(race_df)
        
        # スコアでソート（降順）
        result_df = result_df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
        
        # ランキング記号を付与
        rank_symbols = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        result_df["rank_symbol"] = [
            rank_symbols[i] if i < len(rank_symbols) else str(i + 1)
            for i in range(len(result_df))
        ]
        
        # ランク順位
        result_df["rank_order"] = range(1, len(result_df) + 1)
        
        return result_df
    
    def get_feature_importance(self) -> pd.DataFrame:
        """特徴量重要度を取得"""
        if not self.is_fitted:
            return pd.DataFrame()
        
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        else:
            return pd.DataFrame()
        
        df = pd.DataFrame({
            "feature": self.feature_columns,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        
        return df
    
    def save(self, name: Optional[str] = None) -> Path:
        """
        モデルを保存
        
        Args:
            name: 保存名（省略時は日時）
            
        Returns:
            Path: 保存先パス
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted")
        
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "category": self.category.value,
            "model_type": self.model_type,
            "config": self.config,
        }
        
        save_path = self.model_dir / f"{name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {save_path}")
        return save_path
    
    def load(self, path: Union[str, Path]) -> None:
        """
        モデルを読み込み
        
        Args:
            path: モデルファイルパス
        """
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        self.model = save_data["model"]
        self.scaler = save_data["scaler"]
        self.feature_columns = save_data["feature_columns"]
        self.model_type = save_data.get("model_type", "lightgbm")
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
    
    @classmethod
    def load_latest(cls, category: RaceCategory) -> "KeirinPredictor":
        """最新のモデルを読み込み"""
        model_dir = MODEL_DIR / category.value
        
        if not model_dir.exists():
            raise FileNotFoundError(f"No model directory for {category.value}")
        
        model_files = list(model_dir.glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No model files in {model_dir}")
        
        # 最新ファイルを取得
        latest_file = max(model_files, key=lambda p: p.stat().st_mtime)
        
        predictor = cls(category)
        predictor.load(latest_file)
        return predictor


class ModelManager:
    """
    複数モデルの管理クラス
    
    カテゴリごとのモデルを一元管理
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models: Dict[RaceCategory, KeirinPredictor] = {}
    
    def train_all(
        self,
        df: pd.DataFrame,
        categories: Optional[List[RaceCategory]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        全カテゴリのモデルを学習
        
        Args:
            df: 全データ
            categories: 学習対象カテゴリ（省略時は全て）
            
        Returns:
            Dict: カテゴリごとの評価指標
        """
        from .preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor(self.config)
        
        if categories is None:
            categories = list(RaceCategory)
        
        results = {}
        
        for category in categories:
            logger.info(f"Training model for {category.value}")
            
            # カテゴリでフィルタ
            category_df = preprocessor.filter_by_category(df, category)
            
            if len(category_df) < 100:
                logger.warning(f"Insufficient data for {category.value}: {len(category_df)} rows")
                continue
            
            # モデル作成・学習
            predictor = KeirinPredictor(category, self.config)
            metrics = predictor.train(category_df)
            
            # 保存
            predictor.save()
            
            self.models[category] = predictor
            results[category.value] = metrics
        
        return results
    
    def load_all(self) -> None:
        """全カテゴリの最新モデルを読み込み"""
        for category in RaceCategory:
            try:
                predictor = KeirinPredictor.load_latest(category)
                self.models[category] = predictor
                logger.info(f"Loaded model for {category.value}")
            except FileNotFoundError:
                logger.warning(f"No model found for {category.value}")
    
    def predict(
        self,
        df: pd.DataFrame,
        category: Optional[RaceCategory] = None,
    ) -> pd.DataFrame:
        """
        適切なモデルで予測
        
        Args:
            df: 予測対象データ
            category: 使用カテゴリ（省略時は自動判定）
            
        Returns:
            pd.DataFrame: 予測結果
        """
        if category is None:
            # 自動判定
            category = self._detect_category(df)
        
        if category not in self.models:
            # デフォルトモデルを使用
            if RaceCategory.NINE_CAR in self.models:
                category = RaceCategory.NINE_CAR
            elif self.models:
                category = list(self.models.keys())[0]
            else:
                raise RuntimeError("No models available")
        
        return self.models[category].predict(df)
    
    def _detect_category(self, df: pd.DataFrame) -> RaceCategory:
        """データからカテゴリを自動判定"""
        if "is_girls" in df.columns and df["is_girls"].any():
            return RaceCategory.GIRLS
        if "is_challenge" in df.columns and df["is_challenge"].any():
            return RaceCategory.CHALLENGE
        if "num_cars" in df.columns:
            mode_cars = df["num_cars"].mode().iloc[0] if len(df["num_cars"].mode()) > 0 else 9
            if mode_cars == 7:
                return RaceCategory.SEVEN_CAR
        if "grade_num" in df.columns:
            max_grade = df["grade_num"].max()
            if max_grade >= 6:
                return RaceCategory.G1_SPECIAL
            if max_grade == 4:
                return RaceCategory.G3_SPECIAL
        
        return RaceCategory.NINE_CAR
