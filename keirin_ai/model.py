"""
機械学習モデルモジュール
競輪予測AIのモデル学習・保存・読み込みを行う
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from typing import Dict, Tuple, Optional
import os
from datetime import datetime


class KeirinModel:
    """競輪予測モデル基底クラス"""
    
    def __init__(self, model_name: str, race_category: str, **model_params):
        self.model_name = model_name
        self.race_category = race_category
        self.model_params = model_params
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def build_model(self):
        """モデルの構築"""
        self.model = RandomForestClassifier(**self.model_params)
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
             validation_split: float = 0.2) -> Dict:
        """
        モデルの学習
        
        Returns:
            学習結果の評価指標
        """
        if self.model is None:
            self.build_model()
        
        # 特徴量名を保存
        self.feature_names = X.columns.tolist()
        
        # 学習データと検証データに分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\n=== {self.race_category}モデル学習開始 ===")
        print(f"学習データ: {len(X_train)}行")
        print(f"検証データ: {len(X_val)}行")
        
        # モデル学習
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # 予測
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # 確率予測
        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        
        # 評価指標の計算
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
            'val_precision': precision_score(y_val, y_val_pred, zero_division=0),
            'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
            'val_recall': recall_score(y_val, y_val_pred, zero_division=0),
            'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
            'val_f1': f1_score(y_val, y_val_pred, zero_division=0),
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'val_auc': roc_auc_score(y_val, y_val_proba),
        }
        
        # クロスバリデーション
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        # 結果表示
        print("\n【学習結果】")
        print(f"訓練精度: {metrics['train_accuracy']:.4f}")
        print(f"検証精度: {metrics['val_accuracy']:.4f}")
        print(f"検証AUC: {metrics['val_auc']:.4f}")
        print(f"CV-AUC: {metrics['cv_auc_mean']:.4f} (±{metrics['cv_auc_std']:.4f})")
        
        # 特徴量重要度
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n【特徴量重要度トップ5】")
            print(feature_importance.head())
            metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        確率予測（3着以内に入る確率）
        
        Returns:
            各選手の3着以内に入る確率（0~1）
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")
        
        # 特徴量の順序を確認
        X_ordered = X[self.feature_names]
        
        # 確率予測（クラス1の確率）
        probabilities = self.model.predict_proba(X_ordered)[:, 1]
        
        return probabilities
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        ラベル予測（0 or 1）
        """
        if not self.is_trained:
            raise ValueError("モデルが学習されていません")
        
        X_ordered = X[self.feature_names]
        predictions = self.model.predict(X_ordered)
        
        return predictions
    
    def save_model(self, save_dir: str = './models'):
        """モデルの保存"""
        if not self.is_trained:
            raise ValueError("学習済みモデルがありません")
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.race_category}_{self.model_name}_{timestamp}.pkl"
        filepath = os.path.join(save_dir, filename)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_params': self.model_params,
            'race_category': self.race_category,
            'model_name': self.model_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"\nモデル保存完了: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str):
        """モデルの読み込み"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_params = model_data.get('model_params', {})
        self.race_category = model_data.get('race_category', 'unknown')
        self.model_name = model_data.get('model_name', 'unknown')
        self.is_trained = True
        
        print(f"モデル読み込み完了: {filepath}")
        print(f"カテゴリ: {self.race_category}")
        print(f"特徴量数: {len(self.feature_names)}")


class MultiModelManager:
    """複数モデル管理クラス"""
    
    def __init__(self, model_params: Dict = None):
        self.models = {}
        self.model_params = model_params or {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
    def create_models(self, race_categories: list):
        """カテゴリごとにモデルを作成"""
        for category in race_categories:
            model = KeirinModel(
                model_name='RandomForest',
                race_category=category,
                **self.model_params
            )
            self.models[category] = model
            print(f"モデル作成: {category}")
    
    def train_all_models(self, data_dict: Dict[str, Tuple[pd.DataFrame, pd.Series]]):
        """
        全モデルの学習
        
        Args:
            data_dict: {race_category: (X, y)} の辞書
        """
        results = {}
        
        for category, (X, y) in data_dict.items():
            if category not in self.models:
                print(f"警告: {category}のモデルが存在しません")
                continue
            
            print(f"\n{'='*60}")
            print(f"カテゴリ: {category}")
            print(f"{'='*60}")
            
            metrics = self.models[category].train(X, y)
            results[category] = metrics
        
        return results
    
    def save_all_models(self, save_dir: str = './models'):
        """全モデルの保存"""
        saved_paths = {}
        for category, model in self.models.items():
            if model.is_trained:
                path = model.save_model(save_dir)
                saved_paths[category] = path
        return saved_paths
    
    def load_all_models(self, model_dir: str = './models'):
        """全モデルの読み込み"""
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")
        
        for filename in os.listdir(model_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(model_dir, filename)
                # カテゴリ名を抽出
                category = filename.split('_')[0]
                
                if category not in self.models:
                    self.models[category] = KeirinModel(
                        model_name='RandomForest',
                        race_category=category
                    )
                
                self.models[category].load_model(filepath)
    
    def get_model(self, race_category: str) -> Optional[KeirinModel]:
        """特定カテゴリのモデルを取得"""
        return self.models.get(race_category)


if __name__ == '__main__':
    # テスト実行
    print("=== モデルモジュールテスト ===")
    
    # サンプルデータ生成
    from data_processing import create_sample_data, FeatureEngineering
    
    df = create_sample_data(num_races=100, num_racers_per_race=9)
    
    # 特徴量エンジニアリング
    fe = FeatureEngineering()
    df = fe.create_target_variable(df)
    df = fe.create_ranking_features(df)
    
    # 特徴量とターゲット
    feature_cols = ['car_number', 'race_score', 'back_count', 'score_rank', 'back_rank']
    X = df[feature_cols]
    y = df['target']
    
    # モデル作成・学習
    model = KeirinModel(
        model_name='RandomForest',
        race_category='RACE_9',
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    metrics = model.train(X, y)
    
    # 予測テスト
    probas = model.predict_proba(X[:10])
    print("\n予測確率（最初の10人）:")
    print(probas)
    
    # モデル保存
    model.save_model('./models')
