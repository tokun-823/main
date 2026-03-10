# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」- 機械学習モデルモジュール
Machine Learning Model Module for Keirin Prediction AI "Pasoko"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
import joblib
import os
from tqdm import tqdm
import warnings

from config import MODELS_DIR, MODEL_CATEGORIES

warnings.filterwarnings('ignore')


class KeirinModel:
    """
    競輪予測AIモデルクラス
    
    学習手法: 二値分類（Binary Classification）
    目的変数: 3着以内に入るか（1）、入らないか（0）
    """
    
    def __init__(self, category='general', model_type='random_forest'):
        """
        Args:
            category: モデルカテゴリ（7car, 9car, girls, etc.）
            model_type: モデルタイプ（random_forest, gradient_boosting, logistic）
        """
        self.category = category
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def _create_model(self):
        """
        モデルを作成
        
        【重要】回収率問題への対応
        - 精度を高めすぎないようにハイパーパラメータを調整
        - キャリブレーションカーブを意図的にズラす設計
        """
        if self.model_type == 'random_forest':
            # あえて精度を落とす調整（特徴量の重要度を分散させる）
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=8,  # 深さを制限して過学習を防ぐ
                min_samples_split=20,  # より大きな分割閾値
                min_samples_leaf=10,  # 葉ノードの最小サンプル数
                max_features='sqrt',  # 特徴量のサブサンプリング
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # クラス不均衡対応
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=80,
                max_depth=5,
                learning_rate=0.05,  # 低い学習率で過学習防止
                min_samples_split=15,
                min_samples_leaf=8,
                subsample=0.8,
                random_state=42
            )
        else:
            return LogisticRegression(
                C=0.1,  # 正則化を強めて過学習防止
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
    
    def train(self, df, feature_columns, target_column='target', test_size=0.2):
        """
        モデルを学習
        
        Args:
            df: 学習データ
            feature_columns: 特徴量カラムリスト
            target_column: 目的変数カラム
            test_size: テストデータの割合
            
        Returns:
            dict: 評価メトリクス
        """
        print(f"\n===== カテゴリ: {self.category} のモデル学習開始 =====")
        
        self.feature_columns = feature_columns
        
        # データ準備
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # 欠損値処理
        X = X.fillna(0)
        
        # 数値型に変換
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # スケーリング
        X_scaled = self.scaler.fit_transform(X)
        
        # 訓練/テスト分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # モデル作成と学習
        self.model = self._create_model()
        
        print(f"学習データ: {len(X_train)} 件")
        print(f"テストデータ: {len(X_test)} 件")
        
        self.model.fit(X_train, y_train)
        
        # 予測
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 評価
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        print(f"\n評価結果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # キャリブレーションカーブ確認
        self._check_calibration(y_test, y_proba)
        
        return metrics
    
    def _check_calibration(self, y_true, y_proba, n_bins=10):
        """
        キャリブレーションカーブを確認
        
        【重要】線をぴったり合わせる（高精度）のではなく、
        少しジグザグにズレるようなモデルの方が回収率が高くなる
        """
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_proba, n_bins=n_bins
            )
            
            print("\nキャリブレーションカーブ:")
            for i, (frac, mean_pred) in enumerate(zip(fraction_of_positives, mean_predicted_value)):
                deviation = frac - mean_pred
                status = "◎" if abs(deviation) > 0.05 else "○"  # ズレがある方が良い
                print(f"  Bin {i+1}: 予測={mean_pred:.3f}, 実際={frac:.3f}, 差分={deviation:+.3f} {status}")
                
        except Exception as e:
            print(f"キャリブレーション計算エラー: {e}")
    
    def predict(self, df):
        """
        予測を実行
        
        Args:
            df: 予測対象データ
            
        Returns:
            DataFrame: 予測結果を追加したDataFrame
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        df = df.copy()
        
        # 特徴量準備
        X = df[self.feature_columns].copy()
        X = X.fillna(0)
        
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        X_scaled = self.scaler.transform(X)
        
        # 予測
        labels = self.model.predict(X_scaled)
        probas = self.model.predict_proba(X_scaled)[:, 1]
        
        df['pred_label'] = labels
        df['pred_proba'] = probas
        
        # 【重要】符号反転処理
        # ラベルが「0」の選手のスコアにマイナスをつける
        df['pred_score'] = df.apply(
            lambda row: row['pred_proba'] if row['pred_label'] == 1 else -row['pred_proba'],
            axis=1
        )
        
        return df
    
    def rank_predictions(self, df, race_id_column='race_id'):
        """
        予測結果をランク付け（A, B, C...）
        
        スコアを降順ソートし、ラベル「0」の選手は符号反転により
        自動的に下位に配置される
        
        Args:
            df: 予測結果を含むDataFrame
            race_id_column: レースIDカラム
            
        Returns:
            DataFrame: ランクを追加したDataFrame
        """
        df = df.copy()
        df['rank_label'] = ''
        
        ranks = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        
        for race_id, group in df.groupby(race_id_column):
            # スコア降順でソート
            sorted_group = group.sort_values('pred_score', ascending=False)
            
            for idx, (row_idx, _) in enumerate(sorted_group.iterrows()):
                if idx < len(ranks):
                    df.loc[row_idx, 'rank_label'] = ranks[idx]
        
        return df
    
    def save_model(self):
        """モデルを保存"""
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        model_path = f"{MODELS_DIR}/{self.category}_{self.model_type}.joblib"
        scaler_path = f"{MODELS_DIR}/{self.category}_scaler.joblib"
        features_path = f"{MODELS_DIR}/{self.category}_features.joblib"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_columns, features_path)
        
        print(f"モデル保存: {model_path}")
    
    def load_model(self):
        """モデルを読み込み"""
        model_path = f"{MODELS_DIR}/{self.category}_{self.model_type}.joblib"
        scaler_path = f"{MODELS_DIR}/{self.category}_scaler.joblib"
        features_path = f"{MODELS_DIR}/{self.category}_features.joblib"
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_columns = joblib.load(features_path)
            print(f"モデル読み込み: {model_path}")
            return True
        return False


class ModelManager:
    """複数モデルを管理するマネージャークラス"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.models = {}
        
    def train_all_models(self, df, feature_columns):
        """
        全カテゴリのモデルを学習
        
        Args:
            df: 学習データ
            feature_columns: 特徴量カラムリスト
            
        Returns:
            dict: 各カテゴリの評価メトリクス
        """
        all_metrics = {}
        
        categories = df['category'].unique()
        
        print(f"\n全カテゴリのモデル学習開始")
        print(f"カテゴリ一覧: {categories}")
        
        for category in tqdm(categories, desc="モデル学習"):
            # カテゴリごとにデータ抽出
            cat_df = df[df['category'] == category]
            
            if len(cat_df) < 100:
                print(f"\n{category}: データ不足({len(cat_df)}件)、スキップ")
                continue
            
            # モデル作成・学習
            model = KeirinModel(category=category, model_type=self.model_type)
            metrics = model.train(cat_df, feature_columns)
            
            # 保存
            model.save_model()
            self.models[category] = model
            all_metrics[category] = metrics
        
        return all_metrics
    
    def load_all_models(self):
        """全モデルを読み込み"""
        for category in MODEL_CATEGORIES.keys():
            model = KeirinModel(category=category, model_type=self.model_type)
            if model.load_model():
                self.models[category] = model
    
    def predict(self, df, category=None):
        """
        予測を実行
        
        Args:
            df: 予測対象データ
            category: カテゴリ（Noneの場合は自動判定）
            
        Returns:
            DataFrame: 予測結果
        """
        if category is None:
            category = df['category'].iloc[0] if 'category' in df.columns else 'general'
        
        if category not in self.models:
            # fallback: 汎用モデルまたは9carモデル
            category = '9car' if '9car' in self.models else list(self.models.keys())[0]
        
        model = self.models[category]
        result = model.predict(df)
        result = model.rank_predictions(result)
        
        return result


def train_models(processed_data_path, model_type='random_forest'):
    """
    モデル学習のメイン関数
    
    Args:
        processed_data_path: 処理済みデータのパス
        model_type: モデルタイプ
        
    Returns:
        ModelManager: 学習済みモデルマネージャー
    """
    print("データ読み込み中...")
    df = pd.read_pickle(processed_data_path)
    
    # 特徴量カラムを取得
    from feature_engineering import FeatureEngineer
    engineer = FeatureEngineer()
    feature_columns = engineer.get_feature_columns(df)
    
    print(f"特徴量数: {len(feature_columns)}")
    print(f"特徴量: {feature_columns[:10]}...")
    
    # モデル学習
    manager = ModelManager(model_type=model_type)
    metrics = manager.train_all_models(df, feature_columns)
    
    # 結果サマリー
    print("\n===== 学習結果サマリー =====")
    for category, metric in metrics.items():
        print(f"\n{category}:")
        print(f"  精度: {metric['accuracy']:.4f}")
        print(f"  AUC: {metric['roc_auc']:.4f}")
    
    return manager


if __name__ == "__main__":
    # テスト（ダミーデータ）
    print("モデルモジュールテスト")
    
    # ダミーデータ作成
    np.random.seed(42)
    n_samples = 1000
    
    dummy_df = pd.DataFrame({
        'race_id': [f'race_{i // 7}' for i in range(n_samples)],
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randint(0, 2, n_samples),
        'category': ['7car'] * n_samples
    })
    
    model = KeirinModel(category='7car')
    metrics = model.train(dummy_df, ['feature1', 'feature2', 'feature3'])
    
    print("\nテスト予測:")
    result = model.predict(dummy_df.head(20))
    result = model.rank_predictions(result)
    print(result[['race_id', 'pred_label', 'pred_proba', 'pred_score', 'rank_label']].head())
