"""
モデル学習実行スクリプト

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --data_path data/01_preprocessed/features.csv
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from common.src.utils import load_config, setup_logging, ensure_dir
from v3.src.preprocessing import preprocess_pipeline
from v3.src.feature_engineering import create_features_pipeline
from v3.src.train import train_pipeline

logger = setup_logging()


def run_preprocessing(config: dict, force: bool = False):
    """前処理を実行"""
    raw_df_dir = project_root / config.get('paths', {}).get('raw_df', 'data/raw_df')
    preprocessed_dir = ensure_dir(project_root / config.get('paths', {}).get('preprocessed', 'data/01_preprocessed'))
    
    race_raw_path = raw_df_dir / 'race_results.csv'
    horse_raw_path = raw_df_dir / 'horse_history.csv'
    preprocessed_path = preprocessed_dir / 'preprocessed.csv'
    
    if not race_raw_path.exists():
        logger.error(f"レース生データが見つかりません: {race_raw_path}")
        logger.error("先に scripts/collect_data.py を実行してください")
        return None
    
    if not horse_raw_path.exists():
        logger.error(f"馬成績生データが見つかりません: {horse_raw_path}")
        return None
    
    if preprocessed_path.exists() and not force:
        logger.info(f"前処理済みデータを使用: {preprocessed_path}")
        return str(preprocessed_path)
    
    logger.info("前処理実行...")
    preprocess_pipeline(
        str(race_raw_path),
        str(horse_raw_path),
        str(preprocessed_path),
        config
    )
    
    return str(preprocessed_path)


def run_feature_engineering(config: dict, preprocessed_path: str, force: bool = False):
    """特徴量エンジニアリングを実行"""
    preprocessed_dir = project_root / config.get('paths', {}).get('preprocessed', 'data/01_preprocessed')
    features_path = preprocessed_dir / 'features.csv'
    
    if features_path.exists() and not force:
        logger.info(f"特徴量データを使用: {features_path}")
        return str(features_path)
    
    logger.info("特徴量エンジニアリング実行...")
    create_features_pipeline(
        preprocessed_path,
        str(features_path),
        config=config
    )
    
    return str(features_path)


def run_training(config: dict, features_path: str):
    """モデル学習を実行"""
    models_dir = ensure_dir(project_root / config.get('paths', {}).get('models', 'models'))
    model_path = models_dir / 'lightgbm_model.pkl'
    
    logger.info("モデル学習実行...")
    results = train_pipeline(
        features_path,
        str(model_path),
        config
    )
    
    return results, str(model_path)


def main():
    parser = argparse.ArgumentParser(description='モデル学習スクリプト')
    parser.add_argument('--data_path', type=str, help='特徴量データのパス')
    parser.add_argument('--force', action='store_true', help='キャッシュを無視して再処理')
    parser.add_argument('--step', type=str, choices=['all', 'preprocess', 'features', 'train'],
                        default='all', help='実行ステップ')
    
    args = parser.parse_args()
    
    config = load_config()
    
    try:
        preprocessed_path = None
        features_path = args.data_path
        
        if args.step in ['all', 'preprocess']:
            preprocessed_path = run_preprocessing(config, force=args.force)
            if preprocessed_path is None:
                return
        
        if args.step in ['all', 'features']:
            if preprocessed_path is None:
                preprocessed_dir = project_root / config.get('paths', {}).get('preprocessed', 'data/01_preprocessed')
                preprocessed_path = str(preprocessed_dir / 'preprocessed.csv')
            
            features_path = run_feature_engineering(config, preprocessed_path, force=args.force)
        
        if args.step in ['all', 'train']:
            if features_path is None:
                preprocessed_dir = project_root / config.get('paths', {}).get('preprocessed', 'data/01_preprocessed')
                features_path = str(preprocessed_dir / 'features.csv')
            
            if not Path(features_path).exists():
                logger.error(f"特徴量データが見つかりません: {features_path}")
                return
            
            results, model_path = run_training(config, features_path)
            
            logger.info("=" * 50)
            logger.info("学習完了")
            logger.info(f"モデル保存先: {model_path}")
            logger.info("=" * 50)
            logger.info("テストデータ評価結果:")
            for key, value in results['test_metrics'].items():
                logger.info(f"  {key}: {value:.4f}")
        
        logger.info("処理完了")
        
    except KeyboardInterrupt:
        logger.info("中断されました")
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == '__main__':
    main()
