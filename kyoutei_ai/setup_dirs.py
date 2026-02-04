"""
ディレクトリ初期化スクリプト
"""
from pathlib import Path

# 必要なディレクトリを作成
directories = [
    "data/lzh",
    "data/extracted",
    "data/cache",
    "models",
    "logs"
]

for dir_path in directories:
    path = Path(__file__).parent / dir_path
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created: {path}")

print("\nディレクトリの初期化が完了しました。")
