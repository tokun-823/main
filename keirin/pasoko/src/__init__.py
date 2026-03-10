# -*- coding: utf-8 -*-
"""
競輪予想AI「パソ子」
Keirin Prediction AI "Pasoko"

過去の出走表データとレース結果を学習し、
オッズ確定前の未来レースを予測するAIシステム
"""

from .config import *
from .scraper import KeirinScraper
from .feature_engineering import FeatureEngineer
from .model import KeirinModel, ModelManager
from .prediction_engine import PredictionEngine
from .zone_classifier import ZoneClassifier, TreasureHunter, BettingGenerator
from .ikasama_dice import IkasamaDice

__version__ = '1.0.0'
__author__ = 'Keirin AI Team'
