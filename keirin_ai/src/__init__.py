# -*- coding: utf-8 -*-
"""
競輪予測AI「パソ子」- srcパッケージ初期化
"""

from .config import (
    config,
    Config,
    BankType,
    GradeType,
    RaceCategory,
    ZoneType,
    ScrapingConfig,
    ModelConfig,
    ZoneThresholds,
    OutputConfig,
    DiceConfig,
)

from .utils import (
    setup_logger,
    parse_date,
    get_date_range,
    get_bank_type,
    get_bank_type_numeric,
    get_grade_type,
    get_race_category,
    classify_zone,
)

__version__ = "1.0.0"
__author__ = "競輪予測AI パソ子"

__all__ = [
    # Config
    "config",
    "Config",
    "BankType",
    "GradeType",
    "RaceCategory",
    "ZoneType",
    "ScrapingConfig",
    "ModelConfig",
    "ZoneThresholds",
    "OutputConfig",
    "DiceConfig",
    # Utils
    "setup_logger",
    "parse_date",
    "get_date_range",
    "get_bank_type",
    "get_bank_type_numeric",
    "get_grade_type",
    "get_race_category",
    "classify_zone",
]
