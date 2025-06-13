"""Configuration management for the planning system."""

import os
from dataclasses import dataclass
from typing import Optional

# Constants
MAX_STEPS = 15
BEAM_SIZE = 3
SLEEP_TIME = 2
MAX_SLEEP = 5

SUPPORTED_MODELS = {
    'gpt4o': ['gpt-4o-mini', 'gpt-4o'],
    'gem': ['gemini-1.5-flash'],
    'mistral': ['mistral-7b', 'mistral-8x7b']
}

SUPPORTED_MODES = [
    'vanilla', 'stable', 'tip', 'w_des', 'w_img', 'ours'
    ]

ORDER_DICT = {
    1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
    6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
    11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth', 15: 'fifteenth',
    16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth', 19: 'nineteenth', 20: 'twentieth'
}

@dataclass
class PlanningConfig:
    """Configuration class for planning parameters"""
    backbone: str
    model: str
    temperature: float
    seed: int
    max_steps: int = MAX_STEPS
    
    def __post_init__(self):
        if self.backbone not in SUPPORTED_MODELS:
            raise ValueError(f"Invalid backbone: {self.backbone}. Must be one of {list(SUPPORTED_MODELS.keys())}")
        
        if self.model not in SUPPORTED_MODELS[self.backbone]:
            raise ValueError(f"Invalid model {self.model} for backbone {self.backbone}")

@dataclass
class PathConfig:
    """Configuration for file paths"""
    data_dir: str
    save_dir: str
    cache_dir: str = './cache'
    ref_path: str = "./dataset/wikiHow_articles"
    
    def __post_init__(self):
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset not found: {self.data_dir}")
        
        os.makedirs(self.save_dir, exist_ok=True)