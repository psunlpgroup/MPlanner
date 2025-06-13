"""Main pipeline for processing planning tasks."""

import os
import pandas as pd
from typing import Dict, Optional
from tqdm import tqdm

from config import PlanningConfig, PathConfig
from models import ModelManager
from processors import (
    TaskProcessor, PassageProcessor, StepProcessor, PassageTipProcessor,
    StepTipProcessor, PassageWithDescriptionProcessor, PureImageProcessor,
    ImageToTextProcessor
)

class PlanningPipeline:
    """Main pipeline for processing planning tasks"""
    
    def __init__(self, planning_config: PlanningConfig, path_config: PathConfig):
        self.planning_config = planning_config
        self.path_config = path_config
        self.model_manager = ModelManager(planning_config, path_config)
        self.processors = self._init_processors()
    
    def _init_processors(self) -> Dict[str, TaskProcessor]:
        """Initialize all available processors"""
        processors = {
            'vanilla': PassageProcessor(self.model_manager),
            'stable': StepProcessor(self.model_manager), 
            'tip': PassageTipProcessor(self.model_manager),
            'w_des': PassageWithDescriptionProcessor(self.model_manager),
            'w_img': PureImageProcessor(self.model_manager),
            'ours': StepTipProcessor(self.model_manager),
        }
        
        return processors
    
    def process_dataset(self, mode: str, start_idx: int = 0, end_idx: Optional[int] = None) -> None:
        """Process entire dataset"""
        if mode not in self.processors:
            available_modes = list(self.processors.keys())
            raise ValueError(f"Unknown mode: {mode}. Available modes: {available_modes}")
        
        # Load dataset
        dataset = self._load_dataset()
        if end_idx is None:
            end_idx = len(dataset)
        
        # Validate range
        if not (0 <= start_idx < end_idx <= len(dataset)):
            raise ValueError(f"Invalid range: {start_idx} - {end_idx}")
        
        # Get tasks
        tasks = dataset['task'].tolist()[start_idx:end_idx]
        processor = self.processors[mode]
        
        print(f"🔧 Processing {len(tasks)} tasks using {mode} mode")
        print(f"📝 Backbone: {self.planning_config.backbone}")
        print(f"🤖 Model: {self.planning_config.model}")
        print(f"🌡️  Temperature: {self.planning_config.temperature}")
        print(f"🎲 Seed: {self.planning_config.seed}")
        print(f"💾 Save to: {self.path_config.save_dir}")
        
        # Process tasks
        successful = 0
        failed_tasks = []
        
        for idx, task in tqdm(enumerate(tasks), desc=f"Processing {mode}", total=len(tasks)):
            task_idx = start_idx + idx
            if processor.process_task(task, task_idx, self.path_config.save_dir):
                successful += 1
            else:
                failed_tasks.append(task_idx)
        
        # Report results
        print(f"\n=== Processing Results ===")
        print(f"Successfully processed: {successful}/{len(tasks)} tasks")
        if failed_tasks:
            print(f"Failed tasks: {failed_tasks}")
        
        success_rate = (successful / len(tasks)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    def _load_dataset(self) -> pd.DataFrame:
        """Load dataset from various formats"""
        data_dir = self.path_config.data_dir
        
        if 'jsonl' in data_dir:
            return pd.read_json(data_dir, lines=True)
        elif 'csv' in data_dir:
            return pd.read_csv(data_dir, delimiter=';')
        else:
            raise ValueError(f"Unsupported file format: {data_dir}")
    
    def get_available_modes(self) -> list:
        """Get list of available processing modes"""
        return list(self.processors.keys())
    
    def validate_configuration(self) -> bool:
        """Validate the current configuration"""
        try:
            # Test model connectivity
            test_response = self.model_manager.get_text_response("Hello, this is a test.")
            if test_response is None:
                print("Warning: Model connectivity test failed")
                return False
            
            # Check if dataset is accessible
            dataset = self._load_dataset()
            if len(dataset) == 0:
                print("Warning: Dataset is empty")
                return False
            
            print("Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False