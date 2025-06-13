#!/usr/bin/env python3
"""Main entry point for the planning system."""

import argparse
import sys
from pathlib import Path

from config import PlanningConfig, PathConfig, SUPPORTED_MODES
from pipeline import PlanningPipeline

def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        'MLLM Planner',
        description='Multi-modal Large Language Model Planning System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration settings
    config_group = parser.add_argument_group('Model Configuration')
    config_group.add_argument(
        '--backbone', 
        type=str, 
        default='gpt4o',
        choices=['gpt4o', 'gem', 'mistral'],
        help='Model backbone to use'
    )
    config_group.add_argument(
        '--model', 
        type=str, 
        default='gpt-4o',
        help='Specific model name (must be compatible with backbone)'
    )
    config_group.add_argument(
        '--temperature', 
        type=float, 
        default=0.1,
        help='Sampling temperature for model responses'
    )
    config_group.add_argument(
        '--seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Processing settings
    process_group = parser.add_argument_group('Processing Configuration')
    process_group.add_argument(
        '--mode', 
        type=str, 
        required=True,
        choices=SUPPORTED_MODES,
        help='Processing mode to use'
    )
    process_group.add_argument(
        '--start_idx', 
        type=int, 
        default=0,
        help='Starting index for dataset processing'
    )
    process_group.add_argument(
        '--end_idx', 
        type=int, 
        default=100,
        help='Ending index for dataset processing'
    )
    
    # Path settings
    path_group = parser.add_argument_group('File Paths')
    path_group.add_argument(
        '--data_dir', 
        type=str,
        default='./dataset/wikiHow_tasks_merge.csv',
        help='Path to the evaluation dataset'
    )
    path_group.add_argument(
        '--save_dir', 
        type=str,
        required=True,
        help='Directory to save the results'
    )
    
    # Utility options
    util_group = parser.add_argument_group('Utility Options')
    util_group.add_argument(
        '--validate_config',
        action='store_true',
        help='Validate configuration and test connectivity without processing'
    )
    util_group.add_argument(
        '--list_modes',
        action='store_true',
        help='List all available processing modes'
    )
    util_group.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be processed without actually processing'
    )
    
    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments"""
    errors = []
    
    # Check backbone-model compatibility
    backbone_models = {
        'gpt4o': ['gpt-4o-mini', 'gpt-4o'],
        'gem': ['gemini-1.5-flash'],
        'mistral': ['mistral-7b', 'mistral-8x7b']
    }
    
    if args.model not in backbone_models[args.backbone]:
        errors.append(f"Model '{args.model}' is not compatible with backbone '{args.backbone}'. "
                     f"Compatible models: {backbone_models[args.backbone]}")
    
    # Check range validity
    if args.start_idx < 0:
        errors.append("start_idx must be non-negative")
    
    if args.end_idx <= args.start_idx:
        errors.append("end_idx must be greater than start_idx")
    
    # No special requirements for simplified modes
    
    # Check file paths
    if not Path(args.data_dir).exists():
        errors.append(f"Dataset file not found: {args.data_dir}")
    
    # Report errors
    if errors:
        print("❌ Argument validation failed:")
        for error in errors:
            print(f"  • {error}")
        return False
    
    return True

def setup_pipeline(args: argparse.Namespace) -> PlanningPipeline:
    """Setup and configure the pipeline"""
    # Create configurations
    planning_config = PlanningConfig(
        backbone=args.backbone,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed
    )
    
    path_config = PathConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir
    )
    
    # Create pipeline
    pipeline = PlanningPipeline(planning_config, path_config)
    
    return pipeline

def main() -> int:
    """Main execution function"""
    args = parse_args()
    
    # Handle utility options
    if args.list_modes:
        print("Available processing modes:")
        for mode in SUPPORTED_MODES:
            print(f"  • {mode}")
        return 0
    
    # Validate arguments
    if not validate_args(args):
        return 1
    
    try:
        # Setup pipeline
        print("🔧 Setting up pipeline...")
        pipeline = setup_pipeline(args)
        
        # Validate configuration if requested
        if args.validate_config:
            print("🔍 Validating configuration...")
            if pipeline.validate_configuration():
                print("✅ Configuration is valid!")
                return 0
            else:
                print("❌ Configuration validation failed!")
                return 1
        
        # Dry run
        if args.dry_run:
            print("🔍 Dry run mode - showing what would be processed:")
            print(f"  Mode: {args.mode}")
            print(f"  Backbone: {args.backbone}")
            print(f"  Model: {args.model}")
            print(f"  Dataset: {args.data_dir}")
            print(f"  Range: {args.start_idx} - {args.end_idx}")
            print(f"  Save to: {args.save_dir}")
            return 0
        
        # Process dataset
        print("🚀 Starting dataset processing...")
        pipeline.process_dataset(
            mode=args.mode,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )
        
        print("✅ Processing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())