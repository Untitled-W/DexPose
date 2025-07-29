"""
Main interface for dataset processing.
Usage examples and unified data loading.
"""

import logging
from typing import List, Dict, Any, Optional
import argparse

# Handle imports with fallbacks for different execution contexts
try:
    # Try relative imports first (when run as module)
    from .base_dataset import DatasetRegistry, load_multiple_datasets
    from dataset_processors import OAKINKv2Processor, TACOProcessor
    from .config import DATASET_CONFIGS
except ImportError:
    # Fallback to absolute imports (when run directly)
    try:
        from base_dataset import DatasetRegistry, load_multiple_datasets
        from dataset_processors import OAKINKv2Processor, TACOProcessor
        from config import DATASET_CONFIGS
    except ImportError:
        # Final fallback with embedded config
        import sys
        import os
        
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
    
        from base_dataset import DatasetRegistry, load_multiple_datasets
        from dataset_processors import OAKINKv2Processor, TACOProcessor
        from config import DATASET_CONFIGS

def setup_logging(level=logging.ERROR):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('dataset_processing.log'),
            logging.StreamHandler()
        ]
    )


def process_single_dataset(dataset_name: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Process a single dataset.
    
    Args:
        dataset_name: Name of the dataset ('oakinkv2', 'taco', etc.)
        **kwargs: Additional parameters to override default config
    
    Returns:
        List of processed sequences
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name].copy()
    config.update(kwargs)
    
    processor_class = DatasetRegistry.get_processor(config['processor_name'])
    processor = processor_class(**config)
    
    return processor.run()


def process_multiple_datasets(
    dataset_names: List[str], 
    configs: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple datasets and combine them.
    
    Args:
        dataset_names: List of dataset names to process
        configs: Optional list of config overrides for each dataset
    
    Returns:
        Combined list of processed sequences
    """
    if configs is None:
        configs = [{}] * len(dataset_names)
    
    if len(configs) != len(dataset_names):
        raise ValueError("configs length must match dataset_names length")
    
    processor_configs = []
    for name, config_override in zip(dataset_names, configs):
        if name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {name}")
        
        config = DATASET_CONFIGS[name].copy()
        config.update(config_override)
        processor_configs.append(config)
    
    return load_multiple_datasets(processor_configs)


def list_available_datasets() -> List[str]:
    """List all available datasets."""
    return list(DATASET_CONFIGS.keys())


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a specific dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DATASET_CONFIGS[dataset_name].copy()


# Example usage functions
def example_single_dataset():
    """Example: Process a single dataset."""
    setup_logging()
    
    # Process OAKINKv2
    oakink_data = process_single_dataset(
        'oakinkv2',
        task_interval=2,  # Override default
        pt_n=150         # Override default
    )
    print(f"Processed {len(oakink_data)} OAKINKv2 sequences")
    
    return oakink_data


def example_multiple_datasets():
    """Example: Process multiple datasets."""
    setup_logging()
    
    # Process both datasets with different configs
    combined_data = process_multiple_datasets(
        dataset_names=['oakinkv2', 'taco'],
        configs=[
            {'task_interval': 1, 'pt_n': 100},  # OAKINKv2 config
            {'task_interval': 2, 'pt_n': 80}    # TACO config
        ]
    )
    print(f"Combined {len(combined_data)} sequences from both datasets")
    
    return combined_data


def example_custom_dataset():
    """Example: Add a custom dataset processor."""
    from .base_dataset import BaseDatasetProcessor
    
    @DatasetRegistry.register('custom')
    class CustomProcessor(BaseDatasetProcessor):
        """Custom dataset processor."""
        
        def _setup_paths(self):
            # Setup custom paths
            pass
        
        def _get_data_list(self):
            # Return list of data items
            return []
        
        def _load_sequence_data(self, data_item):
            # Load sequence data
            return []
        
        def _process_hand_data(self, raw_data, side, frame_indices):
            # Process hand data
            return None, None
        
        def _load_object_data(self, raw_data, frame_indices):
            # Load object data
            return None, None, None, None
        
        def _get_task_description(self, raw_data):
            # Get task description
            return ""
    
    # Now you can use it
    # custom_data = process_single_dataset('custom', root_path='/path/to/custom/data')


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Process hand manipulation datasets')
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='Dataset names to process')
    parser.add_argument('--task_interval', type=int, default=1,
                       help='Interval between frames')
    parser.add_argument('--pt_n', type=int, default=100,
                       help='Number of points to sample')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for processed data')
    parser.add_argument('--log_level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)
    
    # Process datasets
    try:
        if len(args.datasets) == 1:
            data = process_single_dataset(
                args.datasets[0],
                task_interval=args.task_interval,
                pt_n=args.pt_n
            )
        else:
            configs = [
                {'task_interval': args.task_interval, 'pt_n': args.pt_n}
                for _ in args.datasets
            ]
            data = process_multiple_datasets(args.datasets, configs)
        
        print(f"Successfully processed {len(data)} sequences")
        
        # Save combined data if output directory specified
        if args.output_dir:
            import os
            import joblib
            os.makedirs(args.output_dir, exist_ok=True)
            output_path = os.path.join(args.output_dir, 'combined_sequences.pkl')
            with open(output_path, 'wb') as f:
                joblib.dump(data, f)
            print(f"Saved combined data to {output_path}")
            
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
