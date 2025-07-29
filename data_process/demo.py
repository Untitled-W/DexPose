"""
Example usage of the refactored dataset processing framework.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import (
    process_single_dataset, 
    process_multiple_datasets,
    list_available_datasets,
    get_dataset_info,
    setup_logging
)


'''
filter warning at /home/qianxu/miniconda3/envs/data_process/lib/python3.8/site-packages/torch/hub.py:477
'''

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='open3d')

def main():
    """Demo the refactored dataset processing."""
    setup_logging()
    
    print("=== Refactored Dataset Processing Demo ===\n")
    
    # 1. List available datasets
    print("1. Available datasets:")
    datasets = list_available_datasets()
    for dataset in datasets:
        info = get_dataset_info(dataset)
        print(f"   - {dataset}: {info['root_path']}")
    print()
    
    OAKINK = True
    TACO = True

    # 2. Process a single dataset
    if OAKINK:
        print("2. Processing OAKINKv2 dataset...")
        # try:
        oakink_data = process_single_dataset(
            'oakinkv2',
            task_interval=5,  # Process every 2nd frame
            pt_n=50,         # Use fewer points for demo
            # seq_data_name='oakinkv2_demo'
            seq_data_name='oakinkv2_0704'
        )
        print(f"   ✓ Successfully processed {len(oakink_data)} OAKINKv2 sequences")
        # except Exception as e:
        #     print(f"   ✗ Failed to process OAKINKv2: {e}")
        print()
    

    # 3. Process TACO dataset
    if TACO:
        print("3. Processing TACO dataset...")
        # try:
        taco_data = process_single_dataset(
            'taco',
            task_interval=2,
            pt_n=50,
            seq_data_name='taco_0704'
        )
        print(f"   ✓ Successfully processed {len(taco_data)} TACO sequences")
        # except Exception as e:
            # print(f"   ✗ Failed to process TACO: {e}")
        print()

    return
    
    # 4. Process multiple datasets together
    print("4. Processing multiple datasets together...")
    try:
        combined_data = process_multiple_datasets(
            dataset_names=['oakinkv2', 'taco'],
            configs=[
                {'task_interval': 2, 'pt_n': 50, 'seq_data_name': 'oakinkv2_combined'},
                {'task_interval': 1, 'pt_n': 50, 'seq_data_name': 'taco_combined'}
            ]
        )
        print(f"   ✓ Successfully combined {len(combined_data)} sequences")
        
        # Show statistics
        oakink_count = sum(1 for seq in combined_data if 'oakinkv2' in seq.get('mesh_path', ''))
        taco_count = sum(1 for seq in combined_data if 'taco' in seq.get('mesh_path', ''))
        print(f"   - OAKINKv2: {oakink_count} sequences")
        print(f"   - TACO: {taco_count} sequences")
        
    except Exception as e:
        print(f"   ✗ Failed to process multiple datasets: {e}")
    print()
    
    # 5. Show data format
    if 'combined_data' in locals() and len(combined_data) > 0:
        print("5. Sample data format:")
        sample = combined_data[0]
        print("   Available keys:")
        for key in sorted(sample.keys()):
            value = sample[key]
            if hasattr(value, 'shape'):
                print(f"   - {key}: {type(value).__name__} {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"   - {key}: {type(value).__name__} length {len(value)}")
            else:
                print(f"   - {key}: {type(value).__name__}")
    
    print("\n=== Demo Complete ===")


if __name__ == '__main__':
    main()