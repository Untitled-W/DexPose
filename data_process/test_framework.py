"""
Simple test script to verify the refactored framework works correctly.
"""

import sys
import os
import logging

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        from data_process.base_dataset import BaseDatasetProcessor, DatasetRegistry
        print("✓ Base classes imported successfully")
    except Exception as e:
        print(f"✗ Failed to import base classes: {e}")
        return False
    
    try:
        from dataset_processors import OAKINKv2Processor, TACOProcessor
        print("✓ Dataset processors imported successfully")
    except Exception as e:
        print(f"✗ Failed to import dataset processors: {e}")
        return False
    
    try:
        from data_process.main import (
            process_single_dataset, 
            process_multiple_datasets,
            list_available_datasets
        )
        print("✓ Main functions imported successfully")
    except Exception as e:
        print(f"✗ Failed to import main functions: {e}")
        return False
    
    return True


def test_registry():
    """Test the dataset registry system."""
    print("\nTesting registry system...")
    
    try:
        from data_process.base_dataset import DatasetRegistry
        
        # Check if processors are registered
        processors = DatasetRegistry.list_processors()
        print(f"✓ Found {len(processors)} registered processors: {processors}")
        
        # Test getting processors
        for processor_name in processors:
            processor_class = DatasetRegistry.get_processor(processor_name)
            print(f"✓ Can access {processor_name} processor: {processor_class.__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Registry test failed: {e}")
        return False


def test_configuration():
    """Test configuration system."""
    print("\nTesting configuration...")
    
    try:
        from data_process.main import list_available_datasets, get_dataset_info
        
        datasets = list_available_datasets()
        print(f"✓ Found {len(datasets)} configured datasets: {datasets}")
        
        for dataset in datasets:
            info = get_dataset_info(dataset)
            print(f"✓ {dataset} config: {info['processor_name']} at {info['root_path']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_processor_instantiation():
    """Test that processors can be instantiated."""
    print("\nTesting processor instantiation...")
    
    try:
        from dataset_processors import OAKINKv2Processor, TACOProcessor
        
        # Test OAKINKv2 processor
        oakink_processor = OAKINKv2Processor(
            root_path='/tmp/test',  # Use dummy path
            task_interval=1,
            pt_n=10
        )
        print("✓ OAKINKv2Processor instantiated successfully")
        
        # Test TACO processor
        taco_processor = TACOProcessor(
            root_path='/tmp/test',  # Use dummy path
            task_interval=1,
            pt_n=10
        )
        print("✓ TACOProcessor instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Processor instantiation failed: {e}")
        return False


def test_data_structure():
    """Test the SequenceData structure."""
    print("\nTesting data structure...")
    
    try:
        from data_process.base_dataset import SequenceData
        import torch
        
        # Create a sample SequenceData
        seq_data = SequenceData(
            hand_joints=torch.zeros(10, 21, 3),
            hand_params=torch.zeros(10, 51),
            side=1,
            obj_points=torch.zeros(100, 3),
            obj_points_ori=torch.zeros(1000, 3),
            obj_normals=torch.zeros(100, 3),
            obj_normals_ori=torch.zeros(1000, 3),
            obj_features=torch.zeros(100, 384),
            obj_transformations=torch.eye(4).unsqueeze(0).repeat(10, 1, 1),
            contact_indices=torch.zeros(10, 50, dtype=torch.long),
            mesh_path="/path/to/mesh.obj",
            mesh_norm_transformation=torch.eye(4),
            task_description="test task",
            seq_len=10
        )
        
        print("✓ SequenceData structure created successfully")
        print(f"  - Hand joints shape: {seq_data.hand_joints.shape}")
        print(f"  - Object points shape: {seq_data.obj_points.shape}")
        print(f"  - Task description: {seq_data.task_description}")
        
        return True
        
    except Exception as e:
        print(f"✗ Data structure test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Testing Refactored Dataset Framework ===\n")
    
    tests = [
        test_imports,
        test_registry, 
        test_configuration,
        test_processor_instantiation,
        test_data_structure
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! The refactored framework is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the issues above.")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
