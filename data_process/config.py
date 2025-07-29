"""
Configuration file for dataset processing.
"""

# Dataset configurations
DATASET_CONFIGS = {
    'oakinkv2': {
        'processor_name': 'oakinkv2',
        'root_path': '/home/qianxu/dataset/oakinkv2',
        # 'root_path': '/home/qianxu/Desktop/New_Folder/OakInk2/OakInk-v2-hub',
        'task_interval': 1,
        'pt_n': 100,
        'cpt_n': 100,
        'seq_data_name': 'oakinkv2',
        'contact_threshold': 0.1,
    },
    
    'taco': {
        'processor_name': 'taco',
        'root_path': '/home/qianxu/project/TACO-Instructions/data/',
        # 'root_path': '/home/qianxu/Desktop/New_Folder/TACO-Instructions/data',
        'task_interval': 1,
        'pt_n': 100,
        'cpt_n': 100,
        'seq_data_name': 'taco',
        'contact_threshold': 0.1,
    }
}

# Working directory
WORKING_DIR = '/home/qianxu/Project24/interaction_pose/'

# Default processing parameters
DEFAULT_PARAMS = {
    'task_interval': 1,
    'pt_n': 100,
    'cpt_n': 100,
    'contact_threshold': 0.1,
}
