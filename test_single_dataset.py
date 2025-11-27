#!/usr/bin/env python3
"""Simple test to verify dataset loading works"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing dataset loading...\n")

try:
    from data.pyg_standardizer import DatasetLoader
    
    loader = DatasetLoader(base_path="datasets")
    
    # Test Cora
    print("Loading Cora...")
    data = loader.load('cora', verbose=False)
    print(f"  ✓ Loaded successfully")
    print(f"    Nodes: {data.num_nodes}")
    print(f"    Features shape: {data.x.shape}")
    print(f"    Labels shape: {data.y.shape}")
    print(f"    Labels unique: {data.y.unique().tolist()}")
    print(f"    Hyperedge_index shape: {data.hyperedge_index.shape}")
    print(f"    Train/val/test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
