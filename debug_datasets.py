#!/usr/bin/env python3
"""Debug dataset loading issues"""

import sys
import pickle
import torch
import numpy as np
from pathlib import Path

# Check cora structure
cora_path = Path("datasets/classification/cora")
print("=== CORA DIRECTORY ===")
print(f"Path exists: {cora_path.exists()}")

if cora_path.exists():
    for item in cora_path.iterdir():
        print(f"  {item.name}")
        if item.is_dir():
            for subitem in item.iterdir():
                print(f"    {subitem.name}")

# Try loading with basic methods
print("\n=== TESTING PYATORCH GEOMETRIC LOADER ===")

raw_path = cora_path / 'raw'
if raw_path.exists():
    print(f"Raw path exists: {raw_path}")
    ind_files = list(raw_path.glob('ind.*'))
    print(f"Found {len(ind_files)} ind.* files:")
    for f in ind_files:
        print(f"  {f.name}")
    
    # Try loading x
    print("\nTrying to load features...")
    x_file = raw_path / 'ind.cora.x'
    if x_file.exists():
        with open(x_file, 'rb') as f:
            x = pickle.load(f, encoding='latin1')
            print(f"  x type: {type(x)}")
            if hasattr(x, 'toarray'):
                x_dense = x.toarray()
                print(f"  x shape (dense): {x_dense.shape}")
            else:
                print(f"  x shape: {np.array(x).shape}")
    
    # Try loading y
    print("\nTrying to load labels...")
    y_file = raw_path / 'ind.cora.y'
    if y_file.exists():
        with open(y_file, 'rb') as f:
            y = pickle.load(f, encoding='latin1')
            print(f"  y type: {type(y)}")
            print(f"  y shape: {y.shape if hasattr(y, 'shape') else len(y)}")
            if hasattr(y, 'shape') and len(y.shape) > 1:
                print(f"  y labels: {y.argmax(axis=1)[:10]}")
    
    # Try loading graph
    print("\nTrying to load graph...")
    graph_file = raw_path / 'ind.cora.graph'
    if graph_file.exists():
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f, encoding='latin1')
            print(f"  graph type: {type(graph)}")
            print(f"  graph nodes: {len(graph)}")
            num_edges = sum(len(neighbors) for neighbors in graph.values())
            print(f"  graph edges: {num_edges}")
else:
    print(f"Raw path does not exist: {raw_path}")

# Check processed path
processed_path = cora_path / 'processed'
if processed_path.exists():
    print(f"\nProcessed path exists: {processed_path}")
    for item in processed_path.iterdir():
        print(f"  {item.name} ({item.stat().st_size} bytes)")
