#!/usr/bin/env python3
"""
Dataset organization and information script for HyperGRAND.

This script helps understand how datasets are organized by task type
(classification, clustering, partitioning) and provides utilities for
managing and exploring dataset metadata.
"""

import json
from pathlib import Path
from typing import Dict, List
import argparse


class DatasetOrganizer:
    """Helper class for managing dataset organization and metadata"""
    
    def __init__(self, metadata_path: str = "datasets/DATASET_METADATA.json"):
        self.metadata_path = Path(metadata_path)
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def print_organization(self):
        """Print the full dataset organization"""
        print("\n" + "="*80)
        print("HYPERGRAPH GRAND - DATASET ORGANIZATION")
        print("="*80 + "\n")
        
        for task_type, task_data in self.metadata.items():
            if task_type == 'description':
                continue
            
            print(f"\n{task_type.upper()}")
            print("-" * 80)
            print(f"Description: {task_data.get('description', 'N/A')}\n")
            
            datasets = task_data.get('datasets', {})
            
            for dataset_name, dataset_info in datasets.items():
                print(f"  📊 {dataset_name}")
                print(f"     Path: {dataset_info.get('path', 'N/A')}")
                print(f"     Nodes: {dataset_info.get('num_nodes', '?')}, "
                      f"Hyperedges: {dataset_info.get('num_hyperedges', '?')}, "
                      f"Classes: {dataset_info.get('num_classes', '?')}")
                print(f"     Source: {dataset_info.get('source', 'N/A')}")
                print()
    
    def get_task_summary(self):
        """Get a summary of datasets per task type"""
        summary = {}
        
        for task_type, task_data in self.metadata.items():
            if task_type == 'description':
                continue
            
            datasets = task_data.get('datasets', {})
            summary[task_type] = {
                'count': len(datasets),
                'description': task_data.get('description', ''),
                'datasets': list(datasets.keys())
            }
        
        return summary
    
    def print_summary(self):
        """Print a concise summary"""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80 + "\n")
        
        summary = self.get_task_summary()
        
        for task_type, info in summary.items():
            print(f"{task_type.upper()} ({info['count']} datasets)")
            print(f"  {info['description']}")
            for dataset in info['datasets']:
                print(f"    - {dataset}")
            print()
    
    def get_usage_guide(self):
        """Return a usage guide for the datasets"""
        guide = """
DATASET USAGE GUIDE FOR HYPERGRAPH GRAND
=========================================

1. CLASSIFICATION DATASETS (7 classes per task)
   These datasets are designed for supervised node classification where each node
   has a single class label.
   
   Examples:
   - planetoid_cora: Citation network (7 classes)
   - planetoid_citeseer: Citation network (6 classes)
   - planetoid_pubmed: Citation network (3 classes)
   - house_committees: Binary classification (2 classes)
   
   Usage:
     python main.py --dataset planetoid_cora --strategy classification
   
2. CLUSTERING DATASETS (Multiple clusters/groups)
   These datasets are designed for unsupervised or semi-supervised clustering
   where nodes form natural groups or clusters.
   
   Examples:
   - contact_high_school: 9 classroom groups
   - contact_primary_school: 10 classroom groups
   - walmart_trips: 8 product categories
   - amazon_reviews: 29 review categories
   - stackoverflow_answers: 18 tag categories
   
   Usage:
     python main.py --dataset contact_high_school --strategy clustering
   
3. PARTITIONING DATASETS (Structural/attribute-based grouping)
   These datasets use hyperedges to represent partitions or structural groupings
   (e.g., biological attributes, shape categories, or skeletal joints).
   
   Examples:
   - zoo: 7 animal types with 17 biological attributes
   - mushroom: 2 classes with 126 attribute groups
   - ntu2012: 50 action classes with 400 skeleton groups
   - modelnet40: 40 3D shape categories
   
   Usage:
     python main.py --dataset zoo --strategy clustering
     python main.py --dataset mushroom --strategy clustering

4. OTHER DATASETS (Mixed/experimental)
   These datasets have unique characteristics and may require custom handling.
   
   Examples:
   - 20newsW100: Text documents with top 100 words
   - coauthorship: Co-authorship networks (DBLP, Cora)
   - cocitation: Co-citation networks
   - yelp: Restaurant reviews with multiple features
   
   Usage:
     python main.py --dataset coauthorship --strategy clustering

COMPREHENSIVE EVALUATION
=========================

To test across all datasets and validate the hypothesis that HyperGRAND
performs well on clustering but struggles with classification:

    python comprehensive_evaluation.py \\
        --num_epochs 500 \\
        --patience 100 \\
        --hidden_dim 32 \\
        --num_layers 2 \\
        --scheme explicit \\
        --edge_dropout_rate 0.5

This will test all datasets organized by task type and save results to
'comprehensive_results.json' with metrics broken down by classification/clustering/partitioning.

HYPOTHESIS TESTING
==================

Based on the hypothesis that HyperGRAND:
✓ Performs WELL on clustering tasks
✗ Struggles with classification tasks
? Uncertain performance on partitioning tasks

The comprehensive evaluation will help validate this by:
1. Testing all classification datasets and measuring accuracy
2. Testing all clustering datasets and measuring ARI/accuracy
3. Testing all partitioning datasets and measuring clustering metrics
4. Comparing performance across task types

Results will be saved and can be analyzed to confirm or refute the hypothesis.
"""
        return guide


def main():
    parser = argparse.ArgumentParser(
        description="Dataset organization and information tool for HyperGRAND"
    )
    parser.add_argument('--format', choices=['full', 'summary', 'guide'], default='summary',
                       help='Output format')
    
    args = parser.parse_args()
    
    organizer = DatasetOrganizer()
    
    if args.format == 'full':
        organizer.print_organization()
    elif args.format == 'summary':
        organizer.print_summary()
    elif args.format == 'guide':
        print(organizer.get_usage_guide())


if __name__ == '__main__':
    main()
