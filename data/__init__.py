#!/usr/bin/env python3
"""
Data module for HyperGRAND - Unified data loading interface
"""

from .pyg_standardizer import (
    UniversalDataConverter,
    DatasetLoader,
    DatasetMetadata
)


def load_dataset(dataset_name: str, verbose: bool = True):
    """Load a single dataset as PyG Data object"""
    loader = DatasetLoader()
    return loader.load(dataset_name, verbose=verbose)


def load_datasets_by_task(task_type: str):
    """Load all datasets of a specific task type"""
    loader = DatasetLoader()
    return loader.load_by_task(task_type)


def list_datasets():
    """List all available datasets"""
    loader = DatasetLoader()
    return loader.list_datasets()


__all__ = [
    'UniversalDataConverter',
    'DatasetLoader',
    'DatasetMetadata',
    'load_dataset',
    'load_datasets_by_task',
    'list_datasets'
]
