# HyperGRAND: Hypergraph Graph Neural Diffusion# HyperGRAND: Hypergraph Graph Neural Diffusion



Implementation of GRAND (Graph Neural Diffusion) for hypergraph data with support for classification, clustering, and partitioning tasks.Implementation of GRAND (Graph Neural Diffusion) for hypergraph data with support for classification, clustering, and partitioning tasks.



## Quick Start (5 minutes)## Quick Start



### 1. Setup Environment### Setup

```bash```bash

make install    # Create venv and install dependenciesmake install    # Create venv and install dependencies

``````



### 2. Verify System### Run Experiments

```bash

python test_pyg_system.pyLoad a single dataset:

``````bash

Expected output: `✓ ALL TESTS PASSED (5/5)`python main.py --dataset planetoid_cora --strategy classification

```

### 3. Load Your First Dataset

```pythonTest hypothesis across all datasets:

from data import load_dataset```bash

python comprehensive_evaluation.py --num_epochs 500 --patience 100

data = load_dataset('planetoid_cora')```

print(f"Nodes: {data.num_nodes}, Classes: {data.num_classes}")

```Explore available datasets:

```bash

### 4. Run Single Experimentpython dataset_info.py --format summary

```bash```

python main.py --dataset planetoid_cora --strategy classification

```### Documentation



### 5. Test Hypothesis (All Datasets)- **`PYG_QUICKSTART.md`** - Quick reference and examples (START HERE)

```bash- **`IMPLEMENTATION_COMPLETE.md`** - Complete implementation guide

python comprehensive_evaluation.py --num_epochs 500 --patience 100- **`requirements.txt`** - Dependencies

```

## Project Structure

---

```

## How to Work with This ProjectHyperGRAND/

├── main.py                          # CLI entry point

### Load Datasets├── comprehensive_evaluation.py       # Hypothesis testing

├── test_pyg_system.py              # System verification

```python├── dataset_info.py                 # Dataset exploration

# Single dataset│

from data import load_dataset├── models/

data = load_dataset('planetoid_cora')│   ├── hypergrand.py               # Main model

│   └── layers.py                   # Diffusion layers

# Multiple datasets│

from data import load_datasets├── training/

datasets = load_datasets(['planetoid_cora', 'contact_high_school'])│   ├── trainer.py                  # Training logic

│   └── loss.py                     # Loss functions

# All datasets of a task type│

from data import UnifiedDataManager├── data/

manager = UnifiedDataManager()│   ├── manager.py                  # Unified data manager

clustering_data = manager.load_by_task('clustering')│   ├── pyg_converter.py           # PyG conversion

```│   └── dataset.py                  # Dataset loaders

│

### Available Datasets (16 Total)├── approaches/

│   └── transductive.py             # Training approach

**Classification** (4 datasets)│

- `planetoid_cora`, `planetoid_citeseer`, `planetoid_pubmed` - Citation networks├── datasets/                        # Dataset storage

- `house_committees` - Political network├── examples/                        # Example scripts

└── utils/                          # Utilities

**Clustering** (5 datasets)```

- `contact_high_school`, `contact_primary_school` - Contact networks

- `walmart_trips`, `amazon_reviews`, `stackoverflow_answers` - Large-scale networks## Features



**Partitioning** (4 datasets)✓ **16 Datasets** - Classification, clustering, and partitioning datasets  

- `zoo`, `mushroom`, `ntu2012`, `modelnet40` - Different domains✓ **Unified Data System** - All data standardized as PyTorch Geometric objects  

✓ **Multiple Integration Schemes** - Explicit, implicit, multistep, adaptive  

**Other** (3 datasets)✓ **Comprehensive Evaluation** - Test across datasets by task type  

- `news_20w100`, `coauthorship_*`, `cocitation_*`✓ **Hypothesis Testing** - Validate performance across task types  



### Data Object Structure## Key Commands



```python```bash

data = load_dataset('planetoid_cora')# Load and test a dataset

python main.py --dataset planetoid_cora --strategy classification

# Node features and labels

data.x                      # [num_nodes, num_features]# Comprehensive evaluation

data.y                      # [num_nodes]python comprehensive_evaluation.py --output results.json



# Hyperedge connectivity# Verify system

data.hyperedge_index        # [2, num_incidences]python test_pyg_system.py



# Data splits# List datasets

data.train_mask             # [num_nodes]python dataset_info.py --format summary

data.val_mask               # [num_nodes]```

data.test_mask              # [num_nodes]

## Research Questions

# Metadata

data.metadata.task_type     # 'classification', 'clustering', 'partitioning'This project investigates whether HyperGRAND:

data.metadata.num_classes   # int- ✓ Performs **well on clustering** tasks

data.label_names            # list of class names- ✗ **Struggles with classification** tasks

```- ? Has **unknown performance on partitioning** tasks



### Training ExampleRun `comprehensive_evaluation.py` to test this hypothesis.



```python## Contact

from data import load_dataset

from models import create_hypergrand_modelAuthor: Atishaya Maharjan  

from training.trainer import create_hypergraph_trainerEmail: maharjaa@myumanitoba.ca

import torch

# Load data
data = load_dataset('planetoid_cora').to('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = create_hypergrand_model(
    input_dim=data.x.shape[1],
    hidden_dim=32,
    num_layers=2
)

# Create trainer
trainer = create_hypergraph_trainer(
    task_type=data.metadata.strategy,
    model=model,
    num_classes=data.metadata.num_classes
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    metrics = trainer.train_epoch(data, data.train_mask, data.val_mask, optimizer)
```

---

## Command Reference

### Data Exploration
```bash
# List all datasets by task type
python dataset_info.py --format summary

# Detailed dataset information
python dataset_info.py --format full

# Usage examples
python dataset_info.py --format guide
```

### Experiments
```bash
# Single dataset experiment
python main.py --dataset planetoid_cora --strategy classification

# Comprehensive evaluation (all datasets, all task types)
python comprehensive_evaluation.py \
    --num_epochs 500 \
    --patience 100 \
    --hidden_dim 32 \
    --learning_rate 0.01

# View results
cat comprehensive_results.json | python -m json.tool
```

### Testing
```bash
# Verify system functionality
python test_pyg_system.py

# Run unit tests (if available)
python -m pytest tests/
```

---

## Project Structure

```
HyperGRAND/
├── main.py                          # Single experiment CLI
├── comprehensive_evaluation.py       # Hypothesis testing (all datasets)
├── test_pyg_system.py              # System verification tests
├── dataset_info.py                 # Dataset exploration
│
├── models/
│   ├── hypergrand.py               # Main model
│   └── layers.py                   # Diffusion integration layers
│
├── training/
│   ├── trainer.py                  # Training classes
│   └── loss.py                     # Loss functions
│
├── data/
│   ├── manager.py                  # Unified data loading
│   ├── pyg_converter.py           # PyG standardization
│   └── dataset.py                  # Dataset implementations
│
├── approaches/
│   └── transductive.py             # Training strategies
│
├── datasets/                        # 16 datasets organized by task
├── examples/                        # Example usage
├── saved_models/                   # Pre-trained models
└── utils/                          # Utilities
```

---

## Key Features

✅ **16 Datasets** - Classification, clustering, and partitioning  
✅ **Unified PyG Interface** - Standardized data loading for all datasets  
✅ **Multiple Integration Schemes** - Explicit, implicit, multistep, adaptive  
✅ **Hypothesis Testing** - Automated evaluation across task types  
✅ **Production-Ready** - Clean code, comprehensive tests, full documentation  

---

## Research Hypothesis

**Question**: Does HyperGRAND perform differently across task types?

**Expected Results**:
- ✓ Good performance on **clustering** tasks
- ✗ Poor performance on **classification** tasks
- ? Unknown performance on **partitioning** tasks

**Test It**:
```bash
python comprehensive_evaluation.py --num_epochs 500 --patience 100
# Compare classification vs clustering vs partitioning accuracy
```

---

## Documentation

- **README.md** (this file) - How to work with the project
- **CHANGES.md** - What was changed and why
- **examples/pyg_example.py** - Code examples
- **requirements.txt** - Python dependencies

See **CHANGES.md** for:
- Complete list of changes
- Architecture overview
- Before/after comparisons
- Technical details

---

## Common Tasks

### Load a single dataset and examine it
```python
from data import load_dataset
data = load_dataset('planetoid_cora')
print(f"Nodes: {data.num_nodes}, Features: {data.x.shape[1]}, Classes: {data.num_classes}")
```

### Load all datasets of one task type
```python
from data import UnifiedDataManager
manager = UnifiedDataManager()
for name, data in manager.load_by_task('clustering').items():
    print(f"{name}: {data.num_nodes} nodes")
```

### Train on custom dataset
```python
from data import load_dataset
data = load_dataset('my_dataset')  # Must exist in datasets/
# Your training code here
```

### Add a new dataset
1. Place files in `datasets/my_dataset/` (hyperedges, labels)
2. Add entry to `datasets/DATASET_METADATA.json`
3. Load with: `load_dataset('my_dataset')`

### Run experiment with different parameters
```bash
python main.py \
    --dataset planetoid_cora \
    --strategy classification \
    --hidden_dim 64 \
    --num_layers 3 \
    --learning_rate 0.001 \
    --num_epochs 1000
```

---

## Troubleshooting

### ImportError when loading data?
```bash
python test_pyg_system.py  # This will show what's missing
```

### Dataset not found?
```bash
python dataset_info.py --format summary  # List available datasets
```

### Model not training?
Check that data loads correctly:
```python
from data import load_dataset
data = load_dataset('planetoid_cora')
print(data)  # Should show Data object with all attributes
```

---

## Performance Tips

1. **Use caching** (default) - Datasets loaded once and cached
2. **Batch operations** - `load_datasets([...])` is faster for multiple
3. **Check metadata** - Use `manager.get_dataset_info()` before loading
4. **Clear cache** - If memory is tight: `manager.clear_cache()`

---

## Integration Schemes

HyperGRAND supports multiple numerical integration schemes:

- **Explicit** - Simple, fast, limited stability
- **Implicit** - Stable, more computation
- **Multistep** - Better accuracy
- **Adaptive** - Automatic step size adjustment

See `models/layers.py` for implementation details.

---

## Contact & Attribution

Author: Atishaya Maharjan  
Email: maharjaa@myumanitoba.ca

Based on GRAND (Graph Neural Diffusion) adapted for hypergraph data.
