# HyperGRAND Diffusion Studies - Quick Start Guide

## Overview

This guide shows how to run systematic diffusion parameter studies to validate the hypotheses in your HyperGRAND paper.

## Quick Commands

### 1. Fast Validation (~4 hours)
Test the framework on representative datasets with reduced configurations:

```bash
make diffusion-fast
```

This runs:
- 3 representative datasets (cora, contact_high_school, zoo)
- Reduced configuration set (explicit/implicit, 1/3/5 layers)
- 3 random seeds per configuration
- ~50 total experiments

### 2. Full Integration Scheme Study (~8-12 hours)
Compare explicit vs implicit vs adaptive integration:

```bash
python main.py --mode diffusion-study \
  --study-dimension integration_scheme \
  --num-seeds 5 \
  --output integration_results.json
```

### 3. Full Diffusion Depth Study (~8-12 hours)
Test impact of 1, 2, 3, 4, 5 diffusion layers:

```bash
python main.py --mode diffusion-study \
  --study-dimension diffusion_depth \
  --num-seeds 5 \
  --output depth_results.json
```

### 4. Complete Study (All Dimensions) (~20-40 hours)
Run all parameter studies:

```bash
make diffusion-study DIMENSION=all SEEDS=5
```

## Analyzing Results

### Generate Reports

After running studies, generate comprehensive analysis:

```bash
make analyze
```

This creates:
- `analysis_output/hypothesis_validation_report.json` - Tests your paper's hypotheses
- `analysis_output/integration_scheme_table.tex` - LaTeX table for paper
- `analysis_output/integration_scheme_heatmap.png` - Visualization

### View in MLflow UI

Start the MLflow UI to explore results interactively:

```bash
make mlflow
```

Navigate to `http://localhost:5000` to:
- Compare runs across configurations
- View learning curves
- Filter by tags (study_dimension, config_variant, task_type)
- Examine statistical aggregations (mean ± std, 95% CI)

## Understanding the Results

### Hypothesis Validation

The framework tests these hypotheses from your paper:

**H1: Clustering Excellence**
- Expected: NMI/ARI > 0.9 on clustering datasets
- Tests: contact_high_school, contact_primary_school, walmart_trips

**H2: Classification Struggle**
- Expected: Accuracy < 0.75 on classification datasets
- Tests: cora, citeseer, pubmed

**H3: Integration Scheme Impact**
- Expected: Significant difference between explicit and implicit
- Tests: Paired t-tests comparing schemes

### Configuration Variants

**Integration Schemes:**
- `explicit`: Forward Euler (fast, may be unstable)
- `implicit`: Backward Euler (stable, slower)
- `adaptive`: RK45-style (dynamic step size)

**Diffusion Depths:**
- `1_layer`: Minimal diffusion
- `3_layers`: Baseline
- `5_layers`: Maximum diffusion

**Attention Mechanisms:**
- `full_attention`: Learned G matrix (best)
- `no_attention`: Identity diffusion
- `simplified_attention`: Dot product only

## Advanced Usage

### Custom Configuration

Use a config file to specify all parameters:

```bash
python main.py --mode train \
  --dataset cora \
  --config configs/classification.yaml
```

### Parallel Workers

Control parallelization (default: auto-detect):

```bash
python main.py --mode diffusion-study \
  --parallel-workers 4 \
  --representative-only
```

### Representative Datasets Only

Quick testing on one dataset per task type:

```bash
python main.py --mode diffusion-study \
  --representative-only \
  --fast-mode \
  --num-seeds 3
```

## Troubleshooting

### Out of Memory

Reduce batch size or use CPU:
```bash
python main.py --mode diffusion-study --device cpu
```

### MLflow Connection Error

Start MLflow server first:
```bash
make mlflow &  # Run in background
python main.py --mode diffusion-study
```

### Resume Interrupted Study

MLflow tracks completed runs automatically. Just re-run the same command:
```bash
python main.py --mode diffusion-study --study-dimension integration_scheme
```

## Expected Runtime

Approximate times (8-core CPU, single GPU):

| Configuration | Experiments | Time |
|--------------|-------------|------|
| Fast mode | ~50 | 4 hours |
| Single dimension | ~240 | 12 hours |
| All dimensions | ~720 | 36 hours |

Times scale linearly with `--num-seeds` and dataset count.

## Output Files

- `diffusion_study_results.json` - Raw results
- `analysis_output/` - Reports and visualizations
- `mlruns/` - MLflow tracking data

## Next Steps

1. Run `make diffusion-fast` to validate the framework
2. Review results in MLflow UI (`make mlflow`)
3. Run full studies for paper: `make diffusion-study`
4. Generate paper figures: `make analyze`
5. Include LaTeX tables from `analysis_output/` in paper
