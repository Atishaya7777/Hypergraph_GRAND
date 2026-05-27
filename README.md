# HyperGRAND

## Validate Datasets

Verify all 16 datasets load correctly with proper hyperedges:

```bash
make test
```

## Train Single Dataset

Train on a specific dataset:

```bash
python main.py --dataset cora
python main.py --dataset news_20w100
python main.py --dataset zoo
```

## Train All Datasets

Run the hypothesis test across all 16 datasets:

```bash
make train-all
```

Or with results saved to file:

```bash
python main.py --mode batch --output results.json
```

## Run Diffusion Studies

Systematic parameter studies with statistical analysis:

```bash
# Fast mode: Representative datasets (cora, contact_high_school, zoo) with reduced configs
python main.py --mode diffusion-study --fast-mode --representative-only --num-seeds 3

# Full study: All integration schemes
python main.py --mode diffusion-study --study-dimension integration_scheme --num-seeds 5

# All study dimensions
python main.py --mode diffusion-study --study-dimension all --num-seeds 5
```

## View Results

Check metrics in terminal:

```bash
cat results.json
```

View in MLflow UI:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000`

## Analyze Study Results

Generate comprehensive reports and visualizations:

```bash
python experiments/analyze_diffusion_results.py --generate-report --generate-latex --plot-heatmap
```
