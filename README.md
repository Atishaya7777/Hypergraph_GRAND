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

Resume training from checkpoint:

```bash
python main.py --mode batch --output results.json --save-results
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
