# Experiment Runner Usage Guide

## Quick Start

### 1. Prepare Data with Features

First, generate your base data and add all the features you want to test:

```bash
# Generate base data
python makeGraph.py

# Add all features to both training and test sets
python feature_manager.py --add closeness --input data/training.pickle
python feature_manager.py --add closeness --input data/test.pickle

python feature_manager.py --add degree --input data/training.pickle
python feature_manager.py --add degree --input data/test.pickle

python feature_manager.py --add in_degree --input data/training.pickle
python feature_manager.py --add in_degree --input data/test.pickle

python feature_manager.py --add out_degree --input data/training.pickle
python feature_manager.py --add out_degree --input data/test.pickle

python feature_manager.py --add pagerank --input data/training.pickle
python feature_manager.py --add pagerank --input data/test.pickle
```

### 2. Run All Experiments

```bash
python experiment_runner.py --data_path ./data/ --output_dir ./experiments/
```

This will test all feature combinations defined in `FEATURE_SETS`:
- Baseline (no features)
- Individual features (closeness, degree, in_degree, out_degree, pagerank)
- Feature combinations (closeness+degree, closeness+pagerank, etc.)
- All features combined

### 3. Run Single Experiment

```bash
# Test only closeness centrality
python experiment_runner.py --feature_set closeness

# Test closeness + degree combination
python experiment_runner.py --feature_set closeness+degree
```

## Output Files

After running experiments, you'll find in `./experiments/`:

1. **Individual Results**: `result_<feature_set>.json`
   - Detailed metrics for each feature combination
   - Per-graph metrics
   - Training time

2. **Combined Results**: `all_results.json`
   - All experiments in one file
   - Easy to parse programmatically

3. **Comparison Table**: `comparison_table.md`
   - Markdown table comparing all experiments
   - Key metrics side-by-side

## Metrics Explained

### Ranking Metrics
- **Kendall's Tau**: Correlation between predicted and true rankings (-1 to 1, higher is better)
- **Spearman's Rho**: Another rank correlation measure

### Top-K Metrics
- **Precision@K**: Of the top-K predicted nodes, how many are actually in the true top-K?
- **Recall@K**: Of the true top-K nodes, how many did we capture?
- **F1@K**: Harmonic mean of Precision and Recall
- **NDCG@K**: Normalized Discounted Cumulative Gain (accounts for position)
- **Overlap@K**: Jaccard similarity of top-K sets

### Regression Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination

## Customization

### Add New Feature Combinations

Edit `experiment_runner.py`:

```python
FEATURE_SETS = {
    'baseline': [],
    'closeness': ['closeness'],
    'degree': ['degree'],
    # Add your custom combination
    'my_combo': ['closeness', 'in_degree', 'pagerank'],
}
```

### Change K Values

In `experiment_runner.py`, modify the `evaluate_model` method:

```python
metrics = evaluate_all_metrics(y_pred, y_true, k_values=[5, 10, 15, 20, 30, 50])
```

### Adjust Training Parameters

```bash
python experiment_runner.py \
    --num_epochs 20 \
    --hidden_dim 40 \
    --device cuda
```

## Example Output

```
==============================================================
Experiment: closeness+degree
Features: ['closeness', 'degree']
==============================================================

Loading data for feature set: closeness+degree
  Feature dimension: 2
  Training graphs: 16
  Test graphs: 4

Training model...
  Epoch 5/10, Loss: 45.2341
  Epoch 10/10, Loss: 32.1234

Evaluating model...

==============================================================
Results Summary for closeness+degree:
==============================================================
Kendall's Tau: 0.7845 ± 0.0234
Recall@10: 0.8500 ± 0.0156
NDCG@10: 0.9123 ± 0.0089
Training time: 45.67s

Saved results to experiments/result_closeness+degree.json
```

## Tips

1. **Start Small**: Test with `--feature_set baseline` first to ensure everything works
2. **GPU Recommended**: Use `--device cuda` for faster training
3. **Adjust Epochs**: For quick testing, use `--num_epochs 5`
4. **Feature Order Matters**: Add features in the same order to training and test sets
