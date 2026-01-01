# Feature Manager Usage Examples

## Basic Usage

### 1. List Current Features
```bash
python feature_manager.py --list --input data/training.pickle
```

### 2. Add Closeness Centrality
```bash
python feature_manager.py --add closeness --input data/training.pickle
```

### 3. Add Degree Centrality
```bash
python feature_manager.py --add degree --input data/training.pickle
```

### 4. Add Multiple Features (chain commands)
```bash
# Add closeness
python feature_manager.py --add closeness --input data/training.pickle

# Then add degree
python feature_manager.py --add degree --input data/training.pickle

# Then add PageRank
python feature_manager.py --add pagerank --input data/training.pickle
```

### 5. Remove a Feature by Index
```bash
# Remove feature at index 1 (second feature)
python feature_manager.py --remove 1 --input data/training.pickle
```

### 6. Save to Different File
```bash
python feature_manager.py --add closeness --input data/training.pickle --output data/training_with_closeness.pickle
```

## Available Features

- `closeness`: Closeness Centrality (weighted)
- `degree`: Degree Centrality (normalized)
- `in_degree`: In-Degree Centrality
- `out_degree`: Out-Degree Centrality
- `pagerank`: PageRank (weighted)

## Workflow Example

```bash
# Start with data from makeGraph.py (has BC labels but no features)
python makeGraph.py

# Add closeness centrality to training data
python feature_manager.py --add closeness --input data/training.pickle

# Add closeness centrality to test data
python feature_manager.py --add closeness --input data/test.pickle

# Add degree as second feature
python feature_manager.py --add degree --input data/training.pickle
python feature_manager.py --add degree --input data/test.pickle

# Check what features we have
python feature_manager.py --list --input data/training.pickle

# Train model (update node_feat_dim in main.py to match number of features)
python main.py
```

## Important Notes

1. **Update `main.py`**: After adding features, update the model initialization:
   ```python
   # If you have 2 features (e.g., closeness + degree)
   model = GNN_Bet(ninput=model_size, nhid=hidden, node_feat_dim=2, dropout=0.6)
   ```

2. **Consistent Features**: Make sure training and test sets have the **same features in the same order**.

3. **Backup**: The script modifies files in-place by default. Use `--output` to save to a new file if you want to keep the original.
