# File Scoring

- `file_features.py`              ← Shared foundation, imported by both sides
- `generate_training_data.py`     ← Generates training data
- `train_ml_scorer.py`            ← Training + Inference + collector.py integration

```bash
uv pip install scikit-learn
```

# Step 1: Generate training data for multiple repos (labeled by LLM)
```bash
git clone https://github.com/ggml-org/llama.cpp.git source/llama.cpp

export LOCAL_LLM_BASE_URL=http://localhost:19001/v1
export LOCAL_LLM_MODEL=sonnet
python -m model.generate_training_data --root ./source --output ./model/tmp/llm_training_data.jsonl

# If you don't have an LLM, you can use rule-based labeling first for a cold start
python -m model.generate_training_data --root ./source --output ./model/tmp/training_data.jsonl --rule-only
```

# Step 2: Train
```bash
python -m model.train_ml_scorer train --data ./model/tmp/training_data.jsonl --model scorer.pkl
```

# Step 3: Inference (integrated with git ls-files)
```bash
git -C source/llama.cpp ls-files | python -m model.train_ml_scorer score --model scorer.pkl --top-k 30
```

# Step 4: Integrate into collector.py

```py
from train_ml_scorer import FileScorer
scorer = FileScorer.load("scorer.pkl")
files_to_scan = [f for f in all_files if scorer.filter(f, threshold=0.5)]
```