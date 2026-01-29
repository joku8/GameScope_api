Prediction package

Quick start:

1. Install requirements (preferably in your venv):

```bash
pip install -r prediction/requirements.txt
```

2. Train and save model artifacts (reads dataset at `data_collection/out/igdb_games_all.csv` and tries to load embeddings from `data_visualization/embeddings/summary_embeddings.npy`):

```bash
python -m prediction.train
```

3. Use inference:

```python
from prediction.infer import recommend
res = recommend('Game Name', personal_rating=8.0, n=3, model_dir='prediction/models')
print(res)
```

Notes:

- The system uses a hybrid similarity-based approach: semantic summary embeddings + genre/theme overlap + game rating influence.
- If precomputed embeddings are missing, a TF-IDF + SVD fallback is used.
- Training now caches each run under `prediction/cache/<timestamp>/` with subfolders `raw/`, `preprocess/`, and `model/` for provenance and reproducibility. The inference wrapper will load the latest model from the cache by default.
