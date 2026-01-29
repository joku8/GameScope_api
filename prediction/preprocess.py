import ast
import json
import os
from typing import Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler


def _parse_list_field(val):
    if pd.isna(val):
        return []
    if isinstance(val, list):
        # ensure elements are strings
        return [str(x) for x in val]
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x) for x in list(parsed)]
    except Exception:
        pass
    # fallback: split by common delimiters
    if isinstance(val, str):
        for sep in [";", "|", ","]:
            if sep in val:
                return [str(x.strip()) for x in val.split(sep) if x.strip()]
        return [str(val.strip())] if val.strip() else []
    # if it's a number or other type, coerce to string in a single-item list
    return [str(val)]


def load_enum_mappings(enum_dir: str = "data_visualization/igdb_enum") -> tuple[dict, dict]:
    """Load genre and theme mapping JSON files (id->name).

    Returns (genre_map, theme_map) where keys are string ids.
    """
    genre_map = {}
    theme_map = {}
    try:
        with open(os.path.join(enum_dir, "genres.json"), "r", encoding="utf-8") as f:
            genre_map = json.load(f)
            # ensure keys are strings
            genre_map = {str(k): v for k, v in genre_map.items()}
    except Exception:
        pass
    try:
        with open(os.path.join(enum_dir, "themes.json"), "r", encoding="utf-8") as f:
            theme_map = json.load(f)
            theme_map = {str(k): v for k, v in theme_map.items()}
    except Exception:
        pass
    return genre_map, theme_map


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def load_or_compute_embeddings(df: pd.DataFrame,
                               embeddings_path: Optional[str] = None,
                               summary_col: str = "summary",
                               embed_dim: int = 128,
                               genre_map: Optional[dict] = None,
                               theme_map: Optional[dict] = None,
                               genre_col: str = "genres",
                               theme_col: str = "themes") -> np.ndarray:
    # Try provided embeddings file first
    if embeddings_path and os.path.exists(embeddings_path):
        emb = np.load(embeddings_path)
        if emb.shape[0] != len(df):
            raise ValueError("Embeddings length does not match dataframe rows")
        return emb

    # Fallback: compute TF-IDF + SVD as lightweight embedding
    # When computing embeddings from text, enrich the summary with genre/theme names
    summaries = df[summary_col].fillna("").astype(str)
    texts = []
    for i, s in summaries.iteritems():
        parts = [s]
        # add mapped genre names
        try:
            gids = df.at[i, genre_col]
        except Exception:
            gids = None
        if genre_map and gids:
            # gids may be list of ids or strings
            names = []
            for g in gids:
                gk = str(g)
                if gk in genre_map:
                    names.append(genre_map[gk])
            if names:
                parts.append(" ".join(names))
        # add mapped theme names
        try:
            tids = df.at[i, theme_col]
        except Exception:
            tids = None
        if theme_map and tids:
            names = []
            for t in tids:
                tk = str(t)
                if tk in theme_map:
                    names.append(theme_map[tk])
            if names:
                parts.append(" ".join(names))

        texts.append(" ".join(parts))
    tf = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tf.fit_transform(texts)
    svd = TruncatedSVD(n_components=embed_dim, random_state=42)
    emb = svd.fit_transform(X)
    return emb


def prepare_features(df: pd.DataFrame,
                     embeddings: np.ndarray,
                     genre_col: str = "genres",
                     theme_col: str = "themes",
                     rating_col: str = "rating") -> Tuple[dict, dict]:
    # Parse multi-label fields
    df = df.copy()
    # Only parse fields if they are not already lists of strings
    sample_genre = df[genre_col].dropna().iloc[0] if not df[genre_col].dropna().empty else None
    if sample_genre is None or not isinstance(sample_genre, list):
        df[genre_col] = df[genre_col].apply(_parse_list_field)
    else:
        # ensure elements are strings
        df[genre_col] = df[genre_col].apply(lambda lst: [str(x) for x in lst])

    sample_theme = df[theme_col].dropna().iloc[0] if not df[theme_col].dropna().empty else None
    if sample_theme is None or not isinstance(sample_theme, list):
        df[theme_col] = df[theme_col].apply(_parse_list_field)
    else:
        df[theme_col] = df[theme_col].apply(lambda lst: [str(x) for x in lst])

    # If the entries are numeric ids (strings of numbers), keep them as-is here;
    # callers can map ids to names before computing embeddings.

    mlb_genre = MultiLabelBinarizer(sparse_output=False)
    mlb_theme = MultiLabelBinarizer(sparse_output=False)

    genres_mat = mlb_genre.fit_transform(df[genre_col])
    themes_mat = mlb_theme.fit_transform(df[theme_col])

    # rating scaler (game's rating in dataset)
    rating_vals = df[rating_col].fillna(0).astype(float).values.reshape(-1, 1)
    scaler = MinMaxScaler()
    rating_scaled = scaler.fit_transform(rating_vals).ravel()

    meta = {
        "df": df.reset_index(drop=True),
        "genres_mat": genres_mat,
        "themes_mat": themes_mat,
        "mlb_genre": mlb_genre,
        "mlb_theme": mlb_theme,
        "rating_scaled": rating_scaled,
        "embeddings": embeddings,
        "rating_scaler": scaler,
    }

    return meta


def save_preprocess_artifacts(out_dir: str, meta: dict):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(meta["mlb_genre"], os.path.join(out_dir, "mlb_genre.joblib"))
    joblib.dump(meta["mlb_theme"], os.path.join(out_dir, "mlb_theme.joblib"))
    joblib.dump(meta["rating_scaler"], os.path.join(out_dir, "rating_scaler.joblib"))
    np.save(os.path.join(out_dir, "embeddings.npy"), meta["embeddings"])
    meta["df"].to_csv(os.path.join(out_dir, "games_table.csv"), index=False)
