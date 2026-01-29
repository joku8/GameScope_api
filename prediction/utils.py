import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import pandas as pd


def avg_cosine_similarity(embeddings, idxs_query, idxs_candidates):
    sims = cosine_similarity(embeddings[idxs_query], embeddings[idxs_candidates])
    return float(sims.mean())


def avg_genre_overlap(genres_mat, idxs_query, idxs_candidates):
    q = genres_mat[idxs_query]
    c = genres_mat[idxs_candidates]
    inter = (q & c).sum(axis=1)
    union = (q | c).sum(axis=1)
    union = np.where(union == 0, 1, union)
    return float((inter / union).mean())


def most_similar_game_names(query: str, n: int = 5) -> list:
    """
    Return top n most similar game names from the IGDB dataset.
    
    Args:
        query: Partial or full game name to search for (case-insensitive)
        n: Number of suggestions to return (default: 5)
    
    Returns:
        List of (game_name, similarity_score, summary) tuples sorted by similarity (highest first)
    
    Example:
        >>> most_similar_game_names("dark", n=3)
        [("Dark Souls III", 0.87, "Action RPG..."), ("Darker Shade of Grey", 0.65, ".."), ...]
    """
    # Load game names from CSV
    df = pd.read_csv("data_collection/out/igdb_games_all.csv")
    available_games = df[["name", "summary"]].dropna(subset=["name"])  # Drop rows with NaN names
    
    query_lower = query.lower()
    
    # Compute similarity for each game name
    similarities = []
    for idx, row in available_games.iterrows():
        game_name = row["name"]
        summary = row["summary"] if pd.notna(row["summary"]) else ""
        game_lower = str(game_name).lower()  # Ensure string
        # Use SequenceMatcher for similarity ratio
        ratio = SequenceMatcher(None, query_lower, game_lower).ratio()
        similarities.append((game_name, ratio, summary))
    
    # Sort by similarity (descending) and return top n
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:n]


