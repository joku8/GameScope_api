
import os
from typing import List

from prediction.model import HybridRecommender
from prediction import cache

_rec = None

def _ensure_loaded(model_dir: str = None):
    global _rec
    if _rec is None:
        _rec = HybridRecommender()
        if model_dir:
            _dir = model_dir
        else:
            _dir = cache.get_latest_model_dir()
            if _dir is None:
                raise FileNotFoundError("No cached model found. Run training first.")
        _rec.load(_dir)
    return _rec

def recommend(game_name: str, personal_rating: float, n: int = 3, model_dir: str = None) -> List[dict]:
    """Get top-n game recommendations for a single game."""
    rec = _ensure_loaded(model_dir)
    return rec.recommend(game_name, personal_rating, n=n)


def recommend_from_profile(rated_games: List[tuple], n: int = 3, model_dir: str = None) -> List[dict]:
    """Get recommendations based on a user's preference profile (multiple rated games).
    
    MULTI-GAME APPROACH:
    Instead of recommending based on a single game, accepts multiple games with different 
    ratings and computes a weighted preference vector that:
    - Pulls TOWARD games the user loved (high ratings)
    - Pushes AWAY from games the user disliked (low ratings)
    - Balances genre/theme preferences across all rated games
    
    Parameters:
    -----------
    rated_games : List[tuple]
        List of (game_name: str, personal_rating: float) tuples.
        personal_rating should be in [0, 10] range.
        Example: [("Baldur's Gate", 9.0), ("Baldur's Gate II", 9.0), ("GTA V", 2.0)]
    n : int
        Number of recommendations to return (default 3)
    model_dir : str, optional
        Path to model directory. If None, uses latest cached model.
    
    Returns:
    --------
    List[dict]
        List of recommendations with keys: name, score, genres, themes
    
    Algorithm:
    ----------
    1. Find embeddings for each rated game
    2. Compute weighted centroid: sum(embedding * weight) where
       weight = (personal_rating - 5) / 5  (range: -1 to +1)
    3. Aggregate genre/theme preferences (liked games boost, disliked suppress)
    4. Find nearest neighbors to weighted centroid
    5. Score considering both centroid similarity and preference direction
    6. Return top-n, excluding already-rated games
    """
    rec = _ensure_loaded(model_dir)
    return rec.recommend_from_profile(rated_games, n=n)

