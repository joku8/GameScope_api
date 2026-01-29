import os
from typing import List

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# optional FAISS (Facebook AI Sim Search) support for fast nearest-neighbor search (CPU/GPU)
try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None
    FAISS_AVAILABLE = False

class HybridRecommender:
    """Hybrid similarity-based recommender combining semantic and categorical signals.

    WORKFLOW:
      fit()      -> Normalize embeddings, build FAISS index, store metadata
      recommend()-> Retrieve 50 nearest neighbors -> compute hybrid score -> return top-n

    TUNABLE PARAMETERS:
    ===================
    See __init__() for weights (w_emb, w_genre, w_theme, w_rating)
    See fit() for FAISS index choice and normalization
    See recommend() for candidate pool size (k=50), rating_factor formula
    """

    def __init__(self, embed_dim: int = 128):
        self.nn = None
        self.kmeans = None
        self.embeddings = None
        self.df = None
        self.genres_mat = None
        self.themes_mat = None
        self.rating_scaled = None
        self.embed_dim = embed_dim
        self.use_faiss = FAISS_AVAILABLE
        self.faiss_index = None
        self.faiss_gpu = False
        self.embeddings_norm = None

        # TUNABLE WEIGHTS: Control recommendation behavior. Must sum to 1.0.
        # Increasing w_emb (>0.6) emphasizes semantic similarity (summary text).
        # Increasing w_genre/w_theme enforces stricter tag matching.
        # Increasing w_rating prioritizes well-rated games over similarity.
        self.w_emb = 0.6       # Embedding similarity (dominant signal)
        self.w_genre = 0.2     # Genre tag Jaccard overlap
        self.w_theme = 0.15    # Theme tag Jaccard overlap
        self.w_rating = 0.05   # Game's dataset rating (popularity)

    def fit(self, embeddings: np.ndarray, df, genres_mat: np.ndarray, themes_mat: np.ndarray, rating_scaled: np.ndarray):
        """Build FAISS index and store metadata for recommendations.
        
        IMPLEMENTATION NOTES:
        - Embeddings are normalized to L2 norm=1, so inner-product = cosine similarity.
        - FAISS.IndexFlatIP does O(d) inner-product search (fast, ~1-10ms per query on CPU).
        - Normalizing by L2: vec_norm = vec / ||vec||_2 makes cosine = inner-product.
        - Falls back to sklearn.NearestNeighbors if FAISS unavailable (slower but reliable).
        - KMeans clustering is optional and unused in current recommend() but available
          for offline analysis or cold-start scenarios.
        """
        self.embeddings = embeddings
        self.df = df.reset_index(drop=True)
        self.genres_mat = genres_mat
        self.themes_mat = themes_mat
        self.rating_scaled = rating_scaled

        # Prefer FAISS for fast nearest-neighbor search if available.
        # TUNABLE: Can replace IndexFlatIP (linear scan O(d)) with:
        #   - IndexIVFFlat (inverted file)
        #   - IndexHNSW (hierarchical navigable small world, sub-linear)
        #   - IndexPQ (product quantization, lower memory)
        if self.use_faiss:
            try:
                # Normalize embeddings to unit L2 norm.
                # This transforms cosine_sim(a,b) = (a·b) / (||a|| ||b||)
                # into inner_product(a_norm, b_norm) = (a/||a||) · (b/||b||) on normalized vectors.
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
                self.embeddings_norm = (self.embeddings / norms).astype('float32')
                d = self.embeddings_norm.shape[1]
                index = faiss.IndexFlatIP(d)
                # try to move index to GPU if supported
                # Unsure why this doesnt work dispite nvidia-smi showing GPU present
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    self.faiss_gpu = True
                except Exception:
                    self.faiss_gpu = False
                index.add(self.embeddings_norm)
                self.faiss_index = index
            except Exception:
                # fallback to sklearn
                self.use_faiss = False
                self.nn = NearestNeighbors(n_neighbors=50, metric="cosine")
                self.nn.fit(self.embeddings)
        else:
            self.nn = NearestNeighbors(n_neighbors=50, metric="cosine")
            self.nn.fit(self.embeddings)

        # cluster embeddings for optional use / diagnostics
        try:
            self.kmeans = KMeans(n_clusters=min(50, max(2, len(self.embeddings)//50)), random_state=42)
            self.kmeans.fit(self.embeddings)
        except Exception:
            self.kmeans = None

    def save(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # joblib dump sklearn NN only if present
        if self.nn is not None:
            try:
                joblib.dump(self.nn, os.path.join(out_dir, "nn.joblib"))
            except Exception:
                pass
        joblib.dump(self.kmeans, os.path.join(out_dir, "kmeans.joblib"))
        np.save(os.path.join(out_dir, "embeddings.npy"), self.embeddings)
        self.df.to_csv(os.path.join(out_dir, "games_table.csv"), index=False)
        np.save(os.path.join(out_dir, "genres_mat.npy"), self.genres_mat)
        np.save(os.path.join(out_dir, "themes_mat.npy"), self.themes_mat)
        np.save(os.path.join(out_dir, "rating_scaled.npy"), self.rating_scaled)

    def load(self, out_dir: str):
        # load sklearn neural net if present
        try:
            self.nn = joblib.load(os.path.join(out_dir, "nn.joblib"))
        except Exception:
            self.nn = None
        try:
            self.kmeans = joblib.load(os.path.join(out_dir, "kmeans.joblib"))
        except Exception:
            self.kmeans = None
        self.embeddings = np.load(os.path.join(out_dir, "embeddings.npy"))
        import pandas as pd

        self.df = pd.read_csv(os.path.join(out_dir, "games_table.csv"))
        self.genres_mat = np.load(os.path.join(out_dir, "genres_mat.npy"))
        self.themes_mat = np.load(os.path.join(out_dir, "themes_mat.npy"))
        self.rating_scaled = np.load(os.path.join(out_dir, "rating_scaled.npy"))

        # If FAISS is available, rebuild the index from loaded embeddings
        if FAISS_AVAILABLE:
            try:
                norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                self.embeddings_norm = (self.embeddings / norms).astype('float32')
                d = self.embeddings_norm.shape[1]
                index = faiss.IndexFlatIP(d)
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    self.faiss_gpu = True
                except Exception:
                    self.faiss_gpu = False
                index.add(self.embeddings_norm)
                self.faiss_index = index
                self.use_faiss = True
            except Exception:
                self.use_faiss = False

    def _genre_theme_similarity(self, idxs: List[int], query_idx: int):
        """Compute Jaccard (tag overlap) similarity for genre and theme.
        
        IMPLEMENTATION NOTES:
        - Genres/themes are binary one-hot encoded (e.g., [1,0,1,0,...] = RPG+Fantasy)
        - Jaccard(A,B) = |A ∩ B| / |A ∪ B| measures overlap: 1.0=identical, 0.0=disjoint
        - Used as categorical constraint alongside semantic embeddings.
        - All-zero inputs are handled (set union=1 to return 0 instead of NaN).
        - TUNABLE: Could replace Jaccard with cosine similarity on binary vectors
          or other metrics like Tanimoto or Hamming distance.
        """
        # Compute normalized overlap: intersection / union (Jaccard similarity)
        g = self.genres_mat
        t = self.themes_mat
        qg = g[query_idx:query_idx+1]
        qt = t[query_idx:query_idx+1]
        
        def overlap(mat, q):
            """Jaccard: |intersection| / |union| for binary one-hot vectors."""
            inter = (mat & q).sum(axis=1)  # Bitwise AND counts shared tags
            union = (mat | q).sum(axis=1)  # Bitwise OR counts all tags
            union = np.where(union == 0, 1, union)  # Avoid division by zero
            return inter / union

        return overlap(g[idxs], qg.ravel()), overlap(t[idxs], qt.ravel())

    def recommend(self, game_name: str, personal_rating: float, n: int = 3):
        # find game index
        matches = self.df[self.df["name"].str.lower() == game_name.lower()]
        if matches.empty:
            # try partial match
            matches = self.df[self.df["name"].str.lower().str.contains(game_name.lower())]
            if matches.empty:
                raise ValueError(f"Game '{game_name}' not found in catalog")
        query_idx = int(matches.index[0])

        q_emb = self.embeddings[query_idx:query_idx+1]

        # RETRIEVE CANDIDATES: Get nearest neighbors via embedding similarity.
        # TUNABLE: Candidate pool size k=50 balances diversity vs speed.
        # Increase to 100+ for broader search space (slower, more diversity).
        # Decrease to 20 for faster, more focused results.
        if self.use_faiss and self.faiss_index is not None:
            qn = q_emb.astype('float32')
            qn = qn / np.linalg.norm(qn, axis=1, keepdims=True)  # Normalize query
            k = min(50, len(self.embeddings))  # TUNABLE: Candidate pool size
            D, I = self.faiss_index.search(qn, k)  # Returns (inner-products, indices)
            cand_idxs = I.ravel().tolist()
            sims = D.ravel().astype(float)
        else:
            distances, neighbors = self.nn.kneighbors(q_emb, n_neighbors=min(50, len(self.embeddings)))
            cand_idxs = neighbors.ravel().tolist()
            sims = cosine_similarity(q_emb, self.embeddings[cand_idxs]).ravel()

        # Compute categorical similarities (genre and theme tag overlaps)
        g_sims, t_sims = self._genre_theme_similarity(cand_idxs, query_idx)

        # TUNABLE: Rating factor applies user personalization to recommendation scores.
        # Formula: rating_factor = 1.0 + (personal_rating - 5.0) / 10.0
        #   personal_rating=0   -> factor=0.5   (halve scores, favor diversity)
        #   personal_rating=5   -> factor=1.0   (no scaling)
        #   personal_rating=10  -> factor=1.5   (boost scores, favor similarity+popularity)
        # Alternative formulas to experiment with:
        #   - Sigmoid: 1 + tanh((personal_rating - 5) / 5) / 2 for smoother curve
        #   - Quadratic: 0.5 + (personal_rating / 10) ** 2 for non-linear scaling
        #   - Flat: 1.0 (ignore personal rating, purely content-based)
        rating_factor = 1.0 + (float(personal_rating) - 5.0) / 10.0

        # HYBRID SCORE: Weighted combination of four signals.
        # All components are normalized to [0,1] range for interpretability.
        # TUNABLE: Adjust w_* weights in __init__() to change recommendation focus:
        #   - Increase w_emb for semantic-driven (text-based) recommendations
        #   - Increase w_genre/w_theme for stricter tag matching
        #   - Increase w_rating to emphasize well-rated games regardless of similarity
        # Example: For a tag-heavy system, try w_emb=0.4, w_genre=0.3, w_theme=0.2, w_rating=0.1
        combined = (
            self.w_emb * sims
            + self.w_genre * g_sims
            + self.w_theme * t_sims
            + self.w_rating * self.rating_scaled[cand_idxs]
        )
        # Apply personalization multiplier
        combined = combined * rating_factor

        # assemble results excluding the query itself
        import numpy as _np

        cand_arr = _np.array(cand_idxs)
        # remove the query index
        mask = cand_arr != query_idx
        cand_arr = cand_arr[mask]
        combined = combined[mask]

        top_idx = combined.argsort()[::-1][:n]
        results = []
        for i in top_idx:
            idx = int(cand_arr[i])
            results.append({
                "name": str(self.df.loc[idx, "name"]),
                "score": float(combined[i]),
                "genres": self.df.loc[idx, "genres"],
                "themes": self.df.loc[idx, "themes"],
            })

        return results
    
    def recommend_from_profile(self, rated_games: List[tuple], n: int = 3):
        """Recommend games based on a user's preference profile (multiple rated games).

        MULTI-GAME APPROACH:
        ====================
        Instead of querying a single game, accepts a list of (game_name, personal_rating) tuples
        and constructs a weighted preference vector that:
        - Pulls TOWARD games with high ratings (positive influence)
        - Pushes AWAY from games with low ratings (negative influence)
        - Aggregates genre/theme preferences across all rated games

        This enables queries like:
          "I loved Baldur's Gate 1 & 2 (9/10) but hated GTA (2/10)"
          -> Recommend RPGs with strategy, avoid open-world crime games

        ALGORITHM:
        ==========
        1. Find embedding for each rated game
        2. Compute weighted centroid: sum(embedding * rating_weight) / sum(weights)
           where rating_weight = (personal_rating - 5) / 5 (range: -1 to +1)
           - Positive weights (liked games): pull recommendations toward them
           - Negative weights (disliked games): push recommendations away
        3. Aggregate genre/theme preferences:
           - For liked games: boost their genres/themes
           - For disliked games: suppress their genres/themes
        4. Find nearest neighbors to weighted embedding
        5. Compute hybrid score considering preference direction
        6. Return top-n recommendations

        PARAMETERS:
        ===========
        rated_games: List of (game_name: str, personal_rating: float) tuples
                     personal_rating should be in [0, 10] range
                     Example: [("Baldur's Gate", 9.0), ("GTA V", 2.0)]
        n: Number of recommendations to return

        RETURNS:
        ========
        List of dicts with keys: name, score, genres, themes
        """
        if not rated_games:
            raise ValueError("Must provide at least one rated game")

        # Find game indices and compute rating weights
        query_idxs = []
        rating_weights = []  # Weights in range [-1, +1]
        
        for game_name, personal_rating in rated_games:
            matches = self.df[self.df["name"].str.lower() == game_name.lower()]
            if matches.empty:
                matches = self.df[self.df["name"].str.lower().str.contains(game_name.lower())]
                if matches.empty:
                    raise ValueError(f"Game '{game_name}' not found in catalog")
            query_idx = int(matches.index[0])
            query_idxs.append(query_idx)
            
            # Convert rating (0-10) to weight (-1 to +1)
            # rating=0 -> weight=-1 (dislike, push away)
            # rating=5 -> weight=0 (neutral, ignore)
            # rating=10 -> weight=+1 (like, pull toward)
            weight = (float(personal_rating) - 5.0) / 5.0
            rating_weights.append(weight)

        # Compute weighted centroid embedding
        weighted_embs = self.embeddings[query_idxs] * np.array(rating_weights).reshape(-1, 1)
        centroid_emb = weighted_embs.sum(axis=0) / np.array(rating_weights).sum()
        centroid_emb = centroid_emb.reshape(1, -1)

        # Normalize centroid for FAISS search
        centroid_norm = np.linalg.norm(centroid_emb, axis=1, keepdims=True)
        centroid_norm = np.where(centroid_norm == 0, 1.0, centroid_norm)
        centroid_emb_norm = (centroid_emb / centroid_norm).astype('float32')

        # Retrieve candidates near the weighted centroid
        if self.use_faiss and self.faiss_index is not None:
            k = min(100, len(self.embeddings))  # Use larger pool for multi-game context
            D, I = self.faiss_index.search(centroid_emb_norm, k)
            cand_idxs = I.ravel().tolist()
            sims = D.ravel().astype(float)
        else:
            distances, neighbors = self.nn.kneighbors(centroid_emb, n_neighbors=min(100, len(self.embeddings)))
            cand_idxs = neighbors.ravel().tolist()
            sims = cosine_similarity(centroid_emb, self.embeddings[cand_idxs]).ravel()

        # Compute aggregate genre/theme preferences
        # Liked games boost their genres/themes, disliked games suppress theirs
        preferred_genres = np.zeros(self.genres_mat.shape[1])
        preferred_themes = np.zeros(self.themes_mat.shape[1])
        
        for idx, weight in zip(query_idxs, rating_weights):
            if weight > 0:  # User liked this game
                preferred_genres += self.genres_mat[idx] * weight
                preferred_themes += self.themes_mat[idx] * weight
            else:  # User disliked this game
                # Invert: subtract to make disliked genres/themes less favorable
                preferred_genres += self.genres_mat[idx] * weight  # weight is negative
                preferred_themes += self.themes_mat[idx] * weight

        # Normalize preferences to [0, 1]
        pref_g_max = np.abs(preferred_genres).max()
        pref_t_max = np.abs(preferred_themes).max()
        if pref_g_max > 0:
            preferred_genres = (preferred_genres + pref_g_max) / (2 * pref_g_max)
        else:
            preferred_genres = np.ones_like(preferred_genres) * 0.5
        if pref_t_max > 0:
            preferred_themes = (preferred_themes + pref_t_max) / (2 * pref_t_max)
        else:
            preferred_themes = np.ones_like(preferred_themes) * 0.5

        # Compute genre/theme similarity to the aggregated preference profile
        g_prefs = preferred_genres  # Candidate's genre match to preference profile
        t_prefs = preferred_themes  # Candidate's theme match to preference profile
        
        # Match each candidate's genres/themes to the preference profile
        g_sims = (self.genres_mat[cand_idxs] * g_prefs).sum(axis=1) / (self.genres_mat[cand_idxs].sum(axis=1) + 1e-6)
        t_sims = (self.themes_mat[cand_idxs] * t_prefs).sum(axis=1) / (self.themes_mat[cand_idxs].sum(axis=1) + 1e-6)

        # Compute average game rating for overall preference direction
        avg_rating_weight = np.mean(rating_weights)  # In [-1, +1]
        rating_factor = 1.0 + avg_rating_weight * 0.5  # Map to [0.5, 1.5] range

        # Compute hybrid score
        combined = (
            self.w_emb * sims
            + self.w_genre * g_sims
            + self.w_theme * t_sims
            + self.w_rating * self.rating_scaled[cand_idxs]
        )
        combined = combined * rating_factor

        # Exclude already-rated games from recommendations
        cand_arr = np.array(cand_idxs)
        mask = ~np.isin(cand_arr, query_idxs)
        cand_arr = cand_arr[mask]
        combined = combined[mask]

        # Return top-n
        if len(combined) == 0:
            return []
        
        top_idx = combined.argsort()[::-1][:n]
        results = []
        for i in top_idx:
            idx = int(cand_arr[i])
            results.append({
                "name": str(self.df.loc[idx, "name"]),
                "score": float(combined[i]),
                "genres": self.df.loc[idx, "genres"],
                "themes": self.df.loc[idx, "themes"],
            })

        return results