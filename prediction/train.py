import os
import shutil
import datetime

from prediction.preprocess import (
    load_dataset,
    load_or_compute_embeddings,
    prepare_features,
    save_preprocess_artifacts,
    load_enum_mappings,
    _parse_list_field
)
from prediction.model import HybridRecommender
from prediction import cache


def train_pipeline(csv_path: str = "data_collection/out/igdb_games_all.csv",
                   embeddings_path: str = "data_visualization/embeddings/summary_embeddings.npy",
                   cache_root: str = "prediction/cache"):
    print("Loading dataset...")
    df = load_dataset(csv_path)
    print(f"Rows: {len(df)}")

    print("Creating run directory for caching artifacts...")
    run = cache.make_run_dir(cache_root)

    # copy raw CSV into run/raw for provenance
    try:
        shutil.copy(csv_path, run["raw"])
    except Exception:
        pass

    print("Loading enum mappings (genres/themes) ...")
    genre_map, theme_map = load_enum_mappings()

    print("Loading or computing embeddings...")
    emb = load_or_compute_embeddings(df, embeddings_path, genre_map=genre_map, theme_map=theme_map)

    print("Preparing features...")
    # map genre/theme ids to names for encoding to provide textual context
    if genre_map or theme_map:
        # create new columns with names where possible
        def map_names(lst, mapping):
            if not lst:
                return []
            out = []
            for v in lst:
                key = str(v)
                if mapping and key in mapping:
                    out.append(mapping[key])
                else:
                    out.append(str(v))
            return out

        df = df.copy()
        df["genres_parsed"] = df["genres"].apply(_parse_list_field)
        df["themes_parsed"] = df["themes"].apply(_parse_list_field)
        df["genres_names"] = df["genres_parsed"].apply(lambda lst: map_names(lst, genre_map))
        df["themes_names"] = df["themes_parsed"].apply(lambda lst: map_names(lst, theme_map))
        meta = prepare_features(df, emb, genre_col="genres_names", theme_col="themes_names")
    else:
        meta = prepare_features(df, emb)

    print("Fitting recommender...")
    rec = HybridRecommender(embed_dim=emb.shape[1])
    rec.fit(emb, meta["df"], meta["genres_mat"], meta["themes_mat"], meta["rating_scaled"])

    print(f"Saving artifacts to run dirs under {run['root']}...")
    rec.save(run["model"])
    save_preprocess_artifacts(run["preprocess"], meta)

    # metadata
    run_meta = {
        "csv": csv_path,
        "embeddings_path": embeddings_path,
        "rows": len(df),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    cache.write_run_metadata(run["root"], run_meta)

    print("Training complete. Artifacts cached at:")
    print(run["root"])


if __name__ == "__main__":
    os.makedirs("prediction/models", exist_ok=True)
    train_pipeline()
