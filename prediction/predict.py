from prediction.model import HybridRecommender
from prediction import cache
from prediction.utils import most_similar_game_names
from prediction.preprocess import load_enum_mappings, _parse_list_field

if __name__=="__main__":
    model = HybridRecommender()
    model_dir = cache.get_latest_model_dir()
    if model_dir is None:
        raise FileNotFoundError("No cached model found. Run training first.")
    model.load(model_dir)

    # Load enum mappings for genre/theme names
    genre_map, theme_map = load_enum_mappings()

    # Input
    game_preference_profile = [
        ("Peak", 7.0),
        ("R.E.P.O.", 7.0),
        ("Teamfight Tactics", 9.0),
        ("Bloons TD 6", 8.0),
        ("Minecraft", 9.0),
    ]

    recommendations = model.recommend_from_profile(game_preference_profile, n=5)

    print("Top Recommendations:")
    for rec in recommendations:
        # Convert genre/theme IDs to names
        genre_ids = _parse_list_field(rec['genres']) if isinstance(rec['genres'], str) else rec['genres']
        theme_ids = _parse_list_field(rec['themes']) if isinstance(rec['themes'], str) else rec['themes']
        
        genre_names = [genre_map.get(str(g), str(g)) for g in genre_ids] if genre_map else genre_ids
        theme_names = [theme_map.get(str(t), str(t)) for t in theme_ids] if theme_map else theme_ids
        
        genres_str = ", ".join(genre_names)
        themes_str = ", ".join(theme_names)
        
        print(f"- {rec['name']} (Score: {rec['score']:.4f}, Genres: {genres_str}, Themes: {themes_str})")

