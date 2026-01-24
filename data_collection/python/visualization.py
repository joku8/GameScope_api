
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import os

def fetch_igdb_genres():
    load_dotenv('../../.env')

    client_id = os.getenv("IGDB_CLIENT_ID")
    client_secret = os.getenv("IGDB_CLIENT_SECRET")

    token_resp = requests.post(
        "https://id.twitch.tv/oauth2/token",
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        }
    ).json()

    token = token_resp["access_token"]

    query = "fields id, name; limit 500;"

    resp = requests.post(
        "https://api.igdb.com/v4/genres",
        data=query,
        headers={
            "Client-ID": client_id,
            "Authorization": f"Bearer {token}",
        }
    )

    genres = resp.json()
    return {g["id"]: g["name"] for g in genres}

def map_genre_names(df, genre_map):
    df = df.copy()
    df["genre_names"] = df["genres"].apply(
        lambda lst: [genre_map.get(g, f"Unknown({g})") for g in lst]
    )
    return df

def load_data(filename):
    df = pd.read_csv(filename)

    # Columns that contain comma-separated integers
    list_cols = ["genres", "platforms", "themes"]

    for col in list_cols:
        df[col] = (
            df[col]
            .fillna("")                      # handle empty cells
            .apply(lambda x: [int(v) for v in x.split(",")] if x else [])
        )

    return df

def plot_release_year_histogram(df):
    # Convert to datetime
    df['first_release_date'] = pd.to_datetime(df['first_release_date'], errors='coerce')
    df = df.dropna(subset=['first_release_date'])

    # Extract year
    df['release_year'] = df['first_release_date'].dt.year

    # Count games per year
    year_counts = df['release_year'].value_counts().sort_index()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(year_counts.index, year_counts.values, color='skyblue', edgecolor='black')
    plt.title('Distribution of Game Release Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Games')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

def rating_histogram(df):
    # Remove zero ratings
    df = df[df['rating'] != 0]

    plt.figure(figsize=(10, 5))
    sns.histplot(
        df['rating'].dropna(),
        bins=20,
        kde=False,
        color='lightgreen',
        edgecolor='black'
    )

    plt.title('Distribution of Game Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Number of Games')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    plt.savefig('../out/rating_histogram.png')
    plt.show()

def genre_heatmap(df):
    # Use genre_names instead of raw IDs
    df = df[df["genre_names"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    # Build binary matrix
    all_genres = sorted({g for sub in df["genre_names"] for g in sub})
    genre_matrix = pd.DataFrame(0, index=df.index, columns=all_genres)

    for idx, genre_list in df["genre_names"].items():
        genre_matrix.loc[idx, genre_list] = 1

    # Correlation
    corr = genre_matrix.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.set_theme(style="white")
    plt.figure(figsize=(14, 12))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="viridis",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.7},
    )

    plt.title("Genre Co‑Occurrence Heatmap")
    plt.tight_layout()
    plt.show()

def visualize_data():
    df = load_data("../out/igdb_games_all.csv")
    # print("Data Summary:")
    # print(df.describe(include='all'))
    # print("\nFirst 5 Rows:")
    # print(df.head())
    # plot_release_year_histogram(df)
    # rating_histogram(df)
    genre_map = fetch_igdb_genres()
    df = map_genre_names(df, genre_map)
    genre_heatmap(df)

if __name__ == "__main__":
    visualize_data()