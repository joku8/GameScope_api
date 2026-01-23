# ingest_data_igdb.py
import os
import requests
import csv
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

CLIENT_ID = os.getenv("IGDB_CLIENT_ID")
CLIENT_SECRET = os.getenv("IGDB_CLIENT_SECRET")

BASE_URL = "https://api.igdb.com/v4"

def get_access_token(client_id, client_secret):
    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    resp = requests.post(url, params=params)
    resp.raise_for_status()
    return resp.json()["access_token"]

def igdb_post(endpoint, query, access_token):
    headers = {
        "Client-ID": CLIENT_ID,
        "Authorization": f"Bearer {access_token}"
    }
    resp = requests.post(f"{BASE_URL}/{endpoint}", headers=headers, data=query)
    resp.raise_for_status()
    return resp.json()

def fetch_games(access_token, limit=50):
    query = f"""
    fields id, name, first_release_date, genres, platforms, rating, rating_count, summary, url;
    limit {limit};
    """
    return igdb_post("games", query, access_token)

def fetch_all_names(access_token, endpoint, all_ids):
    """Batch fetch all names for given IDs"""
    if not all_ids:
        return {}
    ids_str = ",".join(map(str, all_ids))
    query = f"fields id, name; where id = ({ids_str}); limit {len(all_ids)};"
    items = igdb_post(endpoint, query, access_token)
    return {item["id"]: item["name"] for item in items}

def main():
    print("Getting access token...")
    token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    print("Access token acquired!")

    print("Fetching games...")
    games = fetch_games(token, limit=50)

    # Collect all unique genre & platform IDs
    all_genre_ids = set()
    all_platform_ids = set()
    for game in games:
        all_genre_ids.update(game.get("genres") or [])
        all_platform_ids.update(game.get("platforms") or [])

    # Batch fetch names
    print("Fetching genre names...")
    genre_map = fetch_all_names(token, "genres", all_genre_ids)
    print("Fetching platform names...")
    platform_map = fetch_all_names(token, "platforms", all_platform_ids)

    print("Processing games...")
    for game in games:
        # Replace IDs with names
        game["genres"] = [genre_map.get(gid, "") for gid in game.get("genres") or []]
        game["platforms"] = [platform_map.get(pid, "") for pid in game.get("platforms") or []]

        # Format release date
        ts = game.get("first_release_date")
        if ts:
            game["first_release_date"] = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            game["first_release_date"] = ""

    # Write CSV
    csv_file = "out/igdb_games_phase1.csv"
    print(f"Writing CSV to {csv_file}...")
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "name", "first_release_date", "genres", "platforms",
            "rating", "rating_count", "summary", "url"
        ])
        writer.writeheader()
        for game in games:
            writer.writerow({
                "id": game.get("id"),
                "name": game.get("name"),
                "first_release_date": game.get("first_release_date"),
                "genres": ", ".join(game.get("genres", [])),
                "platforms": ", ".join(game.get("platforms", [])),
                "rating": game.get("rating", ""),
                "rating_count": game.get("rating_count", ""),
                "summary": game.get("summary", ""),
                "url": game.get("url", "")
            })

    print("Done!")

if __name__ == "__main__":
    main()
