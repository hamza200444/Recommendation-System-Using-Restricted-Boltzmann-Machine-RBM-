import requests
import pandas as pd
import time

API_KEY = "Your API Key"

BASE_URL = "https://api.themoviedb.org/3"

movies = []

# Placeholder image
PLACEHOLDER = "https://via.placeholder.com/500x750?text=No+Poster"

# -----------------------------
# GET GENRE MAP
# -----------------------------
def get_genre_map():

    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}"

    response = requests.get(url)

    data = response.json()

    return {
        g["id"]: g["name"]
        for g in data["genres"]
    }

genre_map = get_genre_map()

# -----------------------------
# FETCH POPULAR MOVIES
# -----------------------------
def fetch_popular_movies(page):

    url = (
        f"{BASE_URL}/movie/popular"
        f"?api_key={API_KEY}"
        f"&language=en-US"
        f"&page={page}"
    )

    response = requests.get(url)

    return response.json()["results"]

print("Fetching 500 movies from TMDB...")

page = 1

while len(movies) < 500:

    results = fetch_popular_movies(page)

    for movie in results:

        tmdb_id = movie.get("id")

        title = movie.get("title", "")

        overview = movie.get("overview", "")

        genre_ids = movie.get("genre_ids", [])

        genres = "|".join([
            genre_map.get(g, "")
            for g in genre_ids
        ])

        poster_path = movie.get("poster_path")

        # Build valid image URL
        if poster_path:

            poster_url = (
                f"https://image.tmdb.org/t/p/w500{poster_path}"
            )

        else:

            poster_url = PLACEHOLDER

        movies.append({

            "tmdb_id": tmdb_id,

            "title": title,

            "clean_title": (
                title.lower()
                .replace(":", "")
                .replace(",", "")
                .replace("-", "")
                .strip()
            ),

            "genres": genres,

            "overview": overview,

            "poster_url": poster_url

        })

        if len(movies) >= 500:
            break

    page += 1

    print(f"Fetched page {page}")

    time.sleep(0.25)

# -----------------------------
# SAVE DATASET
# -----------------------------
df = pd.DataFrame(movies)

# Remove duplicate titles
df = df.drop_duplicates(subset=["clean_title"])

df.to_csv(
    "tmdb_500_movies.csv",
    index=False,
    encoding="utf-8"
)

print("Dataset saved successfully!")

print(df.head())
