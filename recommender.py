"""
recommender.py
──────────────
Content-Based Movie Recommendation Engine
Uses TF-IDF on genres + Cosine Similarity to find similar movies.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class MovieRecommender:
    """
    Content-Based Movie Recommender.

    How it works:
    1. Load movies.csv and ratings.csv
    2. Build a TF-IDF matrix from movie genres (+ title keywords)
    3. Compute cosine similarity between all movie pairs
    4. For a given movie, return the N most similar movies

    Attributes:
        movies      : pd.DataFrame  — cleaned movie metadata
        tfidf_matrix: np.ndarray    — TF-IDF vectors (one per movie)
        similarity  : np.ndarray    — cosine similarity matrix
        indices     : pd.Series     — maps movie title → row index
    """

    def __init__(self):
        self.movies       = None
        self.tfidf_matrix = None
        self.similarity   = None
        self.indices      = None

    # ── STEP 1: LOAD DATA ──────────────────────────────────────
    def load_data(self, movies_path: str, ratings_path: str) -> None:
        """
        Load and clean the MovieLens CSV files.

        movies.csv columns : movieId, title, genres
        ratings.csv columns: userId, movieId, rating, timestamp
        """
        # Load CSVs
        movies  = pd.read_csv(movies_path)
        ratings = pd.read_csv(ratings_path)

        # ── Clean movies ──────────────────────────────────────
        # Extract year from title: "Toy Story (1995)" → year=1995, title="Toy Story"
        movies["year"] = movies["title"].str.extract(r"\((\d{4})\)").astype(float)
        movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()

        # Replace pipe separator in genres with space (for TF-IDF)
        # "Action|Adventure|Sci-Fi" → "Action Adventure SciFi"
        movies["genres_clean"] = (
            movies["genres"]
            .str.replace("|", " ", regex=False)
            .str.replace("-", "", regex=False)
            .str.replace("(no genres listed)", "", regex=False)
        )

        # Drop movies with no genre info
        movies = movies[movies["genres"] != "(no genres listed)"].copy()
        movies = movies.dropna(subset=["year"]).copy()

        # ── Merge average ratings ─────────────────────────────
        avg_ratings = (
            ratings.groupby("movieId")["rating"]
            .agg(avg_rating="mean", n_ratings="count")
            .reset_index()
        )
        movies = movies.merge(avg_ratings, on="movieId", how="left")
        movies["avg_rating"] = movies["avg_rating"].fillna(0).round(2)
        movies["n_ratings"]  = movies["n_ratings"].fillna(0).astype(int)

        # Reset index
        movies = movies.reset_index(drop=True)
        self.movies = movies
        print(f"✅ Loaded {len(movies):,} movies")

    # ── STEP 2: BUILD MODEL ────────────────────────────────────
    def build_model(self) -> None:
        """
        Build the TF-IDF matrix and cosine similarity matrix.

        TF-IDF = Term Frequency–Inverse Document Frequency
        ─────────────────────────────────────────────────────
        - Each movie is a "document"
        - Each genre word is a "term"
        - TF-IDF gives higher weight to genres that are rare across movies
          (e.g. "Musical" is more distinctive than "Drama")

        Cosine Similarity
        ─────────────────
        - Measures the angle between two TF-IDF vectors
        - cos(angle) = 1.0  → same direction = very similar movies
        - cos(angle) = 0.0  → perpendicular = completely different
        """
        # Build feature string: genres + repeated title words
        # Repeating title words gives them more weight in TF-IDF
        self.movies["features"] = (
            self.movies["genres_clean"] + " " +
            self.movies["title"].str.replace(r"[^a-zA-Z0-9 ]", " ", regex=True)
        )

        # TF-IDF Vectorizer
        # stop_words='english' removes common words like "the", "a", "of"
        # ngram_range=(1,2) also captures 2-word phrases like "Sci Fi"
        tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )

        # fit_transform: learn vocabulary + convert all movies to vectors
        # Result: matrix of shape (n_movies, n_features)
        self.tfidf_matrix = tfidf.fit_transform(self.movies["features"])

        # Compute cosine similarity between ALL pairs of movies
        # Result: square matrix of shape (n_movies, n_movies)
        # similarity[i][j] = how similar movie i is to movie j
        self.similarity = cosine_similarity(self.tfidf_matrix)

        # Build a title → index lookup for fast searching
        self.indices = pd.Series(
            self.movies.index,
            index=self.movies["title"].str.lower()
        )

        print(f"✅ Model built — similarity matrix: {self.similarity.shape}")

    # ── STEP 3: RECOMMEND ─────────────────────────────────────
    def recommend(self, title: str, n: int = 6) -> list[tuple[str, float]]:
        """
        Return top-N movie recommendations for a given title.

        Args:
            title : str  — movie title to base recommendations on
            n     : int  — number of recommendations to return

        Returns:
            List of (title, similarity_score) tuples, sorted by score desc.
        """
        # Find the movie's row index (case-insensitive)
        title_lower = title.lower().strip()

        if title_lower not in self.indices:
            # Try partial match
            matches = [t for t in self.indices.index if title_lower in t]
            if not matches:
                return []
            title_lower = matches[0]

        idx = self.indices[title_lower]

        # Handle duplicate titles (take first match)
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        # Get similarity scores for this movie vs all others
        # sim_scores is a list of (movie_index, similarity_score)
        sim_scores = list(enumerate(self.similarity[idx]))

        # Sort by similarity score — highest first
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Skip index 0 (that's the movie itself, similarity=1.0)
        # Take the next N most similar
        sim_scores = sim_scores[1: n + 1]

        # Get movie titles and scores
        results = []
        for movie_idx, score in sim_scores:
            movie_title = self.movies.iloc[movie_idx]["title"]
            results.append((movie_title, float(score)))

        return results

    # ── BONUS: SEARCH ─────────────────────────────────────────
    def search(self, query: str, limit: int = 10) -> list[str]:
        """Return movie titles matching a search query."""
        query = query.lower()
        matches = [
            t for t in self.movies["title"].tolist()
            if query in t.lower()
        ]
        return matches[:limit]
