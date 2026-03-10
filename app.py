"""
🎬 Movie Recommendation System
Content-Based Filtering + OMDB API
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from recommender import MovieRecommender

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    .movie-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid #0f3460;
        color: white;
    }
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        color: #e94560;
        margin-bottom: 6px;
    }
    .movie-meta {
        font-size: 13px;
        color: #a8b2d8;
        margin: 3px 0;
    }
    .score-badge {
        background: #e94560;
        color: white;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #e94560, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)


# ── OMDB API HELPER ────────────────────────────────────────────
def fetch_movie_details(title: str, api_key: str) -> dict:
    """Fetch movie poster + details from OMDB API."""
    try:
        url = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&apikey={api_key}&plot=short"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        if data.get("Response") == "True":
            return data
    except Exception:
        pass
    return {}


def display_movie_card(movie: dict, score: float = None, api_key: str = ""):
    """Render a single movie card with poster."""
    details = fetch_movie_details(movie["title"], api_key) if api_key else {}
    poster  = details.get("Poster", "")
    year    = details.get("Year",   movie.get("year", "N/A"))
    genre   = details.get("Genre",  movie.get("genres", "N/A"))
    rating  = details.get("imdbRating", "N/A")
    plot    = details.get("Plot",   "No description available.")
    director= details.get("Director", "N/A")

    col_img, col_info = st.columns([1, 3])
    with col_img:
        if poster and poster != "N/A":
            st.image(poster, width=120)
        else:
            st.markdown("🎬", unsafe_allow_html=True)

    with col_info:
        score_html = f'<span class="score-badge">Match {score:.0%}</span>' if score else ""
        st.markdown(f"""
        <div class="movie-card">
            <div class="movie-title">{movie['title']} {score_html}</div>
            <div class="movie-meta">📅 {year} &nbsp;|&nbsp; 🎭 {genre}</div>
            <div class="movie-meta">⭐ IMDB: {rating} &nbsp;|&nbsp; 🎬 {director}</div>
            <div class="movie-meta" style="margin-top:8px">{plot}</div>
        </div>
        """, unsafe_allow_html=True)


# ── LOAD MODEL ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading recommendation engine...")
def load_recommender():
    rec = MovieRecommender()
    rec.load_data("data/movies.csv", "data/ratings.csv")
    rec.build_model()
    return rec


# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input(
        "🔑 OMDB API Key",
        type="password",
        placeholder="Get free key at omdbapi.com",
        help="Free key from omdbapi.com — enables posters & details"
    )
    n_recs = st.slider("Number of recommendations", 3, 15, 6)
    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    try:
        rec = load_recommender()
        st.success(f"✅ {len(rec.movies):,} movies loaded")
        st.info(f"🧠 Model: TF-IDF + Cosine Similarity")
    except Exception as e:
        st.error(f"Error: {e}")
    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[Get OMDB API Key](https://www.omdbapi.com/apikey.aspx)")
    st.markdown("[View on GitHub](https://github.com)")


# ── MAIN PAGE ──────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Content-Based Filtering using TF-IDF & Cosine Similarity</p>", unsafe_allow_html=True)
st.markdown("---")

# ── TABS ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "📊 Explore Dataset", "ℹ️ How It Works"])

# ── TAB 1: RECOMMENDATIONS ────────────────────────────────────
with tab1:
    try:
        rec = load_recommender()

        col1, col2 = st.columns([3, 1])
        with col1:
            movie_input = st.selectbox(
                "🎬 Search and select a movie you like:",
                options=[""] + sorted(rec.movies["title"].tolist()),
                index=0
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_btn = st.button("🚀 Recommend!", use_container_width=True, type="primary")

        if movie_input and (search_btn or movie_input):
            with st.spinner("Finding similar movies..."):
                recommendations = rec.recommend(movie_input, n=n_recs)

            if recommendations:
                st.markdown(f"### 🎯 Because you liked **{movie_input}**:")

                # Show the selected movie first
                st.markdown("#### Your Movie:")
                selected = rec.movies[rec.movies["title"] == movie_input].iloc[0].to_dict()
                display_movie_card(selected, api_key=api_key)

                st.markdown("---")
                st.markdown("#### You might also like:")

                for i, (title, score) in enumerate(recommendations):
                    movie_row = rec.movies[rec.movies["title"] == title]
                    if not movie_row.empty:
                        display_movie_card(movie_row.iloc[0].to_dict(), score=score, api_key=api_key)
            else:
                st.warning("Movie not found. Try another title.")

    except Exception as e:
        st.error(f"Could not load recommender: {e}")

# ── TAB 2: EXPLORE ────────────────────────────────────────────
with tab2:
    try:
        rec = load_recommender()
        movies_df = rec.movies

        st.markdown("### 📊 Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Movies", f"{len(movies_df):,}")
        col2.metric("Genres", movies_df["genres"].str.split("|").explode().nunique())
        col3.metric("Years", f"{movies_df['year'].min():.0f} – {movies_df['year'].max():.0f}")

        st.markdown("---")

        # Genre distribution
        st.markdown("### 🎭 Genre Distribution")
        genre_counts = (
            movies_df["genres"]
            .str.split("|")
            .explode()
            .value_counts()
            .head(15)
        )
        st.bar_chart(genre_counts)

        # Movies by decade
        st.markdown("### 📅 Movies by Decade")
        movies_df["decade"] = (movies_df["year"] // 10 * 10).astype(int)
        decade_counts = movies_df["decade"].value_counts().sort_index()
        st.bar_chart(decade_counts)

        # Raw data
        st.markdown("### 🗃️ Browse Movies")
        genre_filter = st.multiselect(
            "Filter by genre:",
            options=sorted(movies_df["genres"].str.split("|").explode().unique())
        )
        if genre_filter:
            mask = movies_df["genres"].apply(lambda g: any(gf in g for gf in genre_filter))
            st.dataframe(movies_df[mask][["title", "year", "genres"]].reset_index(drop=True), use_container_width=True)
        else:
            st.dataframe(movies_df[["title", "year", "genres"]].head(100).reset_index(drop=True), use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data: {e}")

# ── TAB 3: HOW IT WORKS ───────────────────────────────────────
with tab3:
    st.markdown("""
    ## 🧠 How This Recommender Works

    ### Step-by-step:

    **1. Data Collection**
    > We use the MovieLens dataset — 9,000+ movies with titles, genres, and user ratings.

    **2. Feature Extraction (TF-IDF)**
    > Each movie is described by its genres (e.g. `Action|Adventure|Sci-Fi`).
    > TF-IDF converts these text descriptions into a numeric vector for each movie.
    > Think of it as giving each movie a "DNA fingerprint" made of numbers.

    **3. Similarity Calculation (Cosine Similarity)**
    > We measure the angle between two movie vectors.
    > Movies with similar genres have vectors pointing in similar directions → small angle → high similarity.
    > Score of 1.0 = identical, 0.0 = completely different.

    **4. Recommendation**
    > When you pick a movie, we find its vector, then find the N closest vectors.
    > Those closest movies = our recommendations!

    **5. OMDB API**
    > We enrich each recommendation with live data from OMDB:
    > posters, IMDB ratings, directors, plot summaries.

    ---
    ### 🔬 Tech Stack

    | Component | Technology |
    |---|---|
    | Language | Python 3.10+ |
    | ML | scikit-learn (TF-IDF, Cosine Similarity) |
    | Data | pandas, numpy |
    | API | OMDB API |
    | Frontend | Streamlit |
    | Dataset | MovieLens (GroupLens) |

    ---
    ### 📁 Project Structure
    ```
    movie_recommender/
    ├── app.py              ← Streamlit frontend (this file)
    ├── recommender.py      ← ML recommendation engine
    ├── data/
    │   ├── movies.csv      ← Movie metadata
    │   └── ratings.csv     ← User ratings
    ├── requirements.txt    ← Python dependencies
    └── README.md           ← GitHub documentation
    ```
    """)
