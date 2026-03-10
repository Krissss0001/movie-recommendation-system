# 🎬 Movie Recommendation System

A content-based movie recommender built with Python, scikit-learn, and Streamlit — powered by the OMDB API for live posters and movie details.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🖥️ Demo

> Enter any movie → get 6 similar recommendations with posters, ratings, and plot summaries.

---

## 🧠 How It Works

### Content-Based Filtering
Instead of using other users' behavior, this system recommends movies based on **what a movie IS** — its genres, themes, and title keywords.

```
Movie Input: "Toy Story"
         ↓
   TF-IDF Vector
  [0.2, 0.0, 0.8, ...]   ← "Animation", "Comedy", "Children"
         ↓
  Cosine Similarity
  Compare to all 9,000+ movies
         ↓
  Top 6 closest vectors = Recommendations
```

### Algorithm Steps
1. **Feature Extraction** — genres are cleaned and converted to text features
2. **TF-IDF Vectorization** — each movie becomes a numeric vector (rare genres weighted higher)
3. **Cosine Similarity** — measures angle between vectors (1.0 = identical, 0.0 = opposite)
4. **Ranking** — sort all movies by similarity score, return top-N
5. **API Enrichment** — OMDB API fetches poster, IMDB rating, director, plot

---

## 📁 Project Structure

```
movie_recommender/
│
├── app.py               ← Streamlit web app (frontend + UI)
├── recommender.py       ← ML engine (TF-IDF + Cosine Similarity)
├── download_data.py     ← Auto-downloads MovieLens dataset
├── requirements.txt     ← Python dependencies
├── README.md            ← This file
│
└── data/                ← Created by download_data.py
    ├── movies.csv       ← 9,000+ movies (MovieLens)
    └── ratings.csv      ← 100,000+ user ratings
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
python download_data.py
```

### 4. Get a free OMDB API key
> Go to [omdbapi.com/apikey.aspx](https://www.omdbapi.com/apikey.aspx) → sign up free → check email for key

### 5. Run the app
```bash
streamlit run app.py
```

> App opens at **http://localhost:8501** 🎉

---

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core logic |
| ML | scikit-learn | TF-IDF + Cosine Similarity |
| Data | pandas, numpy | Data processing |
| API | OMDB API | Movie posters & details |
| Frontend | Streamlit | Interactive web UI |
| Dataset | MovieLens (GroupLens) | 9,000+ movies, 100k+ ratings |

---

## 📊 Dataset

This project uses the **MovieLens Small Dataset** by GroupLens Research:
- 9,742 movies
- 100,836 ratings
- 610 users
- Ratings from 1996–2018

> [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)

---

## 💡 Features

- 🔍 **Search** any of 9,000+ movies
- 🎯 **Content-based recommendations** using TF-IDF + Cosine Similarity
- 🖼️ **Live movie posters** via OMDB API
- ⭐ **IMDB ratings** and plot summaries
- 📊 **Dataset explorer** with genre charts
- 🌙 **Dark themed** modern UI

---

## 🔮 Future Improvements

- [ ] Add collaborative filtering (user-based)
- [ ] Hybrid model (content + collaborative)
- [ ] User login + personal watchlist
- [ ] Deploy to Streamlit Cloud (free hosting)
- [ ] Add more features: cast, director, keywords

---

## 📝 License

MIT License — free to use and modify.

---

## 👤 Author

Built as a portfolio project for job preparation.
