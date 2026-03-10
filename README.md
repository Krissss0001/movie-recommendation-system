# 🎬 Movie Recommendation System

## Project Overview

This project is a **simple Netflix-style movie recommendation system** built using Python.

The system recommends movies based on **similarity between movies** using machine learning techniques.

When a user enters a movie name, the system finds **similar movies** and recommends them.

---

## Features

* Movie recommendation based on similarity
* Uses machine learning algorithms
* Simple and easy interface
* Built with Python

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit (for web interface)

---

## How It Works

1. Load movie dataset (movies and ratings)
2. Create a **movie-user matrix**
3. Calculate **similarity between movies**
4. When a movie is selected, the system finds the most similar movies
5. Show recommended movies to the user

---

## Example

Input movie:

Inception

Recommended movies:

* Interstellar
* The Dark Knight
* Tenet
* The Matrix

---

## Project Structure

```
movie-recommendation-system
│
├── app.py
├── recommender.py
├── movies.csv
├── ratings.csv
├── requirements.txt
└── README.md
```

---

## How to Run the Project

Install required libraries:

```
pip install pandas numpy scikit-learn streamlit
```

Run the application:

```
streamlit run app.py
```

Then open the browser and enter a movie name to get recommendations.

---

## Future Improvements

* Add better recommendation algorithms
* Improve UI design
* Add more movie datasets
* Deploy the project online

---

## Author

Kris
