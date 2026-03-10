"""
download_data.py
────────────────
Downloads the MovieLens small dataset automatically.
Run this ONCE before starting the app:  python download_data.py
"""

import urllib.request
import zipfile
import os
import shutil

DATA_DIR = "data"
URL      = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ZIP_PATH = "ml-latest-small.zip"

def download():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("📥 Downloading MovieLens dataset (~3 MB)...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("✅ Downloaded!")

    print("📦 Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(".")

    # Move the files we need into /data
    shutil.copy("ml-latest-small/movies.csv",  f"{DATA_DIR}/movies.csv")
    shutil.copy("ml-latest-small/ratings.csv", f"{DATA_DIR}/ratings.csv")

    # Clean up
    os.remove(ZIP_PATH)
    shutil.rmtree("ml-latest-small")

    print(f"✅ Data ready in /{DATA_DIR}/")
    print("   → data/movies.csv")
    print("   → data/ratings.csv")
    print()
    print("🚀 Now run:  streamlit run app.py")

if __name__ == "__main__":
    download()
