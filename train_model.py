import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
books = pd.read_csv("data/books.csv")
ratings = pd.read_csv("data/ratings.csv")

# Content-Based TF-IDF
books['combined'] = books['title'].fillna('') + ' ' + books['authors'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['combined'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(books.index, index=books['title'].str.lower())

# Collaborative Filtering
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)
trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
svd = SVD(n_factors=100, reg_all=0.1)
svd.fit(trainset)

# Save models
joblib.dump(svd, 'models/svd_model.pkl')
joblib.dump(tfidf_matrix, 'models/tfidf_matrix.pkl')
joblib.dump(indices, 'models/indices.pkl')
joblib.dump(books, 'models/books_df.pkl')
