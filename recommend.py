import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load models
svd = joblib.load("models/svd_model.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
indices = joblib.load("models/indices.pkl")
books = joblib.load("models/books_df.pkl")

def hybrid_recommendation(user_id, title, top_n=10):
    idx = indices.get(title.lower())
    if idx is None:
        return pd.DataFrame()
    
    # Content-based part
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:50]
    book_indices = [i[0] for i in sim_scores]

    book_candidates = books.iloc[book_indices].copy()
    book_candidates['est_rating'] = book_candidates['book_id'].apply(
        lambda x: svd.predict(user_id, x).est
    )
    return book_candidates.sort_values('est_rating', ascending=False).head(top_n)

