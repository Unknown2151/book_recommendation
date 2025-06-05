import streamlit as st
import pandas as pd
import joblib
from recommend import hybrid_recommendation

st.set_page_config(page_title="📚 Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")

# Load data
books = joblib.load("models/books_df.pkl")
titles = books['title'].dropna().unique()

# Sidebar
st.sidebar.header("User Input")
user_id = st.sidebar.number_input("User ID", min_value=1, step=1)
book_name = st.sidebar.selectbox("Choose a Book", sorted(titles))

if st.sidebar.button("Recommend"):
    with st.spinner("Finding recommendations..."):
        results = hybrid_recommendation(user_id, book_name)
    
    if results.empty:
        st.warning("No recommendations found. Try another book.")
    else:
        st.success("Here are your recommendations:")
        for _, row in results.iterrows():
            st.markdown(f"### {row['title']}")
            st.markdown(f"**Author:** {row['authors']}  \n**Rating (predicted):** {row['est_rating']:.2f}")
