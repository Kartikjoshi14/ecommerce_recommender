from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def build_similarity_matrix(train_data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def recommend_items(train_data, cosine_sim_matrix, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        raise ValueError(f"Item '{item_name}' not found in dataset.")

    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_sim_matrix[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[1:top_n+1]
    recommended_indices = [i[0] for i in similar_items]

    return train_data.iloc[recommended_indices][['Name', 'Brand', 'Rating']].assign(
        Similarity=[i[1] for i in similar_items]
    )
