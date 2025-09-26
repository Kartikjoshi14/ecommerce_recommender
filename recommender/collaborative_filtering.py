import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(train_data):
    """
    Build user-item matrix (users x products).
    """
    # Safer: cast IDs to string to avoid float/scientific notation
    user_item_matrix = (
        train_data
        .pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean')
        .fillna(0)
    )
    return user_item_matrix

def compute_user_similarity(user_item_matrix):
    """
    Compute cosine similarity between users.
    Returns a DataFrame with user-user similarities.
    """
    cosine_sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(cosine_sim,
                        index=user_item_matrix.index,
                        columns=user_item_matrix.index)

def recommend_for_user(user_id, train_data, user_item_matrix, user_similarity, top_n=10):
    """
    Recommend items for a given user based on user-user collaborative filtering.
    - Finds most similar users
    - Aggregates their ratings
    - Excludes items already rated by the target user
    """
    user_id = float(user_id)
    if user_id not in user_item_matrix.index:
        raise ValueError(f"User {user_id} not found in dataset")

    # Similarity scores for target user
    sim_scores = user_similarity.loc[user_id].drop(user_id)  # drop self
    similar_users = sim_scores.sort_values(ascending=False)

    # Weighted sum of ratings from similar users
    weighted_scores = (
        user_item_matrix.loc[similar_users.index]
        .T.dot(similar_users)
        / similar_users.sum()
    )

    # Remove products the user already rated
    already_rated = user_item_matrix.loc[user_id]
    recommendations = weighted_scores[already_rated == 0].sort_values(ascending=False)

    # Get top-N product IDs
    recommended_ids = recommendations.head(top_n).index.tolist()

    # Fetch product details
    item_details = (
        train_data[train_data['ProdID'].isin(recommended_ids)]
        .drop_duplicates('ProdID')[['ProdID', 'Name', 'Brand', 'ReviewsCount', 'ImageURL', 'Rating']]
    )

    return item_details.reset_index(drop=True)