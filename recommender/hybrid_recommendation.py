from recommender.content_based import recommend_items
from recommender.collaborative_filtering import recommend_for_user
import pandas as pd

def hybrid_recommendations(train_data, target_user_id, item_name, 
                           cosine_sim_matrix, user_item_matrix, user_similarity, 
                           top_n=10, weight_content=0.5, weight_collab=0.5):
    """
    Hybrid recommendation combining content-based and collaborative filtering.
    """

    # Content-based recommendations
    content_based_rec = recommend_items(train_data, cosine_sim_matrix, item_name, top_n=top_n)
    content_based_rec["Score"] = content_based_rec["Rating"] * weight_content

    # Collaborative filtering recommendations
    collaborative_filtering_recs = recommend_for_user(target_user_id, train_data, user_item_matrix, user_similarity, top_n=top_n)
    collaborative_filtering_recs["Score"] = collaborative_filtering_recs["Rating"] * weight_collab

    # Merge recommendations
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_recs])

    # Group by ProdID and sum scores (in case of duplicates)
    hybrid_rec = hybrid_rec.groupby("ProdID", as_index=False).agg({"Score": "sum", "Rating": "mean"})

    # Sort by score
    hybrid_rec = hybrid_rec.sort_values("Score", ascending=False).head(top_n)

    return hybrid_rec
