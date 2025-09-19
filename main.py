from recommender.data_preprocessing import load_and_clean_data
from recommender.feature_engineering import add_tags
from recommender.rating_based import get_top_rated_products
from recommender.content_based import build_similarity_matrix,recommend_items
from recommender.collaborative_filtering import build_user_item_matrix, compute_user_similarity, recommend_for_user
from recommender.hybrid_recommendation import hybrid_recommendations


DATA_PATH = "D:\python workshop\pythonProject\Ecommerce_recommender\data\marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.csv"

if __name__ == "__main__":
    # Step 1: Load + Clean
    train_data = load_and_clean_data(DATA_PATH)

    # Step 2: Add Tags
    train_data = add_tags(train_data)

    # Step 3: Show sample
    #print(train_data[['Category', 'Brand', 'Description', 'Tags']].head(10))

    #get top -rated products
    top_products = get_top_rated_products(train_data,top_n=10,min_reviews = 20)
    print("\n Top Rated Products")
    print(top_products)

    # Build content-based similarity matrix
    cosine_sim_matrix = build_similarity_matrix(train_data)
    #print("Cosine similarity matrix shape:", cosine_sim_matrix)

    # Pick any product name from your dataset
        # item_name = input("Enter a product name: ")
        # top_n = int(input("How many recommendations do you want? "))
        # recommendations = recommend_items(train_data, cosine_sim_matrix, item_name, top_n)
        # print(train_data['Name'].head(10))
        # print("\nRecommendations:\n", recommendations)

    # Build collaborative filtering matrices
    print(train_data['ID'].unique()[:10])
    user_item_matrix = build_user_item_matrix(train_data)   
    user_similarity = compute_user_similarity(user_item_matrix)
    user_id = train_data['ID'].iloc[0]  # first user in the dataset
    item_name = train_data['Name'].iloc[0]

    print("Using user ID:", user_id)

    # Call your recommendation function
    recommendations = hybrid_recommendations(
    train_data=train_data,
    target_user_id=user_id,
    item_name=item_name,
    cosine_sim_matrix=cosine_sim_matrix,
    user_item_matrix=user_item_matrix,
    user_similarity=user_similarity,
    top_n=10
    )

    print(f"\nHybrid Recommendations for user {int(user_id)} based on '{item_name}':")
    print(recommendations)