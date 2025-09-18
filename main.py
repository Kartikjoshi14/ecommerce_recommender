from recommender.data_preprocessing import load_and_clean_data
from recommender.feature_engineering import add_tags

DATA_PATH = "D:\python workshop\pythonProject\Ecommerce_recommender\data\marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.csv"

if __name__ == "__main__":
    # Step 1: Load + Clean
    train_data = load_and_clean_data(DATA_PATH)

    # Step 2: Add Tags
    train_data = add_tags(train_data)

    # Step 3: Show sample
    print(train_data[['Category', 'Brand', 'Description', 'Tags']].head(10))
