#Rating based recommendation system 
import pandas as pd

def get_top_rated_products(train_data, top_n=10, min_reviews=10):
    average_ratings = train_data.groupby(['Name','ReviewsCount','Brand','ImageURL'])['Rating'].mean().reset_index()
    top_rated_items = average_ratings.sort_values(by='Rating',ascending = False)
    rating_based_recommendation = top_rated_items.head(10)
    rating_based_recommendation = rating_based_recommendation.copy()
    rating_based_recommendation['Rating'] = rating_based_recommendation['Rating'].astype(int)
    rating_based_recommendation['ReviewsCount'] = rating_based_recommendation['ReviewsCount'].astype(int)
    rating_based_recommendation[['Name','Rating','ReviewsCount','Brand','ImageURL']] = rating_based_recommendation[['Name','Rating','ReviewsCount','Brand','ImageURL']]
    return rating_based_recommendation