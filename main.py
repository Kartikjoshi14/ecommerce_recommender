import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv('D:\python workshop\pythonProject\Ecommerce_recommender\marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.csv')

important_columns = [
    'Uniq Id','Product Id', 'Product Name', 'Product Brand',
    'Product Category', 'Product Description',
    'Product Price', 'Product Rating', 'Product Reviews Count',
    # optional columns
    'Product Tags', 'Product Available Inventory',
    'Product Image Url', 'Product Url','Product Contents'
]
train_data = train_data[important_columns]

# Text columns → empty string
text_cols = ['Product Name', 'Product Brand', 'Product Category', 'Product Description', 'Product Tags', 'Product Image Url', 'Product Url','Product Contents']
for col in text_cols:
    train_data[col] = train_data[col].fillna("")

# Numeric columns → median (ratings, reviews count, price)
num_cols = ['Product Price', 'Product Rating', 'Product Reviews Count', 'Product Available Inventory']
for col in num_cols:
    train_data[col] = train_data[col].fillna(train_data[col].median())

column_name_mapping  = {
    'Uniq Id' : 'ID',
    'Product Id' : 'ProdID',
    'Product Rating' : 'Rating',
    'Product Reviews Count': 'ReviewsCount',
    'Product Category' : 'Category',
    'Product Brand' : 'Brand',
    'Product Name' : 'Name',
    'Product Image Url': 'ImageURL',
    'Product Description' : 'Description',
    'Product Tags':'Tags',
    'Product Available Inventory' : 'AvailableStock',
    'Product Contents':'Contents'  
}
train_data.rename(columns=column_name_mapping, inplace=True)

train_data['ID'] = train_data['ID'].str.extract(r'(\d+)').astype(float)
train_data['ProdID'] = train_data['ProdID'].str.extract(r'(\d+)').astype(float)
#print(train_data)

#basic statistics
num_users = train_data['ID'].nunique()
num_items = train_data['ProdID'].nunique()
num_ratings = train_data['Rating'].nunique()
#print(f"Number of unique users:{num_users}")
#print(f"Number of unique items:{num_items}")
#print(f"Number of unique ratings:{num_ratings}")

#Pivot the dataframe to create a heatmap
#heatmap_data = train_data.pivot_table('ID','Rating')
#create the heatmap
#plt.figure(figsize=(8,6))
#sns.heatmap(heatmap_data,annot=True,fmt='g',cmap='coolwarm',cbar=True)
#plt.xlabel('Ratings')
#plt.ylabel('User ID')
#plt.show()

nlp = spacy.load("en_core_web_sm")
def process_texts(texts):
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=50, disable=["ner", "parser"]):  
        tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts

# Columns to process
columns_to_extract_tags_from = ['Category', 'Brand', 'Description']

# Apply nlp.pipe efficiently on each column
for column in columns_to_extract_tags_from:
    texts = train_data[column].fillna("").astype(str).tolist()
    train_data[column] = process_texts(texts)

# Create Tags column by merging all processed text
train_data['Tags'] = train_data[columns_to_extract_tags_from].apply(lambda row: ', '.join(row), axis=1)

# Show first few rows
print(train_data[['Category', 'Brand', 'Description', 'Tags']].head(10))