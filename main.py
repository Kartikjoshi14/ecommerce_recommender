import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv('D:\python workshop\pythonProject\Ecommerce_recommender\marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.csv')
print(train_data.shape)