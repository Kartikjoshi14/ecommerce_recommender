import pandas as pd

def load_and_clean_data(path):
    important_columns = [
        'Uniq Id','Product Id', 'Product Name', 'Product Brand',
        'Product Category', 'Product Description',
        'Product Price', 'Product Rating', 'Product Reviews Count',
        'Product Tags', 'Product Available Inventory',
        'Product Image Url', 'Product Url','Product Contents'
    ]

    df = pd.read_csv("D:\python workshop\pythonProject\Ecommerce_recommender\data\marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.csv")
    df = df[important_columns]

    # Fill text columns
    text_cols = ['Product Name', 'Product Brand', 'Product Category',
                 'Product Description', 'Product Tags',
                 'Product Image Url', 'Product Url','Product Contents']
    for col in text_cols:
        df[col] = df[col].fillna("")

    # Fill numeric columns
    num_cols = ['Product Price', 'Product Rating',
                'Product Reviews Count', 'Product Available Inventory']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Rename columns
    column_name_mapping = {
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
    df.rename(columns=column_name_mapping, inplace=True)

    # Convert IDs
    df['ID'] = df['ID'].astype(str).str.extract(r'(\d+)').astype(float)
    df['ProdID'] = df['ProdID'].astype(str).str.extract(r'(\d+)').astype(float)

    return df