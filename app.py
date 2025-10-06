from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from recommender.data_preprocessing import load_and_clean_data
from recommender.feature_engineering import add_tags
from recommender.content_based import build_similarity_matrix
from recommender.collaborative_filtering import build_user_item_matrix, compute_user_similarity
from recommender.hybrid_recommendation import hybrid_recommendations
from recommender.diversify import diversify_recommendations
import mysql.connector


app = Flask(__name__)
app.secret_key = "alskdjfwoeieiurlskdjfslkdjf"

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:Kartik*14@localhost/recommenderdb"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# DB models
class Signup(db.Model):
    __tablename__ = "signup"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(255), nullable=False)  # increased length


class Cart(db.Model):
    __tablename__ = "cart"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    product_name = db.Column(db.String(200), nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    price = db.Column(db.Float, nullable=False)
    quantity = db.Column(db.Integer, default=1)

# === Application context for DB and global preprocessing ===
with app.app_context():
    # Create tables
    db.create_all()

    # Load and prepare data
    DATA_PATH = "data/marketing_sample_for_walmart_com-walmart_com_product_review__20200701_20201231__5k_data.csv"
    train_data = load_and_clean_data(DATA_PATH)
    train_data = add_tags(train_data)

    # Ensure Price column exists
    if 'Price' not in train_data.columns:
        train_data['Price'] = train_data['Rating'].apply(lambda x: round(float(x) * 20, 2))

    # Ensure ImageURL exists
    if 'ImageURL' not in train_data.columns:
        train_data['ImageURL'] = "static/img/img_1.png"

    # Build lookups and matrices
    cosine_sim_matrix = build_similarity_matrix(train_data)
    user_item_matrix = build_user_item_matrix(train_data)
    user_similarity = compute_user_similarity(user_item_matrix)

    # Build lookup dicts
    user_lookup = {uid: idx for idx, uid in enumerate(train_data['ID'].unique())}
    product_lookup = {pid: idx for idx, pid in enumerate(train_data['ProdID'].unique())}

# Helper to truncate text
def truncate(text, length):
    return text[:length] + "..." if len(text) > length else text

# ===== ROUTES =====
@app.route("/")
def index():
    trending_products = train_data.head(8)
    return render_template("index.html",
                           trending_products=trending_products,
                           truncate=truncate)

@app.route("/main")
def main():
    return render_template("main.html", content_based_rec=pd.DataFrame(), message=None)

@app.route("/recommendations", methods=["POST"])
def recommendations():
    prod = request.form.get("prod")
    nbr_str = request.form.get("nbr", "8")
    try:
        nbr = int(float(nbr_str))
    except:
        nbr = 8
    user_id_str = request.form.get("user_id", "95")
    try:
        user_id = int(float(user_id_str))
    except:
        user_id = 95

    # Get hybrid recommendations (content + collaborative)
    recs = hybrid_recommendations(
        train_data=train_data,
        target_user_id=user_id,
        item_name=prod,
        cosine_sim_matrix=cosine_sim_matrix,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        top_n=50  # Get a large candidate pool for post-filtering
    )

    if recs is None or recs.empty:
        return render_template("main.html", content_based_rec=pd.DataFrame(),
                               message="No recommendations found.")

    # Fill missing images if needed
    if 'ImageURL' not in recs.columns:
        recs['ImageURL'] = ["static/img/img_1.png"] * len(recs)

    # Ensure Price column
    if 'Price' not in recs.columns:
        recs['Price'] = recs['Rating'].apply(lambda x: round(float(x) * 20, 2))

    # Enrich recommendations with all product details
    enriched = []
    for _, row in recs.iterrows():
        prod_id = row['ProdID']
        product_info = train_data[train_data['ProdID'] == prod_id]
        if not product_info.empty:
            info = product_info.iloc[0]
            enriched.append({
                "ProdID": prod_id,
                "Name": info.get("Name", "Unknown"),
                "ImageURL": info.get("ImageURL", "static/img/img_1.png"),
                "Price": info.get("Product Price", info.get("Price", row.get("Price", 0.0))),
                "Rating": row.get("Rating", 0.0),
                "Description": info.get("Review Text", ""),
                "Brand": info.get("Brand", ""),
                "Category": info.get("Category", info.get("Tag", "")),
                "Tag": info.get("Tag", ""),
                "Popularity": info.get("Review Count", 0),
            })
        else:
            enriched.append({
                "ProdID": prod_id,
                "Name": str(prod_id),
                "ImageURL": "static/img/img_1.png",
                "Price": row.get("Price", row.get("Rating", 0.0) * 20),
                "Rating": row.get("Rating", 0.0),
                "Description": "",
                "Brand": "",
                "Category": "",
                "Tag": "",
                "Popularity": 0,
            })

    # Diversify and post-filter recommendations using new utility
    final_recs = diversify_recommendations(enriched, nbr)

    recs_df = pd.DataFrame(final_recs)

    return render_template("main.html", content_based_rec=recs_df, truncate=truncate, message=None)

# --- Signup Route ---
@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')

    if not username or not email or not password:
        return jsonify({"status": "error", "message": "Please fill all fields"}), 400

    hashed_password = generate_password_hash(password)

    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="kartikuser",
            password="Kartik*14",
            database="recommenderdb"
        )
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO signup (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )
        conn.commit()
    except mysql.connector.Error as e:
        return jsonify({"status": "error", "message": f"Error creating account: {str(e)}"}), 400
    finally:
        cursor.close()
        conn.close()

    return jsonify({"status": "success", "message": "Account created successfully! Please log in."})

# --- Signin Route ---
@app.route('/signin', methods=['POST'])
def signin():
    identifier = request.form.get('username_or_email')
    password = request.form.get('password')

    if not identifier or not password:
        return jsonify({"status": "error", "message": "Please fill all fields"}), 400

    conn = mysql.connector.connect(
        host="localhost",
        user="kartikuser",
        password="Kartik*14",
        database="recommenderdb"
    )
    cursor = conn.cursor(dictionary=True,buffered=True)
    cursor.execute(
        "SELECT * FROM signup WHERE username = %s OR email = %s",
        (identifier, identifier)
    )
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if user and check_password_hash(user['password'], password):
        session['user'] = user['username']
        return jsonify({"status": "success", "message": f"Welcome back, {user['username']}!"})
    else:
        return jsonify({"status": "error", "message": "Invalid username/email or password"}), 401
    

@app.route("/signup_form")
def signup_form():
    return render_template("register.html")

@app.route("/cart")
def cart():
    cart_items = session.get('cart', [])
    recommendations = []
    recommended_ids = set()
    tag_to_products = {}

    if cart_items:
        cart_product_names = [item['product_name'] for item in cart_items]
        for prod in cart_product_names:
            recs = hybrid_recommendations(
                train_data=train_data,
                target_user_id=95,
                item_name=prod,
                cosine_sim_matrix=cosine_sim_matrix,
                user_item_matrix=user_item_matrix,
                user_similarity=user_similarity,
                top_n=20
            )
            if recs is not None and not recs.empty:
                recs_sorted = recs.sort_values(by="Rating", ascending=False)
                for _, row in recs_sorted.iterrows():
                    prod_id = row['ProdID']
                    if prod_id in recommended_ids:
                        continue
                    product_info = train_data[train_data['ProdID'] == prod_id]
                    if not product_info.empty:
                        product_info = product_info.iloc[0]
                        tag = product_info.get("Tag") or product_info.get("Category") or "Other"
                        # Collect all relevant details for the recommendation
                        details = {
                            "Name": product_info.get("Name", "Unknown"),
                            "ImageURL": product_info.get("ImageURL", "static/img/img_1.png"),
                            "Product_Price": product_info.get("Product Price", product_info.get("Price", 0.0)),
                            "ProdID": prod_id,
                            "Tag": tag,
                            "Rating": row.get("Rating", 0.0),
                            "Description": product_info.get("Review Text", ""),  # Add description if available
                            "Brand": product_info.get("Brand", ""),              # Add brand if available
                            "Category": product_info.get("Category", tag)        # Add category if available
                        }
                        if tag not in tag_to_products:
                            tag_to_products[tag] = []
                        tag_to_products[tag].append(details)
                        recommended_ids.add(prod_id)
                    else:
                        tag = "Other"
                        details = {
                            "Name": str(prod_id),
                            "ImageURL": "static/img/img_1.png",
                            "Product_Price": row.get("Rating", 0.0) * 20,
                            "ProdID": prod_id,
                            "Tag": tag,
                            "Rating": row.get("Rating", 0.0),
                            "Description": "",
                            "Brand": "",
                            "Category": tag
                        }
                        if tag not in tag_to_products:
                            tag_to_products[tag] = []
                        tag_to_products[tag].append(details)
                        recommended_ids.add(prod_id)
        # Diversify: round-robin pick from each tag, prioritize tags with higher ratings
        max_recs = 8
        tags_sorted = sorted(tag_to_products.keys(), key=lambda t: max([p["Rating"] for p in tag_to_products[t]]), reverse=True)
        tag_indices = {tag: 0 for tag in tags_sorted}
        while len(recommendations) < max_recs:
            added = False
            for tag in tags_sorted:
                idx = tag_indices[tag]
                products = tag_to_products[tag]
                if idx < len(products):
                    recommendations.append(products[idx])
                    tag_indices[tag] += 1
                    added = True
                    if len(recommendations) >= max_recs:
                        break
            if not added:
                break

    return render_template(
        "cart.html",
        cart_items=cart_items,
        recommendations=recommendations,
        name_key="Name",
        image_key="ImageURL",
        price_key="Product_Price",
        desc_key="Description",
        brand_key="Brand",
        category_key="Category"
    )

@app.route("/add_to_cart", methods=["POST"])
def add_to_cart():
    product_id = request.form.get("product_id")
    product_name = request.form.get("product_name")
    product_price = request.form.get("product_price")

    if not product_id or not product_name or not product_price:
        return jsonify({"status": "error", "message": "Missing product info"}), 400

    cart = session.get('cart', [])

    # Check if product already in cart, increment quantity if so
    for item in cart:
        if item['product_id'] == product_id:
            item['quantity'] += 1
            break
    else:
        cart.append({
            "product_id": product_id,
            "product_name": product_name,
            "product_price": float(product_price),
            "quantity": 1
        })

    session['cart'] = cart
    session.modified = True  # Ensure session is saved

    return jsonify({"status": "success", "message": "Added to cart"})

@app.route("/remove_from_cart", methods=["POST"])
def remove_from_cart():
    product_id = request.form.get("product_id")
    cart = session.get('cart', [])
    cart = [item for item in cart if item['product_id'] != product_id]
    session['cart'] = cart
    session.modified = True
    return jsonify({"status": "success", "message": "Removed from cart"})

if __name__ == "__main__":
    app.run(debug=True)
    



