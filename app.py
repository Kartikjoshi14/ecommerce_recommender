from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from recommender.data_preprocessing import load_and_clean_data
from recommender.feature_engineering import add_tags
from recommender.content_based import build_similarity_matrix
from recommender.collaborative_filtering import build_user_item_matrix, compute_user_similarity
from recommender.hybrid_recommendation import hybrid_recommendations
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
    nbr_str = request.form.get("nbr", "5")
    try:
        nbr = int(float(nbr_str))
    except:
        nbr = 5
    user_id_str = request.form.get("user_id", "95")
    try:
        user_id = int(float(user_id_str))
    except:
        user_id = 95

    recs = hybrid_recommendations(
        train_data=train_data,
        target_user_id=user_id,
        item_name=prod,
        cosine_sim_matrix=cosine_sim_matrix,
        user_item_matrix=user_item_matrix,
        user_similarity=user_similarity,
        top_n=nbr
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

    return render_template("main.html", content_based_rec=recs, truncate=truncate, message=None)

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
    cursor = conn.cursor(dictionary=True)
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
    return render_template("cart.html", cart_items=cart_items)

if __name__ == "__main__":
    app.run(debug=True)
