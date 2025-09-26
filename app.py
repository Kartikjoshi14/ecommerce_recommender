from flask import Flask, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from recommender.data_preprocessing import load_and_clean_data
from recommender.feature_engineering import add_tags
from recommender.content_based import build_similarity_matrix
from recommender.collaborative_filtering import build_user_item_matrix, compute_user_similarity
from recommender.hybrid_recommendation import hybrid_recommendations

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
    password = db.Column(db.String(100), nullable=False)

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

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username")
    email = request.form.get("email")
    password = request.form.get("password")

    if not username or not email or not password:
        flash("Please fill all fields")
        return redirect(url_for("index"))

    new_user = Signup(username=username, email=email, password=password)
    db.session.add(new_user)
    db.session.commit()
    session['user'] = username
    flash(f"Welcome {username}!")
    return redirect(url_for("main"))

@app.route("/signin", methods=["POST"])
def signin():
    username = request.form.get("signinUsername")
    password = request.form.get("signinPassword")
    user = Signup.query.filter_by(username=username, password=password).first()
    if user:
        session['user'] = username
        flash("Sign in successful!")
        return redirect(url_for("main"))
    else:
        flash("Invalid username or password")
        return redirect(url_for("index"))
    
@app.route("/add_to_cart", methods=["POST"])
def add_to_cart():
    if 'user' not in session:
        return {"status": "error", "message": "Please sign in to add items to cart."}, 401

    username = session['user']
    product_id = request.form.get("product_id")
    product_name = request.form.get("product_name")
    price = float(request.form.get("Product Price") or 0.0)

    # Check if already in cart
    existing_item = Cart.query.filter_by(username=username, product_id=product_id).first()
    if existing_item:
        existing_item.quantity += 1
        message = f"Added another {product_name} to cart!"
    else:
        new_item = Cart(username=username, product_name=product_name, product_id=product_id, price=price)
        db.session.add(new_item)
        message = f"{product_name} added to cart!"

    db.session.commit()
    return {"status": "success", "message": message}


@app.route("/cart")
def cart():
    cart_items = session.get('cart', [])
    return render_template("cart.html", cart_items=cart_items)

if __name__ == "__main__":
    app.run(debug=True)
