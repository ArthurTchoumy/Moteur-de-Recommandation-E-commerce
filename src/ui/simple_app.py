"""
Simple Streamlit App for Testing
Basic interface without complex dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import os
import json
from PIL import Image

# ---------------------------------------------------------------------------
# User data persistence (JSON file per user)
# ---------------------------------------------------------------------------
USER_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'users')
os.makedirs(USER_DATA_DIR, exist_ok=True)


def _user_file(username: str) -> str:
    """Return path to JSON file for a given user."""
    safe = username.replace('/', '_').replace('\\', '_').replace('..', '_')
    return os.path.join(USER_DATA_DIR, f"{safe}.json")


def save_user_data(username: str = None):
    """Persist current session data for the user."""
    uid = username or st.session_state.get('user_id')
    if not uid:
        return
    data = {
        'user_preferences': st.session_state.get('user_preferences', {}),
        'wishlist': st.session_state.get('wishlist', []),
        'purchase_history': st.session_state.get('purchase_history', []),
        'cart': st.session_state.get('cart', []),
        'rec_seed': st.session_state.get('rec_seed', 42),
    }
    try:
        with open(_user_file(uid), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception:
        pass


def load_user_data(username: str):
    """Restore session data for a returning user."""
    path = _user_file(username)
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        st.session_state.user_preferences = data.get('user_preferences', {})
        st.session_state.wishlist = data.get('wishlist', [])
        st.session_state.purchase_history = data.get('purchase_history', [])
        st.session_state.cart = data.get('cart', [])
        st.session_state.rec_seed = data.get('rec_seed', 42)
    except Exception:
        pass

# Page configuration
st.set_page_config(
    page_title="E-commerce Recommendations",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Load real Amazon datasets
def load_real_amazon_data():
    """Load real Amazon product and review data"""
    # Check if data is already in session state
    if 'cached_items_df' in st.session_state and 'cached_interactions_df' in st.session_state:
        return st.session_state.cached_items_df, st.session_state.cached_interactions_df
    
    # Try to load preprocessed data
    items_path = "data/real_items.parquet"
    interactions_path = "data/real_interactions.parquet"
    
    if not os.path.exists(items_path):
        st.error("❌ Données prétraitées non trouvées!")
        st.info("💡 Veuillez d'abord exécuter: `python load_real_amazon_data.py`")
        return pd.DataFrame(), pd.DataFrame()
    
    try:
        # Load preprocessed data directly
        items_df = pd.read_parquet(items_path)
        interactions_df = pd.read_parquet(interactions_path)
        
        # Cache in session state
        st.session_state.cached_items_df = items_df
        st.session_state.cached_interactions_df = interactions_df
        
        return items_df, interactions_df
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'items_data' not in st.session_state:
    st.session_state.items_data = None
if 'interactions_data' not in st.session_state:
    st.session_state.interactions_data = None
if 'interactions_df' not in st.session_state:
    st.session_state.interactions_df = None
if 'users_data' not in st.session_state:
    st.session_state.users_data = None
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []
if 'button_counter' not in st.session_state:
    st.session_state.button_counter = 0
if 'rec_button_counter' not in st.session_state:
    st.session_state.rec_button_counter = 0
if 'search_button_counter' not in st.session_state:
    st.session_state.search_button_counter = 0
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'rec_seed' not in st.session_state:
    st.session_state.rec_seed = 42
if 'purchase_history' not in st.session_state:
    st.session_state.purchase_history = []

CATEGORY_ICONS = {
    "Video_Games": "🎮",
    "Digital_Music": "🎵",
    "Software": "💻",
    "Electronics": "📱",
    "Clothing": "👕",
    "Books": "📚",
    "Home": "🏠",
    "Garden": "🌱",
    "Sports": "⚽",
    "Beauty": "💄",
    "Appliances": "🔌",
    "Gift_Cards": "🎁",
    "Industrial": "🏭",
    "Magazine": "📰",
    "Pantry": "🥫",
}


def get_image_or_icon(image_urls, category):
    """Try to get a valid image URL, return (url, is_image) or (icon, False)."""
    if hasattr(image_urls, 'tolist'):
        image_urls = image_urls.tolist()
    if (isinstance(image_urls, (list, tuple))
            and len(image_urls) > 0
            and image_urls[0]
            and isinstance(image_urls[0], str)
            and image_urls[0].startswith(('http://', 'https://'))):
        return image_urls[0], True
    icon = CATEGORY_ICONS.get(category, '📦')
    return icon, False


def show_image_col(image_urls, category):
    """Render image or fallback icon in current column."""
    url_or_icon, is_image = get_image_or_icon(image_urls, category)
    if is_image:
        try:
            st.image(url_or_icon, width=100, use_column_width=False)
        except Exception:
            st.markdown(
                f"<div style='font-size:4rem;text-align:center;padding:20px;'>"
                f"{CATEGORY_ICONS.get(category, '📦')}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f"<div style='font-size:4rem;text-align:center;padding:20px;'>{url_or_icon}</div>",
            unsafe_allow_html=True,
        )


def login_page():
    """Display beautiful login/registration page"""
    # Custom CSS for beautiful design
    st.markdown("""
    <style>
    .login-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        margin: 2rem 0;
    }
    .auth-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .auth-subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .divider {
        text-align: center;
        margin: 1.5rem 0;
        position: relative;
    }
    .divider::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 1px;
        background: #ddd;
    }
    .divider span {
        background: white;
        padding: 0 1rem;
        position: relative;
        color: #999;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Main login container
    with st.container():
        st.markdown("""
        <div class="login-container">
            <div class="auth-header">🛍️ Smart Shopping</div>
            <div class="auth-subtitle">Discover products tailored just for you</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="auth-card">
                <h3 style="color: #333; margin-bottom: 1.5rem;">🔐 Welcome Back</h3>
            </div>
            """, unsafe_allow_html=True)

            username = st.text_input("👤 Username", key="login_username",
                                     help="Enter your username")
            password = st.text_input("🔒 Password", type="password", key="login_password",
                                     help="Enter your password")

            if st.button("🚀 Login", key="login_button", use_container_width=True):
                if username and password:
                    st.session_state.user_id = username
                    st.session_state.is_authenticated = True
                    # Restore saved data for this user
                    load_user_data(username)
                    # Clear cached recommendations so they regenerate for this user
                    st.session_state.pop('cached_recommendations', None)
                    st.session_state.pop('rec_params', None)
                    st.success(f"🎉 Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("⚠️ Please enter username and password")

        with col2:
            st.markdown("""
            <div class="auth-card">
                <h3 style="color: #333; margin-bottom: 1.5rem;">✨ Create Account</h3>
            </div>
            """, unsafe_allow_html=True)

            with st.form("registration_form"):
                new_username = st.text_input("👤 Choose Username", help="Pick a unique username")
                email = st.text_input("📧 Email Address", help="We'll never share your email")
                reg_password = st.text_input("🔒 Password", type="password",
                                             help="Choose a strong password")
                confirm_password = st.text_input("🔒 Confirm Password", type="password",
                                                 help="Re-enter your password")

                st.markdown('<div class="divider"><span>Optional</span></div>',
                            unsafe_allow_html=True)

                # Simple preferences
                st.markdown("**🎯 Quick Preferences** *(optional)*")
                categories = st.multiselect(
                    "🛍️ Favorite Categories",
                    [
                        "Video_Games",
                        "Digital_Music",
                        "Software",
                        "Appliances",
                        "Gift_Cards",
                        "Industrial_and_Scientific",
                        "Magazine_Subscriptions",
                        "Prime_Pantry",
                    ],
                    format_func=lambda x: {
                        "Video_Games": "🎮 Video Games",
                        "Digital_Music": "🎵 Digital Music",
                        "Software": "💻 Software",
                        "Appliances": "🔌 Appliances",
                        "Gift_Cards": "🎁 Gift Cards",
                        "Industrial_and_Scientific": "🏭 Industrial & Scientific",
                        "Magazine_Subscriptions": "📰 Magazine Subscriptions",
                        "Prime_Pantry": "🥫 Prime Pantry",
                    }.get(x, x),
                    help="Select categories you're interested in (leave empty for general recommendations)",
                )

                register_button = st.form_submit_button("🎉 Create Account",
                                                        use_container_width=True)

                if register_button:
                    if new_username and email and reg_password and reg_password == confirm_password:
                        # Save user preferences to disk (do NOT auto-login)
                        user_preferences = {
                            'username': new_username,
                            'email': email,
                            'favorite_categories': categories if categories else [],
                            'registration_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                        }
                        # Temporarily set state to persist, then clear
                        st.session_state.user_preferences = user_preferences
                        st.session_state.wishlist = []
                        st.session_state.purchase_history = []
                        st.session_state.cart = []
                        save_user_data(new_username)
                        # Reset — user must login manually
                        st.session_state.pop('user_preferences', None)
                        st.success(f"🎊 Account created successfully for **{new_username}**!")
                        if categories:
                            st.info("🎯 Your preferences have been saved.")
                        st.info("👈 Please log in with your username and password.")
                    else:
                        if not new_username or not email or not reg_password:
                            st.error("⚠️ Please fill username, email and password")
                        elif reg_password != confirm_password:
                            st.error("⚠️ Passwords do not match")


# ---------------------------------------------------------------------------
# Helper: sample products with image priority
# ---------------------------------------------------------------------------
def _sample_with_image_priority(df, n, seed=42):
    """Return up to *n* products from *df*, preferring rows with images."""
    has_img = df['valid_image_urls'].apply(
        lambda x: len(x) > 0 if hasattr(x, '__len__') else False
    )
    with_img = df[has_img]
    without_img = df[~has_img]

    if len(with_img) >= n:
        return with_img.sample(n, random_state=seed)

    need = n - len(with_img)
    parts = []
    if len(with_img) > 0:
        parts.append(with_img)
    if need > 0 and len(without_img) > 0:
        parts.append(without_img.sample(min(need, len(without_img)), random_state=seed))
    if parts:
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df.sample(min(n, len(df)), random_state=seed)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------
def main_dashboard():
    """Main dashboard with recommendations, search, and activity"""
    # Header bar with logout and cart
    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    with h1:
        st.markdown('<h1 class="main-header">🛍️ Smart Shopping</h1>', unsafe_allow_html=True)
    with h2:
        cart_count = len(st.session_state.cart)
        if st.button(f"🛒 Cart ({cart_count})", use_container_width=True):
            st.session_state.show_cart = not st.session_state.get('show_cart', False)
    with h3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.pop('cached_recommendations', None)
            st.session_state.pop('rec_params', None)
            st.session_state.rec_seed += 1
            st.rerun()
    with h4:
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            # Save user data before clearing session
            save_user_data()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown(f"Welcome back, **{st.session_state.user_id}**!")

    # ---------- Cart sidebar panel ----------
    if st.session_state.get('show_cart', False):
        st.markdown("---")
        st.subheader("🛒 Your Cart")
        if not st.session_state.cart:
            st.info("Your cart is empty.")
        else:
            total = 0.0
            for ci, c_item in enumerate(st.session_state.cart):
                cc1, cc2, cc3 = st.columns([3, 1, 1])
                with cc1:
                    st.markdown(f"**{c_item['title'][:50]}...**")
                with cc2:
                    st.markdown(f"${c_item.get('price', 0.0):.2f}")
                with cc3:
                    if st.button("❌", key=f"rm_cart_{ci}"):
                        st.session_state.cart.pop(ci)
                        st.rerun()
                total += c_item.get('price', 0.0)
            st.markdown(f"**Total: ${total:.2f}**")
            if st.button("✅ Checkout", use_container_width=True, type="primary"):
                # Move cart items to purchase history
                for c_item in st.session_state.cart:
                    c_item['purchase_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
                    st.session_state.purchase_history.append(c_item)
                st.session_state.cart = []
                # Invalidate recommendations so they update based on new purchases
                st.session_state.pop('cached_recommendations', None)
                st.session_state.pop('rec_params', None)
                save_user_data()
                st.success("🎉 Order placed successfully! Check your purchase history.")
        st.markdown("---")

    # Fixed settings for cleaner interface
    model_type = "hybrid"
    num_recs = 10
    include_explanations = False

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🏠 Recommendations", "🔍 Search Items", "� Purchase History", "❤️ Wishlist"]
    )

    # ------------------------------------------------------------------
    # TAB 1 – Recommendations
    # ------------------------------------------------------------------
    with tab1:
        # Load real Amazon data
        if st.session_state.items_data is None:
            with st.spinner("Loading Amazon product data..."):
                items_df, interactions_df = load_real_amazon_data()
                st.session_state.items_data = items_df
                st.session_state.interactions_df = interactions_df

        items_df = st.session_state.items_data
        interactions_df = st.session_state.interactions_df

        if items_df is None or items_df.empty:
            st.error("No product data available. Please check your data files.")
            st.info("Expected files in data/ directory: real_items.parquet, real_interactions.parquet")
            return

        # ---------------------------------------------------------------
        # Adaptive recommendation logic
        #   Priority: purchases > wishlist > registration prefs > general
        # ---------------------------------------------------------------
        available_categories = items_df['category'].unique().tolist()

        # 1) Gather categories from purchase history
        purchase_cats = list({
            it.get('category', '') for it in st.session_state.purchase_history
            if it.get('category', '') in available_categories
        })

        # 2) Gather categories from wishlist
        wishlist_cats = list({
            it.get('category', '') for it in st.session_state.wishlist
            if it.get('category', '') in available_categories
        })

        # 3) Registration preferences
        user_prefs = st.session_state.get('user_preferences', {})
        reg_cats_raw = user_prefs.get('favorite_categories', [])
        reg_cats = [c for c in reg_cats_raw if c in available_categories]

        # Decide which categories to use & source label
        if purchase_cats:
            target_cats = purchase_cats
            rec_source = "purchase history"
            # Merge wishlist cats that aren't already present
            for wc in wishlist_cats:
                if wc not in target_cats:
                    target_cats.append(wc)
        elif wishlist_cats:
            target_cats = wishlist_cats
            rec_source = "wishlist"
        elif reg_cats:
            target_cats = reg_cats
            rec_source = "registration preferences"
        else:
            target_cats = []
            rec_source = "general"

        has_targeted = len(target_cats) > 0

        st.subheader(f"🎁 Top {num_recs} Recommendations for You")
        if has_targeted:
            st.info(f"🎯 Based on your **{rec_source}**: {', '.join(target_cats)}")
        else:
            st.info("💡 Add products to your wishlist or make purchases to get personalized recommendations!")

        # Generate or use cached recommendations
        if 'cached_recommendations' not in st.session_state:
            recommendations = []

            if not items_df.empty:
                current_seed = st.session_state.rec_seed
                if has_targeted:
                    filtered = items_df[items_df['category'].isin(target_cats)]
                    if filtered.empty:
                        filtered = items_df
                    sample_products = _sample_with_image_priority(filtered, num_recs, seed=current_seed)
                else:
                    sample_products = _sample_with_image_priority(items_df, num_recs, seed=current_seed)

                for idx, (_, product) in enumerate(sample_products.iterrows()):
                    product_id = product.get('asin', product.get('item_id', ''))
                    product_interactions = (
                        interactions_df[interactions_df['asin'] == product_id]
                        if interactions_df is not None and not interactions_df.empty
                        else pd.DataFrame()
                    )

                    if not product_interactions.empty:
                        avg_rating = product_interactions['overall'].mean() if 'overall' in product_interactions.columns else 4.0
                        review_count = len(product_interactions)
                    else:
                        avg_rating = 4.0
                        review_count = 0

                    title = product['title']
                    price = product.get('price', 0.0)
                    if price == 0.0:
                        price = np.random.uniform(10, 100)

                    item = {
                        "item_id": product_id,
                        "asin": product_id,
                        "title": str(title),
                        "category": product['category'],
                        "brand": product.get('brand', 'Unknown Brand'),
                        "price": price,
                        "rating": avg_rating,
                        "review_count": review_count,
                        "score": np.random.uniform(0.7, 0.95),
                        "explanation": f"Recommended based on your interests in {product['category']}",
                        "valid_image_urls": product.get('valid_image_urls', []),
                    }
                    recommendations.append(item)

            st.session_state.cached_recommendations = recommendations
            st.session_state.rec_params = num_recs
        else:
            recommendations = st.session_state.cached_recommendations

        # Display recommendations
        for idx, item in enumerate(recommendations):
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])

                with col1:
                    show_image_col(item.get('valid_image_urls', []), item.get('category', ''))

                with col2:
                    st.markdown(f"### {item['title']}")
                    stars = "⭐" * int(item['rating'])
                    st.markdown(f"{stars} {item['rating']:.1f}/5.0 ({item['review_count']} reviews)")
                    st.markdown(f"**Price:** ${item['price']:.2f}")
                    st.markdown(f"**Category:** {item['category']}")
                    st.markdown(f"**Brand:** {item['brand']}")
                    if include_explanations:
                        st.markdown(
                            f"<p style='font-style:italic;color:#666;font-size:0.9rem;'>{item['explanation']}</p>",
                            unsafe_allow_html=True,
                        )

                with col3:
                    st.markdown(
                        f"<span class='score-badge'>Score: {item['score']:.3f}</span>",
                        unsafe_allow_html=True,
                    )
                    if st.button("🛒 Add to Cart", key=f"cart_{idx}_{item['item_id']}"):
                        is_in_cart = any(c.get('asin', '') == item.get('asin', '') for c in st.session_state.cart)
                        if not is_in_cart:
                            st.session_state.cart.append(item)
                            save_user_data()
                            st.success(f"Added to cart!")
                        else:
                            st.info("Already in cart!")

                    if st.button("❤️ Save", key=f"save_rec_{idx}_{item.get('asin', 'unknown')}"):
                        product_id = item.get('asin', '')
                        is_dup = any(
                            (product_id and w.get('asin', '') == product_id)
                            or (not product_id and w.get('title', '') == item.get('title', '')
                                and abs(w.get('price', 0) - item.get('price', 0)) < 0.01)
                            for w in st.session_state.wishlist
                        )
                        if not is_dup:
                            st.session_state.wishlist.append(item)
                            # Invalidate recommendations so they adapt to new wishlist
                            st.session_state.pop('cached_recommendations', None)
                            st.session_state.pop('rec_params', None)
                            save_user_data()
                            st.success("Added to wishlist! Recommendations will update.")
                            st.session_state.rec_button_counter += 1
                        else:
                            st.info("Already in wishlist!")

                st.markdown("---")

    # ------------------------------------------------------------------
    # TAB 2 – Search
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("🔍 Search for Items")

        s_col1, s_col2 = st.columns([3, 1])
        with s_col1:
            search_query = st.text_input("Search for products...",
                                         placeholder="Enter product name or category")
        with s_col2:
            st.write("")
            st.write("")
            search_button = st.button("🔍 Search")

        # Advanced search filters
        with st.expander("Advanced Filters"):
            filter_col1, filter_col2 = st.columns(2)

            if st.session_state.items_data is not None and not st.session_state.items_data.empty:
                avail_cats = sorted(st.session_state.items_data['category'].unique().tolist())
                cat_options = ["All"] + avail_cats
            else:
                cat_options = ["All", "Video_Games", "Digital_Music", "Software"]

            if st.session_state.items_data is not None and not st.session_state.items_data.empty:
                avail_brands = sorted(st.session_state.items_data['brand'].unique().tolist())
                brand_options = ["All"] + avail_brands[:20]
            else:
                brand_options = ["All", "Apple", "Samsung", "Nike", "Sony"]

            with filter_col1:
                category_filter = st.selectbox("Category", cat_options)
                min_price = st.number_input("Min Price", min_value=0, value=0, step=10)
            with filter_col2:
                brand_filter = st.selectbox("Brand", brand_options)
                max_price = st.number_input("Max Price", min_value=0, value=1000, step=10)

        if search_button and not search_query.strip():
            st.warning("⚠️ Please enter a search term")

        # Run search and cache results in session_state
        if search_button and search_query.strip():
            q = search_query.strip().lower()

            if st.session_state.items_data is None:
                with st.spinner("Loading data..."):
                    items_df, interactions_df = load_real_amazon_data()
                    st.session_state.items_data = items_df
                    st.session_state.interactions_df = interactions_df

            items_df = st.session_state.items_data

            if items_df is None or items_df.empty:
                st.error("No data available for search")
            else:
                # Safe string conversion for description (may contain lists)
                desc_col = items_df['description'].apply(
                    lambda x: ' '.join(x) if isinstance(x, list) else str(x) if x else ''
                ).str.lower()

                mask = (
                    items_df['title'].fillna('').str.lower().str.contains(q, na=False, regex=False)
                    | desc_col.str.contains(q, na=False, regex=False)
                    | items_df['brand'].fillna('').str.lower().str.contains(q, na=False, regex=False)
                    | items_df['category'].fillna('').str.lower().str.contains(q, na=False, regex=False)
                )
                search_results = items_df[mask]

                # Apply filters
                if category_filter != "All":
                    search_results = search_results[search_results['category'] == category_filter]
                if brand_filter != "All":
                    search_results = search_results[search_results['brand'] == brand_filter]
                if min_price > 0:
                    search_results = search_results[search_results['price'] >= min_price]
                if max_price > 0:
                    search_results = search_results[search_results['price'] <= max_price]

                # Cache search results as list of dicts so they persist across reruns
                if not search_results.empty:
                    display_df = search_results.sample(min(20, len(search_results)), random_state=42)
                    cached = []
                    for _, product in display_df.iterrows():
                        pid = product.get('asin', product.get('item_id', ''))
                        cached.append({
                            "item_id": pid,
                            "asin": pid,
                            "title": str(product['title']),
                            "category": product['category'],
                            "brand": product.get('brand', 'Unknown'),
                            "price": product.get('price', 0.0),
                            "valid_image_urls": product.get('valid_image_urls', []),
                        })
                    st.session_state.search_results = cached
                    st.session_state.search_total = len(search_results)
                    st.session_state.last_search_query = q
                else:
                    st.session_state.search_results = []
                    st.session_state.search_total = 0
                    st.session_state.last_search_query = q

        # Display cached search results (persists after button clicks)
        if st.session_state.get('search_results') is not None and len(st.session_state.get('search_results', [])) > 0:
            st.success(f"Found {st.session_state.search_total} matching products (showing top 20)")

            for idx, item in enumerate(st.session_state.search_results):
                with st.container():
                    c1, c2, c3 = st.columns([1, 3, 1])

                    with c1:
                        show_image_col(item.get('valid_image_urls', []), item.get('category', ''))

                    with c2:
                        title = item['title']
                        if len(title) > 100:
                            title = title[:100] + "..."
                        st.markdown(f"**{title}**")
                        st.markdown(f"**Category:** {item['category']}")
                        st.markdown(f"**Brand:** {item.get('brand', 'Unknown')}")
                        st.markdown(f"**Price:** ${item.get('price', 0.0):.2f}")

                        pid = item.get('asin', '')
                        p_int = (
                            st.session_state.interactions_df[
                                st.session_state.interactions_df['asin'] == pid
                            ]
                            if st.session_state.interactions_df is not None
                            and not st.session_state.interactions_df.empty
                            else pd.DataFrame()
                        )
                        if not p_int.empty:
                            ar = p_int['overall'].mean() if 'overall' in p_int.columns else 4.0
                            rc = len(p_int)
                            st.markdown(f"{'⭐' * int(ar)} {ar:.1f}/5.0 ({rc} reviews)")

                    with c3:
                        if st.button("🛒 Add to Cart", key=f"cart_search_{idx}_{pid}"):
                            is_in_cart = any(c.get('asin', '') == pid for c in st.session_state.cart)
                            if not is_in_cart:
                                st.session_state.cart.append(item)
                                save_user_data()
                                st.success("Added to cart!")
                            else:
                                st.info("Already in cart!")

                        if st.button("❤️ Wishlist", key=f"save_search_{idx}_{pid}"):
                            is_dup = any(
                                (pid and w.get('asin', '') == pid)
                                for w in st.session_state.wishlist
                            )
                            if not is_dup:
                                st.session_state.wishlist.append(item)
                                st.session_state.pop('cached_recommendations', None)
                                st.session_state.pop('rec_params', None)
                                save_user_data()
                                st.success(f"Added to wishlist!")
                            else:
                                st.info("Already in wishlist!")

                    st.markdown("---")

        elif st.session_state.get('last_search_query') and st.session_state.get('search_total', -1) == 0:
            st.warning(f"No products found matching '{st.session_state.last_search_query}'")
            st.info("Try different keywords or check your spelling")

    # ------------------------------------------------------------------
    # TAB 3 – Purchase History
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("� Purchase History")

        if not st.session_state.purchase_history:
            st.info("No purchases yet. Add items to your cart and checkout!")
        else:
            # Summary metrics
            ph = st.session_state.purchase_history
            total_spent = sum(i.get('price', 0) for i in ph)
            ph_cats = set(i.get('category', '') for i in ph)

            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                st.metric("Total Orders", len(ph))
            with mc2:
                st.metric("Total Spent", f"${total_spent:.2f}")
            with mc3:
                st.metric("Categories", len(ph_cats))

            st.markdown("---")

            # List purchases
            for pi, p_item in enumerate(reversed(ph)):
                with st.container():
                    pc1, pc2 = st.columns([1, 4])
                    with pc1:
                        show_image_col(p_item.get('valid_image_urls', []), p_item.get('category', ''))
                    with pc2:
                        title = p_item['title']
                        if len(title) > 80:
                            title = title[:80] + "..."
                        st.markdown(f"**{title}**")
                        st.markdown(f"**Category:** {p_item['category']} | **Brand:** {p_item.get('brand', 'Unknown')} | **Price:** ${p_item.get('price', 0.0):.2f}")
                        st.markdown(f"🗓️ Purchased on: {p_item.get('purchase_date', 'N/A')}")
                    st.markdown("---")

    # ------------------------------------------------------------------
    # TAB 4 – Wishlist
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("❤️ Your Wishlist")

        if not st.session_state.wishlist:
            st.info("Your wishlist is empty. Start adding products you love!")
        else:
            for idx, item in enumerate(st.session_state.wishlist):
                with st.container():
                    c1, c2, c3 = st.columns([1, 3, 1])

                    with c1:
                        show_image_col(item.get('valid_image_urls', []), item.get('category', ''))

                    with c2:
                        title = item['title']
                        if len(title) > 80:
                            title = title[:80] + "..."
                        st.markdown(f"**{title}**")
                        st.markdown(f"**Category:** {item['category']}")
                        st.markdown(f"**Brand:** {item.get('brand', 'Unknown')}")
                        st.markdown(f"**Price:** ${item.get('price', 0.0):.2f}")

                        pid = item.get('asin', item.get('item_id', ''))
                        if st.session_state.interactions_df is not None and not st.session_state.interactions_df.empty:
                            p_int = st.session_state.interactions_df[
                                st.session_state.interactions_df['asin'] == pid
                            ]
                            if not p_int.empty:
                                ar = p_int['overall'].mean() if 'overall' in p_int.columns else 4.0
                                rc = len(p_int)
                                st.markdown(f"{'⭐' * int(ar)} {ar:.1f}/5.0 ({rc} reviews)")

                    with c3:
                        w_pid = item.get('asin', item.get('item_id', ''))
                        if st.button("🛒 Add to Cart", key=f"cart_wish_{idx}_{w_pid}"):
                            is_in_cart = any(c.get('asin', '') == w_pid for c in st.session_state.cart)
                            if not is_in_cart:
                                st.session_state.cart.append(item)
                                save_user_data()
                                st.success("Added to cart!")
                            else:
                                st.info("Already in cart!")
                        if st.button("🗑️ Remove", key=f"remove_{idx}_{item['item_id']}"):
                            st.session_state.wishlist = [
                                w for w in st.session_state.wishlist
                                if w['item_id'] != item['item_id']
                            ]
                            st.session_state.pop('cached_recommendations', None)
                            st.session_state.pop('rec_params', None)
                            save_user_data()
                            st.success(f"Removed from wishlist!")
                            st.rerun()

                    st.markdown("---")

            # Wishlist summary
            st.subheader("📊 Wishlist Summary")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.metric("Total Items", len(st.session_state.wishlist))
            with sc2:
                total_val = sum(i.get('price', 0) for i in st.session_state.wishlist)
                st.metric("Total Value", f"${total_val:.2f}")
            with sc3:
                cats = set(i.get('category', '') for i in st.session_state.wishlist)
                st.metric("Categories", len(cats))

            if st.button("🗑️ Clear Wishlist", type="secondary"):
                st.session_state.wishlist = []
                st.success("Wishlist cleared!")


def main():
    """Main application router"""
    if not st.session_state.is_authenticated:
        login_page()
    else:
        main_dashboard()


if __name__ == "__main__":
    main()
