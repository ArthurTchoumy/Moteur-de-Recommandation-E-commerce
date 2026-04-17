"""
Streamlit User Interface for E-commerce Recommendation Engine
Provides user-facing interface for recommendations and account management
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

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
    .explanation-text {
        font-style: italic;
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'is_authenticated' not in st.session_state:
    st.session_state.is_authenticated = False
if 'preferences' not in st.session_state:
    st.session_state.preferences = {
        'categories': [],
        'price_range': [0, 1000],
        'brands': []
    }

# Helper functions
def api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
    """Make API request to recommendation service"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        else:
            return None
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

def format_price(price: float) -> str:
    """Format price with currency symbol"""
    return f"${price:.2f}"

def display_recommendation_card(item: Dict, show_explanation: bool = False):
    """Display a single recommendation card"""
    with st.container():
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Placeholder for product image
            st.image("https://via.placeholder.com/100x100.png?text=Product", width=100)
        
        with col2:
            st.markdown(f"### {item.get('item_id', 'Unknown Product')}")
            
            # Rating
            if 'rating' in item:
                stars = "⭐" * int(item['rating'])
                st.markdown(f"{stars} {item['rating']:.1f}/5.0 ({item.get('review_count', 0)} reviews)")
            
            # Price
            if 'price' in item:
                st.markdown(f"**Price:** {format_price(item['price'])}")
            
            # Category and brand
            if 'category' in item:
                st.markdown(f"**Category:** {item['category']}")
            if 'brand' in item:
                st.markdown(f"**Brand:** {item['brand']}")
            
            # Explanation
            if show_explanation and 'explanation' in item:
                st.markdown(f"<p class='explanation-text'>{item['explanation']}</p>", 
                           unsafe_allow_html=True)
        
        with col3:
            # Score
            if 'score' in item:
                st.markdown(f"<span class='score-badge'>Score: {item['score']:.3f}</span>", 
                           unsafe_allow_html=True)
            
            # Add to cart button
            if st.button("Add to Cart", key=f"cart_{item.get('item_id')}"):
                st.success(f"Added {item.get('item_id')} to cart!")
        
        st.markdown("---")

def login_page():
    """Display login/registration page"""
    st.markdown('<h1 class="main-header">🛍️ Welcome to Smart Shopping</h1>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 Login")
        with st.form("login_form"):
            username = st.text_input("Username or Email")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if username and password:
                    # Simulate login (in real app, this would authenticate)
                    st.session_state.user_id = username
                    st.session_state.is_authenticated = True
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Please enter username and password")
    
    with col2:
        st.subheader("📝 New User Registration")
        with st.form("register_form"):
            new_username = st.text_input("Choose Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            # Preferences for cold start
            st.subheader("Tell us about your preferences:")
            categories = st.multiselect(
                "Favorite Categories",
                ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty"],
                help="Select categories you're interested in"
            )
            
            price_range = st.slider(
                "Price Range",
                min_value=0,
                max_value=2000,
                value=(0, 500),
                step=50,
                format="$%d"
            )
            
            brands = st.multiselect(
                "Favorite Brands",
                ["Apple", "Samsung", "Nike", "Adidas", "Sony", "LG"],
                help="Select your preferred brands"
            )
            
            register_button = st.form_submit_button("Create Account")
            
            if register_button:
                if new_username and email and password == confirm_password:
                    # Create new user profile
                    st.session_state.user_id = new_username
                    st.session_state.is_authenticated = True
                    st.session_state.preferences = {
                        'categories': categories,
                        'price_range': list(price_range),
                        'brands': brands
                    }
                    
                    # Call API for new user recommendations
                    new_user_data = {
                        "user_id": new_username,
                        "preferences": st.session_state.preferences,
                        "num_recommendations": 10
                    }
                    
                    response = api_request("/recommendations/new-user", "POST", new_user_data)
                    if response:
                        st.success(f"Account created! Welcome, {new_username}!")
                        st.rerun()
                else:
                    st.error("Please fill all fields correctly")

def main_dashboard():
    """Display main dashboard with recommendations"""
    # Header
    st.markdown('<h1 class="main-header">🛍️ Your Personalized Recommendations</h1>', 
                unsafe_allow_html=True)
    
    # User info and logout
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"**Welcome back, {st.session_state.user_id}!**")
    with col2:
        if st.button("🔄 Refresh Recommendations"):
            st.rerun()
    with col3:
        if st.button("🚪 Logout"):
            st.session_state.is_authenticated = False
            st.session_state.user_id = None
            st.rerun()
    
    # Sidebar for preferences and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Recommendation Model",
            ["hybrid", "collaborative", "deep_learning"],
            help="Choose which algorithm to use for recommendations"
        )
        
        # Number of recommendations
        num_recs = st.slider(
            "Number of Recommendations",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        
        # Include explanations
        include_explanations = st.checkbox(
            "Show Explanations",
            value=True,
            help="Display why each item was recommended"
        )
        
        st.header("🎯 Your Preferences")
        
        # Update preferences
        with st.expander("Update Preferences"):
            categories = st.multiselect(
                "Categories",
                ["Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty"],
                default=st.session_state.preferences['categories']
            )
            
            price_range = st.slider(
                "Price Range",
                min_value=0,
                max_value=2000,
                value=tuple(st.session_state.preferences['price_range']),
                step=50,
                format="$%d"
            )
            
            brands = st.multiselect(
                "Brands",
                ["Apple", "Samsung", "Nike", "Adidas", "Sony", "LG"],
                default=st.session_state.preferences['brands']
            )
            
            if st.button("Update Preferences"):
                st.session_state.preferences = {
                    'categories': categories,
                    'price_range': list(price_range),
                    'brands': brands
                }
                st.success("Preferences updated!")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🏠 Recommendations", "🔍 Search Items", "📊 Your Activity"])
    
    with tab1:
        # Get recommendations
        with st.spinner("Generating personalized recommendations..."):
            request_data = {
                "user_id": st.session_state.user_id,
                "num_recommendations": num_recs,
                "model_type": model_type,
                "include_explanation": include_explanations
            }
            
            response = api_request("/recommendations", "POST", request_data)
            
            if response:
                # Display performance metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Model Used", response.get('model_used', 'Unknown'))
                with col2:
                    latency = response.get('latency_ms', 0)
                    st.metric("Response Time", f"{latency:.1f} ms")
                with col3:
                    cached = response.get('cached', False)
                    st.metric("From Cache", "Yes" if cached else "No")
                
                # Display recommendations
                st.subheader(f"🎁 Top {len(response.get('recommendations', []))} Recommendations for You")
                
                recommendations = response.get('recommendations', [])
                if recommendations:
                    for item in recommendations:
                        display_recommendation_card(item, include_explanations)
                else:
                    st.info("No recommendations available. Please check back later.")
            else:
                st.error("Failed to load recommendations. Please try again.")
    
    with tab2:
        st.subheader("🔍 Search for Items")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("Search for products...", placeholder="Enter product name or category")
        
        with col2:
            st.write("")
            st.write("")
            search_button = st.button("🔍 Search")
        
        # Advanced search filters
        with st.expander("Advanced Filters"):
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                category_filter = st.selectbox(
                    "Category",
                    ["All", "Electronics", "Clothing", "Books", "Home & Garden", "Sports", "Beauty"]
                )
                min_price = st.number_input("Min Price", min_value=0, value=0, step=10)
            
            with filter_col2:
                brand_filter = st.selectbox(
                    "Brand",
                    ["All", "Apple", "Samsung", "Nike", "Adidas", "Sony", "LG"]
                )
                max_price = st.number_input("Max Price", min_value=0, value=1000, step=10)
        
        if search_button or search_query:
            st.info(f"Searching for '{search_query}'...")
            # This would integrate with your product catalog
            st.success("Search results would appear here")
    
    with tab3:
        st.subheader("📊 Your Shopping Activity")
        
        # Activity metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Purchases", "42")
        with col2:
            st.metric("Items Viewed", "156")
        with col3:
            st.metric("Avg Order Value", "$127.50")
        with col4:
            st.metric("Saved Items", "8")
        
        # Recent activity chart
        st.subheader("📈 Recent Activity")
        
        # Sample data - in real app, this would come from user's activity log
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        activity = np.random.randint(0, 10, size=30)
        
        activity_df = pd.DataFrame({
            'date': dates,
            'activity': activity
        })
        
        fig = px.line(activity_df, x='date', y='activity', 
                     title='Your Shopping Activity (Last 30 Days)')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Category preferences
        st.subheader("🎯 Your Category Preferences")
        
        category_data = {
            'Category': ['Electronics', 'Clothing', 'Books', 'Home & Garden'],
            'Preference': [35, 25, 20, 20]
        }
        
        category_df = pd.DataFrame(category_data)
        
        fig = px.pie(category_df, values='Preference', names='Category',
                    title='Your Shopping Preferences by Category')
        st.plotly_chart(fig, use_container_width=True)

def item_details_page(item_id: str):
    """Display detailed item page"""
    st.markdown(f'<h1 class="main-header">Product Details</h1>', 
                unsafe_allow_html=True)
    
    # Get item details
    response = api_request(f"/items/{item_id}/features")
    
    if response:
        item = response.get('features', {})
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image("https://via.placeholder.com/300x300.png?text=Product", width=300)
        
        with col2:
            st.markdown(f"### {item.get('item_id', 'Unknown Product')}")
            
            # Rating and reviews
            if 'avg_rating' in item:
                stars = "⭐" * int(item['avg_rating'])
                st.markdown(f"{stars} {item['avg_rating']:.1f}/5.0 ({item.get('review_count', 0)} reviews)")
            
            # Price
            if 'price' in item:
                st.markdown(f"## {format_price(item['price'])}")
            
            # Product details
            st.markdown("#### Product Details")
            if 'category' in item:
                st.markdown(f"**Category:** {item['category']}")
            if 'brand' in item:
                st.markdown(f"**Brand:** {item['brand']}")
            if 'availability' in item:
                availability_color = "🟢" if item['availability'] == "in_stock" else "🔴"
                st.markdown(f"**Availability:** {availability_color} {item['availability']}")
            
            # Action buttons
            col_btn1, col_btn2, col_btn3 = st.columns(3)
            
            with col_btn1:
                if st.button("🛒 Add to Cart", key="add_cart"):
                    st.success("Added to cart!")
            
            with col_btn2:
                if st.button("❤️ Save for Later", key="save_later"):
                    st.success("Saved to your wishlist!")
            
            with col_btn3:
                if st.button("🔗 Share", key="share"):
                    st.success("Link copied to clipboard!")
        
        # Similar items section
        st.markdown("#### 🔍 Similar Items You Might Like")
        
        similar_response = api_request("/items/similar", "POST", {
            "item_id": item_id,
            "num_similar_items": 5
        })
        
        if similar_response:
            similar_items = similar_response.get('similar_items', [])
            for item in similar_items:
                display_recommendation_card(item)
    
    # Back button
    if st.button("← Back to Recommendations"):
        st.rerun()

# Main app logic
def main():
    """Main application router"""
    
    # Check API health
    health_response = api_request("/health")
    if not health_response:
        st.error("⚠️ Unable to connect to recommendation service. Please check if the API is running.")
        st.info("Make sure the API server is running on http://localhost:8000")
        return
    
    # Display API status in sidebar
    with st.sidebar:
        if health_response:
            status = health_response.get('status', 'unknown')
            status_color = "🟢" if status == "healthy" else "🟡"
            st.markdown(f"**API Status:** {status_color} {status.title()}")
            
            components = health_response.get('components', {})
            for component, comp_status in components.items():
                comp_color = "🟢" if comp_status == "healthy" else "🔴"
                st.markdown(f"{comp_color} {component.title()}: {comp_status}")
    
    # Route based on authentication
    if not st.session_state.is_authenticated:
        login_page()
    else:
        # Check for specific item view
        query_params = st.experimental_get_query_params()
        if 'item' in query_params:
            item_details_page(query_params['item'][0])
        else:
            main_dashboard()

if __name__ == "__main__":
    main()
