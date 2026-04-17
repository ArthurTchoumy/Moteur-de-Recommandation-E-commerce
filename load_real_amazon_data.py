import pandas as pd
import os
import json
from pathlib import Path

def load_real_amazon_data():
    """Load real Amazon product and review data"""
    
    # Load products from different categories
    product_files = [
        "products_Video_Games_cleaned.parquet",
        "products_Digital_Music_cleaned.parquet", 
        "products_Software_cleaned.parquet",
        "products_Appliances_cleaned.parquet",
        "products_Gift_Cards_cleaned.parquet",
        "products_Industrial_and_Scientific_cleaned.parquet",
        "products_Magazine_Subscriptions_cleaned.parquet",
        "products_Prime_Pantry_cleaned.parquet"
    ]
    
    # Filter to only existing files
    product_files = [f for f in product_files if os.path.exists(f)]
    
    # Filter out None values
    product_files = [f for f in product_files if f is not None]
    
    products_dfs = []
    
    for file in product_files:  # Load all available categories
        try:
            print(f"Loading {file}...")
            df = pd.read_parquet(file)
            
            # Extract category from filename
            category = file.replace("products_", "").replace("_cleaned.parquet", "")
            df['category'] = category
            
            # Standardize column names
            if 'title' not in df.columns and 'product_title' in df.columns:
                df['title'] = df['product_title']
            elif 'title' not in df.columns and 'name' in df.columns:
                df['title'] = df['name']
            
            if 'brand' not in df.columns:
                df['brand'] = 'Unknown'
                
            if 'price' not in df.columns:
                df['price'] = 0.0
            
            # Create item_id if not exists
            if 'item_id' not in df.columns:
                if 'asin' in df.columns:
                    df['item_id'] = df['asin']
                else:
                    df['item_id'] = [f"{category}_{i}" for i in range(len(df))]
            
            # Fix image URLs - convert JSON strings to lists
            if 'valid_image_urls' in df.columns:
                df['valid_image_urls'] = df['valid_image_urls'].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x
                )
            
            print(f"Loaded {len(df)} products from {category}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample titles: {df['title'].head(3).tolist()}")
            
            products_dfs.append(df)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Load reviews
    review_files = [
        file.replace("products_", "reviews_") for file in product_files
    ]
    
    reviews_dfs = []
    
    for file in review_files:
        try:
            if os.path.exists(file):
                print(f"Loading {file}...")
                df = pd.read_parquet(file)
                
                # Extract category from filename
                category = file.replace("reviews_", "").replace("_cleaned.parquet", "")
                df['category'] = category
                
                # Standardize column names
                if 'asin' not in df.columns and 'item_id' in df.columns:
                    df['asin'] = df['item_id']
                elif 'asin' not in df.columns and 'product_id' in df.columns:
                    df['asin'] = df['product_id']
                
                if 'overall' not in df.columns and 'rating' in df.columns:
                    df['overall'] = df['rating']
                elif 'overall' not in df.columns and 'star_rating' in df.columns:
                    df['overall'] = df['star_rating']
                
                print(f"Loaded {len(df)} reviews from {category}")
                reviews_dfs.append(df)
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Combine dataframes
    all_products = pd.concat(products_dfs, ignore_index=True) if products_dfs else pd.DataFrame()
    all_reviews = pd.concat(reviews_dfs, ignore_index=True) if reviews_dfs else pd.DataFrame()
    
    print(f"\nTotal: {len(all_products)} products, {len(all_reviews)} reviews")
    
    return all_products, all_reviews

if __name__ == "__main__":
    products, reviews = load_real_amazon_data()
    
    # Save to processed data for the app
    os.makedirs("data", exist_ok=True)
    products.to_parquet("data/real_items.parquet", index=False)
    reviews.to_parquet("data/real_interactions.parquet", index=False)
    
    print("\nSaved to data/real_items.parquet and data/real_interactions.parquet")
    
    # Show sample
    print("\nSample products:")
    for i in range(min(5, len(products))):
        row = products.iloc[i]
        print(f"- {row['title'][:80]}... ({row['category']}) - ${row.get('price', 0):.2f}")
