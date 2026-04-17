import pandas as pd
import json

# Check original Amazon data files
print("=== INVESTIGATING ORIGINAL AMAZON DATA ===")

# Check Video Games data
try:
    vg_df = pd.read_parquet('products_Video_Games_cleaned.parquet')
    print(f"\nVideo Games Products: {len(vg_df)}")
    print(f"Columns: {vg_df.columns.tolist()}")
    
    # Check image-related columns
    image_cols = [col for col in vg_df.columns if 'image' in col.lower() or 'url' in col.lower()]
    print(f"Image-related columns: {image_cols}")
    
    if 'image' in vg_df.columns:
        print(f"Sample 'image' values:")
        for i in range(min(3, len(vg_df))):
            print(f"  {vg_df.iloc[i]['image']}")
    
    if 'imageURL' in vg_df.columns:
        print(f"Sample 'imageURL' values:")
        for i in range(min(3, len(vg_df))):
            print(f"  {vg_df.iloc[i]['imageURL']}")
            
    if 'imUrl' in vg_df.columns:
        print(f"Sample 'imUrl' values:")
        for i in range(min(3, len(vg_df))):
            print(f"  {vg_df.iloc[i]['imUrl']}")
    
except Exception as e:
    print(f"Error loading Video Games: {e}")

# Check Digital Music data
try:
    dm_df = pd.read_parquet('products_Digital_Music_cleaned.parquet')
    print(f"\nDigital Music Products: {len(dm_df)}")
    
    # Check image-related columns
    image_cols = [col for col in dm_df.columns if 'image' in col.lower() or 'url' in col.lower()]
    print(f"Image-related columns: {image_cols}")
    
    if 'image' in dm_df.columns:
        print(f"Sample 'image' values:")
        for i in range(min(3, len(dm_df))):
            val = dm_df.iloc[i]['image']
            print(f"  Type: {type(val)}, Value: {val}")
            
except Exception as e:
    print(f"Error loading Digital Music: {e}")

# Check Software data
try:
    sw_df = pd.read_parquet('products_Software_cleaned.parquet')
    print(f"\nSoftware Products: {len(sw_df)}")
    
    # Check image-related columns
    image_cols = [col for col in sw_df.columns if 'image' in col.lower() or 'url' in col.lower()]
    print(f"Image-related columns: {image_cols}")
    
    if 'image' in sw_df.columns:
        print(f"Sample 'image' values:")
        for i in range(min(3, len(sw_df))):
            val = sw_df.iloc[i]['image']
            print(f"  Type: {type(val)}, Value: {val}")
            
except Exception as e:
    print(f"Error loading Software: {e}")

print("\n" + "="*50)
print("CHECKING PROCESSED DATA")

# Check our processed data
try:
    real_items = pd.read_parquet('data/real_items.parquet')
    print(f"\nReal Items: {len(real_items)}")
    
    if 'valid_image_urls' in real_items.columns:
        print(f"valid_image_urls column exists")
        
        # Count non-empty image URLs
        non_empty = real_items['valid_image_urls'].apply(
            lambda x: len(x) > 0 if isinstance(x, list) else False
        )
        print(f"Products with non-empty image URLs: {non_empty.sum()}")
        
        # Show some examples
        for i in range(min(5, len(real_items))):
            row = real_items.iloc[i]
            image_urls = row['valid_image_urls']
            print(f"Product {i}: {row['title'][:50]}...")
            print(f"  Image URLs: {image_urls}")
            print(f"  Type: {type(image_urls)}, Length: {len(image_urls) if isinstance(image_urls, list) else 'N/A'}")
            print()
    else:
        print("No valid_image_urls column found")
        
except Exception as e:
    print(f"Error checking processed data: {e}")
