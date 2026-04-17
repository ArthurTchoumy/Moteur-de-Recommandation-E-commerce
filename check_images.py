import pandas as pd

# Check image URLs in real Amazon data
print("=== CHECKING IMAGE URLS ===")
items_df = pd.read_parquet('data/real_items.parquet')

print(f"Total products: {len(items_df)}")
print(f"Columns: {items_df.columns.tolist()}")

if 'valid_image_urls' in items_df.columns:
    # Count products with images
    has_images = items_df['valid_image_urls'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    print(f"Products with images: {has_images.sum()}")
    print(f"Products without images: {(~has_images).sum()}")
    
    # Show sample image URLs
    print("\nSample image URLs:")
    for i in range(min(5, len(items_df))):
        row = items_df.iloc[i]
        image_urls = row.get('valid_image_urls', [])
        if image_urls and len(image_urls) > 0:
            print(f"Product {i}: {row['title'][:50]}...")
            print(f"  Image URL: {image_urls[0]}")
            print()
        else:
            print(f"Product {i}: {row['title'][:50]}... - NO IMAGES")
            print()
else:
    print("No 'valid_image_urls' column found")

# Check if image URLs are valid HTTP URLs
print("\n=== CHECKING URL FORMATS ===")
if 'valid_image_urls' in items_df.columns:
    for i in range(min(3, len(items_df))):
        row = items_df.iloc[i]
        image_urls = row.get('valid_image_urls', [])
        if image_urls and len(image_urls) > 0:
            url = image_urls[0]
            print(f"URL: {url}")
            print(f"Starts with http: {url.startswith(('http://', 'https://'))}")
            print()
