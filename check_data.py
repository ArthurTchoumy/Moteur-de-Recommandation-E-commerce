import pandas as pd

# Check items data
print("=== ITEMS DATA ===")
items_df = pd.read_parquet('data/items.parquet')
print(f"Shape: {items_df.shape}")
print(f"Columns: {items_df.columns.tolist()}")
print("\nFirst 5 items:")
for i in range(min(5, len(items_df))):
    row = items_df.iloc[i]
    print(f"Item {i}: {row['title']} - {row['category']} - {row['brand']} - ${row['price']:.2f}")

print("\n=== INTERACTIONS DATA ===")
interactions_df = pd.read_parquet('data/interactions.parquet')
print(f"Shape: {interactions_df.shape}")
print(f"Columns: {interactions_df.columns.tolist()}")
print("\nFirst 3 interactions:")
for i in range(min(3, len(interactions_df))):
    row = interactions_df.iloc[i]
    print(f"Interaction {i}: User {row['user_id']} - Item {row['item_id']} - Rating {row['rating']:.2f}")
