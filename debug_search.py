import pandas as pd

# Check what keys are available in search results
print("VÉRIFICATION DES CLÉS DES PRODUITS DE RECHERCHE")

# Load real data
items_df = pd.read_parquet('data/real_items.parquet')
print(f"Total produits: {len(items_df)}")
print(f"Colonnes disponibles: {items_df.columns.tolist()}")

# Check sample products
print("\nExemples de produits:")
for i in range(3):
    item = items_df.iloc[i]
    print(f"\nProduit {i+1}:")
    print(f"  item_id: {item.get('item_id', 'NOT FOUND')}")
    print(f"  asin: {item.get('asin', 'NOT FOUND')}")
    print(f"  title: {item.get('title', 'NOT FOUND')[:50]}...")
    print(f"  category: {item.get('category', 'NOT FOUND')}")
    print(f"  price: {item.get('price', 'NOT FOUND')}")

# Test search logic
print("\n" + "="*50)
print("TEST DE LOGIQUE DE RECHERCHE")

# Simulate search for "music"
search_query = "music"
search_query_lower = search_query.lower()

mask = (
    items_df['title'].fillna('').str.lower().str.contains(search_query_lower, na=False) |
    items_df['description'].fillna('').str.lower().str.contains(search_query_lower, na=False) |
    items_df['brand'].fillna('').str.lower().str.contains(search_query_lower, na=False) |
    items_df['category'].fillna('').str.lower().str.contains(search_query_lower, na=False)
)

search_results = items_df[mask]
print(f"Résultats pour 'music': {len(search_results)}")

if len(search_results) > 0:
    print("\nPremiers résultats:")
    for i in range(min(3, len(search_results))):
        item = search_results.iloc[i]
        print(f"\nRésultat {i+1}:")
        print(f"  item_id: {item.get('item_id', 'NOT FOUND')}")
        print(f"  asin: {item.get('asin', 'NOT FOUND')}")
        print(f"  title: {item.get('title', 'NOT FOUND')[:50]}...")
        print(f"  category: {item.get('category', 'NOT FOUND')}")
