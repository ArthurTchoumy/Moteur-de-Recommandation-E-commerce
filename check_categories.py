import pandas as pd
import os

# Check all original product files
files = [f for f in os.listdir('.') if f.startswith('products_') and f.endswith('.parquet')]
print('Fichiers produits trouvés:', files)

for file in files:
    try:
        df = pd.read_parquet(file)
        category = file.replace('products_', '').replace('_cleaned.parquet', '')
        print(f'\n{file}: {len(df)} produits - Catégorie: {category}')
        if 'category' in df.columns:
            unique_cats = df['category'].unique()[:5]
            print(f'  Valeurs uniques dans category: {unique_cats}')
        if 'main_category' in df.columns:
            unique_main = df['main_category'].unique()[:5]
            print(f'  Valeurs uniques dans main_category: {unique_main}')
    except Exception as e:
        print(f'  Erreur: {e}')

print("\n" + "="*50)
print("VÉRIFICATION DES DONNÉES COMBINÉES")

# Check our combined data
try:
    combined_df = pd.read_parquet('data/real_items.parquet')
    print(f'\nDonnées combinées: {len(combined_df)} produits')
    print('Catégories uniques:')
    cat_counts = combined_df['category'].value_counts()
    for cat, count in cat_counts.items():
        print(f'  {cat}: {count}')
        
    print('\nMarques uniques (top 10):')
    brand_counts = combined_df['brand'].value_counts().head(10)
    for brand, count in brand_counts.items():
        print(f'  {brand}: {count}')
        
except Exception as e:
    print(f'Erreur lecture données combinées: {e}')
