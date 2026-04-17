import pandas as pd

df = pd.read_parquet('data/real_items.parquet')
print('NOUVELLES CATÉGORIES:')
cats = df['category'].value_counts()
for cat, count in cats.items():
    print(f'  {cat}: {count}')

print(f'\nTOTAL: {len(df)} produits')
