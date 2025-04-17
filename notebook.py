import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat dataset
ecommerce = pd.read_csv('dataset/data.csv', encoding='latin-1')

# Menampilkan informasi dataset
print('Informasi Dataset:')
print(ecommerce.info())

# Menampilkan 5 baris pertama
print('\nDataset (5 baris pertama):')
print(ecommerce.head())

# Visualisasi produk terpopuler
top_products = ecommerce.groupby('Description')['Quantity'].sum().sort_values(ascending=False)[:10]
plt.figure(figsize=(12, 6))
top_products.plot(kind='bar')
plt.title('Top 10 Produk Terpopuler')
plt.xlabel('Produk')
plt.ylabel('Total Kuantitas')
plt.savefig('products.png')
plt.show()

# Visualisasi distribusi pembelian per pelanggan
purchases_per_customer = ecommerce.groupby('CustomerID')['InvoiceNo'].nunique()
plt.figure(figsize=(8, 5))
sns.histplot(purchases_per_customer, bins=20)
plt.title('Distribusi Jumlah Pembelian per Pelanggan')
plt.xlabel('Jumlah Pembelian')
plt.ylabel('Frekuensi')
plt.show()

# Membersihkan data
ecommerce = ecommerce.dropna(subset=['CustomerID', 'Description'])
ecommerce = ecommerce[ecommerce['Quantity'] > 0]
ecommerce = ecommerce[~ecommerce['InvoiceNo'].astype(str).str.startswith('C')]  # Menghapus transaksi batal

# Mengubah CustomerID menjadi string
ecommerce['CustomerID'] = ecommerce['CustomerID'].astype(str)

# Membuat matriks user-item (berdasarkan kuantitas)
user_item_matrix = ecommerce.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)

# Membersihkan deskripsi untuk TF-IDF
ecommerce['Description'] = ecommerce['Description'].str.lower().str.replace('[^a-z0-9 ]', '', regex=True)
product_descriptions = ecommerce.groupby('StockCode')['Description'].first().reset_index()

print('Data setelah pembersihan:')
print(ecommerce.info())
print('\nMatriks User-Item (5 baris pertama):')
print(user_item_matrix.head())

# TF-IDF Vectorizer untuk deskripsi
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(product_descriptions['Description'])

# Menghitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi rekomendasi
def get_content_based_recommendations(stock_code, cosine_sim=cosine_sim, df=product_descriptions, top_n=5):
    idx = df[df['StockCode'] == stock_code].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    product_indices = [i[0] for i in sim_scores]
    return df['Description'].iloc[product_indices]

# Contoh rekomendasi
example_product = product_descriptions['StockCode'].iloc[0]
print(f'Rekomendasi untuk produk {example_product}:')
print(get_content_based_recommendations(example_product))

# SVD
svd = TruncatedSVD(n_components=20, random_state=42)
matrix_svd = svd.fit_transform(user_item_matrix)

# Menghitung similarity antar pelanggan
user_sim = cosine_similarity(matrix_svd)

# Fungsi rekomendasi
def get_collaborative_recommendations(customer_id, user_sim=user_sim, user_item_matrix=user_item_matrix, df=product_descriptions, top_n=5):
    user_idx = user_item_matrix.index.get_loc(customer_id)
    sim_scores = list(enumerate(user_sim[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_users = [i[0] for i in sim_scores[1:11]]  # Top 10 pelanggan serupa
    
    # Mendapatkan produk yang dibeli pelanggan serupa
    sim_user_purchases = user_item_matrix.iloc[sim_users]
    product_scores = sim_user_purchases.mean(axis=0)
    product_scores = product_scores[product_scores > 0]
    top_product_ids = product_scores.sort_values(ascending=False).head(top_n).index
    
    return df[df['StockCode'].isin(top_product_ids)]['Description']

# Contoh rekomendasi
example_customer = user_item_matrix.index[0]
print(f'Rekomendasi untuk pelanggan {example_customer}:')
print(get_collaborative_recommendations(example_customer))

# Fungsi evaluasi Recall@5 untuk content-based
def evaluate_content_based(stock_code, recommendations, df=product_descriptions, ecommerce_df=ecommerce):
    target_desc = set(df[df['StockCode'] == stock_code]['Description'].str.split().iloc[0])
    # Menganggap produk dengan kata serupa dalam deskripsi sebagai relevan
    relevant_items = set(df[df['Description'].apply(lambda x: bool(set(x.split()).intersection(target_desc)))]['StockCode'])
    rec_ids = set(df[df['Description'].isin(recommendations)]['StockCode'])
    hits = len(relevant_items.intersection(rec_ids))
    return hits / len(relevant_items) if relevant_items else 0

# Fungsi evaluasi Recall@5 untuk collaborative
def evaluate_collaborative(customer_id, recommendations, ecommerce_df=ecommerce, product_df=product_descriptions):
    relevant_items = set(ecommerce_df[ecommerce_df['CustomerID'] == customer_id]['StockCode'])
    rec_ids = set(product_df[product_df['Description'].isin(recommendations)]['StockCode'])
    hits = len(relevant_items.intersection(rec_ids))
    return hits / len(relevant_items) if relevant_items else 0

# Contoh evaluasi
cb_recs = get_content_based_recommendations(example_product)
cf_recs = get_collaborative_recommendations(example_customer)

print(f'Recall@5 Content-Based untuk produk {example_product}:', evaluate_content_based(example_product, cb_recs))
print(f'Recall@5 Collaborative untuk pelanggan {example_customer}:', evaluate_collaborative(example_customer, cf_recs))