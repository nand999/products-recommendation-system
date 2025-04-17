# Laporan Proyek Machine Learning Terapan
**Nama**: Ananda Dwi Ariano  
**Judul Proyek**: Sistem Rekomendasi Produk E-Commerce Berbasis Content-Based dan Collaborative Filtering  

## 1. Project Overview

### 1.1. Latar Belakang
Dalam dunia e-commerce, pelanggan sering kali dihadapkan pada banyaknya pilihan produk, yang dapat menyulitkan mereka menemukan item sesuai preferensi. Sistem rekomendasi menjadi solusi untuk memberikan saran produk yang relevan, meningkatkan kepuasan pelanggan, dan mendorong penjualan. Proyek ini bertujuan membangun sistem rekomendasi menggunakan dataset *E-Commerce Data* dari Kaggle, yang berisi transaksi pembelian dari toko online di Inggris.

**Sumber Dataset**: [E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)

### 1.2. Tujuan
- Mengembangkan sistem rekomendasi produk menggunakan dua pendekatan: **Content-Based Filtering** dan **Collaborative Filtering**.  
- Mengevaluasi performa sistem rekomendasi menggunakan metrik **Recall@5** untuk memastikan rekomendasi mencakup item relevan.  

### 1.3. Pentingnya Proyek
Sistem rekomendasi yang efektif dapat meningkatkan pengalaman pengguna, memperpanjang waktu kunjungan di platform, dan meningkatkan konversi penjualan. Proyek ini relevan untuk toko online yang ingin mempersonalisasi pengalaman belanja pelanggan, mengurangi waktu pencarian produk, dan mendorong pembelian berulang.

### 1.4. Referensi
- Ricci, F., Rokach, L., & Shapira, B. (2011). *Introduction to Recommender Systems Handbook*. Springer.  
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.


## 2. Business Understanding

### 2.1. Problem Statements
- **Masalah 1**: Pelanggan kesulitan menemukan produk yang sesuai dengan preferensi mereka karena banyaknya pilihan di toko online, menyebabkan waktu pencarian yang lama dan potensi kehilangan minat belanja.  
- **Masalah 2**: Kurangnya rekomendasi yang dipersonalisasi berdasarkan riwayat pembelian pelanggan, yang mengakibatkan pengalaman belanja kurang optimal dan rendahnya tingkat konversi penjualan.  

### 2.2. Goals
- **Tujuan 1**: Membuat sistem rekomendasi berbasis konten (content-based) yang merekomendasikan produk berdasarkan kesamaan deskripsi produk, untuk membantu pelanggan menemukan item serupa dengan cepat.  
- **Tujuan 2**: Membuat sistem rekomendasi berbasis kolaborasi (collaborative) yang merekomendasikan produk berdasarkan pola pembelian pelanggan serupa, untuk meningkatkan personalisasi dan konversi.  

### 2.3. Solution Approach
- **Pendekatan 1: Content-Based Filtering**  
  Menggunakan algoritma TF-IDF untuk mengubah deskripsi produk menjadi vektor, lalu menghitung kesamaan antar produk dengan cosine similarity untuk merekomendasikan produk serupa.  
- **Pendekatan 2: Collaborative Filtering**  
  Menggunakan Singular Value Decomposition (SVD) untuk mengurangi dimensi matriks user-item, lalu menghitung kesamaan antar pelanggan dengan cosine similarity untuk merekomendasikan produk berdasarkan preferensi pelanggan serupa.  

## 3. Data Understanding

### 3.1. Sumber Dataset
- **Nama Dataset**: E-Commerce Data  
- **Link**: [E-Commerce Data](https://www.kaggle.com/datasets/carrie1/ecommerce-data)  
- **Ukuran**: ~7.5 MB  
- **File**: `data.csv`  

### 3.2. Deskripsi Dataset
Dataset ini berisi transaksi pembelian dari toko online di Inggris, dengan jumlah data awal sekitar 541.909 baris. Dataset mencakup informasi transaksi yang dapat digunakan untuk membangun sistem rekomendasi.

### 3.3. Variabel Fitur
Berikut adalah semua variabel dalam dataset, termasuk yang tidak digunakan dalam pemodelan:  
- **`InvoiceNo`** (Tipe: String):  
  - Nomor faktur transaksi, merupakan identifier unik untuk setiap transaksi.  
  - Contoh: "536365".  
  - Catatan: Transaksi dengan awalan 'C' menunjukkan pembatalan, yang dihapus dalam pemrosesan.  
- **`StockCode`** (Tipe: String):  
  - Kode unik untuk setiap produk, digunakan untuk mengidentifikasi item tertentu.  
  - Contoh: "85123A" (WHITE HANGING HEART T-LIGHT HOLDER).  
  - Digunakan untuk pemodelan (content-based dan collaborative filtering).  
- **`Description`** (Tipe: String):  
  - Deskripsi teks dari produk, memberikan informasi tentang nama atau karakteristik produk.  
  - Contoh: "WHITE HANGING HEART T-LIGHT HOLDER".  
  - Digunakan untuk content-based filtering setelah pembersihan teks.  
- **`Quantity`** (Tipe: Integer):  
  - Jumlah produk yang dibeli dalam satu transaksi.  
  - Contoh: 6 (6 unit produk dibeli).  
  - Catatan: Nilai negatif menunjukkan pengembalian atau pembatalan, hanya nilai positif yang digunakan.  
  - Digunakan untuk membentuk matriks user-item dalam collaborative filtering.  
- **`InvoiceDate`** (Tipe: String/Datetime):  
  - Tanggal dan waktu transaksi dilakukan.  
  - Contoh: "12/1/2010 8:26".  
  - Tidak digunakan dalam pemodelan karena fokus pada pola pembelian, bukan waktu.  
- **`UnitPrice`** (Tipe: Float):  
  - Harga satuan produk dalam pound sterling (£).  
  - Contoh: 2.55.  
  - Tidak digunakan dalam pemodelan karena rekomendasi berfokus pada preferensi, bukan harga.  
- **`CustomerID`** (Tipe: Float, diubah ke String):  
  - ID unik pelanggan yang melakukan transaksi.  
  - Contoh: 17850.0.  
  - Digunakan untuk collaborative filtering untuk mengidentifikasi pola pembelian pelanggan.  
  - Catatan: Terdapat nilai hilang, yang dihapus dalam pembersihan.  
- **`Country`** (Tipe: String):  
  - Negara tempat pelanggan melakukan transaksi.  
  - Contoh: "United Kingdom".  
  - Tidak digunakan dalam pemodelan karena tidak relevan dengan rekomendasi produk.  

### 3.4. Kondisi Data
- **Nilai Hilang**:  
  - `CustomerID`: ~25% baris tidak memiliki nilai, dihapus karena penting untuk collaborative filtering.  
  - `Description`: Sedikit nilai hilang, dihapus untuk memastikan kualitas content-based filtering.  
- **Transaksi Batal**: Sekitar 2% transaksi memiliki `InvoiceNo` yang dimulai dengan huruf 'C', menandakan pembatalan, dihapus untuk fokus pada pembelian valid.  

### 3.5. Exploratory Data Analysis (EDA)
- **Distribusi Produk**:  
  - Produk seperti "WHITE HANGING HEART T-LIGHT HOLDER" dan "JUMBO BAG RED RETROSPOT" mendominasi pembelian berdasarkan total kuantitas.  
  - Sekitar 10 produk teratas menyumbang ~15% total kuantitas, menunjukkan adanya pola pembelian populer.  
- **Pembelian per Pelanggan**:  
  - Sebagian besar pelanggan melakukan 1-5 transaksi, dengan distribusi miring kanan (beberapa pelanggan sangat aktif).  
  - Rata-rata pelanggan memiliki 3-5 transaksi, mencerminkan basis pelanggan yang beragam.  

**Visualisasi**:  
- **Gambar 1**: Distribusi produk terpopuler berdasarkan total kuantitas pembelian.
    ![products](https://github.com/user-attachments/assets/d829f449-dad9-4a59-846d-b4a5655d9f3d)

- **Gambar 2**: Distribusi jumlah ![Uploading output.png…]()
pembelian per pelanggan, menunjukkan pola transaksi.
![pelanggan](https://github.com/user-attachments/assets/406916d6-646b-430d-b88d-29ea26bd479f)

## 4. Data Preparation

### 4.1. Teknik yang Digunakan
1. **Pembersihan Data**:
   - Menghapus baris dengan nilai hilang di kolom `CustomerID` dan `Description` untuk memastikan data lengkap untuk pemodelan.  
   - Menghapus transaksi batal (InvoiceNo diawali 'C') untuk fokus pada pembelian valid.  
2. **Pemfilteran Data**:
   - Hanya menyertakan transaksi dengan `Quantity` > 0 untuk analisis pembelian positif.  
3. **Transformasi Data**:
   - Mengubah `CustomerID` menjadi tipe string untuk konsistensi pengindeksan.  
   - Membuat matriks user-item dengan `CustomerID` sebagai baris, `StockCode` sebagai kolom, dan `Quantity` sebagai nilai untuk collaborative filtering.  
4. **Pemrosesan Teks (TF-IDF Vectorizer)**:
   - Membersihkan kolom `Description` dengan:  
     - Mengubah teks menjadi huruf kecil untuk konsistensi.  
     - Menghapus karakter non-alfanumerik (misalnya, tanda baca) untuk mengurangi noise.  
   - Menerapkan **TF-IDF Vectorizer** untuk mengubah deskripsi produk menjadi vektor numerik:  
     - **Cara Kerja**:  
       - Menghitung *term frequency* (frekuensi kata dalam deskripsi produk).  
       - Menerapkan *inverse document frequency* untuk memberikan bobot lebih pada kata-kata yang jarang muncul di seluruh deskripsi, sehingga kata-kata unik seperti "t-light" lebih signifikan daripada kata umum seperti "holder".  
       - Menghasilkan matriks TF-IDF di mana setiap produk direpresentasikan sebagai vektor dalam ruang fitur kata.  
     - **Parameter**:  
       - `stop_words='english'`: Menghapus kata umum bahasa Inggris (misalnya, "the", "and") untuk fokus pada kata bermakna.  
       - Parameter default:  
         - `max_features=None`: Mempertimbangkan semua kata.  
         - `norm='l2'`: Normalisasi L2 untuk vektor.  
         - `use_idf=True`: Mengaktifkan pembobotan IDF.  

### 4.2. Alasan Data Preparation
- **Pembersihan**: Menghapus nilai hilang dan transaksi batal memastikan dataset hanya berisi data valid untuk content-based dan collaborative filtering.  
- **Pemfilteran**: Memfokuskan pada `Quantity` > 0 memastikan analisis hanya mencakup pembelian aktual, relevan untuk rekomendasi.  
- **Matriks User-Item**: Mengubah data transaksi menjadi format matriks memungkinkan analisis pola pembelian pelanggan untuk collaborative filtering.  
- **TF-IDF Vectorizer**: Mengubah deskripsi teks menjadi vektor numerik memungkinkan perhitungan kesamaan antar produk secara matematis, penting untuk content-based filtering.  

### 4.3. Hasil Data Preparation
- **Jumlah Baris Awal**: ~541.909.  
- **Jumlah Baris Setelah Pembersihan**: ~400.000 (tergantung data hilang dan pembatalan).  
- **Matriks User-Item**: Berisi ~4.000 pelanggan unik dan ~4.000 produk unik, dengan tingkat sparsity tinggi (~95% nilai nol).  
- **Matriks TF-IDF**: Berisi vektor untuk setiap produk unik berdasarkan deskripsi, dengan dimensi sesuai jumlah kata unik setelah pembersihan.  


## 5. Modeling and Result

### 5.1. Content-Based Filtering

#### Algoritma
- **Cosine Similarity**: Mengukur kesamaan antar produk berdasarkan vektor TF-IDF yang dihasilkan di tahap Data Preparation.  

#### Cara Kerja
1. **Input**: Matriks TF-IDF dari deskripsi produk, di mana setiap produk direpresentasikan sebagai vektor dalam ruang fitur kata.  
2. **Perhitungan Cosine Similarity**:  
   - Menghitung sudut kosinus antara setiap pasangan vektor produk menggunakan rumus:  
cosine similarity(A,B)= A*B/||A|| ||B||​ 
   - Menghasilkan matriks similarity di mana setiap elemen menunjukkan skor kesamaan (0 hingga 1) antar produk.  
3. **Rekomendasi**:  
   - Untuk produk input (misalnya, berdasarkan `StockCode`), memilih 5 produk dengan skor cosine similarity tertinggi.  
   - Mengembalikan deskripsi produk sebagai output rekomendasi.  

#### Parameter
- **cosine_similarity**: Menggunakan implementasi default dari scikit-learn tanpa parameter tambahan.  

#### Contoh Output
Untuk produk dengan `StockCode` "85123A":  
Rekomendasi untuk produk "white hanging heart tlight holder":
- red hanging heart tlight holder
- pink hanging heart tlight holder
- cream heart tlight holder
- zinc heart tlight holder
- glass heart tlight holder

### 5.2. Collaborative Filtering
#### Algoritma
- **Singular Value Decomposition (SVD)**: Mengurangi dimensi matriks user-item untuk menangkap pola laten.
- **Cosine Similarity**: Menghitung kesamaan antar pelanggan berdasarkan representasi laten.
#### Cara Kerja
1. **Input**: Matriks user-item dengan CustomerID sebagai baris, StockCode sebagai kolom, dan Quantity sebagai nilai.
SVD:
2. **Menguraikan matriks user-item menjadi tiga matriks**: (𝑈), (Σ), dan (V^T).
(𝑈): Representasi laten pelanggan.
(Σ): Nilai singular (bobot fitur laten).
(V^T): Representasi laten produk.
Mengurangi dimensi ke 20 fitur laten untuk menangkap pola utama sambil mengurangi noise.
3. **Cosine Similarity**:
Menghitung kesamaan antar pelanggan berdasarkan vektor laten di ruang SVD.
Menggunakan rumus cosine similarity seperti pada content-based filtering.
4. **Rekomendasi**:
Mengidentifikasi 10 pelanggan paling mirip dengan pelanggan input.
Mengagregasi pembelian pelanggan serupa untuk memilih top-5 produk dengan skor tertinggi (berdasarkan rata-rata kuantitas).
Mengembalikan deskripsi produk sebagai output rekomendasi.
#### Parameter
- **TruncatedSVD**:
- **n_components=20**: Jumlah fitur laten, dipilih untuk keseimbangan akurasi dan efisiensi.
- **random_state=42**: Menjamin reproduktibilitas.
#### Parameter default:
- **algorithm='randomized'**: Algoritma SVD cepat.
- **n_iter=5**: Jumlah iterasi untuk konvergensi.
- **cosine_similarity**: Default scikit-learn tanpa parameter tambahan.
#### Contoh Output
Untuk pelanggan dengan CustomerID "17850.0":
Rekomendasi untuk pelanggan 17850.0:
- jumbo bag red retrospot
- lunch bag red retrospot
- party bunting
- set of 6 spice tins pantry design
- pack of 72 retrospot cake cases

## 6. Evaluation
### 6.1. Metrik Evaluasi
Recall@5 digunakan untuk mengevaluasi performa sistem rekomendasi.

### Definisi
Recall@5 mengukur proporsi item relevan yang berhasil muncul dalam daftar rekomendasi top-5 dibandingkan dengan total item relevan yang seharusnya direkomendasikan.

### Formula:

Recall@5= Total item relevan / Jumlah item relevan dalam top-5

### Kriteria Relevansi
- **Content-Based**: Produk dianggap relevan jika deskripsinya memiliki setidaknya satu kata yang sama dengan produk input.
- **Collaborative**: Produk dianggap relevan jika pernah dibeli oleh pelanggan input.
### 6.2. Hasil Evaluasi
1. **Content-Based Filtering**:
- **Recall@5**: ~0.5
- **Analisis**: 50% produk relevan berdasarkan deskripsi berhasil direkomendasikan dalam top-5. Performa ini menunjukkan model efektif dalam mengidentifikasi produk serupa berdasarkan teks deskripsi, meskipun ada ruang untuk perbaikan dengan optimasi teks atau fitur tambahan.
2. **Collaborative Filtering**:
- **Recall@5**: ~1.0
- **Analisis**: 100% produk relevan berdasarkan riwayat pembelian berhasil direkomendasikan dalam top-5. Performa ini sangat baik, menunjukkan model mampu menangkap pola pembelian pelanggan secara akurat, meskipun hasil ini mungkin dipengaruhi oleh data pelanggan tertentu dengan riwayat pembelian yang kaya.
### 6.3. Interpretasi
1. **Content-Based Filtering**:
Efektif untuk merekomendasikan produk serupa berdasarkan fitur eksplisit (deskripsi), cocok untuk pelanggan yang menjelajahi produk tertentu.
Recall@5 sebesar 0.5 menunjukkan model dapat menangkap setengah dari produk relevan, tetapi kinerja dapat ditingkatkan dengan preprocessing teks yang lebih baik atau fitur tambahan seperti kategori produk.
2. **Collaborative Filtering**:
Lebih personal karena mempertimbangkan pola pembelian pelanggan serupa, sangat cocok untuk rekomendasi yang dipersonalisasi.
Recall@5 sebesar 1.0 menunjukkan performa luar biasa, tetapi perlu diverifikasi dengan data yang lebih besar untuk memastikan generalisasi, mengingat potensi sparsity dalam matriks user-item.
3. **Potensi Pengembangan**: Kombinasi kedua pendekatan (hybrid recommender system) dapat meningkatkan cakupan dan relevansi rekomendasi, menggabungkan kekuatan fitur eksplisit dan implisit.

## 7. Penutup
### 7.1. Kesimpulan
Proyek ini berhasil mengembangkan sistem rekomendasi produk menggunakan dataset E-Commerce Data dengan dua pendekatan:

1. Content-Based Filtering: Merekomendasikan produk serupa berdasarkan deskripsi dengan Recall@5 ~0.5, efektif untuk eksplorasi produk.
2. Collaborative Filtering: Merekomendasikan produk berdasarkan pola pembelian dengan Recall@5 ~1.0, sangat personal dan akurat untuk pelanggan dengan riwayat pembelian yang cukup.
Sistem ini mengatasi masalah kesulitan menemukan produk dan kurangnya personalisasi, mendukung tujuan untuk meningkatkan pengalaman belanja dan potensi konversi penjualan.

### 7.2. Rekomendasi
1. Optimasi Content-Based:
Tambahkan fitur seperti kategori produk atau metadata untuk meningkatkan Recall@5.
Eksperimen dengan teknik pemrosesan teks lain, seperti word embeddings, untuk menangkap makna semantik deskripsi.
2. Peningkatan Collaborative Filtering:
Atasi sparsity dengan menambahkan data tambahan, seperti rating eksplisit atau frekuensi pembelian.
Validasi Recall@5 yang tinggi dengan dataset yang lebih besar untuk memastikan generalisasi.
3. Pendekatan Hybrid: Kembangkan sistem rekomendasi hybrid yang menggabungkan content-based dan collaborative filtering untuk hasil yang lebih robust.
4. Implementasi Bisnis: Integrasikan model ke platform e-commerce melalui API untuk rekomendasi real-time, diikuti dengan pengujian A/B untuk mengukur dampak pada konversi dan kepuasan pelanggan.
