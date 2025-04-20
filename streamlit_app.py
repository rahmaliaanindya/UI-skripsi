import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide"
)

# CSS Styling
def local_css():
    st.markdown(
        """
        <style>
            body {
                background-color: #fdf0ed;
            }
            .main {
                background: linear-gradient(to bottom right, #e74c3c, #f39c12, #f8c471);
            }
            .block-container {
                padding-top: 1rem;
                background-color: transparent;
            }
            h1, h2, h3, h4, h5, h6, p, div, span {
                color: #2c3e50 !important;
            }
            .title {
                font-family: 'Helvetica', sans-serif;
                color: #1f3a93;
                font-size: 38px;
                font-weight: bold;
                text-align: center;
                padding: 30px 0 10px 0;
            }
            .sidebar .sidebar-content {
                background-color: #fef5e7;
            }
            .legend-box {
                padding: 15px;
                border-radius: 10px;
                background-color: #ffffffdd;
                box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
                margin-top: 20px;
            }
            .info-card {
                background-color: #ffffffaa;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 25px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Terapkan CSS
local_css()

# === Navigasi Menu di Atas ===
menu = st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering", "Step 5: Analisis Hasil"),
    horizontal=True
)

# === Konten berdasarkan Menu ===
if menu == "Home":
    st.markdown(""" 
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊
    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode **Spectral Clustering**
    - 📈 Mengevaluasi hasil pengelompokan
    📌 Silakan pilih menu di atas untuk memulai analisis.
    """)

# 2. UPLOAD DATA
elif menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")

    # Deskripsi tentang data yang harus diunggah
    st.markdown("""
    ### Ketentuan Data:
    - Data berupa file **Excel (.xlsx)**.
    - Data mencakup kolom-kolom berikut:
        1. **Persentase Penduduk Miskin (%)**
        2. **Jumlah Penduduk Miskin (ribu jiwa)**
        3. **Harapan Lama Sekolah (Tahun)**
        4. **Rata-Rata Lama Sekolah (Tahun)**
        5. **Tingkat Pengangguran Terbuka (%)**
        6. **Tingkat Partisipasi Angkatan Kerja (%)**
        7. **Angka Harapan Hidup (Tahun)**
        8. **Garis Kemiskinan (Rupiah/Bulan/Kapita)**
        9. **Indeks Pembangunan Manusia**
        10. **Rata-rata Upah/Gaji Bersih Pekerja Informal Berdasarkan Lapangan Pekerjaan Utama (Rp)**
        11. **Rata-rata Pendapatan Bersih Sebulan Pekerja Informal berdasarkan Pendidikan Tertinggi - Jumlah (Rp)**
    """)

    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil dimuat!")
        st.write(df)

# 3. PREPROCESSING
elif menu == "Step 2: Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    if 'df' in st.session_state:
        df = st.session_state.df
        st.subheader("Cek Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Cek Duplikat")
        st.write(f"Jumlah duplikat: {df.duplicated().sum()}")

        st.subheader("Statistik Deskriptif")
        st.write(df.describe())

        st.subheader("Normalisasi dan Seleksi Fitur")
        features = df.select_dtypes(include=['float64', 'int64']).columns
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.session_state.X_scaled = X_scaled
        st.write("Fitur telah dinormalisasi dan disimpan.")
    else:
        st.warning("Silakan upload data terlebih dahulu.")

# 4. VISUALISASI DATA
elif menu == "Step 3: Visualisasi Data":
    st.header("📊 Visualisasi Data")
    if 'df' in st.session_state:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])

        st.subheader("Heatmap Korelasi")
        plt.figure(figsize=(10, 5))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()

    else:
        st.warning("Silakan upload data terlebih dahulu.")

# 5. HASIL CLUSTERING
elif menu == "Step 4: Hasil Clustering":
    st.header("🧩 Hasil Clustering")
    
    if 'X_scaled' in st.session_state:
        X_scaled = st.session_state.X_scaled
        st.subheader("Evaluasi Jumlah Cluster (Silhouette & DBI)")

        clusters_range = range(2, 10)
        silhouette_scores = {}
        dbi_scores = {}

        for k in clusters_range:
            clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
            labels = clustering.fit_predict(X_scaled)
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
            dbi_scores[k] = davies_bouldin_score(X_scaled, labels)

        # Tampilkan grafik evaluasi
        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': dbi_scores
        })
        st.line_chart(score_df)

        # Menentukan cluster terbaik dari dua metrik
        best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
        best_k_dbi = min(dbi_scores, key=dbi_scores.get)

        st.success(f"🔹 Jumlah cluster optimal berdasarkan **Silhouette Score**: {best_k_silhouette}")
        st.success(f"🔸 Jumlah cluster optimal berdasarkan **Davies-Bouldin Index**: {best_k_dbi}")

        # Pilihan manual untuk k_final atau default ke Silhouette
        st.subheader("Pilih Jumlah Cluster untuk Clustering Final")
        k_final = st.number_input("Jumlah Cluster (k):", min_value=2, max_value=10, value=best_k_silhouette, step=1)

        # Final Clustering
        final_cluster = SpectralClustering(n_clusters=k_final, affinity='nearest_neighbors', random_state=42)
        labels = final_cluster.fit_predict(X_scaled)
        st.session_state.labels = labels

        # Visualisasi 2D menggunakan PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.subheader("Visualisasi Clustering (PCA)")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
        plt.title("Visualisasi Clustering dengan Spectral Clustering")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        st.pyplot(plt.gcf())
        plt.clf()

        # Menampilkan hasil clustering
        if 'df' in st.session_state:
            df = st.session_state.df.copy()
            df['Cluster'] = labels

            st.subheader("📄 Hasil Cluster pada Data")

            # Urutkan data berdasarkan 'Cluster'
            df_sorted = df.sort_values(by='Cluster')

            # Tampilkan DataFrame yang sudah diurutkan
            st.dataframe(df_sorted)

            # Tampilkan jumlah anggota tiap cluster
            st.subheader("📊 Jumlah Anggota per Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)

    else:
        st.warning("⚠️ Data belum diproses. Silakan lakukan preprocessing terlebih dahulu.")

# 6. ANALISIS HASIL
elif menu == "Step 5: Analisis Hasil":
    st.header("🔍 Analisis Hasil Clustering")

    if 'labels' in st.session_state and 'df' in st.session_state:
        df = st.session_state.df.copy()
        labels = st.session_state.labels
        df['Cluster'] = labels

        # Hanya ambil kolom numerik untuk analisis
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_numeric = df[numeric_cols]

        # Analisis distribusi cluster
        st.subheader("Analisis Distribusi Cluster")
        cluster_summary = df_numeric.groupby('Cluster').mean()
        st.write(cluster_summary)

        # Visualisasi distribusi cluster
        st.subheader("Distribusi Cluster (Bar Chart)")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        # Insight berdasarkan cluster
        st.subheader("Insight per Cluster")
        for cluster_num in df['Cluster'].unique():
            st.write(f"Cluster {cluster_num}:")
            cluster_data = df[df['Cluster'] == cluster_num]
            st.write(cluster_data.describe())

        # --- Feature Importance using RandomForestClassifier ---
        # Cek apakah 'cluster_sorted' ada dalam dataframe
        if "cluster_sorted" in df.columns:  # Change df_cleaned to df
            df = df.drop(columns=["cluster_sorted"])

        # Definisi variabel X dan y
        X = df.drop(columns=["Kabupaten/Kota", "Cluster"], errors="ignore")  # Change df_cleaned to df, Hapus "Cluster" jika ada
        y = df["Cluster"]  # Change df_cleaned to df, Gunakan "Cluster" sebagai target

        # Import the RandomForestClassifier
        from sklearn.ensemble import RandomForestClassifier

        # Inisialisasi dan pelatihan model RandomForest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        # Menampilkan feature importance
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

        # Visualisasi Feature Importance
        st.subheader("Pentingnya Fitur (Feature Importance)")
        plt.figure(figsize=(10,5))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis")
        plt.title("Feature Importance (Indikator Paling Berpengaruh)")
        plt.xlabel("Tingkat Pengaruh")
        st.pyplot()

        # --- Principal Component Analysis (PCA) ---
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Standarisasi data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA dengan 2 komponen utama
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(X_scaled)

        # Visualisasi kontribusi indikator terhadap komponen utama
        st.subheader("Kontribusi Indikator terhadap Komponen Utama dengan PCA")
        plt.figure(figsize=(8,6))
        plt.bar(range(len(X.columns)), pca.components_[0], tick_label=X.columns)
        plt.xticks(rotation=90)
        plt.title("Kontribusi Indikator terhadap Komponen Utama")
        plt.ylabel("Kontribusi")
        st.pyplot()

        # --- Menampilkan 3 Kota Termiskin dan 3 Kota Terbaik ---
        # Menampilkan 3 Kota Termiskin dan 3 Kota Terbaik berdasarkan "Cluster"
        cluster_0 = df[df['Cluster'] == 0]  # Cluster 0 adalah kelompok yang memiliki tingkat kemiskinan tinggi
        cluster_1 = df[df['Cluster'] == 1]  # Cluster 1 adalah kelompok yang memiliki tingkat kemiskinan rendah

        # 3 Kota Termiskin (Cluster 0)
        st.subheader("3 Kota Termiskin (Cluster 0)")
        top_3_poor = cluster_0[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)']].sort_values(by="Persentase Penduduk Miskin (%)", ascending=False).head(3)
        st.write(top_3_poor)

        # 3 Kota Tidak Miskin (Cluster 1)
        st.subheader("3 Kota Tidak Miskin (Cluster 1)")
        top_3_rich = cluster_1[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)']].sort_values(by="Persentase Penduduk Miskin (%)", ascending=True).head(3)
        st.write(top_3_rich)

    else:
        st.warning("⚠️ Hasil clustering belum ada. Silakan lakukan clustering terlebih dahulu.")
