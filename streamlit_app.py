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

# === Variabel Navigasi Manual ===
if 'stage' not in st.session_state:
    st.session_state.stage = 0  # Mulai dari tahap 0 (Home)

# Fungsi untuk melanjutkan ke tahap berikutnya
def next_stage():
    if st.session_state.stage < 4:
        st.session_state.stage += 1

# === Menu Navigasi di Sidebar ===
menu = ["Home", "Upload Data", "Preprocessing", "Visualisasi", "Hasil Clustering"]
selection = st.sidebar.radio("Pilih Tahap", menu, index=st.session_state.stage)

# Fungsi untuk menentukan tampilan berdasarkan menu yang dipilih
if selection == "Home":
    st.session_state.stage = 0
elif selection == "Upload Data":
    st.session_state.stage = 1
elif selection == "Preprocessing":
    st.session_state.stage = 2
elif selection == "Visualisasi":
    st.session_state.stage = 3
elif selection == "Hasil Clustering":
    st.session_state.stage = 4

# === Konten berdasarkan Menu ===
if st.session_state.stage == 0:
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
    if st.button('Next'):
        next_stage()

elif st.session_state.stage == 1:
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

    if st.button('Next'):
        next_stage()

elif st.session_state.stage == 2:
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
    
    if st.button('Next'):
        next_stage()

elif st.session_state.stage == 3:
    st.header("📊 Visualisasi Data")
    if 'df' in st.session_state:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])

        st.subheader("Heatmap Korelasi")
        plt.figure(figsize=(10, 5))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()
    
    if st.button('Next'):
        next_stage()

elif st.session_state.stage == 4:
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
