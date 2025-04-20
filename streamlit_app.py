import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

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
            .nav-button {
                display: flex;
                justify-content: space-between;
                margin-top: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Terapkan CSS
local_css()

# === Navigasi Menu di Atas ===
menu_options = ["Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", 
                "Step 3: Visualisasi Data", "Step 4: Hasil Clustering", "Step 5: Analisis Hasil"]
menu = st.radio(
    "Navigasi Aplikasi:",
    menu_options,
    horizontal=True
)

# Fungsi navigasi
def create_nav_buttons(current_step):
    if current_step == "Home":
        if st.button("Mulai Analisis →", key="home_next"):
            st.session_state.menu = "Step 1: Upload Data"
            st.experimental_rerun()
    
    elif current_step in menu_options[1:-1]:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Kembali", key=f"{current_step}_back"):
                prev_index = menu_options.index(current_step) - 1
                st.session_state.menu = menu_options[prev_index]
                st.experimental_rerun()
        with col2:
            if st.button("Lanjut →", key=f"{current_step}_next"):
                next_index = menu_options.index(current_step) + 1
                st.session_state.menu = menu_options[next_index]
                st.experimental_rerun()
    
    elif current_step == "Step 5: Analisis Hasil":
        if st.button("← Kembali ke Clustering", key="analysis_back"):
            st.session_state.menu = "Step 4: Hasil Clustering"
            st.experimental_rerun()

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
    - 🔍 Menganalisis karakteristik cluster

    📌 Silakan pilih menu di atas atau klik tombol di bawah untuk memulai analisis.
    """)
    create_nav_buttons(menu)

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
    
    create_nav_buttons(menu)

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
        st.session_state.features = features
        st.success("Fitur telah dinormalisasi dan disimpan.")
    else:
        st.warning("Silakan upload data terlebih dahulu.")
    
    create_nav_buttons(menu)

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
    
    create_nav_buttons(menu)

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

        # Tentukan jumlah cluster final
        optimal_k = best_k_silhouette
        clustering = SpectralClustering(n_clusters=optimal_k, affinity='nearest_neighbors', random_state=42)
        labels = clustering.fit_predict(X_scaled)
        df['Cluster'] = labels

        st.write(df)
        
        st.session_state.df_clustered = df
    else:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu.")
    
    create_nav_buttons(menu)

# 6. ANALISIS HASIL CLUSTERING
elif menu == "Step 5: Analisis Hasil":
    st.header("🔍 Analisis Hasil Clustering")
    
    if 'df_clustered' in st.session_state:
        df = st.session_state.df_clustered

        st.subheader("Visualisasi Hasil Clustering")

        # Visualisasi dengan scatter plot atau box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="Harapan Lama Sekolah (Tahun)", y="Tingkat Pengangguran Terbuka (%)", hue="Cluster", palette="viridis", ax=ax)
        st.pyplot(fig)

        # Menampilkan karakteristik setiap cluster
        st.subheader("Karakteristik Cluster")

        for cluster_num in df['Cluster'].unique():
            st.markdown(f"### Cluster {cluster_num}")
            cluster_data = df[df['Cluster'] == cluster_num]
            mean_values = cluster_data.mean()
            st.write(f"Rata-rata nilai fitur pada Cluster {cluster_num}:")
            st.write(mean_values)

            st.markdown("**Fitur tertinggi:**")
            for i, (ind, val) in enumerate(mean_values.head(3).items()):
                st.write(f"{i+1}. {ind}: {val:.2f}")
            
            st.markdown("**Fitur terendah:**")
            for i, (ind, val) in enumerate(mean_values.tail(3).items()):
                st.write(f"{i+1}. {ind}: {val:.2f}")

    else:
        st.warning("⚠️ Hasil clustering belum tersedia. Silakan lakukan clustering terlebih dahulu.")
    
    create_nav_buttons(menu)
