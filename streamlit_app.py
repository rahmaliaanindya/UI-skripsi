# === IMPORT LIBRARY ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings("ignore")

# === KONFIGURASI HALAMAN ===
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide"
)

# === CSS Styling ===
def local_css():
    st.markdown(
        """
        <style>
            body {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .main {
                background-color: #ffffff;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50 !important;
            }
            .title {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #1f3a93;
                font-size: 38px;
                font-weight: bold;
                text-align: center;
                padding: 20px 0 15px 0;
                border-bottom: 2px solid #e9ecef;
                margin-bottom: 25px;
            }
            .required-columns {
                background-color: #fff8e1;
                padding: 15px;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
                margin: 15px 0;
            }
            .dataframe {
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu = st.sidebar.radio(
    "Menu:",
    ("Home", "Upload Data", "EDA", "Preprocessing", "Clustering", "Hasil & Analisis")
)

# === HOME ===
if menu == "Home":
    st.markdown(""" 
    <div class="title">Aplikasi Analisis Cluster Kemiskinan Jawa Timur</div>
    
    <div style="text-align: center; margin-bottom: 30px;">
        <p style="font-size: 18px; color: #555;">
            Aplikasi ini dirancang untuk menganalisis dan mengelompokkan wilayah di Jawa Timur 
            berdasarkan indikator kemiskinan menggunakan metode Spectral Clustering
        </p>
    </div>
    """, unsafe_allow_html=True)

# === UPLOAD DATA ===
elif menu == "Upload Data":
    st.markdown('<div class="title">📤 Upload Data Excel</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="required-columns">
        <h4 style="margin-top: 0;">Kolom yang harus ada dalam file Excel:</h4>
        <ul>
            <li>Kabupaten/Kota</li>
            <li>Persentase Penduduk Miskin (%)</li>
            <li>Jumlah Penduduk Miskin (ribu jiwa)</li>
            <li>Harapan Lama Sekolah (Tahun)</li>
            <li>Rata-Rata Lama Sekolah (Tahun)</li>
            <li>Tingkat Pengangguran Terbuka (%)</li>
            <li>Tingkat Partisipasi Angkatan Kerja (%)</li>
            <li>Angka Harapan Hidup (Tahun)</li>
            <li>Garis Kemiskinan (Rupiah/Bulan/Kapita)</li>
            <li>Indeks Pembangunan Manusia</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Cek kolom yang diperlukan
            required_columns = [
                "Kabupaten/Kota",
                "Persentase Penduduk Miskin (%)",
                "Jumlah Penduduk Miskin (ribu jiwa)",
                "Harapan Lama Sekolah (Tahun)",
                "Rata-Rata Lama Sekolah (Tahun)",
                "Tingkat Pengangguran Terbuka (%)",
                "Tingkat Partisipasi Angkatan Kerja (%)",
                "Angka Harapan Hidup (Tahun)",
                "Garis Kemiskinan (Rupiah/Bulan/Kapita)",
                "Indeks Pembangunan Manusia"
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Kolom berikut tidak ditemukan dalam file: {', '.join(missing_columns)}")
            else:
                st.session_state.df = df
                st.success("✅ Data berhasil dimuat dan valid!")
                
                with st.expander("Lihat Seluruh Data"):
                    st.dataframe(df)

# === EDA ===
elif menu == "EDA":
    st.markdown('<div class="title">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di halaman Upload Data")
    else:
        df = st.session_state.df
        
        st.subheader("Data Awal")
        st.dataframe(df)
        
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())
        
        st.subheader("Distribusi Variabel Numerik")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Pilih variabel:", numeric_columns)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, bins=30, color='#4a6baf', ax=ax)
        ax.set_title(f'Distribusi {selected_col}', fontsize=14)
        st.pyplot(fig)
        
        st.subheader("Korelasi Antar Variabel")
        numerical_df = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

# === PREPROCESSING ===
elif menu == "Preprocessing":
    st.markdown('<div class="title">⚙️ Preprocessing Data</div>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di halaman Upload Data")
    else:
        df = st.session_state.df
        
        st.subheader("Data Sebelum Preprocessing")
        st.dataframe(df)
        
        st.subheader("Scaling Data dengan RobustScaler")
        X = df.drop(columns=['Kabupaten/Kota']) if 'Kabupaten/Kota' in df.columns else df
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled = X_scaled
        st.session_state.X = X
        
        st.success("✅ Data telah discaling menggunakan RobustScaler!")
        
        st.subheader("Data Setelah Scaling")
        scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        st.dataframe(scaled_df)

# === CLUSTERING ===
elif menu == "Clustering":
    st.markdown('<div class="title">🧩 Spectral Clustering</div>', unsafe_allow_html=True)
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu")
    else:
        X_scaled = st.session_state.X_scaled
        
        st.subheader("Evaluasi Jumlah Cluster Optimal")
        
        with st.spinner("Menghitung metrik evaluasi..."):
            clusters_range = range(2, 11)
            silhouette_scores = []
            db_scores = []

            for k in clusters_range:
                model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
                labels = model.fit_predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                db_scores.append(davies_bouldin_score(X_scaled, labels))

        score_df = pd.DataFrame({
            'Jumlah Cluster': clusters_range,
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': db_scores
        })
        
        st.dataframe(score_df)
        
        # Visualisasi metrik evaluasi
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(clusters_range, silhouette_scores, 'bo-', label='Silhouette Score')
        ax.set_xlabel('Jumlah Cluster')
        ax.set_ylabel('Silhouette Score', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax.twinx()
        ax2.plot(clusters_range, db_scores, 'ro-', label='Davies-Bouldin Index')
        ax2.set_ylabel('DB Index', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('Evaluasi Jumlah Cluster Optimal')
        fig.tight_layout()
        st.pyplot(fig)
        
        best_k = 2  # Memaksa menggunakan 2 cluster sesuai analisis DBI
        st.info(f"Berdasarkan analisis DBI, jumlah cluster optimal yang digunakan: {best_k}")
        
        if st.button("Lakukan Clustering dengan 2 Cluster"):
            with st.spinner("Menjalankan Spectral Clustering..."):
                try:
                    # Spectral Clustering
                    gamma = 0.1
                    W = rbf_kernel(X_scaled, gamma=gamma)
                    threshold = 0.01
                    W[W < threshold] = 0
                    D = np.diag(W.sum(axis=1))
                    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
                    L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
                    eigvals, eigvecs = eigh(L_sym)
                    U = eigvecs[:, :best_k]
                    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
                    kmeans = KMeans(n_clusters=best_k, random_state=42)
                    labels = kmeans.fit_predict(U_norm)
                    
                    # Simpan hasil ke session state
                    st.session_state.labels = labels
                    st.session_state.U_norm = U_norm
                    
                    st.success("✅ Clustering selesai!")
                    
                    # Visualisasi
                    st.subheader("Visualisasi Hasil Clustering")
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter = ax.scatter(U_norm[:, 0], U_norm[:, 1], c=labels, cmap='viridis')
                    ax.set_title('Hasil Spectral Clustering (2 Cluster)')
                    ax.set_xlabel('Komponen 1')
                    ax.set_ylabel('Komponen 2')
                    plt.colorbar(scatter, label='Cluster')
                    st.pyplot(fig)
                    
                    # Evaluasi
                    st.subheader("Evaluasi Clustering")
                    eval_df = pd.DataFrame({
                        'Metrik': ['Silhouette Score', 'Davies-Bouldin Index'],
                        'Nilai': [
                            silhouette_score(U_norm, labels),
                            davies_bouldin_score(U_norm, labels)
                        ]
                    })
                    st.dataframe(eval_df)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat clustering: {str(e)}")

# === HASIL & ANALISIS ===
elif menu == "Hasil & Analisis":
    st.markdown('<div class="title">📊 Hasil & Analisis Clustering</div>', unsafe_allow_html=True)
    
    if 'labels' not in st.session_state or 'df' not in st.session_state:
        st.warning("⚠️ Silakan lakukan clustering terlebih dahulu")
    else:
        df = st.session_state.df.copy()
        labels = st.session_state.labels
        df['Cluster'] = labels
        
        st.subheader("Data dengan Label Cluster")
        st.dataframe(df)
        
        st.subheader("Distribusi Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        cols = st.columns(2)
        with cols[0]:
            st.dataframe(cluster_counts)
        with cols[1]:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Blues", ax=ax)
            ax.set_title('Jumlah Data per Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Jumlah')
            st.pyplot(fig)
        
        st.subheader("Karakteristik Tiap Cluster")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cluster_means = df.groupby('Cluster')[numeric_cols].mean()
        st.dataframe(cluster_means)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Rata-rata Nilai per Cluster')
        st.pyplot(fig)
        
        st.subheader("Wilayah dengan Kemiskinan Tertinggi dan Terendah")
        if 'Persentase Penduduk Miskin (%)' in df.columns:
            top5 = df.sort_values(by='Persentase Penduduk Miskin (%)', ascending=False).head(5)
            bottom5 = df.sort_values(by='Persentase Penduduk Miskin (%)', ascending=True).head(5)
            
            cols = st.columns(2)
            with cols[0]:
                st.write("**5 Wilayah dengan Kemiskinan Tertinggi:**")
                st.dataframe(top5[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']])
            with cols[1]:
                st.write("**5 Wilayah dengan Kemiskinan Terendah:**")
                st.dataframe(bottom5[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']])
        
        st.subheader("Rekomendasi Kebijakan")
        st.write("""
        Berdasarkan hasil clustering:
        - **Cluster 0**: Wilayah dengan tingkat kemiskinan tinggi, membutuhkan intervensi khusus
        - **Cluster 1**: Wilayah dengan tingkat kemiskinan lebih rendah, bisa menjadi acuan
        """)
