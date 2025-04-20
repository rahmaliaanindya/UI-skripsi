import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from PIL import Image

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
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# Menu navigasi disimpan di session_state
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

menu = st.session_state.menu

# === Navigasi Radio Menu (hanya untuk menampilkan posisi sekarang) ===
st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Upload Data", "Preprocessing Data", "Visualisasi Data", "Hasil Clustering"),
    horizontal=True,
    index=["Home", "Upload Data", "Preprocessing Data", "Visualisasi Data", "Hasil Clustering"].index(menu),
    key="menu_display",  # Tidak akan mempengaruhi session_state.menu
    disabled=True
)

# === 1. HOME ===
if menu == "Home":
    st.markdown("""
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊

    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode **Spectral Clustering**
    - 📈 Mengevaluasi hasil pengelompokan

    📌 Silakan klik tombol **Next ➡️** untuk memulai proses analisis.
    """)
    if st.button("Next ➡️"):
        st.session_state.menu = "Upload Data"
        st.experimental_rerun()

# === 2. UPLOAD DATA ===
elif menu == "Upload Data":
    st.header("📤 Upload Data Excel")

    st.markdown("""
    ### Ketentuan Data:
    - File **Excel (.xlsx)**
    - Kolom harus mencakup indikator kemiskinan sesuai petunjuk
    """)

    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        st.write(df)

        if st.button("Next ➡️"):
            st.session_state.menu = "Preprocessing Data"
            st.experimental_rerun()

# === 3. PREPROCESSING DATA ===
elif menu == "Preprocessing Data":
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
        st.success("✅ Fitur telah dinormalisasi dan disimpan.")

        if st.button("Next ➡️"):
            st.session_state.menu = "Visualisasi Data"
            st.experimental_rerun()
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === 4. VISUALISASI DATA ===
elif menu == "Visualisasi Data":
    st.header("📊 Visualisasi Data")

    if 'df' in st.session_state:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])

        st.subheader("Heatmap Korelasi")
        plt.figure(figsize=(10, 5))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()

        if st.button("Next ➡️"):
            st.session_state.menu = "Hasil Clustering"
            st.experimental_rerun()
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === 5. HASIL CLUSTERING ===
elif menu == "Hasil Clustering":
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

        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': dbi_scores
        })
        st.line_chart(score_df)

        best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
        best_k_dbi = min(dbi_scores, key=dbi_scores.get)

        st.success(f"🔹 Optimal cluster (Silhouette): {best_k_silhouette}")
        st.success(f"🔸 Optimal cluster (DBI): {best_k_dbi}")

        st.subheader("Pilih Jumlah Cluster untuk Clustering Final")
        k_final = st.number_input("Jumlah Cluster (k):", min_value=2, max_value=10, value=best_k_silhouette, step=1)

        final_cluster = SpectralClustering(n_clusters=k_final, affinity='nearest_neighbors', random_state=42)
        labels = final_cluster.fit_predict(X_scaled)
        st.session_state.labels = labels

        # Visualisasi dengan PCA
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

        # Tampilkan DataFrame hasil clustering
        if 'df' in st.session_state:
            df = st.session_state.df.copy()
            df['Cluster'] = labels

            st.subheader("📄 Hasil Cluster pada Data")
            df_sorted = df.sort_values(by='Cluster')
            st.dataframe(df_sorted)

            st.subheader("📊 Jumlah Anggota per Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
    else:
        st.warning("⚠️ Data belum diproses. Silakan lakukan preprocessing terlebih dahulu.")
