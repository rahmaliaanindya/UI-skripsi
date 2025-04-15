import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Konfigurasi halaman
st.set_page_config(
    page_title="Insight Predict",
    page_icon="📊",
    layout="wide"
)

# CSS Styling
def local_css():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                background-color: #cdd4b1;
            }
            [data-testid="stAppViewContainer"] {
                background-color: #feecd0;
            }
            h1, h2, h3, h4, h5, h6, p, div, span {
                color: #4a4a4a !important;
            }
            .welcome-text {
                font-size: 22px;
                font-weight: bold;
                color: #4a4a4a;
                text-align: center;
                background-color: #f5deb3;
                padding: 15px;
                border-radius: 10px;
            }
        </style>
        """, unsafe_allow_html=True
    )

local_css()

# Sidebar Menu
menu = st.sidebar.radio("Navigasi", (
    "Beranda", 
    "Upload Data", 
    "Preprocessing Data", 
    "Visualisasi Data", 
    "Hasil Clustering"
))

# Session state untuk menyimpan data antar halaman
if "df" not in st.session_state:
    st.session_state.df = None
if "X_scaled" not in st.session_state:
    st.session_state.X_scaled = None
if "labels" not in st.session_state:
    st.session_state.labels = None

# 1. BERANDA
if menu == "Beranda":
    st.markdown('<div class="welcome-text">Selamat datang di aplikasi Insight Predict 📊</div>', unsafe_allow_html=True)
    st.markdown("Aplikasi ini membantu analisis dan klasterisasi data indikator kemiskinan secara interaktif.")

# 2. UPLOAD DATA
elif menu == "Upload Data":
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil dimuat!")
        st.write(df)

# 3. PREPROCESSING
elif menu == "Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    if st.session_state.df is not None:
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
elif menu == "Visualisasi Data":
    st.header("📊 Visualisasi Data")
    if st.session_state.df is not None:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])

        st.subheader("Heatmap Korelasi")
        plt.figure(figsize=(10, 5))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()

        st.subheader("Distribusi Fitur (Histogram)")
        selected_feature = st.selectbox("Pilih fitur:", numerical_df.columns)
        plt.hist(df[selected_feature], bins=20, color='skyblue', edgecolor='black')
        plt.title(f'Distribusi: {selected_feature}')
        st.pyplot(plt.gcf())
        plt.clf()
    else:
        st.warning("Silakan upload data terlebih dahulu.")

# 5. HASIL CLUSTERING
elif menu == "Hasil Clustering":
    st.header("🧩 Hasil Clustering")
    if st.session_state.X_scaled is not None:
        X_scaled = st.session_state.X_scaled
        st.subheader("Evaluasi Jumlah Cluster (Silhouette & DBI)")

        clusters_range = range(2, 6)
        silhouette_scores = {}
        dbi_scores = {}

        for k in clusters_range:
            clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
            labels = clustering.fit_predict(X_scaled)
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
            dbi_scores[k] = davies_bouldin_score(X_scaled, labels)

        st.line_chart(pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': dbi_scores
        }))

        best_k = max(silhouette_scores, key=silhouette_scores.get)
        st.success(f"Jumlah cluster optimal berdasarkan Silhouette Score: {best_k}")

        # Final Clustering
        final_cluster = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', random_state=42)
        labels = final_cluster.fit_predict(X_scaled)
        st.session_state.labels = labels

        # Visualisasi 2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
        plt.title("Visualisasi Clustering (PCA)")
        st.pyplot(plt.gcf())
        plt.clf()

        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            df['Cluster'] = labels
            st.dataframe(df[['Cluster'] + list(df.columns[:3])])
    else:
        st.warning("Data belum diproses. Silakan lakukan preprocessing terlebih dahulu.")
