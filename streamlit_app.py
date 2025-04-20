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

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide"
)

# CSS Styling
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        h1, h2, h3 {
            color: #2c3e50;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        .stRadio > div {
            flex-direction: row !important;
        }
        .stRadio > div > label {
            margin-right: 15px;
        }
        .info-box {
            background-color: #e8f4fc;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# =============================================
# NAVIGASI MENU
# =============================================
menu = st.radio(
    "PILIH MENU:",
    ("Home", "Upload Data", "Preprocessing", "Visualisasi", "Clustering", "Analisis Hasil"),
    horizontal=True
)

# =============================================
# HOME PAGE
# =============================================
if menu == "Home":
    st.title("📊 Aplikasi Analisis Cluster Kemiskinan Jawa Timur")
    
    st.markdown("""
    <div class="info-box">
    <h3>🛠️ Panduan Penggunaan Aplikasi</h3>
    <ol>
        <li><b>Upload Data</b>: Unggah dataset dalam format Excel</li>
        <li><b>Preprocessing</b>: Bersihkan dan persiapkan data</li>
        <li><b>Visualisasi</b>: Eksplorasi data dengan grafik</li>
        <li><b>Clustering</b>: Kelompokkan data menggunakan Spectral Clustering</li>
        <li><b>Analisis Hasil</b>: Interpretasi hasil clustering</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("https://i.imgur.com/Jb7WQBW.png", caption="Workflow Analisis Data")

# =============================================
# UPLOAD DATA
# =============================================
elif menu == "Upload Data":
    st.header("📤 Upload Data Excel")
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"])
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("Data berhasil diunggah!")
            
            st.subheader("Preview Data")
            st.dataframe(df.head())
            
            st.subheader("Informasi Dataset")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dimensi Data:**")
                st.write(f"{df.shape[0]} baris × {df.shape[1]} kolom")
                
            with col2:
                st.write("**Kolom Numerik:**")
                st.write(df.select_dtypes(include=['int64','float64']).columns.tolist())
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# =============================================
# PREPROCESSING
# =============================================
elif menu == "Preprocessing":
    st.header("⚙️ Preprocessing Data")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu di menu Upload Data")
    else:
        df = st.session_state.df.copy()
        
        st.subheader("1. Pengecekan Data")
        tab1, tab2, tab3 = st.tabs(["Missing Values", "Duplikat", "Statistik"])
        
        with tab1:
            st.write("**Jumlah Missing Values per Kolom:**")
            st.write(df.isnull().sum())
            
        with tab2:
            st.write(f"**Jumlah Data Duplikat:** {df.duplicated().sum()}")
            
        with tab3:
            st.write("**Statistik Deskriptif:**")
            st.write(df.describe())
        
        st.subheader("2. Normalisasi Data")
        if st.button("Lakukan Normalisasi"):
            numeric_cols = df.select_dtypes(include=['float64','int64']).columns
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            
            st.session_state.df_normalized = df
            st.success("Data berhasil dinormalisasi!")
            st.write(df.head())

# =============================================
# VISUALISASI DATA
# =============================================
elif menu == "Visualisasi":
    st.header("📊 Visualisasi Data")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
    else:
        df = st.session_state.df
        
        st.subheader("1. Heatmap Korelasi")
        plt.figure(figsize=(12,8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
        
        st.subheader("2. Distribusi Variabel")
        selected_col = st.selectbox("Pilih kolom:", df.select_dtypes(include=['float64','int64']).columns)
        
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        sns.histplot(df[selected_col], kde=True, ax=ax[0])
        sns.boxplot(x=df[selected_col], ax=ax[1])
        st.pyplot(fig)

# =============================================
# CLUSTERING
# =============================================
elif menu == "Clustering":
    st.header("🧩 Spectral Clustering")
    
    if 'df_normalized' not in st.session_state:
        st.warning("Silakan lakukan normalisasi data terlebih dahulu")
    else:
        df = st.session_state.df_normalized
        X = df.select_dtypes(include=['float64','int64'])
        
        st.subheader("1. Evaluasi Jumlah Cluster Optimal")
        
        silhouette_scores = []
        db_scores = []
        cluster_range = range(2, 10)
        
        for k in cluster_range:
            sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
            labels = sc.fit_predict(X)
            silhouette_scores.append(silhouette_score(X, labels))
            db_scores.append(davies_bouldin_score(X, labels))
        
        fig, ax = plt.subplots(1,2, figsize=(15,5))
        ax[0].plot(cluster_range, silhouette_scores, marker='o')
        ax[0].set_title("Silhouette Score")
        ax[1].plot(cluster_range, db_scores, marker='o')
        ax[1].set_title("Davies-Bouldin Score")
        st.pyplot(fig)
        
        optimal_k = st.slider("Pilih jumlah cluster:", 2, 10, 3)
        
        if st.button("Lakukan Clustering"):
            sc = SpectralClustering(n_clusters=optimal_k, affinity='nearest_neighbors', random_state=42)
            labels = sc.fit_predict(X)
            
            st.session_state.labels = labels
            st.session_state.optimal_k = optimal_k
            st.success(f"Clustering berhasil dengan {optimal_k} cluster!")
            
            # Visualisasi PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            plt.figure(figsize=(10,6))
            sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette="viridis", s=100)
            plt.title("Visualisasi Cluster (PCA)")
            st.pyplot(plt)

# =============================================
# ANALISIS HASIL
# =============================================
elif menu == "Analisis Hasil":
    st.header("🔍 Analisis Hasil Clustering")
    
    if 'labels' not in st.session_state:
        st.warning("Silakan lakukan clustering terlebih dahulu")
    else:
        df = st.session_state.df.copy()
        labels = st.session_state.labels
        optimal_k = st.session_state.optimal_k
        
        df['Cluster'] = labels
        
        st.subheader("1. Distribusi Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Jumlah Data per Cluster:**")
            st.write(cluster_counts)
        
        with col2:
            plt.figure(figsize=(8,4))
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis")
            plt.title("Distribusi Cluster")
            st.pyplot(plt)
        
        st.subheader("2. Karakteristik Cluster")
        
        # Analisis fitur penting
        X = df.select_dtypes(include=['float64','int64']).drop('Cluster', axis=1, errors='ignore')
        y = df['Cluster']
        
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X,y)
        
        importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        plt.figure(figsize=(10,6))
        sns.barplot(x=importance.values, y=importance.index, palette="viridis")
        plt.title("Feature Importance")
        st.pyplot(plt)
        
        st.subheader("3. Profil Tiap Cluster")
        
        for cluster in range(optimal_k):
            with st.expander(f"Cluster {cluster}"):
                cluster_data = df[df['Cluster'] == cluster]
                
                st.write(f"**Jumlah Wilayah:** {len(cluster_data)}")
                st.write("**Contoh Wilayah:**")
                st.write(cluster_data['Kabupaten/Kota'].head().tolist())
                
                st.write("**Rata-rata Indikator:**")
                st.write(cluster_data.mean(numeric_only=True))
