# === IMPORT LIBRARY ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
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
            /* CSS styling tetap sama */
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

try:
    # === HOME ===
    if menu == "Home":
        st.markdown(""" 
        <div class="title">Aplikasi Analisis Cluster Kemiskinan Jawa Timur</div>
        """, unsafe_allow_html=True)

    # === UPLOAD DATA ===
    elif menu == "Upload Data":
        st.markdown('<div class="title">📤 Upload Data Excel</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.success("✅ Data berhasil dimuat!")
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")

    # === EDA ===
    elif menu == "EDA":
        st.markdown('<div class="title">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        if 'df' in st.session_state:
            df = st.session_state.df
            st.dataframe(df)
            
            # Visualisasi EDA
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            selected_col = st.selectbox("Pilih variabel:", numeric_cols)
            
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)

    # === PREPROCESSING ===
    elif menu == "Preprocessing":
        st.markdown('<div class="title">⚙️ Preprocessing Data</div>', unsafe_allow_html=True)
        
        if 'df' in st.session_state:
            df = st.session_state.df
            X = df.select_dtypes(include=['float64', 'int64'])
            
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            st.session_state.X_scaled = X_scaled
            st.write("Data setelah scaling:", pd.DataFrame(X_scaled, columns=X.columns))

    # === CLUSTERING ===
    elif menu == "Clustering":
        st.markdown('<div class="title">🧩 Spectral Clustering</div>', unsafe_allow_html=True)
        
        if 'X_scaled' in st.session_state:
            X_scaled = st.session_state.X_scaled
            
            # Spectral Clustering
            sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
            labels = sc.fit_predict(X_scaled)
            st.session_state.labels = labels
            
            # Visualisasi
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            fig, ax = plt.subplots()
            ax.scatter(X_pca[:,0], X_pca[:,1], c=labels)
            st.pyplot(fig)

    # === HASIL & ANALISIS ===
    elif menu == "Hasil & Analisis":
        st.markdown('<div class="title">📊 Hasil & Analisis Clustering</div>', unsafe_allow_html=True)
        
        if 'labels' in st.session_state and 'df' in st.session_state:
            df = st.session_state.df.copy()
            df['Cluster'] = st.session_state.labels
            st.dataframe(df)
            
            # Analisis cluster
            st.write("Rata-rata per cluster:")
            st.write(df.groupby('Cluster').mean())

except Exception as e:
    st.error(f"Terjadi kesalahan: {str(e)}")
