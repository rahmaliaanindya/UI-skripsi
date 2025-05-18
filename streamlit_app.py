import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import warnings
from io import StringIO
import sys
import random
import os

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
warnings.filterwarnings("ignore")

# ======================
# STREAMLIT UI SETUP
# ======================
st.set_page_config(page_title="Spectral Clustering with PSO", layout="wide", page_icon="📊")

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 15px;
    }
    .plot-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    /* Navigation menu at the top */
    .stHorizontalBlock {
        display: flex;
        justify-content: center;
        margin-bottom: 30px;
    }
    .stHorizontalBlock [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #e8f5e9;
        border-radius: 5px;
        margin: 0 5px;
    }
    .stHorizontalBlock [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    /* Landing page styling */
    .landing-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .feature-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# MAIN APP FUNCTIONS
# ======================

def landing_page():
    st.markdown('<div class="landing-header">', unsafe_allow_html=True)
    st.title("🔍 Spectral Clustering with PSO Optimization")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Selamat Datang di Aplikasi Spectral Clustering dengan Optimasi PSO</h3>
        <p>Aplikasi ini dirancang untuk membantu Anda melakukan analisis clustering menggunakan metode Spectral Clustering yang dioptimasi dengan Particle Swarm Optimization (PSO).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Fitur Utama</h4>
            <ul>
                <li>Exploratory Data Analysis</li>
                <li>Preprocessing Data Otomatis</li>
                <li>Spectral Clustering</li>
                <li>Optimasi Parameter dengan PSO</li>
                <li>Visualisasi Hasil Clustering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>🔧 Cara Menggunakan</h4>
            <ol>
                <li>Upload dataset Anda (format Excel)</li>
                <li>Lakukan eksplorasi data</li>
                <li>Bersihkan dan normalisasi data</li>
                <li>Tentukan parameter clustering</li>
                <li>Jalankan optimasi PSO</li>
                <li>Analisis hasil clustering</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>📈 Keunggulan</h4>
            <ul>
                <li>Antarmuka yang mudah digunakan</li>
                <li>Optimasi parameter otomatis</li>
                <li>Visualisasi interaktif</li>
                <li>Metrik evaluasi clustering</li>
                <li>Analisis feature importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4>📌 Contoh Penggunaan</h4>
        <p>Aplikasi ini cocok untuk berbagai kasus seperti:</p>
        <ul>
            <li>Segmentasi pelanggan</li>
            <li>Pengelompokan wilayah berdasarkan indikator</li>
            <li>Analisis pola data kompleks</li>
            <li>Eksplorasi struktur data</li>
        </ul>
        <p>Gunakan menu navigasi di atas untuk memulai analisis Anda!</p>
    </div>
    """, unsafe_allow_html=True)

def upload_data():
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil dimuat!")
        
        with st.expander("Lihat Data Mentah"):
            st.dataframe(df)

def exploratory_data_analysis():
    st.header("🔍 Exploratory Data Analysis (EDA)")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        return
    
    df = st.session_state.df
    
    # Dataset Info
    st.subheader("Informasi Dataset")
    buffer = StringIO()
    sys.stdout = buffer
    df.info()
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())
    
    # Descriptive Statistics
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe().style.format("{:.2f}"))
    
    # Missing Values
    st.subheader("Pengecekan Nilai Kosong")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("Tidak ada nilai kosong pada dataset")
    else:
        st.dataframe(missing_values[missing_values > 0].to_frame("Jumlah Nilai Kosong"))
    
    # Data Distribution
    st.subheader("Distribusi Variabel Numerik")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Pilih variabel:", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[selected_col], kde=True, bins=30, color='skyblue')
    ax.set_title(f'Distribusi {selected_col}')
    st.pyplot(fig)
    
    # Correlation Matrix
    st.subheader("Matriks Korelasi")
    numerical_df = df.select_dtypes(include=['number'])
    if len(numerical_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig)
    else:
        st.warning("Tidak cukup variabel numerik untuk menampilkan matriks korelasi")

def data_preprocessing():
    st.header("⚙️ Data Preprocessing")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        return
    
    df = st.session_state.df.copy()
    
    # Data Cleaning
    st.subheader("Pembersihan Data")
    
    if st.button("Bersihkan Data"):
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        st.session_state.df_cleaned = df
        st.success("Data berhasil dibersihkan!")
    
    if 'df_cleaned' in st.session_state:
        # Data Scaling
        st.subheader("Normalisasi Data dengan RobustScaler")
        
        X = st.session_state.df_cleaned.drop(columns=['Kabupaten/Kota'], errors='ignore')
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        st.session_state.X_scaled = X_scaled
        st.session_state.feature_names = X.columns.tolist()
        
        st.success("Data berhasil dinormalisasi!")
        
        # Show scaled data sample
        st.subheader("Contoh Data setelah Scaling")
        st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    if 'X_scaled' not in st.session_state:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # Optimal Cluster Evaluation
    st.subheader("Evaluasi Jumlah Cluster Optimal")
    
    k_range = range(2, 11)
    silhouette_scores = []
    db_scores = []
    
    with st.spinner("Menghitung metrik untuk berbagai jumlah cluster..."):
        for k in k_range:
            model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
            labels = model.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            db_scores.append(davies_bouldin_score(X_scaled, labels))
    
    # Plot evaluation metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(k_range, silhouette_scores, 'bo-')
    ax1.set_xlabel('Jumlah Cluster')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Evaluasi Silhouette Score')
    ax1.grid(True)
    
    ax2.plot(k_range, db_scores, 'ro-')
    ax2.set_xlabel('Jumlah Cluster')
    ax2.set_ylabel('Davies-Bouldin Index')
    ax2.set_title('Evaluasi Davies-Bouldin Index')
    ax2.grid(True)
    
    st.pyplot(fig)
    
    # Determine optimal clusters
    optimal_k_sil = k_range[np.argmax(silhouette_scores)]
    optimal_k_dbi = k_range[np.argmin(db_scores)]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Optimal Cluster (Silhouette)", optimal_k_sil)
    with col2:
        st.metric("Optimal Cluster (DBI)", optimal_k_dbi)
    
    # Clustering with PSO optimization
    st.subheader("Optimasi Parameter dengan PSO")
    k_final = st.number_input("Pilih jumlah cluster (k):", min_value=2, max_value=10, value=optimal_k_sil)
    
    if st.button("Jalankan Spectral Clustering dengan PSO"):
        with st.spinner("Menjalankan optimasi PSO..."):
            # PSO optimization function
            def evaluate_gamma(gamma_array):
                scores = []
                for gamma in gamma_array:
                    gamma_val = gamma[0]
                    try:
                        W = rbf_kernel(X_scaled, gamma=gamma_val)
                        
                        # Apply threshold
                        W[W < 0.01] = 0
                        
                        L = laplacian(W, normed=True)
                        eigvals, eigvecs = eigsh(L, k=k_final, which='SM', tol=1e-6)
                        U = normalize(eigvecs, norm='l2')
                        kmeans = KMeans(n_clusters=k_final, random_state=SEED).fit(U)
                        labels = kmeans.labels_
                        
                        sil = silhouette_score(U, labels)
                        dbi = davies_bouldin_score(U, labels)
                        scores.append(-sil + dbi)  # Minimize this
                    except:
                        scores.append(10)  # Penalty for failed cases
                return np.array(scores)
            
            # Run PSO
            options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
            bounds = (np.array([0.001]), np.array([5.0]))
            optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(evaluate_gamma, iters=100)
            best_gamma = best_pos[0]
            
            # Store results
            st.session_state.best_gamma = best_gamma
            
            # Final clustering with optimal gamma
            W = rbf_kernel(X_scaled, gamma=best_gamma)
            W[W < 0.01] = 0  # Thresholding
            L = laplacian(W, normed=True)
            eigvals, eigvecs = eigsh(L, k=k_final, which='SM', tol=1e-6)
            U = normalize(eigvecs, norm='l2')
            kmeans = KMeans(n_clusters=k_final, random_state=SEED).fit(U)
            labels = kmeans.labels_
            
            # Store results
            st.session_state.U = U
            st.session_state.labels = labels
            st.session_state.eigvecs = eigvecs
            
            # Save to dataframe
            df = st.session_state.df_cleaned.copy()
            df['Cluster'] = labels
            st.session_state.df_clustered = df
            
            # Metrics
            sil_score = silhouette_score(U, labels)
            dbi_score = davies_bouldin_score(U, labels)
            
            st.success(f"Optimasi selesai! Gamma optimal: {best_gamma:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", f"{sil_score:.4f}")
            with col2:
                st.metric("Davies-Bouldin Index", f"{dbi_score:.4f}")
            
            # Cluster visualization
            st.subheader("Visualisasi Cluster")
            pca = PCA(n_components=2)
            U_pca = pca.fit_transform(U)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(U_pca[:, 0], U_pca[:, 1], c=labels, cmap='viridis')
            ax.set_title(f"Hasil Clustering (k={k_final})")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)

def results_analysis():
    st.header("📊 Hasil Analisis Cluster")
    
    if 'df_clustered' not in st.session_state:
        st.warning("Silakan jalankan clustering terlebih dahulu")
        return
    
    df = st.session_state.df_clustered
    
    # Cluster distribution
    st.subheader("Distribusi Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    # Cluster characteristics
    st.subheader("Karakteristik per Cluster")
    cluster_means = df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_means.style.format("{:.2f}"))
    
    # Feature importance
    st.subheader("Feature Importance")
    X = df.drop(columns=['Cluster', 'Kabupaten/Kota'], errors='ignore')
    y = df['Cluster']
    
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    
    # Cluster mapping
    if 'Kabupaten/Kota' in df.columns:
        st.subheader("Pemetaan Kabupaten/Kota per Cluster")
        st.dataframe(df[['Kabupaten/Kota', 'Cluster']].sort_values(['Cluster', 'Kabupaten/Kota']))
    
    # Before-after PSO comparison
    if 'U' in st.session_state and 'eigvecs' in st.session_state:
        st.subheader("Perbandingan Sebelum dan Sesudah Optimasi")
        
        # Before PSO (using default gamma=0.1)
        W_before = rbf_kernel(st.session_state.X_scaled, gamma=0.1)
        W_before[W_before < 0.01] = 0
        L_before = laplacian(W_before, normed=True)
        eigvals_before, eigvecs_before = eigsh(L_before, k=2, which='SM', tol=1e-6)
        U_before = normalize(eigvecs_before, norm='l2')
        kmeans_before = KMeans(n_clusters=2, random_state=SEED).fit(U_before)
        labels_before = kmeans_before.labels_
        
        # After PSO
        U_after = st.session_state.U
        labels_after = st.session_state.labels
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.scatter(U_before[:, 0], U_before[:, 1], c=labels_before, cmap='viridis')
        ax1.set_title("Sebelum Optimasi (γ=0.1)")
        
        ax2.scatter(U_after[:, 0], U_after[:, 1], c=labels_after, cmap='viridis')
        ax2.set_title(f"Sesudah Optimasi (γ={st.session_state.best_gamma:.4f})")
        
        st.pyplot(fig)

# ======================
# APP LAYOUT
# ======================

# Navigation menu at the top
menu_options = {
    "Beranda": landing_page,
    "Upload Data": upload_data,
    "EDA": exploratory_data_analysis,
    "Preprocessing": data_preprocessing,
    "Clustering": clustering_analysis,
    "Results": results_analysis
}

# Display the appropriate page based on menu selection
menu_selection = st.radio(
    "Menu Navigasi",
    list(menu_options.keys()),
    index=0,
    key="menu",
    horizontal=True,
    label_visibility="hidden"
)

# Execute the selected page function
menu_options[menu_selection]()
