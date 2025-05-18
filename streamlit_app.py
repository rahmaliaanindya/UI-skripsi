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
from scipy.linalg import eigh  # Untuk matriks symmetric
from scipy.sparse.linalg import eigsh  # Untuk matriks sparse
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import rbf_kernel
from pyswarms.single import GlobalBestPSO
import streamlit as st
from collections import Counter
import time

# Set random seed untuk reproduktibilitas
SEED = 42
np.random.seed(SEED)

def spectral_clustering_consistent(X, gamma, n_clusters):
    """Implementasi konsisten untuk spectral clustering"""
    try:
        # 1. Affinity matrix
        W = rbf_kernel(X, gamma=gamma)
        
        # 2. Thresholding
        W[W < 0.01] = 0
        
        # 3. Laplacian
        L = laplacian(W, normed=True)
        
        # 4. Eigen decomposition
        eigvals, eigvecs = eigsh(L, k=n_clusters, which='SM', tol=1e-6)
        U = normalize(eigvecs, norm='l2')
        
        # 5. Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=20)
        labels = kmeans.fit_predict(U)
        
        return U, labels, W, True
    
    except Exception as e:
        st.error(f"Error dalam spectral clustering: {str(e)}")
        return None, None, None, False

def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    # Pastikan data sudah di-preprocessing
    if 'X_scaled' not in st.session_state:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # =============================================
    # 1. EVALUASI JUMLAH CLUSTER OPTIMAL
    # =============================================
    st.subheader("1. Evaluasi Jumlah Cluster Optimal")
    
    k_range = range(2, 11)
    silhouette_scores = []
    db_scores = []
    
    with st.expander("Detail Evaluasi Cluster", expanded=False):
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for i, k in enumerate(k_range):
            progress_text.text(f"Menghitung untuk k={k}...")
            progress_bar.progress((i+1)/len(k_range))
            
            # Gunakan implementasi konsisten
            _, labels, _, success = spectral_clustering_consistent(X_scaled, gamma=0.1, n_clusters=k)
            
            if success and len(np.unique(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
                dbi = davies_bouldin_score(X_scaled, labels)
                st.write(f'k={k} | Silhouette: {sil:.4f} | DBI: {dbi:.4f}')
                silhouette_scores.append(sil)
                db_scores.append(dbi)
            else:
                silhouette_scores.append(0)
                db_scores.append(float('inf'))
        
        progress_text.empty()
        progress_bar.empty()
    
    # Visualisasi metrik
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(k_range, silhouette_scores, 'bo-')
    ax1.set_title('Silhouette Score (higher better)')
    ax1.set_xlabel('Jumlah Cluster')
    ax1.grid(True)
    
    ax2.plot(k_range, db_scores, 'ro-')
    ax2.set_title('Davies-Bouldin Index (lower better)')
    ax2.set_xlabel('Jumlah Cluster')
    ax2.grid(True)
    
    st.pyplot(fig)
    
    # =============================================
    # 2. PILIH CLUSTER OPTIMAL
    # =============================================
    best_cluster = None
    best_dbi = float('inf')
    best_silhouette = float('-inf')
    
    for n_clusters in k_range:
        U, labels, _, success = spectral_clustering_consistent(X_scaled, gamma=0.1, n_clusters=n_clusters)
        
        if success and len(np.unique(labels)) > 1:
            dbi_score = davies_bouldin_score(X_scaled, labels)
            silhouette_avg = silhouette_score(X_scaled, labels)
            
            if dbi_score < best_dbi and silhouette_avg > best_silhouette:
                best_dbi = dbi_score
                best_silhouette = silhouette_avg
                best_cluster = n_clusters
    
    if best_cluster is None:
        st.error("Tidak dapat menentukan cluster optimal")
        return
    
    st.success(f"**Cluster optimal terpilih:** k={best_cluster} (Silhouette: {best_silhouette:.4f}, DBI: {best_dbi:.4f})")
    
    # =============================================
    # 3. SPECTRAL CLUSTERING BASELINE (γ=0.1)
    # =============================================
    st.subheader("2. Spectral Clustering Baseline (γ=0.1)")
    
    U_before, labels_before, W_before, success = spectral_clustering_consistent(
        X_scaled, gamma=0.1, n_clusters=best_cluster
    )
    
    if not success:
        st.error("Gagal melakukan baseline clustering")
        return
    
    # Simpan hasil
    st.session_state.U_before = U_before
    st.session_state.labels_before = labels_before
    st.session_state.W_before = W_before
    
    # Hitung metrik
    sil_before = silhouette_score(U_before, labels_before)
    dbi_before = davies_bouldin_score(U_before, labels_before)
    
    st.success(f"Baseline berhasil! Silhouette: {sil_before:.4f}, DBI: {dbi_before:.4f}")
    
    # Visualisasi baseline
    fig_before = plt.figure(figsize=(8, 6))
    plt.scatter(U_before[:, 0], U_before[:, 1], c=labels_before, cmap='viridis', alpha=0.7)
    plt.title(f'Sebelum PSO (γ=0.1)\nSilhouette: {sil_before:.4f}, DBI: {dbi_before:.4f}')
    plt.xlabel('Eigenvector 1')
    plt.ylabel('Eigenvector 2')
    st.pyplot(fig_before)
    
    # =============================================
    # 4. OPTIMASI GAMMA DENGAN PSO
    # =============================================
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        with st.spinner("Menjalankan optimasi PSO..."):
            start_time = time.time()
            
            def evaluate_gamma_robust(gamma_array):
                scores = []
                n_runs = 3  # Jumlah run untuk stabilitas
                
                for gamma in gamma_array:
                    gamma_val = gamma[0]
                    sil_list, dbi_list = [], []
                    
                    for _ in range(n_runs):
                        U, labels, _, success = spectral_clustering_consistent(
                            X_scaled, gamma=gamma_val, n_clusters=best_cluster
                        )
                        
                        if success and len(np.unique(labels)) > 1:
                            sil = silhouette_score(U, labels)
                            dbi = davies_bouldin_score(U, labels)
                        else:
                            sil = 0
                            dbi = 10
                        
                        sil_list.append(sil)
                        dbi_list.append(dbi)
                    
                    # Gabungan skor evaluasi
                    mean_sil = np.mean(sil_list)
                    mean_dbi = np.mean(dbi_list)
                    fitness_score = -mean_sil + mean_dbi
                    scores.append(fitness_score)
                
                return np.array(scores)
            
            # Parameter PSO
            options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
            bounds = (np.array([0.001]), np.array([5.0]))
            
            # Inisialisasi PSO
            optimizer = GlobalBestPSO(
                n_particles=15,  # Dikurangi untuk performa
                dimensions=1,
                options=options,
                bounds=bounds
            )
            
            # Jalankan optimasi
            best_cost, best_pos = optimizer.optimize(
                evaluate_gamma_robust,
                iters=50,  # Dikurangi untuk performa
                verbose=False
            )
            
            best_gamma = best_pos[0]
            elapsed_time = time.time() - start_time
            
            st.success(f"**Optimasi selesai!** Waktu: {elapsed_time:.2f} detik | Gamma optimal: {best_gamma:.4f}")
            
            # =============================================
            # 5. CLUSTERING DENGAN GAMMA OPTIMAL
            # =============================================
            U_opt, labels_opt, W_opt, success = spectral_clustering_consistent(
                X_scaled, gamma=best_gamma, n_clusters=best_cluster
            )
            
            if not success:
                st.error("Gagal melakukan clustering dengan gamma optimal")
                return
            
            # Simpan hasil
            st.session_state.best_gamma = best_gamma
            st.session_state.U_opt = U_opt
            st.session_state.labels_opt = labels_opt
            st.session_state.W_opt = W_opt
            
            # Hitung metrik
            sil_opt = silhouette_score(U_opt, labels_opt)
            dbi_opt = davies_bouldin_score(U_opt, labels_opt)
            
            # Tampilkan hasil
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", f"{sil_opt:.4f}", 
                       f"{(sil_opt - sil_before):.4f} vs baseline")
            col2.metric("Davies-Bouldin Index", f"{dbi_opt:.4f}", 
                       f"{(dbi_before - dbi_opt):.4f} vs baseline")
            
            # =============================================
            # 6. VISUALISASI HASIL
            # =============================================
            st.subheader("4. Visualisasi Hasil")
            
            # Plot side-by-side
            fig_compare, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Sebelum PSO
            ax1.scatter(U_before[:, 0], U_before[:, 1], c=labels_before, cmap='viridis', alpha=0.7)
            ax1.set_title(f'Sebelum PSO (γ=0.1)\nSilhouette: {sil_before:.4f}, DBI: {dbi_before:.4f}')
            ax1.set_xlabel('Eigenvector 1')
            ax1.set_ylabel('Eigenvector 2')
            
            # Sesudah PSO
            ax2.scatter(U_opt[:, 0], U_opt[:, 1], c=labels_opt, cmap='viridis', alpha=0.7)
            ax2.set_title(f'Sesudah PSO (γ={best_gamma:.4f})\nSilhouette: {sil_opt:.4f}, DBI: {dbi_opt:.4f}')
            ax2.set_xlabel('Eigenvector 1')
            ax2.set_ylabel('Eigenvector 2')
            
            st.pyplot(fig_compare)
            
            # =============================================
            # 7. SIMPAN HASIL KE DATAFRAME
            # =============================================
            if 'df_cleaned' in st.session_state:
                df = st.session_state.df_cleaned.copy()
                df['Cluster'] = labels_opt
                st.session_state.df_clustered = df
                
                # Tampilkan distribusi cluster
                st.subheader("Distribusi Cluster")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
                # Tampilkan contoh hasil
                if 'Kabupaten/Kota' in df.columns:
                    st.subheader("Pemetaan Cluster")
                    st.dataframe(df[['Kabupaten/Kota', 'Cluster']].sort_values('Cluster'))
                
                # Interpretasi hasil
                st.subheader("Karakteristik Cluster")
                numeric_cols = df.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    cluster_stats = df.groupby('Cluster')[numeric_cols].mean().T
                    st.dataframe(cluster_stats.style.background_gradient(cmap='viridis'))
            
            # Informasi tambahan
            st.subheader("Informasi Tambahan")
            st.write(f"- Gamma optimal: {best_gamma:.6f}")
            st.write(f"- Jumlah iterasi PSO: 50")
            st.write(f"- Jumlah partikel: 15")
            st.write(f"- Waktu eksekusi: {elapsed_time:.2f} detik")

# Untuk menjalankan di Streamlit
if __name__ == "__main__":
    clustering_analysis()
    
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
