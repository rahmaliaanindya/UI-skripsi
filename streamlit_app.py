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

def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    # Pastikan data sudah di-preprocessing
    if 'X_scaled' not in st.session_state:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # =============================================
    # 1. EVALUASI JUMLAH CLUSTER OPTIMAL DENGAN SPECTRALCLUSTERING
    # =============================================
    st.subheader("1. Evaluasi Jumlah Cluster Optimal")
    
    # --- Tentukan jumlah cluster optimal dengan Spectral Clustering ---
    silhouette_scores = []
    db_scores = []
    k_range = range(2, 11)

    for k in k_range:
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
        labels = model.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    # Visualisasi hasil evaluasi jumlah cluster
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(k_range, silhouette_scores, 'bo-', label='Silhouette Score')
    ax1.set_xlabel('Jumlah Cluster')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Evaluasi Silhouette Score')
    ax1.legend()

    ax2.plot(k_range, db_scores, 'ro-', label='Davies-Bouldin Index')
    ax2.set_xlabel('Jumlah Cluster')
    ax2.set_ylabel('DB Index')
    ax2.set_title('Evaluasi Davies-Bouldin Index')
    ax2.legend()

    st.pyplot(fig)

    # Pilih jumlah cluster optimal
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # =============================================
    # 2. PILIH CLUSTER OPTIMAL
    # =============================================
    best_cluster = None
    best_dbi = float('inf')
    best_silhouette = float('-inf')

    clusters_range = range(2, 11)  # You can adjust the range as needed

    for n_clusters in clusters_range:
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=SEED)
        clusters = spectral.fit_predict(X_scaled)

        # Evaluasi clustering dengan Davies-Bouldin Index dan Silhouette Score
        dbi_score = davies_bouldin_score(X_scaled, clusters)
        silhouette_avg = silhouette_score(X_scaled, clusters)

        st.write(f'Jumlah Cluster: {n_clusters} | DBI: {dbi_score:.4f} | Silhouette Score: {silhouette_avg:.4f}')

        if dbi_score < best_dbi and silhouette_avg > best_silhouette:
            best_dbi = dbi_score
            best_silhouette = silhouette_avg
            best_cluster = n_clusters
    
    if best_cluster is None:
        st.error("Tidak dapat menentukan cluster optimal")
        return
    
    st.success(f"**Cluster optimal terpilih:** k={best_cluster} (Silhouette: {best_silhouette:.4f}, DBI: {best_dbi:.4f})")
    
    # =============================================
    # 3. SPECTRAL CLUSTERING MANUAL DENGAN GAMMA=0.1
    # =============================================
    st.subheader("2. Spectral Clustering Manual (γ=0.1)")
    
    # Hitung affinity matrix
    gamma_before = 0.1
    W_before = rbf_kernel(X_scaled, gamma=gamma_before)
    
    # Threshold untuk mengurangi noise
    W_before[W_before < 0.01] = 0
    
    # Hitung degree matrix dan Laplacian
    D = np.diag(W_before.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W_before.sum(axis=1)))
    L_sym = np.eye(W_before.shape[0]) - D_inv_sqrt @ W_before @ D_inv_sqrt
    
    # Eigen decomposition
    eigvals_before, eigvecs_before = eigh(L_sym)
    
    # Ambil k eigenvector terkecil
    U_before = eigvecs_before[:, :best_cluster]
    
    # Normalisasi
    U_before_norm = U_before / np.linalg.norm(U_before, axis=1, keepdims=True)
    
    # KMeans clustering
    kmeans_before = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=20)
    labels_before = kmeans_before.fit_predict(U_before_norm)
    
    # Simpan hasil
    st.session_state.U_before = U_before_norm
    st.session_state.labels_before = labels_before
    
    # Hitung metrik
    sil_before = silhouette_score(U_before_norm, labels_before)
    dbi_before = davies_bouldin_score(U_before_norm, labels_before)
    
    st.success(f"Clustering manual berhasil! Silhouette: {sil_before:.4f}, DBI: {dbi_before:.4f}")
    
    # Visualisasi baseline
    fig_before = plt.figure(figsize=(8, 6))
    plt.scatter(U_before_norm[:, 0], U_before_norm[:, 1], c=labels_before, cmap='viridis', alpha=0.7)
    plt.title(f'Spectral Clustering Manual (γ=0.1)\nSilhouette: {sil_before:.4f}, DBI: {dbi_before:.4f}')
    plt.xlabel('Eigenvector 1')
    plt.ylabel('Eigenvector 2')
    st.pyplot(fig_before)
    
    # =============================================
    # 4. OPTIMASI GAMMA DENGAN PSO
    # =============================================
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        with st.spinner("Menjalankan optimasi PSO (mungkin memakan waktu beberapa menit)..."):
            try:
                # Fungsi evaluasi gamma
                def evaluate_gamma_robust(gamma_array):
                    scores = []
                    n_runs = 3  # Jumlah run untuk stabilitas
                    
                    for gamma in gamma_array:
                        gamma_val = gamma[0]
                        sil_list, dbi_list = [], []
                        
                        for _ in range(n_runs):
                            try:
                                # Hitung affinity matrix
                                W = rbf_kernel(X_scaled, gamma=gamma_val)
                                W[W < 0.01] = 0  # Thresholding
                                
                                # Hitung Laplacian
                                L = laplacian(W, normed=True)
                                
                                # Eigen decomposition
                                eigvals, eigvecs = eigsh(L, k=best_cluster, which='SM', tol=1e-4)
                                U = normalize(eigvecs, norm='l2')
                                
                                # Clustering
                                kmeans = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=10)
                                labels = kmeans.fit_predict(U)
                                
                                # Evaluasi
                                if len(np.unique(labels)) > 1:
                                    sil = silhouette_score(U, labels)
                                    dbi = davies_bouldin_score(U, labels)
                                else:
                                    sil = 0
                                    dbi = 10
                                
                                sil_list.append(sil)
                                dbi_list.append(dbi)
                                
                            except:
                                sil_list.append(0)
                                dbi_list.append(10)
                        
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
                    n_particles=20,
                    dimensions=1,
                    options=options,
                    bounds=bounds
                )
                
                # Jalankan optimasi
                best_cost, best_pos = optimizer.optimize(
                    evaluate_gamma_robust,
                    iters=100,
                    verbose=False
                )
                
                best_gamma = best_pos[0]
                
                # =============================================
                # 5. CLUSTERING DENGAN GAMMA OPTIMAL
                # =============================================
                W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
                W_opt[W_opt < 0.01] = 0
                
                # Laplacian
                L_opt = laplacian(W_opt, normed=True)
                eigvals_opt, eigvecs_opt = eigsh(L_opt, k=best_cluster, which='SM', tol=1e-4)
                U_opt = normalize(eigvecs_opt, norm='l2')
                
                # KMeans clustering
                kmeans_opt = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=20)
                labels_opt = kmeans_opt.fit_predict(U_opt)
                
                # Simpan hasil
                st.session_state.best_gamma = best_gamma
                st.session_state.U_opt = U_opt
                st.session_state.labels_opt = labels_opt
                
                # Hitung metrik
                sil_opt = silhouette_score(U_opt, labels_opt)
                dbi_opt = davies_bouldin_score(U_opt, labels_opt)
                
                # Tampilkan hasil
                st.success(f"**Optimasi selesai!** Gamma optimal: {best_gamma:.4f}")
                
                col1, col2 = st.columns(2)
                col1.metric("Silhouette Score", f"{sil_opt:.4f}", 
                           f"{(sil_opt - sil_before):.4f} vs baseline")
                col2.metric("Davies-Bouldin Index", f"{dbi_opt:.4f}", 
                           f"{(dbi_before - dbi_opt):.4f} vs baseline")
                
                # =============================================
                # 6. VISUALISASI HASIL
                # =============================================
                st.subheader("4. Visualisasi Hasil")
                
                # PCA untuk visualisasi 2D
                pca = PCA(n_components=2)
                
                # Sebelum PSO
                U_before_pca = pca.fit_transform(st.session_state.U_before)
                
                # Sesudah PSO
                U_opt_pca = pca.transform(U_opt)
                
                # Plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Plot sebelum PSO
                scatter1 = ax1.scatter(U_before_pca[:,0], U_before_pca[:,1], 
                                     c=st.session_state.labels_before, 
                                     cmap='viridis', s=50, alpha=0.7)
                ax1.set_title(f"Sebelum PSO (γ=0.1)\nSilhouette: {sil_before:.4f}, DBI: {dbi_before:.4f}")
                ax1.set_xlabel("PC1")
                ax1.set_ylabel("PC2")
                plt.colorbar(scatter1, ax=ax1, label='Cluster')
                
                # Plot sesudah PSO
                scatter2 = ax2.scatter(U_opt_pca[:,0], U_opt_pca[:,1], 
                                     c=labels_opt, 
                                     cmap='viridis', s=50, alpha=0.7)
                ax2.set_title(f"Sesudah PSO (γ={best_gamma:.4f})\nSilhouette: {sil_opt:.4f}, DBI: {dbi_opt:.4f}")
                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")
                plt.colorbar(scatter2, ax=ax2, label='Cluster')
                
                st.pyplot(fig)
                
                # =============================================
                # 7. SIMPAN HASIL KE DATAFRAME
                # =============================================
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
                
            except Exception as e:
                st.error(f"Error dalam optimasi PSO: {str(e)}")
                st.stop()
    
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
