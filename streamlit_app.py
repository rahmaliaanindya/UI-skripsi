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
                <li>Bersihkan dan standarisasi data</li>
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
            <li>Pengelompokan wilayah berdasarkan indikator</li>
            <li>Analisis pola data kompleks</li>
            <li>Eksplorasi struktur data</li>
        </ul>
        <p>Gunakan menu navigasi di atas untuk memulai analisis Anda!</p>
    </div>
    """, unsafe_allow_html=True)

def upload_data():
    st.header("📤 Upload Data Excel")

    # Tampilkan kriteria variabel sebelum upload
    with st.expander("ℹ️ Kriteria Variabel yang Harus Diunggah", expanded=True):
        st.markdown("""
        **Pastikan file Excel Anda memiliki kolom-kolom berikut:**

        1. `Kabupaten/Kota`  
        2. `Persentase Penduduk Miskin (%)`  
        3. `Jumlah Penduduk Miskin (ribu jiwa)`  
        4. `Harapan Lama Sekolah (Tahun)`  
        5. `Rata-Rata Lama Sekolah (Tahun)`  
        6. `Tingkat Pengangguran Terbuka (%)`  
        7. `Tingkat Partisipasi Angkatan Kerja (%)`  
        8. `Angka Harapan Hidup (Tahun)`  
        9. `Garis Kemiskinan (Rupiah/Bulan/Kapita)`  
        10. `Indeks Pembangunan Manusia`  
        11. `Rata-rata Upah/Gaji Bersih Pekerja Informal Berdasarkan Lapangan Pekerjaan Utama (Rp)`  
        12. `Rata-rata Pendapatan Bersih Sebulan Pekerja Informal berdasarkan Pendidikan Tertinggi - Jumlah (Rp)`
        """)

    # Upload file Excel
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")

        # Tampilkan data mentah
        with st.expander("📄 Lihat Data Mentah"):
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
    st.dataframe(df.describe())
    
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

    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Silakan upload data terlebih dahulu")
        return

    df = st.session_state.df.copy()
    
    # Simpan dataframe cleaned ke session state
    st.session_state.df_cleaned = df.copy()

    # Hanya buang kolom non-numerik ('Kabupaten/Kota')
    X = df.drop(columns=['Kabupaten/Kota'])  

    # Tampilkan data sebelum scaling
    st.subheader("Contoh Data Sebelum Scaling")
    st.dataframe(X)

    # Scaling
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    X_scaled = scaler.fit_transform(X)

    # Simpan ke session_state
    st.session_state.X_scaled = X_scaled
    st.session_state.feature_names = X.columns.tolist()

    # Tampilkan hasil scaling
    st.subheader("Contoh Data setelah Scaling")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns))


def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    if 'X_scaled' not in st.session_state or st.session_state.X_scaled is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # =============================================
    # 1. EVALUASI JUMLAH CLUSTER OPTIMAL DENGAN SPECTRALCLUSTERING
    # =============================================
    st.subheader("1. Evaluasi Jumlah Cluster Optimal")
    
    silhouette_scores = []
    db_scores = []
    k_range = range(2, 11)

    for k in k_range:
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
        labels = model.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))

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

    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # =============================================
    # 2. PILIH CLUSTER OPTIMAL
    # =============================================
    best_cluster = None
    best_dbi = float('inf')
    best_silhouette = float('-inf')

    clusters_range = range(2, 11)

    for n_clusters in clusters_range:
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=SEED)
        clusters = spectral.fit_predict(X_scaled)

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

    gamma = 0.1
    W = rbf_kernel(X_scaled, gamma=gamma)
    threshold = 0.01
    W[W < threshold] = 0

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = eigh(L_sym)
    k = best_cluster  # Gunakan jumlah cluster optimal yang sudah ditemukan
    U = eigvecs[:, :k]
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(U_norm)

    st.session_state.U_before = U_norm
    st.session_state.labels_before = labels

    sil_score = silhouette_score(U_norm, labels)
    dbi_score = davies_bouldin_score(U_norm, labels)

    st.success(f"Clustering manual berhasil! Silhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(U_norm[:, 0], U_norm[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f'Spectral Clustering Manual (γ=0.1)\nSilhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}')
    plt.xlabel('Eigenvector 1')
    plt.ylabel('Eigenvector 2')
    st.pyplot(fig)

    def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    if 'X_scaled' not in st.session_state or st.session_state.X_scaled is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # =============================================
    # 1. EVALUASI JUMLAH CLUSTER OPTIMAL DENGAN SPECTRALCLUSTERING
    # =============================================
    st.subheader("1. Evaluasi Jumlah Cluster Optimal")
    
    silhouette_scores = []
    db_scores = []
    k_range = range(2, 11)

    for k in k_range:
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
        labels = model.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))

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

    optimal_k = k_range[np.argmax(silhouette_scores)]
    st.success(f"Jumlah cluster optimal berdasarkan Silhouette Score: {optimal_k}")
    
    # =============================================
    # 2. SPECTRAL CLUSTERING MANUAL DENGAN GAMMA=0.1
    # =============================================
    st.subheader("2. Spectral Clustering Manual (γ=0.1)")

    gamma = 0.1
    W = rbf_kernel(X_scaled, gamma=gamma)
    threshold = 0.01
    W[W < threshold] = 0

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = eigh(L_sym)
    k = optimal_k  # Gunakan jumlah cluster optimal yang sudah ditemukan
    U = eigvecs[:, :k]
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(U_norm)

    st.session_state.U_before = U_norm
    st.session_state.labels_before = labels

    sil_score = silhouette_score(U_norm, labels)
    dbi_score = davies_bouldin_score(U_norm, labels)

    st.success(f"Clustering manual berhasil! Silhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(U_norm[:, 0], U_norm[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f'Spectral Clustering Manual (γ=0.1)\nSilhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}')
    plt.xlabel('Eigenvector 1')
    plt.ylabel('Eigenvector 2')
    st.pyplot(fig)

    # =============================================
    # 3. OPTIMASI GAMMA DENGAN PSO
    # =============================================
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        with st.spinner("Menjalankan optimasi PSO (mungkin memakan waktu beberapa menit)..."):
            try:
                # Fungsi evaluasi yang lebih sederhana
                def evaluate_gamma_robust(gamma_array):
                    scores = []
                    data_for_kernel = X_scaled
                    n_runs = 3  # Jumlah run untuk stabilitas

                    for gamma in gamma_array:
                        gamma_val = gamma[0]
                        sil_list, dbi_list = [], []

                        for _ in range(n_runs):
                            try:
                                W = rbf_kernel(data_for_kernel, gamma=gamma_val)

                                if np.allclose(W, 0) or np.any(np.isnan(W)) or np.any(np.isinf(W)):
                                    raise ValueError("Invalid kernel matrix.")

                                L = laplacian(W, normed=True)

                                if np.any(np.isnan(L.data)) or np.any(np.isinf(L.data)):
                                    raise ValueError("Invalid Laplacian.")

                                eigvals, eigvecs = eigsh(L, k=optimal_k, which='SM', tol=1e-6)
                                U = normalize(eigvecs, norm='l2')

                                if np.isnan(U).any() or np.isinf(U).any():
                                    raise ValueError("Invalid U.")

                                kmeans = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=10)
                                labels = kmeans.fit_predict(U)

                                if len(np.unique(labels)) < 2:
                                    raise ValueError("Only one cluster.")

                                sil = silhouette_score(U, labels)
                                dbi = davies_bouldin_score(U, labels)

                                sil_list.append(sil)
                                dbi_list.append(dbi)

                            except Exception:
                                # Penalti jika gagal
                                sil_list.append(0.0)
                                dbi_list.append(10.0)

                        mean_sil = np.mean(sil_list)
                        mean_dbi = np.mean(dbi_list)
                        fitness_score = -mean_sil + mean_dbi
                        scores.append(fitness_score)

                    return np.array(scores)
                
                # Konfigurasi PSO
                options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
                bounds = (np.array([0.001]), np.array([5.0]))  # Range gamma

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
                    iters=50,
                    verbose=False
                )
                
                best_gamma = best_pos[0]
                st.session_state.best_gamma = best_gamma
                
                # Evaluasi hasil optimal
                W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
                L_opt = laplacian(W_opt, normed=True)
                eigvals_opt, eigvecs_opt = eigsh(L_opt, k=optimal_k, which='SM', tol=1e-6)
                U_opt = normalize(eigvecs_opt, norm='l2')
                kmeans_opt = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=10)
                labels_opt = kmeans_opt.fit_predict(U_opt)
                
                st.session_state.U_opt = U_opt
                st.session_state.labels_opt = labels_opt
                
                sil_opt = silhouette_score(U_opt, labels_opt)
                dbi_opt = davies_bouldin_score(U_opt, labels_opt)
                
                st.success(f"**Optimasi selesai!** Gamma optimal: {best_gamma:.4f}")
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                col1.metric("Silhouette Score", f"{sil_opt:.4f}", 
                            f"{(sil_opt - sil_score):.4f} vs baseline")
                col2.metric("Davies-Bouldin Index", f"{dbi_opt:.4f}", 
                            f"{(dbi_score - dbi_opt):.4f} vs baseline")
                
                # Visualisasi perbandingan
                pca = PCA(n_components=2)
                U_before_pca = pca.fit_transform(st.session_state.U_before)
                U_opt_pca = pca.transform(U_opt)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                scatter1 = ax1.scatter(U_before_pca[:,0], U_before_pca[:,1], 
                                     c=st.session_state.labels_before, 
                                     cmap='viridis', s=50, alpha=0.7)
                ax1.set_title(f"Sebelum PSO (γ=0.1)\nSilhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")
                ax1.set_xlabel("PC1")
                ax1.set_ylabel("PC2")
                plt.colorbar(scatter1, ax=ax1, label='Cluster')
                
                scatter2 = ax2.scatter(U_opt_pca[:,0], U_opt_pca[:,1], 
                                     c=labels_opt, 
                                     cmap='viridis', s=50, alpha=0.7)
                ax2.set_title(f"Sesudah PSO (γ={best_gamma:.4f})\nSilhouette: {sil_opt:.4f}, DBI: {dbi_opt:.4f}")
                ax2.set_xlabel("PC1")
                ax2.set_ylabel("PC2")
                plt.colorbar(scatter2, ax=ax2, label='Cluster')
                
                st.pyplot(fig)
                
                # Simpan hasil ke dataframe
                if 'df_cleaned' in st.session_state and st.session_state.df_cleaned is not None:
                    df = st.session_state.df_cleaned.copy()
                else:
                    df = st.session_state.df.copy()
                
                df['Cluster'] = labels_opt
                st.session_state.df_clustered = df
                
                # Tampilkan distribusi cluster
                st.subheader("Distribusi Cluster")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
                if 'Kabupaten/Kota' in df.columns:
                    st.subheader("Pemetaan Cluster")
                    st.dataframe(df[['Kabupaten/Kota', 'Cluster']].sort_values('Cluster'))
                
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam optimasi PSO: {str(e)}")

                
def results_analysis():
    st.header("📊 Hasil Analisis Cluster")
    
    if 'df_clustered' not in st.session_state:
        st.warning("Silakan jalankan clustering terlebih dahulu")
        return
    
    df = st.session_state.df_clustered
    
    # 1. Distribusi Cluster
    st.subheader("1. Distribusi Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    # 2. Karakteristik Cluster
    st.subheader("2. Karakteristik per Cluster")
    
    # Pastikan kolom Cluster ada di original_df
    if 'df_cleaned' in st.session_state:
        original_df = st.session_state.df_cleaned.copy()
        
        # Gabungkan dengan hasil clustering
        if 'Kabupaten/Kota' in original_df.columns and 'Kabupaten/Kota' in df.columns:
            original_df = original_df.merge(
                df[['Kabupaten/Kota', 'Cluster']],
                on='Kabupaten/Kota',
                how='left'
            )
            
            numeric_cols = original_df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col != 'Cluster']  # Exclude Cluster jika ada
            
            if 'Cluster' in original_df.columns:
                cluster_means = original_df.groupby('Cluster')[numeric_cols].mean()
                
                # Urutkan cluster dari termiskin (asumsi kolom 'PDRB' sebagai indikator)
                if 'PDRB' in numeric_cols:
                    cluster_order = cluster_means['PDRB'].sort_values().index
                    cluster_means = cluster_means.loc[cluster_order]
                
                st.dataframe(cluster_means.style.format("{:.2f}").background_gradient(cmap='Blues'))
            else:
                st.warning("Kolom 'Cluster' tidak ditemukan di data asli")
        else:
            st.warning("Tidak dapat menggabungkan data karena kolom 'Kabupaten/Kota' tidak ditemukan")
    
    # 3. Feature Importance
    st.subheader("3. Feature Importance")
    X = df.drop(columns=['Cluster', 'Kabupaten/Kota'], errors='ignore')
    y = df['Cluster']
    
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    ax.set_title("Faktor Paling Berpengaruh dalam Clustering")
    st.pyplot(fig)
    
        # 4. Pemetaan Daerah per Cluster
    if 'Kabupaten/Kota' in df.columns:
        st.subheader("4. Pemetaan Daerah per Cluster")
        
        try:
            # Gabungkan dengan data asli
            if 'df_cleaned' in st.session_state:
                merged_df = pd.merge(
                    df[['Kabupaten/Kota', 'Cluster']],
                    st.session_state.df_cleaned,
                    on='Kabupaten/Kota',
                    how='left'
                )
                
                # Daftar variabel yang tersedia
                kemiskinan_vars = [
                    'Persentase Penduduk Miskin (%)',
                    'Jumlah Penduduk Miskin (ribu jiwa)',
                    'Garis Kemiskinan (Rupiah/Bulan/Kapita)'
                ]
                
                pendidikan_vars = [
                    'Harapan Lama Sekolah (Tahun)',
                    'Rata-Rata Lama Sekolah (Tahun)'
                ]
                
                ketenagakerjaan_vars = [
                    'Tingkat Pengangguran Terbuka (%)',
                    'Tingkat Partisipasi Angkatan Kerja (%)',
                    'Rata-rata Upah/Gaji Bersih Pekerja Informal Berdasarkan Lapangan Pekerjaan Utama (Rp)',
                    'Rata-rata Pendapatan Bersih Sebulan Pekerja Informal berdasarkan Pendidikan Tertinggi - Jumlah (Rp)'
                ]
                
                kesehatan_vars = [
                    'Angka Harapan Hidup (Tahun)'
                ]
                
                ipm_vars = [
                    'Indeks Pembangunan Manusia'
                ]
                
                # Cari variabel yang ada di dataset
                available_vars = {
                    'Kemiskinan': [v for v in kemiskinan_vars if v in merged_df.columns],
                    'Pendidikan': [v for v in pendidikan_vars if v in merged_df.columns],
                    'Ketenagakerjaan': [v for v in ketenagakerjaan_vars if v in merged_df.columns],
                    'Kesehatan': [v for v in kesehatan_vars if v in merged_df.columns],
                    'IPM': [v for v in ipm_vars if v in merged_df.columns]
                }
                
                # Pilih 1 variabel utama per kategori untuk ditampilkan
                display_cols = ['Kabupaten/Kota', 'Cluster']
                sort_by = 'Cluster'
                
                # Tambahkan variabel terpilih
                for category, vars_list in available_vars.items():
                    if vars_list:
                        display_cols.append(vars_list[0])  # Ambil variabel pertama yang tersedia
                        if category == 'Kemiskinan':
                            sort_by = vars_list[0]  # Default sort by first poverty variable
                
                # Urutkan data
                merged_df = merged_df.sort_values([sort_by, 'Kabupaten/Kota'], ascending=[False, True])
                
                # Tampilkan data
                st.dataframe(
                    merged_df[display_cols],
                    height=600,
                    column_config={
                        'Persentase Penduduk Miskin (%)': st.column_config.NumberColumn(format="%.2f %%"),
                        'Garis Kemiskinan (Rupiah/Bulan/Kapita)': st.column_config.NumberColumn(format="%,d")
                    }
                )
                
                # Analisis sederhana per cluster
                st.subheader("Analisis Indikator per Cluster")
                
                # Pilih indikator untuk analisis
                analysis_var = st.selectbox(
                    "Pilih indikator untuk analisis cluster:",
                    options=[v for vars_list in available_vars.values() for v in vars_list]
                )
                
                if analysis_var in merged_df.columns:
                    cluster_stats = merged_df.groupby('Cluster')[analysis_var].describe()
                    st.write(f"Statistik {analysis_var} per Cluster:")
                    st.dataframe(cluster_stats.style.format("{:.2f}"))
                    
                    # Visualisasi
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=merged_df, x='Cluster', y=analysis_var, palette='viridis')
                    plt.title(f'Distribusi {analysis_var} per Cluster')
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam pemetaan: {str(e)}")
            if 'merged_df' in locals():
                st.write("Kolom yang tersedia:", merged_df.columns.tolist())

        # 5. Ranking Kota (Termiskin & Paling Tidak Miskin)
    st.subheader("5. Ranking Kota Berdasarkan Indikator Kemiskinan")
    
    if 'df_cleaned' in st.session_state:
        merged_df = st.session_state.df_cleaned.merge(
            df[['Kabupaten/Kota', 'Cluster']],
            on='Kabupaten/Kota',
            how='left'
        )
        
        kemiskinan_indicators = [
            'Persentase Penduduk Miskin (%)',
            'Jumlah Penduduk Miskin (ribu jiwa)',
            'Garis Kemiskinan (Rupiah/Bulan/Kapita)'
        ]
        
        available_indicators = [col for col in kemiskinan_indicators if col in merged_df.columns]
        
        if available_indicators:
            main_indicator = available_indicators[0]
            
            # Tampilkan 3 Kota Termiskin
            st.markdown("**3 Kota Kemiskinan Tinggi:**")
            poorest = merged_df.nlargest(3, main_indicator)[['Kabupaten/Kota', 'Cluster', main_indicator]]
            st.dataframe(
                poorest.style.format({
                    main_indicator: "{:.2f} %" if "%" in main_indicator else "Rp {:,}" if "Rupiah" in main_indicator else "{:.2f}"
                }),
                hide_index=True
            )
            
            # Tampilkan 3 Kota Paling Tidak Miskin
            st.markdown("**3 Kota Kemiskinan Rendah:**")
            least_poor = merged_df.nsmallest(3, main_indicator)[['Kabupaten/Kota', 'Cluster', main_indicator]]
            st.dataframe(
                least_poor.style.format({
                    main_indicator: "{:.2f} %" if "%" in main_indicator else "Rp {:,}" if "Rupiah" in main_indicator else "{:.2f}"
                }),
                hide_index=True
            )
    
    # 6. Perbandingan Sebelum-Sesudah PSO
    st.subheader("6. Perbandingan Hasil Sebelum dan Sesudah Optimasi")
    
    if all(key in st.session_state for key in ['U_before', 'labels_before', 'U_opt', 'labels_opt']):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sebelum Optimasi (γ=0.1):**")
            st.write(f"- Silhouette Score: {silhouette_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
            st.write(f"- Davies-Bouldin Index: {davies_bouldin_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
            
        with col2:
            st.markdown(f"**Sesudah Optimasi (γ={st.session_state.get('best_gamma', 0):.4f}):**")
            st.write(f"- Silhouette Score: {silhouette_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
            st.write(f"- Davies-Bouldin Index: {davies_bouldin_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
        
        # Visualisasi
        fig = plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plt.scatter(st.session_state.U_before[:, 0], st.session_state.U_before[:, 1], 
                    c=st.session_state.labels_before, cmap='viridis')
        plt.title("Sebelum Optimasi")
        
        plt.subplot(122)
        plt.scatter(st.session_state.U_opt[:, 0], st.session_state.U_opt[:, 1], 
                    c=st.session_state.labels_opt, cmap='viridis')
        plt.title("Sesudah Optimasi")
        
        st.pyplot(fig)
    
    # 7. Implementasi dan Rekomendasi
    st.subheader("7. Implementasi dan Rekomendasi Kebijakan")
    
    st.markdown("""
    **Berdasarkan hasil clustering:**
    
    1. **Cluster Termiskin** (Cluster 0):
    - Fokus pada program pengentasan kemiskinan
    - Pengembangan UMKM lokal
    - Peningkatan akses pendidikan dan kesehatan
    
    2. **Cluster Menengah** (Cluster 1):
    - Penguatan sektor produktif
    - Pelatihan keterampilan kerja
    - Infrastruktur dasar
    
    **Strategi Implementasi:**
    - Prioritas anggaran berdasarkan karakteristik cluster
    - Program khusus untuk daerah tertinggal
    - Monitoring evaluasi berbasis indikator cluster
    """)
    
    # Tambahkan tombol download hasil
    st.download_button(
        label="📥 Download Hasil Clustering",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='hasil_clustering.csv',
        mime='text/csv'
    )

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
