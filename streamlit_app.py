import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from numba import njit, prange
import warnings
from io import StringIO
import sys
import random
import os
from sklearn.ensemble import RandomForestClassifier

# Set random seed untuk reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
warnings.filterwarnings("ignore")

# ======================
# KELAS PSO OPTIMIZER DENGAN REAL-TIME TRACKING
# ======================
class PSOOptimizer:
    def __init__(self, X_scaled, best_cluster):
        self.X_scaled = X_scaled
        self.best_cluster = best_cluster
        self.cost_history = []
        self.gamma_history = []
        self.iteration = 0
        
    def evaluate(self, gamma_array):
        scores = np.zeros(gamma_array.shape[0])
        
        for i in range(gamma_array.shape[0]):
            gamma_val = gamma_array[i,0]
            try:
                # Cukup 1 run untuk mempercepat
                W = numba_rbf_kernel(self.X_scaled, gamma_val)
                L = numba_laplacian(W)
                eigvals, eigvecs = numba_eigsh(L, self.best_cluster)
                U = numba_normalize(eigvecs)
                labels = fast_kmeans(U, self.best_cluster, max_iter=5)  # Kurangi iterasi
                
                sil = numba_silhouette_score(U, labels)
                dbi = numba_davies_bouldin_score(U, labels)
                
                scores[i] = -sil + dbi  # Gabungkan metric
            except:
                scores[i] = 10.0  # Nilai penalty jika error
        
        # Update history
        best_idx = np.argmin(scores)
        self.cost_history.append(scores[best_idx])
        self.gamma_history.append(gamma_array[best_idx,0])
        self.iteration += 1
        
        return scores

# ======================
# FUNGSI NUMBA OPTIMIZED 
# ======================
@njit(fastmath=True, parallel=True)
def numba_rbf_kernel(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in prange(n_samples):
        for j in prange(n_samples):
            diff = X[i] - X[j]
            K[i,j] = np.exp(-gamma * np.dot(diff, diff))
    return K

@njit(fastmath=True)
def numba_laplacian(W):
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    return L

@njit(fastmath=True)
def numba_eigsh(L, k):
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals[:k], eigvecs[:, :k]

@njit(fastmath=True)
def numba_normalize(U):
    norms = np.sqrt(np.sum(U**2, axis=1))
    return U / norms.reshape(-1, 1)

@njit(fastmath=True, parallel=True)
def fast_kmeans(U, n_clusters, max_iter=5):  # Kurangi max_iter
    centroids = U[np.random.choice(U.shape[0], n_clusters, replace=False)]
    for _ in range(max_iter):
        distances = np.zeros((U.shape[0], n_clusters))
        for k in prange(n_clusters):
            distances[:, k] = np.sum((U - centroids[k])**2, axis=1)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.zeros((n_clusters, U.shape[1]))
        for k in prange(n_clusters):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(U[labels == k], axis=0)
            else:
                new_centroids[k] = centroids[k]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels

@njit(fastmath=True)
def numba_silhouette_score(U, labels):
    n_samples = U.shape[0]
    n_clusters = len(np.unique(labels))
    silhouette = 0.0
    
    for i in range(n_samples):
        cluster_i = labels[i]
        a_i = 0.0
        count_a = 0
        for j in range(n_samples):
            if labels[j] == cluster_i and i != j:
                a_i += np.sqrt(np.sum((U[i] - U[j])**2))
                count_a += 1
        if count_a > 0:
            a_i /= count_a
        
        b_i = np.inf
        for k in range(n_clusters):
            if k != cluster_i:
                b_k = 0.0
                count_b = 0
                for j in range(n_samples):
                    if labels[j] == k:
                        b_k += np.sqrt(np.sum((U[i] - U[j])**2))
                        count_b += 1
                if count_b > 0:
                    b_k /= count_b
                    if b_k < b_i:
                        b_i = b_k
        
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
            
        silhouette += s_i
    
    return silhouette / n_samples

@njit(fastmath=True)
def numba_davies_bouldin_score(U, labels):
    n_clusters = len(np.unique(labels))
    centroids = np.zeros((n_clusters, U.shape[1]))
    cluster_sizes = np.zeros(n_clusters)
    
    for i in range(U.shape[0]):
        cluster = labels[i]
        centroids[cluster] += U[i]
        cluster_sizes[cluster] += 1
    
    for k in range(n_clusters):
        if cluster_sizes[k] > 0:
            centroids[k] /= cluster_sizes[k]
    
    S = np.zeros(n_clusters)
    for k in range(n_clusters):
        sum_dist = 0.0
        count = 0
        for i in range(U.shape[0]):
            if labels[i] == k:
                sum_dist += np.sqrt(np.sum((U[i] - centroids[k])**2))
                count += 1
        if count > 0:
            S[k] = sum_dist / count
    
    R = np.zeros(n_clusters)
    for i in range(n_clusters):
        max_R = -np.inf
        for j in range(n_clusters):
            if i != j:
                M_ij = np.sqrt(np.sum((centroids[i] - centroids[j])**2))
                if M_ij > 0:
                    R_ij = (S[i] + S[j]) / M_ij
                    if R_ij > max_R:
                        max_R = R_ij
        R[i] = max_R
    
    return np.mean(R)

# ======================
# FUNGSI UTAMA OPTIMASI
# ======================
def run_pso_optimization(X_scaled, best_cluster):
    # Setup container untuk progress
    progress_container = st.empty()
    progress_col1, progress_col2 = st.columns(2)
    
    with progress_col1:
        st.markdown("**Progress Optimasi**")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with progress_col2:
        st.markdown("**Visualisasi Konvergensi**")
        fig_ax = plt.subplots(figsize=(8, 4))
        fig, ax = fig_ax
        chart_placeholder = st.empty()
    
    # Setup optimizer
    pso_optimizer = PSOOptimizer(X_scaled, best_cluster)
    
    # Setup PSO dengan parameter lebih ringan
    optimizer = GlobalBestPSO(
        n_particles=15,  # Lebih sedikit partikel
        dimensions=1,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.7},
        bounds=(np.array([0.001]), np.array([5.0]))
    )
    
    # Jalankan optimasi dengan callback untuk update UI
    def update_progress(optimizer):
        current_iter = pso_optimizer.iteration
        total_iter = 30  # Lebih sedikit iterasi
        
        # Update progress bar
        progress = int((current_iter / total_iter) * 100)
        progress_bar.progress(min(progress, 100))
        
        # Update status text
        if len(pso_optimizer.cost_history) > 0:
            best_cost = min(pso_optimizer.cost_history)
            best_gamma = pso_optimizer.gamma_history[np.argmin(pso_optimizer.cost_history)]
            status_text.markdown(
                f"**Iterasi:** {current_iter}/{total_iter}\n"
                f"**Gamma Terbaik:** {best_gamma:.4f}\n"
                f"**Cost Terbaik:** {best_cost:.4f}"
            )
        
        # Update plot
        ax.clear()
        ax.plot(range(1, current_iter+1), pso_optimizer.cost_history, 'b-', marker='o')
        ax.set_xlabel('Iterasi')
        ax.set_ylabel('Nilai Cost')
        ax.set_title('Konvergensi Optimasi PSO')
        ax.grid(True)
        chart_placeholder.pyplot(fig)
        
        # Beri sedikit delay untuk update UI
        time.sleep(0.1)
    
    # Jalankan optimasi
    best_cost, best_pos = optimizer.optimize(
        pso_optimizer.evaluate,
        iters=30,  # Lebih sedikit iterasi
        n_processes=1,  # Gunakan 1 process untuk stabil
        verbose=False
    )
    
    # Final update
    update_progress(optimizer)
    
    return best_pos[0], pso_optimizer.cost_history, pso_optimizer.gamma_history

# ======================
# STREAMLIT UI SETUP
# ======================
st.set_page_config(page_title="Spectral Clustering with PSO", layout="wide", page_icon="📊")

st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
    .stSelectbox, .stNumberInput { margin-bottom: 15px; }
    .plot-container { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
    .metric-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 15px; }
    .stHorizontalBlock { display: flex; justify-content: center; margin-bottom: 30px; }
    .stHorizontalBlock [data-baseweb="tab"] { padding: 10px 20px; background-color: #e8f5e9; border-radius: 5px; margin: 0 5px; }
    .stHorizontalBlock [aria-selected="true"] { background-color: #4CAF50 !important; color: white !important; }
    .landing-header { text-align: center; margin-bottom: 30px; }
    .feature-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }
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

    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("✅ Data berhasil dimuat!")
            
            with st.expander("📄 Lihat Data Mentah"):
                st.dataframe(df)
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")

def exploratory_data_analysis():
    st.header("🔍 Exploratory Data Analysis (EDA)")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        return
    
    df = st.session_state.df
    
    st.subheader("Informasi Dataset")
    buffer = StringIO()
    sys.stdout = buffer
    df.info()
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())
    
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe().T)
    
    st.subheader("Pengecekan Nilai Kosong")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("Tidak ada nilai kosong pada dataset")
    else:
        st.dataframe(missing_values[missing_values > 0].to_frame("Jumlah Nilai Kosong"))
        if st.button("Hapus Baris dengan Nilai Kosong"):
            df = df.dropna()
            st.session_state.df = df
            st.success("Baris dengan nilai kosong telah dihapus!")
    
    st.subheader("Distribusi Variabel Numerik")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("Pilih variabel:", numeric_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, bins=30, color='skyblue')
        ax.set_title(f'Distribusi {selected_col}')
        st.pyplot(fig)
    else:
        st.warning("Tidak ada kolom numerik dalam dataset")
    
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
    
    # Pilih kolom numerik saja
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("Tidak ada kolom numerik dalam dataset")
        return
    
    X = df[numeric_cols]
    
    st.subheader("Contoh Data Sebelum Scaling")
    st.dataframe(X.head())

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    X_scaled = scaler.fit_transform(X)

    st.session_state.X_scaled = X_scaled
    st.session_state.feature_names = numeric_cols
    st.session_state.df_cleaned = df

    st.subheader("Contoh Data setelah Scaling")
    st.dataframe(pd.DataFrame(X_scaled, columns=numeric_cols).head())

def optimized_clustering_analysis():
    st.header("🚀 Spectral Clustering dengan PSO (Optimized)")
    
    if 'X_scaled' not in st.session_state or st.session_state.X_scaled is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # Evaluasi jumlah cluster optimal
    st.subheader("1. Evaluasi Jumlah Cluster Optimal")
    
    silhouette_scores = []
    db_scores = []
    k_range = range(2, min(11, X_scaled.shape[0]))  # Pastikan tidak melebihi jumlah sampel

    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    for i, k in enumerate(k_range):
        progress_text.text(f"Menghitung untuk k={k}...")
        progress_bar.progress((i+1)/len(k_range))
        
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
        labels = model.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))

    progress_text.empty()
    progress_bar.empty()

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
    st.success(f"Jumlah cluster optimal berdasarkan silhouette score: k={optimal_k}")
    
    # Spectral Clustering Manual
    st.subheader("2. Spectral Clustering Manual (γ=0.1)")

    gamma = 0.1
    W = rbf_kernel(X_scaled, gamma=gamma)
    threshold = 0.01
    W[W < threshold] = 0

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = eigh(L_sym)
    k = optimal_k
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
    
    # Optimasi Gamma dengan PSO
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    if st.button("🚀 Jalankan Optimasi PSO"):
        with st.spinner('Menjalankan optimasi PSO...'):
            best_gamma, cost_history, gamma_history = run_pso_optimization(X_scaled, optimal_k)
            
            st.session_state.best_gamma = best_gamma
            st.session_state.cost_history = cost_history
            st.session_state.gamma_history = gamma_history
            
            # Spectral Clustering dengan gamma optimal
            W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
            W_opt[W_opt < threshold] = 0
            
            D_opt = np.diag(W_opt.sum(axis=1))
            D_inv_sqrt_opt = np.diag(1.0 / np.sqrt(W_opt.sum(axis=1)))
            L_sym_opt = np.eye(W_opt.shape[0]) - D_inv_sqrt_opt @ W_opt @ D_inv_sqrt_opt
            
            eigvals_opt, eigvecs_opt = eigh(L_sym_opt)
            U_opt = eigvecs_opt[:, :k]
            U_norm_opt = U_opt / np.linalg.norm(U_opt, axis=1, keepdims=True)
            
            kmeans_opt = KMeans(n_clusters=k, random_state=SEED, n_init=10)
            labels_opt = kmeans_opt.fit_predict(U_norm_opt)
            
            st.session_state.U_opt = U_norm_opt
            st.session_state.labels_opt = labels_opt
            
            sil_score_opt = silhouette_score(U_norm_opt, labels_opt)
            dbi_score_opt = davies_bouldin_score(U_norm_opt, labels_opt)
            
            st.success(f"Optimasi selesai! Gamma optimal: {best_gamma:.4f}")
            st.success(f"Hasil clustering dengan gamma optimal - Silhouette: {sil_score_opt:.4f}, DBI: {dbi_score_opt:.4f}")
            
            # Visualisasi hasil clustering dengan gamma optimal
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(U_norm_opt[:, 0], U_norm_opt[:, 1], c=labels_opt, cmap='viridis', alpha=0.7)
            plt.title(f'Spectral Clustering dengan Gamma Optimal ({best_gamma:.4f})\nSilhouette: {sil_score_opt:.4f}, DBI: {dbi_score_opt:.4f}')
            plt.xlabel('Eigenvector 1')
            plt.ylabel('Eigenvector 2')
            st.pyplot(fig)
            
            # Simpan hasil clustering ke dataframe
            df = st.session_state.df_cleaned.copy()
            df['Cluster'] = labels_opt
            st.session_state.df_clustered = df

def results_analysis():
    st.header("📊 Hasil Analisis Cluster")
    
    if 'df_clustered' not in st.session_state:
        st.warning("Silakan jalankan clustering terlebih dahulu")
        return
    
    df = st.session_state.df_clustered
    
    st.subheader("1. Distribusi Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    st.subheader("2. Karakteristik per Cluster")
    
    if 'df_cleaned' in st.session_state:
        original_df = st.session_state.df_cleaned.copy()
        
        if 'Kabupaten/Kota' in original_df.columns and 'Kabupaten/Kota' in df.columns:
            original_df = original_df.merge(
                df[['Kabupaten/Kota', 'Cluster']],
                on='Kabupaten/Kota',
                how='left'
            )
            
            numeric_cols = original_df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [col for col in numeric_cols if col != 'Cluster']
            
            if 'Cluster' in original_df.columns and len(numeric_cols) > 0:
                cluster_means = original_df.groupby('Cluster')[numeric_cols].mean()
                
                st.dataframe(cluster_means.style.format("{:.2f}").background_gradient(cmap='Blues'))
            else:
                st.warning("Tidak ada kolom numerik untuk ditampilkan")
        else:
            st.warning("Kolom 'Kabupaten/Kota' tidak ditemukan untuk penggabungan data")
    
    st.subheader("3. Feature Importance")
    if 'feature_names' in st.session_state and 'X_scaled' in st.session_state:
        X = st.session_state.X_scaled
        y = df['Cluster']
        
        rf = RandomForestClassifier(random_state=SEED)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, 
                               index=st.session_state.feature_names).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis")
        ax.set_title("Faktor Paling Berpengaruh dalam Clustering")
        st.pyplot(fig)
    else:
        st.warning("Data tidak tersedia untuk menghitung feature importance")
    
    if 'Kabupaten/Kota' in df.columns:
        st.subheader("4. Pemetaan Daerah per Cluster")
        
        try:
            if 'df_cleaned' in st.session_state:
                merged_df = pd.merge(
                    df[['Kabupaten/Kota', 'Cluster']],
                    st.session_state.df_cleaned,
                    on='Kabupaten/Kota',
                    how='left'
                )
                
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
                
                kesehatan_vars = ['Angka Harapan Hidup (Tahun)']
                ipm_vars = ['Indeks Pembangunan Manusia']
                
                available_vars = {
                    'Kemiskinan': [v for v in kemiskinan_vars if v in merged_df.columns],
                    'Pendidikan': [v for v in pendidikan_vars if v in merged_df.columns],
                    'Ketenagakerjaan': [v for v in ketenagakerjaan_vars if v in merged_df.columns],
                    'Kesehatan': [v for v in kesehatan_vars if v in merged_df.columns],
                    'IPM': [v for v in ipm_vars if v in merged_df.columns]
                }
                
                display_cols = ['Kabupaten/Kota', 'Cluster']
                sort_by = 'Cluster'
                
                for category, vars_list in available_vars.items():
                    if vars_list:
                        display_cols.append(vars_list[0])
                        if category == 'Kemiskinan':
                            sort_by = vars_list[0]
                
                merged_df = merged_df.sort_values([sort_by, 'Kabupaten/Kota'], ascending=[False, True])
                
                st.dataframe(
                    merged_df[display_cols],
                    height=600,
                    column_config={
                        'Persentase Penduduk Miskin (%)': st.column_config.NumberColumn(format="%.2f %%"),
                        'Garis Kemiskinan (Rupiah/Bulan/Kapita)': st.column_config.NumberColumn(format="%,d")
                    }
                )
                
                st.subheader("Analisis Indikator per Cluster")
                
                analysis_var = st.selectbox(
                    "Pilih indikator untuk analisis cluster:",
                    options=[v for vars_list in available_vars.values() for v in vars_list]
                )
                
                if analysis_var in merged_df.columns:
                    cluster_stats = merged_df.groupby('Cluster')[analysis_var].describe()
                    st.write(f"Statistik {analysis_var} per Cluster:")
                    st.dataframe(cluster_stats.style.format("{:.2f}"))
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(data=merged_df, x='Cluster', y=analysis_var, palette='viridis')
                    plt.title(f'Distribusi {analysis_var} per Cluster')
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam pemetaan: {str(e)}")

    if all(key in st.session_state for key in ['U_before', 'labels_before', 'U_opt', 'labels_opt']):
        st.subheader("5. Perbandingan Hasil Sebelum dan Sesudah Optimasi")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sebelum Optimasi (γ=0.1):**")
            st.write(f"- Silhouette Score: {silhouette_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
            st.write(f"- Davies-Bouldin Index: {davies_bouldin_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
            
        with col2:
            st.markdown(f"**Sesudah Optimasi (γ={st.session_state.get('best_gamma', 0):.4f}):**")
            st.write(f"- Silhouette Score: {silhouette_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
            st.write(f"- Davies-Bouldin Index: {davies_bouldin_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
        
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
    
    st.subheader("6. Implementasi dan Rekomendasi Kebijakan")
    
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

    st.download_button(
        label="📥 Download Hasil Clustering",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='hasil_clustering.csv',
        mime='text/csv'
    )

# ======================
# APP LAYOUT
# ======================

menu_options = {
    "Beranda": landing_page,
    "Upload Data": upload_data,
    "EDA": exploratory_data_analysis,
    "Preprocessing": data_preprocessing,
    "Clustering": optimized_clustering_analysis,
    "Results": results_analysis
}

menu_selection = st.radio(
    "Menu Navigasi",
    list(menu_options.keys()),
    index=0,
    key="menu",
    horizontal=True,
    label_visibility="hidden"
)

menu_options[menu_selection]()
