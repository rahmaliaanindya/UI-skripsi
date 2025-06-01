
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
            sil_sum = 0.0
            dbi_sum = 0.0
            
            for _ in range(2):  # 2 runs untuk stabilisasi
                try:
                    W = numba_rbf_kernel(self.X_scaled, gamma_val)
                    L = numba_laplacian(W)
                    eigvals, eigvecs = numba_eigsh(L, self.best_cluster)
                    U = numba_normalize(eigvecs)
                    labels = fast_kmeans(U, self.best_cluster)
                    
                    sil = numba_silhouette_score(U, labels)
                    dbi = numba_davies_bouldin_score(U, labels)
                    
                    sil_sum += sil
                    dbi_sum += dbi
                except:
                    sil_sum += 0.0
                    dbi_sum += 10.0
            
            scores[i] = -sil_sum/2 + dbi_sum/2
        
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
def fast_kmeans(U, n_clusters, max_iter=10):
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
    
    # Setup PSO
    optimizer = GlobalBestPSO(
        n_particles=20,
        dimensions=1,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.7},
        bounds=(np.array([0.001]), np.array([5.0]))
    )
    
    # Jalankan optimasi dengan callback untuk update UI
    def update_progress(optimizer):
        current_iter = pso_optimizer.iteration
        total_iter = 50
        
        # Update progress bar
        progress = int((current_iter / total_iter) * 100)
        progress_bar.progress(progress)
        
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
        iters=50,
        n_processes=4,
        verbose=False
    )
    
    # Final update
    update_progress(optimizer)
    
    return best_pos[0], pso_optimizer

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
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        
        with st.expander("📄 Lihat Data Mentah"):
            st.dataframe(df)

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
    st.dataframe(df.describe())
    
    st.subheader("Pengecekan Nilai Kosong")
    missing_values = df.isnull().sum()
    if missing_values.sum() == 0:
        st.success("Tidak ada nilai kosong pada dataset")
    else:
        st.dataframe(missing_values[missing_values > 0].to_frame("Jumlah Nilai Kosong"))
    
    st.subheader("Distribusi Variabel Numerik")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Pilih variabel:", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[selected_col], kde=True, bins=30, color='skyblue')
    ax.set_title(f'Distribusi {selected_col}')
    st.pyplot(fig)
    
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
    st.session_state.df_cleaned = df.copy()

    X = df.drop(columns=['Kabupaten/Kota'])  
    
    st.subheader("Contoh Data Sebelum Scaling")
    st.dataframe(X)

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    X_scaled = scaler.fit_transform(X)

    st.session_state.X_scaled = X_scaled
    st.session_state.feature_names = X.columns.tolist()

    st.subheader("Contoh Data setelah Scaling")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns))

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
    
    # Pilih cluster optimal
    st.subheader("2. Menentukan Cluster Optimal")
    
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
    
    # Spectral Clustering Manual
    st.subheader("3. Spectral Clustering Manual (γ=0.1)")

    gamma = 0.1
    W = rbf_kernel(X_scaled, gamma=gamma)
    threshold = 0.01
    W[W < threshold] = 0

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    eigvals, eigvecs = eigh(L_sym)
    k = best_cluster
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
    st.subheader("4. Optimasi Gamma dengan PSO")
    
    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        try:
            # Setup progress containers
            progress_container = st.container()
            with progress_container:
                st.write("**Proses Optimasi**")
                progress_col1, progress_col2 = st.columns(2)
                
                with progress_col1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                with progress_col2:
                    st.write("**Konvergensi PSO**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    chart_placeholder = st.empty()
            
            # Initialize optimizer
            pso_optimizer = PSOOptimizer(
                X_scaled=X_scaled,
                best_cluster=best_cluster
            )
            
            # PSO configuration
            options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
            bounds = (np.array([0.001]), np.array([5.0]))
            optimizer = GlobalBestPSO(
                n_particles=20,
                dimensions=1,
                options=options,
                bounds=bounds
            )
            
            # Run optimization with progress updates
            start_time = time.time()
            
            def update_progress(optimizer):
                current_iter = pso_optimizer.iteration
                total_iter = 50
                
                # Update progress bar
                progress_bar.progress(int((current_iter / total_iter) * 100))
                
                # Update status text
                if len(pso_optimizer.cost_history) > 0:
                    best_cost = min(pso_optimizer.cost_history)
                    best_gamma = pso_optimizer.gamma_history[np.argmin(pso_optimizer.cost_history)]
                    status_text.markdown(
                        f"**Iterasi:** {current_iter}/{total_iter}\n"
                        f"**Gamma Terbaik:** {best_gamma:.4f}\n"
                        f"**Cost Terbaik:** {best_cost:.4f}"
                    )
                
                # Update convergence plot
                ax.clear()
                ax.plot(range(1, current_iter+1), pso_optimizer.cost_history, 'b-o', markersize=4)
                ax.set_xlabel('Iterasi')
                ax.set_ylabel('Nilai Cost')
                ax.set_title('Progres Konvergensi PSO')
                ax.grid(True)
                chart_placeholder.pyplot(fig)
                
                # Small delay for UI update
                time.sleep(0.1)
            
            # Run optimization
            best_cost, best_pos = optimizer.optimize(
                pso_optimizer.evaluate,
                iters=50,
                n_processes=1,
                verbose=False
            )
            
            # Final update
            update_progress(optimizer)
            elapsed_time = time.time() - start_time
            
            # Process results
            best_gamma = best_pos[0]
            progress_container.success(
                f"Optimasi selesai dalam {elapsed_time:.2f} detik! "
                f"Gamma optimal: {best_gamma:.4f}"
            )
            
            # Evaluasi hasil optimal
            W_opt = numba_rbf_kernel(X_scaled, best_gamma)
            L_opt = numba_laplacian(W_opt)
            eigvals_opt, eigvecs_opt = numba_eigsh(L_opt, best_cluster)
            U_opt = numba_normalize(eigvecs_opt)
            
            labels_opt = fast_kmeans(U_opt, best_cluster)
            
            # Store results in session state
            st.session_state.update({
                'U_opt': U_opt,
                'labels_opt': labels_opt,
                'best_gamma': best_gamma,
                'optimizer_history': {
                    'cost': pso_optimizer.cost_history,
                    'gamma': pso_optimizer.gamma_history
                }
            })
            
            # Calculate metrics
            sil_opt = numba_silhouette_score(U_opt, labels_opt)
            dbi_opt = numba_davies_bouldin_score(U_opt, labels_opt)
            
            # Show results comparison
            st.subheader("Hasil Optimasi")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Silhouette Score", 
                         f"{sil_opt:.4f}", 
                         f"{(sil_opt - sil_score):.4f} vs baseline")
                
                # Convergence plot
                fig_conv, ax_conv = plt.subplots(figsize=(8, 4))
                ax_conv.plot(pso_optimizer.cost_history, 'b-o', markersize=4)
                ax_conv.set_title('Konvergensi Nilai Cost')
                ax_conv.set_xlabel('Iterasi')
                ax_conv.set_ylabel('Cost')
                ax_conv.grid(True)
                st.pyplot(fig_conv)
                
            with col2:
                st.metric("Davies-Bouldin Index", 
                         f"{dbi_opt:.4f}", 
                         f"{(dbi_score - dbi_opt):.4f} vs baseline")
                
                # Gamma history
                fig_gamma, ax_gamma = plt.subplots(figsize=(8, 4))
                ax_gamma.plot(pso_optimizer.gamma_history, 'r-o', markersize=4)
                ax_gamma.set_title('Perkembangan Gamma')
                ax_gamma.set_xlabel('Iterasi')
                ax_gamma.set_ylabel('Nilai Gamma')
                ax_gamma.grid(True)
                st.pyplot(fig_gamma)
            
            # Cluster visualization
            st.subheader("Visualisasi Cluster")
            pca = PCA(n_components=2)
            U_before_pca = pca.fit_transform(st.session_state.U_before)
            U_opt_pca = pca.transform(U_opt)
            
            fig_cluster, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Before optimization
            scatter1 = ax1.scatter(U_before_pca[:,0], U_before_pca[:,1], 
                                 c=st.session_state.labels_before, 
                                 cmap='viridis', s=50, alpha=0.7)
            ax1.set_title(f"Sebelum PSO (γ=0.1)\nSilhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")
            ax1.set_xlabel("PC1")
            ax1.set_ylabel("PC2")
            plt.colorbar(scatter1, ax=ax1, label='Cluster')
            
            # After optimization
            scatter2 = ax2.scatter(U_opt_pca[:,0], U_opt_pca[:,1], 
                                 c=labels_opt, 
                                 cmap='viridis', s=50, alpha=0.7)
            ax2.set_title(f"Sesudah PSO (γ={best_gamma:.4f})\nSilhouette: {sil_opt:.4f}, DBI: {dbi_opt:.4f}")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            plt.colorbar(scatter2, ax=ax2, label='Cluster')
            
            st.pyplot(fig_cluster)
            
            # Save clustered data
            if 'df_cleaned' in st.session_state:
                df = st.session_state.df_cleaned.copy()
            else:
                df = st.session_state.df.copy()
            
            df['Cluster'] = labels_opt
            st.session_state.df_clustered = df
            
            # Show cluster distribution
            st.subheader("Distribusi Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)
            
            # Show cluster mapping if available
            if 'Kabupaten/Kota' in df.columns:
                st.subheader("Pemetaan Cluster")
                st.dataframe(
                    df[['Kabupaten/Kota', 'Cluster']].sort_values('Cluster'),
                    height=400,
                    hide_index=True
                )
    
        except Exception as e:
            st.error(f"Terjadi kesalahan dalam optimasi PSO: {str(e)}")
            st.text(traceback.format_exc())

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
            
            if 'Cluster' in original_df.columns:
                cluster_means = original_df.groupby('Cluster')[numeric_cols].mean()
                
                if 'PDRB' in numeric_cols:
                    cluster_order = cluster_means['PDRB'].sort_values().index
                    cluster_means = cluster_means.loc[cluster_order]
                
                st.dataframe(cluster_means.style.format("{:.2f}").background_gradient(cmap='Blues'))
            else:
                st.warning("Kolom 'Cluster' tidak ditemukan di data asli")
        else:
            st.warning("Tidak dapat menggabungkan data karena kolom 'Kabupaten/Kota' tidak ditemukan")
    
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
            if 'merged_df' in locals():
                st.write("Kolom yang tersedia:", merged_df.columns.tolist())

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
            
            st.markdown("**3 Kota Kemiskinan Tinggi:**")
            poorest = merged_df.nlargest(3, main_indicator)[['Kabupaten/Kota', 'Cluster', main_indicator]]
            st.dataframe(
                poorest.style.format({
                    main_indicator: "{:.2f} %" if "%" in main_indicator else "Rp {:,}" if "Rupiah" in main_indicator else "{:.2f}"
                }),
                hide_index=True
            )
            
            st.markdown("**3 Kota Kemiskinan Rendah:**")
            least_poor = merged_df.nsmallest(3, main_indicator)[['Kabupaten/Kota', 'Cluster', main_indicator]]
            st.dataframe(
                least_poor.style.format({
                    main_indicator: "{:.2f} %" if "%" in main_indicator else "Rp {:,}" if "Rupiah" in main_indicator else "{:.2f}"
                }),
                hide_index=True
            )
    
    if all(key in st.session_state for key in ['U_before', 'labels_before', 'U_opt', 'labels_opt']):
        st.subheader("6. Perbandingan Hasil Sebelum dan Sesudah Optimasi")
        
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
