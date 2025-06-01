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

    # =============================================
    # 4. OPTIMASI GAMMA DENGAN PSO - REVISI TANPA CALLBACK
    # =============================================
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    def evaluate_gamma_robust(gamma_values):
        """Fungsi evaluasi untuk PSO yang lebih robust terhadap error."""
        try:
            gamma = gamma_values[0] if isinstance(gamma_values, (np.ndarray, list)) else gamma_values
            # Hitung matriks kernel
            W = rbf_kernel(X_scaled, gamma=gamma)
            
            # Handle potential numerical issues
            if np.allclose(W, 0) or np.any(np.isnan(W)) or np.any(np.isinf(W)):
                return np.inf
                
            # Hitung matriks Laplacian
            L = laplacian(W, normed=True)
            
            # Handle potential numerical issues
            if np.any(np.isnan(L.data)) or np.any(np.isinf(L.data)):
                return np.inf
                
            # Hitung eigenvectors
            eigvals, eigvecs = eigsh(L, k=best_cluster, which='SM', tol=1e-6)
            U = normalize(eigvecs, norm='l2')
            
            # Handle potential numerical issues
            if np.isnan(U).any() or np.isinf(U).any():
                return np.inf
                
            # Clustering
            kmeans = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=10)
            labels = kmeans.fit_predict(U)
            
            # Hitung metrics
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(U, labels)
                dbi = davies_bouldin_score(U, labels)
                
                # Gabungkan metrics (kita ingin memaksimalkan silhouette dan meminimalkan DBI)
                fitness = (1 - silhouette) + dbi  # Gabungan kedua metrics
                return fitness
            else:
                return np.inf
                
        except Exception as e:
            st.warning(f"Error pada gamma={gamma:.4f}: {str(e)}")
            return np.inf
    
    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        with st.spinner("Menjalankan optimasi PSO (mungkin memakan waktu beberapa menit)..."):
            try:
                # Dictionary untuk menyimpan history
                history = {
                    'iteration': [],
                    'g_best': [],
                    'best_gamma': [],
                    'silhouette': [],
                    'dbi': [],
                    'pbest_history': [],
                    'gbest_history': []
                }
                
                # Setup optimizer
                options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
                bounds = (np.array([0.001]), np.array([2.0]))
                
                optimizer = GlobalBestPSO(
                    n_particles=10,
                    dimensions=1,
                    options=options,
                    bounds=bounds
                )
                
                # Buat progress bar
                progress_bar = st.progress(0, text="Memulai optimasi...")
                
                # Jalankan optimasi secara manual per iterasi
                for i in range(30):
                    # Jalankan satu iterasi
                    cost, pos = optimizer.optimize(evaluate_gamma_robust, iters=1)
                    
                    # Simpan history
                    best_gamma = pos[0][0]  # Accessing the first element directly
                    
                    # Hitung metrics untuk best gamma
                    try:
                        W = rbf_kernel(X_scaled, gamma=best_gamma)
                        L = laplacian(W, normed=True)
                        eigvals, eigvecs = eigsh(L, k=best_cluster, which='SM', tol=1e-6)
                        U = normalize(eigvecs, norm='l2')
                        kmeans = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=10)
                        labels = kmeans.fit_predict(U)
                        
                        silhouette = silhouette_score(U, labels) if len(np.unique(labels)) > 1 else np.nan
                        dbi = davies_bouldin_score(U, labels) if len(np.unique(labels)) > 1 else np.nan
                    except:
                        silhouette = np.nan
                        dbi = np.nan
                    
                    history['iteration'].append(i)
                    history['g_best'].append(cost)
                    history['best_gamma'].append(best_gamma)
                    history['silhouette'].append(silhouette)
                    history['dbi'].append(dbi)
                    history['pbest_history'].append(optimizer.swarm.pbest_pos.copy())
                    history['gbest_history'].append(optimizer.swarm.best_pos.copy())
                    
                    # Update progress bar
                    progress = (i + 1) / 30
                    progress_bar.progress(progress, text=f"Iterasi {i + 1}/30 - Best Gamma: {best_gamma:.4f}")
                
                best_gamma = history['best_gamma'][np.argmin(history['g_best'])]
                st.session_state.best_gamma = best_gamma
                st.session_state.pso_history = history
                
                st.success(f"**Optimasi selesai!** Gamma optimal: {best_gamma:.4f}")
                
                # =============================================
                # 5. TAMPILKAN HASIL OPTIMASI
                # =============================================
                
                # Visualisasi evolusi PBest dan GBest
                st.subheader("Evolusi PBest dan GBest")
                
                # Buat dataframe untuk visualisasi
                pbest_evolution = []
                for i, pbest_iter in enumerate(history['pbest_history']):
                    for j, pbest in enumerate(pbest_iter):
                        pbest_evolution.append({
                            'Iteration': i,
                            'Particle': j,
                            'Gamma': pbest[0],
                            'Type': 'PBest'
                        })
                
                gbest_evolution = []
                for i, gbest_iter in enumerate(history['gbest_history']):
                    gbest_evolution.append({
                        'Iteration': i,
                        'Particle': 'GBest',
                        'Gamma': gbest_iter[0],
                        'Type': 'GBest'
                    })
                
                evolution_df = pd.DataFrame(pbest_evolution + gbest_evolution)
                
                # Plot evolusi
                fig_evolution, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(
                    data=evolution_df, 
                    x='Iteration', 
                    y='Gamma', 
                    hue='Type', 
                    style='Type',
                    markers=True,
                    dashes=False,
                    ax=ax
                )
                ax.set_title("Evolusi Nilai Gamma pada PBest dan GBest")
                ax.set_xlabel("Iterasi")
                ax.set_ylabel("Nilai Gamma")
                st.pyplot(fig_evolution)

                # Tampilkan grafik konvergensi
                fig_convergence, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history['iteration'], history['g_best'], 'b-', label='Global Best')
                ax.set_xlabel('Iterasi')
                ax.set_ylabel('Nilai Fitness (G-best)')
                ax.set_title('Konvergensi PSO')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig_convergence)
                
                # Tampilkan iterasi terbaik
                best_iter_idx = np.argmin(history['g_best'])
                st.info(f"""
                **Iterasi terbaik:** {history['iteration'][best_iter_idx]}  
                **Gamma terbaik:** {history['best_gamma'][best_iter_idx]:.4f}  
                **Silhouette Score:** {history['silhouette'][best_iter_idx]:.4f}  
                **Davies-Bouldin Index:** {history['dbi'][best_iter_idx]:.4f}
                """)
                
                # Tampilkan tabel history
                st.subheader("History Optimasi PSO")
                history_df = pd.DataFrame({
                    'Iterasi': history['iteration'],
                    'Gamma': history['best_gamma'],
                    'G-best': history['g_best'],
                    'Silhouette': history['silhouette'],
                    'DBI': history['dbi']
                })
                st.dataframe(history_df.style.format({
                    'Gamma': '{:.4f}',
                    'G-best': '{:.4f}',
                    'Silhouette': '{:.4f}',
                    'DBI': '{:.4f}'
                }).highlight_min(subset=['G-best', 'DBI'], color='lightgreen')
                                .highlight_max(subset=['Silhouette'], color='lightgreen'))
                
                # =============================================
                # 6. CLUSTERING DENGAN GAMMA OPTIMAL
                # =============================================
                W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
                
                if not (np.allclose(W_opt, 0) or np.any(np.isnan(W_opt)) or np.any(np.isinf(W_opt))):
                    L_opt = laplacian(W_opt, normed=True)
                    
                    if not (np.any(np.isnan(L_opt.data)) or np.any(np.isinf(L_opt.data))):
                        eigvals_opt, eigvecs_opt = eigsh(L_opt, k=best_cluster, which='SM', tol=1e-6)
                        U_opt = normalize(eigvecs_opt, norm='l2')

                        if not (np.isnan(U_opt).any() or np.isinf(U_opt).any()):
                            kmeans_opt = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=10)
                            labels_opt = kmeans_opt.fit_predict(U_opt)

                            if len(np.unique(labels_opt)) > 1:
                                st.session_state.U_opt = U_opt
                                st.session_state.labels_opt = labels_opt
                                
                                sil_opt = silhouette_score(U_opt, labels_opt)
                                dbi_opt = davies_bouldin_score(U_opt, labels_opt)
                                
                                col1, col2 = st.columns(2)
                                col1.metric("Silhouette Score", f"{sil_opt:.4f}", 
                                           f"{(sil_opt - sil_score):.4f} vs baseline")
                                col2.metric("Davies-Bouldin Index", f"{dbi_opt:.4f}", 
                                           f"{(dbi_score - dbi_opt):.4f} vs baseline")
                                
                                # =============================================
                                # 7. VISUALISASI HASIL
                                # =============================================
                                st.subheader("4. Visualisasi Hasil")
                                
                                pca = PCA(n_components=2)
                                U_before_pca = pca.fit_transform(st.session_state.U_before)
                                U_opt_pca = pca.transform(U_opt)
                                
                                fig_comparison, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
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
                                
                                st.pyplot(fig_comparison)
                                
                                # =============================================
                                # 8. SIMPAN HASIL KE DATAFRAME
                                # =============================================
                                try:
                                    if 'df_cleaned' in st.session_state and st.session_state.df_cleaned is not None:
                                        df = st.session_state.df_cleaned.copy()
                                    else:
                                        df = st.session_state.df.copy()
                                    
                                    df['Cluster'] = labels_opt
                                    st.session_state.df_clustered = df
                                    
                                    st.subheader("Distribusi Cluster")
                                    cluster_counts = df['Cluster'].value_counts().sort_index()
                                    st.bar_chart(cluster_counts)
                                    
                                    if 'Kabupaten/Kota' in df.columns:
                                        st.subheader("Pemetaan Cluster")
                                        st.dataframe(df[['Kabupaten/Kota', 'Cluster']].sort_values('Cluster'))
                                        
                                except Exception as e:
                                    st.error(f"Error dalam menyimpan hasil: {str(e)}")
                            else:
                                st.error("Hanya 1 cluster yang terbentuk, evaluasi gagal.")
                        else:
                            st.error("Matriks fitur U mengandung nilai NaN atau inf.")
                    else:
                        st.error("Matriks Laplacian mengandung nilai NaN atau inf.")
                else:
                    st.error("Matriks kernel W mengandung nilai NaN, inf, atau nol semua.")

            except Exception as e:
                st.error(f"Terjadi kesalahan dalam optimasi PSO: {str(e)}")

def results_analysis():
    st.header("📊 Results Analysis")
    
    if 'df_clustered' not in st.session_state:
        st.warning("Silakan lakukan clustering terlebih dahulu")
        return
    
    df = st.session_state.df_clustered
    
    # Show clustered data
    st.subheader("Clustered Data")
    st.dataframe(df)
    
    # Cluster statistics
    st.subheader("Cluster Statistics")
    cluster_stats = df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
    
    # Feature importance analysis
    if 'X_scaled' in st.session_state and 'labels_opt' in st.session_state:
        st.subheader("Feature Importance Analysis")
        
        X = st.session_state.X_scaled
        y = st.session_state.labels_opt
        
        # Train Random Forest to get feature importance
        rf = RandomForestClassifier(random_state=SEED)
        rf.fit(X, y)
        
        # Get feature importance
        importance = rf.feature_importances_
        feature_names = st.session_state.feature_names
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title('Feature Importance for Clustering')
        st.pyplot(fig)
        
        st.dataframe(importance_df)

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
