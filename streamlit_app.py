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
    st.markdown("""
    "Count" = jumlah kabupaten/kota yang termasuk dalam satu kelompok nilai.
    """)
    
    # Correlation Matrix
    st.subheader("Matriks Korelasi")
    numerical_df = df.select_dtypes(include=['number'])
    if len(numerical_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(fig)
    else:
        st.warning("Tidak cukup variabel numerik untuk menampilkan matriks korelasi")
    st.markdown("""
    - **Heatmap korelasi** menunjukkan hubungan antar variabel dalam bentuk angka dan warna.
    - **Nilai korelasi** berkisar dari **-1 sampai 1**:
      - `+1` → Hubungan positif sempurna (jika satu variabel naik, yang lain ikut naik)
      - `0` → Tidak ada hubungan linear
      - `-1` → Hubungan negatif sempurna (jika satu variabel naik, yang lain turun)
    - **Warna biru** menunjukkan korelasi negatif, **warna merah** menunjukkan korelasi positif.
    - Semakin gelap warnanya, semakin kuat hubungan antar variabel tersebut.
    """)

def data_preprocessing():
    st.header("⚙️ Data Preprocessing")
    st.markdown("""
    Pada tahap ini, dilakukan proses **data preprocessing** dengan tujuan menyiapkan data agar dapat digunakan dalam proses analisis dan pemodelan. Tahapan preprocessing yang dilakukan meliputi:

    1. **Menghapus Kolom Non-Numerik:**  
       Kolom `'Kabupaten/Kota'` dihapus karena bersifat kategorikal dan tidak dibutuhkan dalam proses perhitungan numerik seperti clustering.

    2. **Scaling (Normalisasi Data):**  
       Data dinormalisasi menggunakan **RobustScaler**, yaitu metode scaling yang tidak sensitif terhadap outlier.  
       - Fungsi `RobustScaler` adalah untuk mengubah skala data agar setiap fitur memiliki rentang nilai yang sebanding.  
       - Scaling ini penting agar algoritma seperti **Spectral Clustering** atau optimasi **PSO (Particle Swarm Optimization)** tidak bias terhadap fitur dengan skala lebih besar.

    Metode RobustScaler bekerja dengan cara mengurangi median dan membaginya dengan rentang interkuartil (IQR = Q3 - Q1), sehingga lebih stabil terhadap data ekstrem (outlier).
    """)

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

    st.markdown("""
    - **Plot klasterisasi Spectral Clustering** menampilkan pemetaan data ke ruang eigenvector utama.
    - **Setiap titik** mewakili satu kabupaten/kota yang diproyeksikan ke dua dimensi utama (Eigenvector 1 dan Eigenvector 2).
    - **Warna titik** menunjukkan klaster atau kelompok yang terbentuk berdasarkan kemiripan struktur data.
    
    - **Titik-titik yang berdekatan** dalam visualisasi cenderung memiliki karakteristik yang mirip berdasarkan variabel yang dianalisis.
    - **Nilai Silhouette Score** mengukur kualitas pemisahan klaster:
      - Nilai mendekati **1.0** → Pemisahan antar klaster sangat baik.
      - Nilai di atas **0.5** → Klaster dianggap cukup baik.
    - **Nilai Davies-Bouldin Index (DBI)** menilai kepadatan dan jarak antar klaster:
      - Nilai yang **semakin kecil** → Kualitas klaster semakin baik (klaster kompak dan terpisah jelas).
    """)

    
    # =============================================
    # 4. OPTIMASI GAMMA DENGAN PSO - REVISI TANPA CALLBACK
    # =============================================
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    def evaluate_gamma_robust(gamma_values):
        """Fungsi evaluasi untuk PSO yang lebih robust terhadap error."""
        try:
            # Handle both scalar and array inputs
            if isinstance(gamma_values, (np.ndarray, list)):
                gamma = gamma_values[0] if len(gamma_values) > 0 else 0.1
            else:
                gamma = float(gamma_values)
            
            # Ensure gamma stays within bounds
            gamma = max(0.001, min(gamma, 2.0))
            
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
                bounds = ([0.001], [2.0])  # Format bounds yang lebih jelas
                
                try:
                    optimizer = GlobalBestPSO(
                        n_particles=10,
                        dimensions=1,
                        options=options,
                        bounds=bounds
                    )
                except Exception as e:
                    st.error(f"Gagal inisialisasi PSO: {str(e)}")
                    st.stop()
                
                # Buat progress bar
                progress_bar = st.progress(0, text="Memulai optimasi...")
                
                # Jalankan optimasi secara manual per iterasi
                for i in range(20):
                    try:
                        # Jalankan satu iterasi
                        cost, pos = optimizer.optimize(evaluate_gamma_robust, iters=1)
                        
                        # Ekstrak posisi dengan aman
                        try:
                            best_gamma = float(pos[0])  # Konversi eksplisit ke float
                        except (IndexError, TypeError) as e:
                            st.warning(f"Gagal ekstrak posisi pada iterasi {i}: {str(e)}")
                            best_gamma = 0.1  # Nilai default jika ekstraksi gagal
                        
                        # Pastikan gamma dalam batas yang valid
                        best_gamma = max(0.001, min(best_gamma, 2.0))
                        
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
                        except Exception as e:
                            st.warning(f"Gagal hitung metrics pada iterasi {i}: {str(e)}")
                            silhouette = np.nan
                            dbi = np.nan
                        
                        # Simpan history
                        history['iteration'].append(i)
                        history['g_best'].append(float(cost))
                        history['best_gamma'].append(best_gamma)
                        history['silhouette'].append(silhouette)
                        history['dbi'].append(dbi)
                        history['pbest_history'].append([float(x[0]) for x in optimizer.swarm.pbest_pos])
                        history['gbest_history'].append([float(x[0]) for x in [optimizer.swarm.best_pos]])
                        
                        # Update progress bar
                        progress = (i + 1) / 30
                        progress_bar.progress(progress, text=f"Iterasi {i + 1}/30 - Best Gamma: {best_gamma:.4f}")
                    
                    except Exception as e:
                        st.warning(f"Error pada iterasi {i}: {str(e)}")
                        continue
                
                # Cari gamma terbaik
                if not history['g_best']:
                    st.error("Optimasi gagal - tidak ada hasil yang valid")
                    st.stop()
                
                best_idx = np.argmin(history['g_best'])
                best_gamma = history['best_gamma'][best_idx]
                
                # Validasi gamma akhir
                if not (0.001 <= best_gamma <= 2.0):
                    st.warning(f"Gamma optimal {best_gamma} di luar batas, menggunakan nilai default 0.1")
                    best_gamma = 0.1
                
                st.session_state.best_gamma = best_gamma
                st.session_state.pso_history = history
                
                st.success(f"**Optimasi selesai!** Gamma optimal: {best_gamma:.4f}")
                
                # =============================================
                # 5. TAMPILKAN HASIL OPTIMASI
                # =============================================
                
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
                try:
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
                                    
                                    # Visualisasi hasil
                                    st.subheader("4. Visualisasi Hasil")
                                    
                                    pca = PCA(n_components=2)
                                    U_before_pca = pca.fit_transform(st.session_state.U_before)
                                    U_opt_pca = pca.transform(U_opt)
                                    
                                    fig_comparison, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                    
                                    scatter1 = ax1.scatter(U_before_pca[:,0], U_before_pca[:,1], 
                                                         c=st.session_state.labels_before, 
                                                         cmap='viridis', s=50, alpha=0.7)
                                    ax1.set_title(f"Sebelum PSO (γ=0.1)\nSilhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}")
                                    ax1.set_xlabel("Eigenvector 1")
                                    ax1.set_ylabel("Eigenvector 2")
                                    plt.colorbar(scatter1, ax=ax1, label='Cluster')
                                    
                                    scatter2 = ax2.scatter(U_opt_pca[:,0], U_opt_pca[:,1], 
                                                         c=labels_opt, 
                                                         cmap='viridis', s=50, alpha=0.7)
                                    ax2.set_title(f"Sesudah PSO (γ={best_gamma:.4f})\nSilhouette: {sil_opt:.4f}, DBI: {dbi_opt:.4f}")
                                    ax2.set_xlabel("Eigenvector 1")
                                    ax2.set_ylabel("Eigenvector 2")
                                    plt.colorbar(scatter2, ax=ax2, label='Cluster')
                                    
                                    st.pyplot(fig_comparison)

                                    st.markdown("""
                                    - **Titik-titik pada plot** mewakili setiap observasi (misalnya, Kabupaten/Kota) dalam dataset.
                                    - **Warna Titik (Cluster)** menunjukkan cluster tempat observasi tersebut dikelompokkan oleh algoritma Spectral Clustering.  
                                      Skala warna di sebelah kanan plot (**colorbar**) menunjukkan pemetaan warna ke label cluster (misalnya, 0, 1, 2, dst.).
                                    
                                    - **Metrik Evaluasi Clustering**:  
                                      - **Silhouette Score**  
                                        Merupakan metrik evaluasi yang mengukur seberapa mirip suatu objek dengan clusternya sendiri dibandingkan cluster lain.  
                                        - Nilai berkisar antara **-1 hingga 1**.  
                                        - Nilai mendekati **1** → Objek berada dalam cluster yang tepat dan terpisah dengan baik dari cluster lain.  
                                        - Nilai mendekati **0** → Objek berada di antara dua cluster.  
                                        - Nilai **negatif** → Objek mungkin telah ditetapkan ke cluster yang salah.  
                                        - **Semakin tinggi nilai Silhouette, semakin baik kualitas clustering.**
                                    
                                      - **Davies-Bouldin Index (DBI)**  
                                        Metrik yang mengukur rasio antara dispersi intra-cluster (jarak rata-rata setiap titik ke centroid cluster) dan dispersi inter-cluster (jarak antar centroid cluster).  
                                        - **Semakin rendah nilai DBI, semakin baik kualitas clustering** (klaster lebih kompak dan terpisah jelas).
                                    """)

                                    
                                    # Simpan hasil ke dataframe
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
                    st.error(f"Gagal melakukan clustering dengan gamma optimal: {str(e)}")
    
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam optimasi PSO: {str(e)}")
                st.error("Detail error: " + traceback.format_exc())

def results_analysis():
    st.header("📊 Results Analysis")
    
    if 'df_clustered' not in st.session_state:
        st.warning("Silakan lakukan clustering terlebih dahulu")
        return
    
    df = st.session_state.df_clustered

    st.markdown("""
    ### **Hasil Pengelompokan Data (Clustering)**
    
    Berikut adalah hasil akhir dari proses clustering. Setiap baris data telah diberikan label **Cluster**, yang menunjukkan kelompok atau kategori kemiskinan yang serupa berdasarkan fitur-fitur yang digunakan.  
    Hasil clustering ini akan menjadi dasar dalam analisis lebih lanjut, seperti visualisasi dan evaluasi performa model clustering.
    """)
    
    # Show clustered data
    st.subheader("Clustered Data")
    st.dataframe(df)

    st.markdown("""
    ### **Statistik Tiap Cluster**
    
    Tabel berikut menyajikan **rata-rata nilai setiap indikator** untuk masing-masing klaster.  
    Statistik ini membantu dalam **menginterpretasikan karakteristik dari setiap cluster**, misalnya:
    
    - Cluster dengan nilai **rata-rata persentase penduduk miskin tinggi** bisa diartikan sebagai cluster dengan tingkat kemiskinan relatif lebih parah.
    - Cluster dengan **rata-rata pengeluaran per kapita tinggi dan tingkat pengangguran rendah** dapat menunjukkan wilayah yang relatif lebih sejahtera.
    
    Analisis statistik ini penting untuk memahami profil sosial ekonomi dari masing-masing kelompok hasil clustering.
    """)
    # Cluster statistics
    st.subheader("Cluster Statistics")
    cluster_stats = df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))

    st.markdown("""
    ### **Analisis Feature Importance**
    
    Analisis ini bertujuan untuk mengetahui **seberapa besar pengaruh masing-masing fitur (indikator)** terhadap hasil pengelompokan (clustering) yang telah dilakukan.
    
    Dengan memanfaatkan model **Random Forest**, bisa mengukur tingkat kepentingan relatif dari setiap fitur terhadap label hasil clustering.  
    Fitur dengan nilai importance yang lebih tinggi memiliki kontribusi yang lebih besar dalam menentukan kelompok cluster.
    
    Informasi ini dapat membantu:
    - Mengetahui **indikator utama** dalam klasifikasi kemiskinan.
    - Memberikan **rekomendasi kebijakan** berdasarkan fitur yang paling berpengaruh.
    
    Hasil ditampilkan dalam bentuk visualisasi batang dan tabel pentingnya fitur.
    """)
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

# Pemetaan Daerah per Cluster
    if 'Kabupaten/Kota' in df.columns:
        st.subheader("Pemetaan Daerah per Cluster")

        st.markdown("""
        Berikut ini adalah tabel yang menampilkan daftar Kabupaten/Kota beserta cluster pengelompokan
        dan variabel utama dari kategori yang berbeda (Kemiskinan, Pendidikan, Ketenagakerjaan, Kesehatan, IPM).
        Data diurutkan berdasarkan indikator kemiskinan utama.
        """)
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
                st.markdown("""
                 "Pilih indikator di bawah ini untuk melihat statistik deskriptifnya pada masing-masing cluster."
                """)
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

        except Exception as e:
            st.error(f"Terjadi kesalahan dalam pemetaan: {str(e)}")
            if 'merged_df' in locals():
                st.write("Kolom yang tersedia:", merged_df.columns.tolist())

        # 5. Ranking Kota (Termiskin & Paling Tidak Miskin)
    st.subheader("Ranking Kota Berdasarkan Indikator Kemiskinan")
    
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
            st.markdown("**3 Kota Kemiskinan Tingii:**")
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
    st.subheader("Perbandingan Hasil Sebelum dan Sesudah Optimasi")
    
    if all(key in st.session_state for key in ['U_before', 'labels_before', 'U_opt', 'labels_opt']):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Sebelum Optimasi (γ=0.1):**")
            st.write(f"- Silhouette Score: {silhouette_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
            st.write(f"- Davies-Bouldin Index: {davies_bouldin_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
            
        with col2:
            best_gamma = st.session_state.get('best_gamma', 0)
            st.markdown(f"**Sesudah Optimasi (γ={best_gamma:.4f}):**")
            st.write(f"- Silhouette Score: {silhouette_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
            st.write(f"- Davies-Bouldin Index: {davies_bouldin_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
        
        # Visualisasi dengan format yang sama seperti di PSO
        pca = PCA(n_components=2)
        U_before_pca = pca.fit_transform(st.session_state.U_before)
        U_opt_pca = pca.transform(st.session_state.U_opt)
        
        fig_comparison, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        scatter1 = ax1.scatter(U_before_pca[:,0], U_before_pca[:,1], 
                             c=st.session_state.labels_before, 
                             cmap='viridis', s=50, alpha=0.7)
        ax1.set_title(f"Sebelum PSO (γ=0.1)\nSilhouette: {silhouette_score(st.session_state.U_before, st.session_state.labels_before):.4f}, DBI: {davies_bouldin_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
        ax1.set_xlabel("Eigenvector 1")
        ax1.set_ylabel("Eigenvector 2")
        plt.colorbar(scatter1, ax=ax1, label='Cluster')
        
        scatter2 = ax2.scatter(U_opt_pca[:,0], U_opt_pca[:,1], 
                             c=st.session_state.labels_opt, 
                             cmap='viridis', s=50, alpha=0.7)
        ax2.set_title(f"Sesudah PSO (γ={best_gamma:.4f})\nSilhouette: {silhouette_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}, DBI: {davies_bouldin_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
        ax2.set_xlabel("PC1")
        ax2.set_ylabel("PC2")
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        
        st.pyplot(fig_comparison)

        st.markdown("""
                                    - **Titik-titik pada plot** mewakili setiap observasi (misalnya, Kabupaten/Kota) dalam dataset.
                                    - **Warna Titik (Cluster)** menunjukkan cluster tempat observasi tersebut dikelompokkan oleh algoritma Spectral Clustering.  
                                      Skala warna di sebelah kanan plot (**colorbar**) menunjukkan pemetaan warna ke label cluster (misalnya, 0, 1, 2, dst.).
                                    
                                    - **Metrik Evaluasi Clustering**:  
                                      - **Silhouette Score**  
                                        Merupakan metrik evaluasi yang mengukur seberapa mirip suatu objek dengan clusternya sendiri dibandingkan cluster lain.  
                                        - Nilai berkisar antara **-1 hingga 1**.  
                                        - Nilai mendekati **1** → Objek berada dalam cluster yang tepat dan terpisah dengan baik dari cluster lain.  
                                        - Nilai mendekati **0** → Objek berada di antara dua cluster.  
                                        - Nilai **negatif** → Objek mungkin telah ditetapkan ke cluster yang salah.  
                                        - **Semakin tinggi nilai Silhouette, semakin baik kualitas clustering.**
                                    
                                      - **Davies-Bouldin Index (DBI)**  
                                        Metrik yang mengukur rasio antara dispersi intra-cluster (jarak rata-rata setiap titik ke centroid cluster) dan dispersi inter-cluster (jarak antar centroid cluster).  
                                        - **Semakin rendah nilai DBI, semakin baik kualitas clustering** (klaster lebih kompak dan terpisah jelas).
                                    """)
    
    # 7. Implementasi dan Rekomendasi
    st.subheader("Implementasi dan Rekomendasi Kebijakan")
    
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
