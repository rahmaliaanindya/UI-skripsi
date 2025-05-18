# === IMPORT LIBRARY ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
import warnings
import random
import os

warnings.filterwarnings("ignore")

# === KONFIGURASI HALAMAN ===
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS Styling Modern ===
def local_css():
    st.markdown(
        """
        <style>
            /* Font dan warna dasar */
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                color: #333;
            }
            
            /* Container utama */
            .main {
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            /* Judul */
            .title {
                color: #2c3e50;
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e9ecef;
            }
            
            /* Subjudul */
            h2 {
                color: #3498db;
                font-size: 1.5rem;
                margin-top: 1.5rem;
            }
            
            /* Menu navigasi atas */
            .nav-menu {
                display: flex;
                justify-content: center;
                margin-bottom: 2rem;
                background-color: white;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .nav-item {
                margin: 0 10px;
                padding: 8px 16px;
                border-radius: 5px;
                cursor: pointer;
                font-weight: 500;
                transition: all 0.3s;
                text-align: center;
            }
            
            .nav-item:hover {
                background-color: #f0f0f0;
            }
            
            .nav-item.active {
                background-color: #3498db;
                color: white;
            }
            
            .nav-number {
                background-color: #3498db;
                color: white;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                margin-right: 8px;
                font-size: 14px;
            }
            
            /* Tombol */
            .stButton>button {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.3s;
            }
            
            .stButton>button:hover {
                background-color: #2980b9;
                transform: translateY(-2px);
            }
            
            /* Dataframe */
            .dataframe {
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            /* Card */
            .card {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
            }
            
            /* Tabs */
            .stTabs [role="tablist"] {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 4px;
            }
            
            .stTabs [role="tab"][aria-selected="true"] {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
            }
            
            /* Metric cards */
            .metric-card {
                background: white;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ATAS ===
menu_items = [
    {"number": 1, "name": "Home", "icon": "🏠"},
    {"number": 2, "name": "Upload Data", "icon": "📤"},
    {"number": 3, "name": "EDA", "icon": "🔍"},
    {"number": 4, "name": "Preprocessing", "icon": "⚙️"},
    {"number": 5, "name": "Clustering", "icon": "🧩"},
    {"number": 6, "name": "Hasil Analisis", "icon": "📊"}
]

# Create navigation menu
cols = st.columns(len(menu_items))
current_page = st.session_state.get("current_page", "Home")

for i, item in enumerate(menu_items):
    with cols[i]:
        if st.button(
            f"<span class='nav-number'>{item['number']}</span> {item['icon']} {item['name']}",
            key=f"nav_{item['name']}",
            help=f"Langkah {item['number']}: {item['name']}"
        ):
            current_page = item["name"]
            st.session_state.current_page = current_page

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# === FUNGSI UTAMA ===
def main():
    try:
        # Set random seed
        SEED = 42
        np.random.seed(SEED)
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        
        # === HOME ===
        if current_page == "Home":
            show_home_page()
            
        # === UPLOAD DATA ===
        elif current_page == "Upload Data":
            show_upload_page()
            
        # === EDA ===
        elif current_page == "EDA":
            show_eda_page()
            
        # === PREPROCESSING ===
        elif current_page == "Preprocessing":
            show_preprocessing_page()
            
        # === CLUSTERING ===
        elif current_page == "Clustering":
            show_clustering_page(SEED)
            
        # === HASIL ANALISIS ===
        elif current_page == "Hasil Analisis":
            show_results_page(SEED)
            
    except Exception as e:
        st.error(f"Terjadi kesalahan sistem: {str(e)}")

def show_home_page():
    st.markdown("""
    <div class="title">ANALISIS KLUSTER KEMISKINAN JAWA TIMUR</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Tentang Aplikasi</h3>
        <p>Aplikasi ini membantu analisis pola kemiskinan di Jawa Timur menggunakan metode Spectral Clustering dengan alur kerja berikut:</p>
        
        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
            <div style="text-align: center; width: 16%;">
                <div style="background-color: #3498db; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">1</div>
                <div>Upload Data</div>
            </div>
            <div style="text-align: center; width: 16%;">
                <div style="background-color: #3498db; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">2</div>
                <div>Exploratory Data Analysis</div>
            </div>
            <div style="text-align: center; width: 16%;">
                <div style="background-color: #3498db; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">3</div>
                <div>Preprocessing</div>
            </div>
            <div style="text-align: center; width: 16%;">
                <div style="background-color: #3498db; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">4</div>
                <div>Clustering</div>
            </div>
            <div style="text-align: center; width: 16%;">
                <div style="background-color: #3498db; color: white; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; margin: 0 auto 10px; font-weight: bold;">5</div>
                <div>Analisis Hasil</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>Petunjuk Penggunaan</h3>
        <ol>
            <li>Ikuti langkah-langkah sesuai nomor urut di menu atas</li>
            <li>Pastikan data yang diupload sesuai dengan format yang ditentukan</li>
            <li>Gunakan menu navigasi untuk berpindah antar langkah</li>
            <li>Untuk clustering, sistem akan menggunakan Spectral Clustering dengan optimasi PSO</li>
            <li>Hasil analisis akan menampilkan perbandingan sebelum dan sesudah optimasi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

def show_upload_page():
    st.markdown('<div class="title">LANGKAH 1: UPLOAD DATA</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <h3>Persyaratan Data</h3>
        <p>File Excel harus mengandung kolom-kolom berikut:</p>
        <ul>
            <li>Kabupaten/Kota</li>
            <li>Persentase Penduduk Miskin (%)</li>
            <li>Jumlah Penduduk Miskin (ribu jiwa)</li>
            <li>Harapan Lama Sekolah (Tahun)</li>
            <li>Rata-Rata Lama Sekolah (Tahun)</li>
            <li>Tingkat Pengangguran Terbuka (%)</li>
            <li>Garis Kemiskinan (Rupiah/Bulan/Kapita)</li>
            <li>Indeks Pembangunan Manusia</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"], key="file_uploader")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            
            st.success("✅ Data berhasil dimuat!")
            
            with st.expander("LIHAT DATA", expanded=True):
                st.dataframe(df.style.highlight_max(axis=0, color='#e6f3ff'))
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Jumlah Kabupaten/Kota", df.shape[0])
                with cols[1]:
                    st.metric("Jumlah Variabel", df.shape[1])
        
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")

def show_eda_page():
    st.markdown('<div class="title">LANGKAH 2: EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di Langkah 1")
    else:
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📈 Distribusi", "🔗 Korelasi"])
        
        with tab1:
            st.subheader("Data Lengkap")
            st.dataframe(df)
            
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe().style.background_gradient(cmap='Blues'))
        
        with tab2:
            st.subheader("Distribusi Variabel")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            selected_col = st.selectbox("Pilih variabel:", numeric_cols)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[selected_col], kde=True, color='#3498db', bins=30, ax=ax)
            ax.set_title(f'Distribusi {selected_col}', fontweight='bold')
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Matriks Korelasi")
            numeric_df = df.select_dtypes(include=['number'])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', linewidths=.5, ax=ax)
            ax.set_title('Korelasi Antar Variabel', fontweight='bold')
            st.pyplot(fig)

def show_preprocessing_page():
    st.markdown('<div class="title">LANGKAH 3: PREPROCESSING DATA</div>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di Langkah 1")
    else:
        df = st.session_state.df
        
        st.markdown("""
        <div class="card">
            <h3>Metode Preprocessing</h3>
            <p>Data akan diproses menggunakan <strong>RobustScaler</strong> untuk menangani outlier.</p>
            <p>Proses ini akan menormalisasi data sehingga memiliki mean=0 dan variance=1, tetapi lebih robust terhadap outlier dibanding StandardScaler.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("PROSES DATA", key="preprocess_btn"):
            with st.spinner("Sedang memproses data..."):
                try:
                    X = df.select_dtypes(include=['float64', 'int64'])
                    
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.X_scaled = X_scaled
                    st.session_state.feature_names = X.columns.tolist()
                    
                    st.success("✅ Preprocessing data selesai!")
                    
                    scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                    
                    st.subheader("Hasil Scaling")
                    st.dataframe(scaled_df)
                    
                    # Visualisasi perbandingan sebelum/sesudah scaling
                    st.subheader("Perbandingan Distribusi")
                    selected_col = st.selectbox("Pilih variabel untuk visualisasi:", X.columns)
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    # Sebelum scaling
                    sns.histplot(X[selected_col], kde=True, color='#3498db', ax=ax1)
                    ax1.set_title(f'Sebelum Scaling\n({selected_col})')
                    
                    # Sesudah scaling
                    sns.histplot(scaled_df[selected_col], kde=True, color='#2ecc71', ax=ax2)
                    ax2.set_title(f'Sesudah Scaling\n({selected_col})')
                    
                    st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"Error saat preprocessing: {str(e)}")

def show_clustering_page(SEED):
    st.markdown('<div class="title">LANGKAH 4: SPECTRAL CLUSTERING</div>', unsafe_allow_html=True)
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu di Langkah 3")
    else:
        X_scaled = st.session_state.X_scaled
        
        st.markdown("""
        <div class="card">
            <h3>Clustering dengan Spectral Clustering</h3>
            <p>Analisis clustering akan dilakukan dengan 2 cluster berdasarkan evaluasi DBI dan Silhouette Score.</p>
            <p>Metode ini cocok untuk data yang struktur clusternya non-convex.</p>
            <p>Kami juga akan melakukan optimasi parameter gamma menggunakan Particle Swarm Optimization (PSO).</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Clustering Dasar", "Optimasi dengan PSO"])
        
        with tab1:
            st.subheader("Clustering Dasar")
            
            if st.button("PROSES CLUSTERING DASAR", key="basic_cluster_btn"):
                with st.spinner("Sedang melakukan clustering dasar..."):
                    try:
                        # Spectral Clustering
                        sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', 
                                              random_state=SEED)
                        labels = sc.fit_predict(X_scaled)
                        st.session_state.basic_labels = labels
                        
                        # Visualisasi dengan PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        fig, ax = plt.subplots(figsize=(10, 7))
                        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', s=100)
                        
                        # Tambahkan label jika ada kolom Kabupaten/Kota
                        if 'df' in st.session_state and 'Kabupaten/Kota' in st.session_state.df.columns:
                            df = st.session_state.df
                            for i, txt in enumerate(df['Kabupaten/Kota']):
                                ax.annotate(txt, (X_pca[i,0], X_pca[i,1]), 
                                           fontsize=8, alpha=0.7)
                        
                        ax.set_title('Visualisasi Cluster Dasar (PCA)', fontweight='bold')
                        ax.set_xlabel('Principal Component 1')
                        ax.set_ylabel('Principal Component 2')
                        plt.colorbar(scatter, label='Cluster')
                        st.pyplot(fig)
                        
                        # Evaluasi clustering
                        st.subheader("Evaluasi Clustering Dasar")
                        
                        silhouette = silhouette_score(X_scaled, labels)
                        dbi = davies_bouldin_score(X_scaled, labels)
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Silhouette Score", f"{silhouette:.4f}")
                        with cols[1]:
                            st.metric("Davies-Bouldin Index", f"{dbi:.4f}")
                        
                        st.session_state.basic_silhouette = silhouette
                        st.session_state.basic_dbi = dbi
                    
                    except Exception as e:
                        st.error(f"Error saat clustering dasar: {str(e)}")
        
        with tab2:
            st.subheader("Optimasi dengan PSO")
            
            if st.button("OPTIMASI DENGAN PSO", key="pso_btn"):
                with st.spinner("Sedang melakukan optimasi PSO..."):
                    try:
                        def evaluate_gamma_robust(gamma_array):
                            scores = []
                            data_for_kernel = X_scaled
                            n_runs = 3  # bisa ditambah untuk stabilitas

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

                                        eigvals, eigvecs = eigsh(L, k=2, which='SM', tol=1e-6)
                                        U = normalize(eigvecs, norm='l2')

                                        if np.isnan(U).any() or np.isinf(U).any():
                                            raise ValueError("Invalid U.")

                                        kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(U)
                                        labels = kmeans.labels_

                                        if len(set(labels)) < 2:
                                            raise ValueError("Only one cluster.")

                                        sil = silhouette_score(U, labels)
                                        dbi = davies_bouldin_score(U, labels)

                                        sil_list.append(sil)
                                        dbi_list.append(dbi)

                                    except Exception:
                                        # Penalti berat jika gagal
                                        sil_list.append(0.0)
                                        dbi_list.append(10.0)

                                # Hitung skor rata-rata dari n_runs
                                mean_sil = np.mean(sil_list)
                                mean_dbi = np.mean(dbi_list)

                                # Gabungan skor evaluasi (Semakin kecil lebih baik untuk PSO)
                                fitness_score = -mean_sil + mean_dbi
                                scores.append(fitness_score)

                            return np.array(scores)

                        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
                        bounds = (np.array([0.001]), np.array([5.0]))  # range gamma

                        optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=bounds)
                        best_cost, best_pos = optimizer.optimize(evaluate_gamma_robust, iters=100)
                        best_gamma = best_pos[0]
                        
                        st.session_state.best_gamma = best_gamma
                        
                        # Final Evaluation
                        W_opt = rbf_kernel(X_scaled, gamma=best_gamma)

                        if not (np.allclose(W_opt, 0) or np.any(np.isnan(W_opt)) or np.any(np.isinf(W_opt))):
                            L_opt = laplacian(W_opt, normed=True)
                            if not (np.any(np.isnan(L_opt.data)) or np.any(np.isinf(L_opt.data))):
                                eigvals_opt, eigvecs_opt = eigsh(L_opt, k=2, which='SM', tol=1e-6)
                                U_opt = normalize(eigvecs_opt, norm='l2')

                                if not (np.isnan(U_opt).any() or np.isinf(U_opt).any()):
                                    kmeans_opt = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit(U_opt)
                                    labels_opt = kmeans_opt.labels_

                                    if len(set(labels_opt)) > 1:
                                        silhouette = silhouette_score(U_opt, labels_opt)
                                        dbi = davies_bouldin_score(U_opt, labels_opt)
                                        
                                        st.session_state.opt_labels = labels_opt
                                        st.session_state.U_opt = U_opt
                                        st.session_state.opt_silhouette = silhouette
                                        st.session_state.opt_dbi = dbi
                                        
                                        st.success(f"Optimasi selesai! Gamma optimal: {best_gamma:.4f}")
                                        
                                        # Visualisasi hasil optimasi
                                        fig, ax = plt.subplots(figsize=(10, 7))
                                        scatter = ax.scatter(U_opt[:, 0], U_opt[:, 1], c=labels_opt, cmap='viridis', s=100)
                                        
                                        # Tambahkan label jika ada kolom Kabupaten/Kota
                                        if 'df' in st.session_state and 'Kabupaten/Kota' in st.session_state.df.columns:
                                            df = st.session_state.df
                                            for i, txt in enumerate(df['Kabupaten/Kota']):
                                                ax.annotate(txt, (U_opt[i,0], U_opt[i,1]), 
                                                           fontsize=8, alpha=0.7)
                                        
                                        ax.set_title(f'Visualisasi Cluster dengan Gamma Optimal {best_gamma:.4f}', fontweight='bold')
                                        ax.set_xlabel('Eigenvector 1')
                                        ax.set_ylabel('Eigenvector 2')
                                        plt.colorbar(scatter, label='Cluster')
                                        st.pyplot(fig)
                                        
                                        # Evaluasi clustering
                                        st.subheader("Evaluasi Clustering Optimal")
                                        
                                        cols = st.columns(2)
                                        with cols[0]:
                                            st.metric("Silhouette Score", f"{silhouette:.4f}")
                                        with cols[1]:
                                            st.metric("Davies-Bouldin Index", f"{dbi:.4f}")
                                        
                                        st.write(f"Distribusi Cluster: {Counter(labels_opt)}")
                                        
                                        # Perbandingan dengan clustering dasar
                                        if 'basic_silhouette' in st.session_state:
                                            st.subheader("Perbandingan dengan Clustering Dasar")
                                            
                                            comparison_data = {
                                                'Metrik': ['Silhouette Score', 'Davies-Bouldin Index'],
                                                'Sebelum PSO': [
                                                    st.session_state.basic_silhouette,
                                                    st.session_state.basic_dbi
                                                ],
                                                'Sesudah PSO': [
                                                    silhouette,
                                                    dbi
                                                ]
                                            }
                                            
                                            st.dataframe(pd.DataFrame(comparison_data))
                                            
                                            # Visualisasi perbandingan
                                            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                                            
                                            # Sebelum PSO
                                            pca = PCA(n_components=2)
                                            X_pca = pca.fit_transform(X_scaled)
                                            axes[0].scatter(X_pca[:,0], X_pca[:,1], c=st.session_state.basic_labels, cmap='viridis')
                                            axes[0].set_title('Sebelum PSO')
                                            axes[0].set_xlabel('Principal Component 1')
                                            axes[0].set_ylabel('Principal Component 2')
                                            
                                            # Sesudah PSO
                                            axes[1].scatter(U_opt[:,0], U_opt[:,1], c=labels_opt, cmap='viridis')
                                            axes[1].set_title(f'Sesudah PSO (gamma={best_gamma:.4f})')
                                            axes[1].set_xlabel('Eigenvector 1')
                                            axes[1].set_ylabel('Eigenvector 2')
                                            
                                            plt.suptitle('Perbandingan Embedding Sebelum dan Sesudah PSO')
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                    else:
                                        st.error("Hanya 1 cluster yang terbentuk, evaluasi gagal.")
                                else:
                                    st.error("Embedding tidak valid.")
                            else:
                                st.error("Matriks Laplacian tidak valid.")
                        else:
                            st.error("Matriks kernel tidak valid.")
                    
                    except Exception as e:
                        st.error(f"Error saat optimasi PSO: {str(e)}")

def show_results_page(SEED):
    st.markdown('<div class="title">LANGKAH 5: HASIL ANALISIS CLUSTERING</div>', unsafe_allow_html=True)
    
    if 'opt_labels' not in st.session_state or 'df' not in st.session_state:
        st.warning("⚠️ Silakan lakukan clustering terlebih dahulu di Langkah 4")
    else:
        df = st.session_state.df.copy()
        df['Cluster'] = st.session_state.opt_labels
        
        st.markdown("""
        <div class="card">
            <h3>Hasil Clustering</h3>
            <p>Berikut adalah hasil pengelompokan wilayah berdasarkan indikator kemiskinan setelah optimasi PSO.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tampilkan data dengan cluster
        st.subheader("Data dengan Label Cluster")
        st.dataframe(df.style.background_gradient(subset=['Cluster'], cmap='viridis'))
        
        # Analisis cluster
        st.subheader("Karakteristik Cluster")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        cluster_means = df.groupby('Cluster')[numeric_cols].mean()
        
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
        
        # Visualisasi heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', fmt='.2f', 
                   linewidths=.5, ax=ax)
        ax.set_title('Perbandingan Rata-rata Indikator per Cluster', fontweight='bold')
        st.pyplot(fig)
        
        # Wilayah dengan kemiskinan tertinggi/terendah
        if 'Persentase Penduduk Miskin (%)' in df.columns:
            st.subheader("Wilayah dengan Tingkat Kemiskinan Ekstrim")
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown("**5 Wilayah dengan Kemiskinan Tertinggi**")
                top5 = df.nlargest(5, 'Persentase Penduduk Miskin (%)')
                st.dataframe(top5[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']]
                           .style.background_gradient(subset=['Persentase Penduduk Miskin (%)'], cmap='Reds'))
            
            with cols[1]:
                st.markdown("**5 Wilayah dengan Kemiskinan Terendah**")
                bottom5 = df.nsmallest(5, 'Persentase Penduduk Miskin (%)')
                st.dataframe(bottom5[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']]
                           .style.background_gradient(subset=['Persentase Penduduk Miskin (%)'], cmap='Greens'))
        
        # Feature Importance
        st.subheader("Analisis Feature Importance")
        
        X = df.drop(columns=['Cluster', 'Kabupaten/Kota'])
        y = df['Cluster']
        
        rf = RandomForestClassifier(random_state=SEED)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        
        feat_importance = pd.DataFrame({
            'Fitur': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        # Tampilkan hasil
        st.dataframe(feat_importance)
        
        # Visualisasi
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feat_importance['Fitur'], feat_importance['Importance'], color='skyblue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance - Random Forest')
        ax.invert_yaxis()  # Biar fitur dengan importance tertinggi di atas
        plt.tight_layout()
        st.pyplot(fig)
        
        # Rekomendasi kebijakan
        st.subheader("Rekomendasi Kebijakan")
        
        st.markdown("""
        <div class="card">
            <h4>Cluster 0 (Kemiskinan Tinggi):</h4>
            <ul>
                <li>Prioritaskan program bantuan sosial</li>
                <li>Tingkatkan akses pendidikan dan kesehatan</li>
                <li>Program pelatihan keterampilan</li>
                <li>Penguatan ekonomi lokal</li>
                <li>Peningkatan infrastruktur dasar</li>
            </ul>
            
            <h4>Cluster 1 (Kemiskinan Rendah):</h4>
            <ul>
                <li>Pertahankan program yang sudah berjalan</li>
                <li>Fokus pada peningkatan kualitas hidup</li>
                <li>Pengembangan ekonomi kreatif</li>
                <li>Inovasi teknologi untuk pemerintahan</li>
                <li>Perluasan lapangan kerja berkualitas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
