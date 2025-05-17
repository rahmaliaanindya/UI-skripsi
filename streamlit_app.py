# === IMPORT LIBRARY ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
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
            body {
                background-color: #fdf0ed;
            }
            .main {
                background: linear-gradient(to bottom right, #e74c3c, #f39c12, #f8c471);
            }
            .block-container {
                padding-top: 1rem;
                background-color: transparent;
            }
            h1, h2, h3, h4, h5, h6, p, div, span {
                color: #2c3e50 !important;
            }
            .title {
                font-family: 'Helvetica', sans-serif;
                color: #1f3a93;
                font-size: 38px;
                font-weight: bold;
                text-align: center;
                padding: 30px 0 10px 0;
            }
            .sidebar .sidebar-content {
                background-color: #fef5e7;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu = st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Upload Data", "EDA", "Preprocessing", "Clustering", "Hasil & Analisis"),
    horizontal=True
)

# === HOME ===
if menu == "Home":
    st.markdown(""" 
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊
    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode **Spectral Clustering dengan Optimasi PSO**
    - 📈 Mengevaluasi hasil pengelompokan
    """)

# === UPLOAD DATA ===
elif menu == "Upload Data":
    st.header("📤 Upload Data Excel")
    st.markdown("""
    **Petunjuk:**
    1. File yang diunggah harus berupa file **Excel** dengan ekstensi `.xlsx`.
    2. Data yang diunggah harus memuat variabel berikut:
        - **Persentase Penduduk Miskin (%)**
        - **Jumlah Penduduk Miskin (ribu jiwa)**
        - **Harapan Lama Sekolah (Tahun)**
        - **Rata-Rata Lama Sekolah (Tahun)**
        - **Tingkat Pengangguran Terbuka (%)**
        - **Tingkat Partisipasi Angkatan Kerja (%)**
        - **Angka Harapan Hidup (Tahun)**
        - **Garis Kemiskinan (Rupiah/Bulan/Kapita)**
        - **Indeks Pembangunan Manusia**
        - **Rata-rata Upah/Gaji Bersih Pekerja Informal Berdasarkan Lapangan Pekerjaan Utama (Rp)**
        - **Rata-rata Pendapatan Bersih Sebulan Pekerja Informal berdasarkan Pendidikan Tertinggi - Jumlah (Rp)**
    """)

    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        st.write(df)

# === EDA ===
elif menu == "EDA":
    st.header("🔍 Exploratory Data Analysis (EDA)")
    if 'df' in st.session_state:
        df = st.session_state.df
        
        st.subheader("Info Dataset")
        st.write(df.info())
        
        st.subheader("Statistik Deskriptif")
        st.write(df.describe())
        
        st.subheader("Cek Nilai Null")
        st.write(df.isnull().sum())
        
        st.subheader("Distribusi Variabel Numerik")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Pilih variabel untuk dilihat distribusinya:", numeric_columns)
        
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, bins=30, color='skyblue', ax=ax)
        ax.set_title(f'Distribusi Variabel: {selected_col}')
        st.pyplot(fig)
        
        st.subheader("Heatmap Korelasi")
        numerical_df = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === PREPROCESSING ===
elif menu == "Preprocessing":
    st.header("⚙️ Preprocessing Data")
    if 'df' in st.session_state:
        df = st.session_state.df
        
        st.subheader("Scaling Data")
        X = df.drop(columns=['Kabupaten/Kota']) if 'Kabupaten/Kota' in df.columns else df
        
        scaler_type = st.radio("Pilih metode scaling:", ("Standard Scaler", "Robust Scaler"))
        
        if scaler_type == "Standard Scaler":
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
            
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled = X_scaled
        st.session_state.X = X
        
        st.success("✅ Data telah discaling!")
        st.write("5 baris pertama data setelah scaling:", pd.DataFrame(X_scaled).head())
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === CLUSTERING ===
elif menu == "Clustering":
    st.header("🧩 Spectral Clustering dengan Optimasi PSO")
    if 'X_scaled' in st.session_state:
        X_scaled = st.session_state.X_scaled
        
        st.subheader("Evaluasi Jumlah Cluster Optimal")
        clusters_range = range(2, 11)
        silhouette_scores = []
        db_scores = []

        for k in clusters_range:
            model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
            labels = model.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            db_scores.append(davies_bouldin_score(X_scaled, labels))

        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': db_scores
        }, index=clusters_range)
        
        st.line_chart(score_df)
        
        best_k_silhouette = max(zip(clusters_range, silhouette_scores), key=lambda x: x[1])[0]
        best_k_dbi = min(zip(clusters_range, db_scores), key=lambda x: x[1])[0]
        
        st.success(f"🔹 Jumlah cluster optimal berdasarkan Silhouette Score: {best_k_silhouette}")
        st.success(f"🔸 Jumlah cluster optimal berdasarkan Davies-Bouldin Index: {best_k_dbi}")
        
        k_final = st.number_input("Pilih jumlah cluster (k):", min_value=2, max_value=10, value=best_k_silhouette)
        
        if st.button("Lakukan Spectral Clustering"):
            # Spectral Clustering tanpa optimasi
            gamma = 0.1
            W = rbf_kernel(X_scaled, gamma=gamma)
            threshold = 0.01
            W[W < threshold] = 0
            D = np.diag(W.sum(axis=1))
            D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
            L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
            eigvals, eigvecs = eigh(L_sym)
            U = eigvecs[:, :k_final]
            U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
            kmeans = KMeans(n_clusters=k_final, random_state=42)
            labels = kmeans.fit_predict(U_norm)
            
            # Optimasi PSO
            def evaluate_gamma_robust(gamma_array):
                scores = []
                for gamma in gamma_array:
                    gamma_val = gamma[0]
                    try:
                        W = rbf_kernel(X_scaled, gamma=gamma_val)
                        L = laplacian(W, normed=True)
                        eigvals, eigvecs = eigsh(L, k=2, which='SM')
                        U = normalize(eigvecs, norm='l2')
                        kmeans = KMeans(n_clusters=k_final, random_state=42).fit(U)
                        labels = kmeans.labels_
                        sil = silhouette_score(U, labels)
                        dbi = davies_bouldin_score(U, labels)
                        fitness_score = -sil + dbi
                    except:
                        fitness_score = 10.0  # penalty for failed cases
                    scores.append(fitness_score)
                return np.array(scores)
            
            options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
            bounds = (np.array([0.001]), np.array([5.0]))
            
            with st.spinner("Menjalankan optimasi PSO..."):
                optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=bounds)
                best_cost, best_pos = optimizer.optimize(evaluate_gamma_robust, iters=50)
                best_gamma = best_pos[0]
                
                # Spectral Clustering dengan gamma optimal
                W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
                L_opt = laplacian(W_opt, normed=True)
                eigvals_opt, eigvecs_opt = eigsh(L_opt, k=k_final, which='SM')
                U_opt = normalize(eigvecs_opt, norm='l2')
                kmeans_opt = KMeans(n_clusters=k_final, random_state=42).fit(U_opt)
                labels_opt = kmeans_opt.labels_
                
                # Simpan hasil ke session state
                st.session_state.labels = labels_opt
                st.session_state.U_opt = U_opt
                st.session_state.best_gamma = best_gamma
                st.session_state.labels_before = labels
                st.session_state.U_norm = U_norm
                
                st.success("✅ Clustering selesai!")
                
                # Visualisasi
                st.subheader("Visualisasi Hasil Clustering")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Sebelum optimasi
                ax1.scatter(U_norm[:, 0], U_norm[:, 1], c=labels_before)
                ax1.set_title(f'Sebelum Optimasi PSO (gamma={gamma})')
                
                # Sesudah optimasi
                ax2.scatter(U_opt[:, 0], U_opt[:, 1], c=labels_opt)
                ax2.set_title(f'Sesudah Optimasi PSO (gamma={best_gamma:.4f})')
                
                st.pyplot(fig)
                
                # Perbandingan evaluasi
                silhouette_before = silhouette_score(U_norm, labels_before)
                dbi_before = davies_bouldin_score(U_norm, labels_before)
                silhouette_after = silhouette_score(U_opt, labels_opt)
                dbi_after = davies_bouldin_score(U_opt, labels_opt)
                
                eval_df = pd.DataFrame({
                    'Metrik': ['Silhouette Score', 'Davies-Bouldin Index'],
                    'Sebelum Optimasi': [silhouette_before, dbi_before],
                    'Sesudah Optimasi': [silhouette_after, dbi_after]
                })
                
                st.subheader("Perbandingan Evaluasi Clustering")
                st.table(eval_df)
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")

# === HASIL & ANALISIS ===
elif menu == "Hasil & Analisis":
    st.header("📊 Hasil & Analisis Clustering")
    if 'labels' in st.session_state and 'df' in st.session_state:
        df = st.session_state.df.copy()
        labels = st.session_state.labels
        df['Cluster'] = labels
        
        st.subheader("Distribusi Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        st.subheader("Data dengan Label Cluster")
        st.dataframe(df.sort_values(by='Cluster'))
        
        st.subheader("Rata-rata Indikator per Cluster")
        cluster_means = df.groupby('Cluster').mean(numeric_only=True)
        st.dataframe(cluster_means)
        
        # Visualisasi heatmap
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(cluster_means, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        st.subheader("Feature Importance")
        X = df.drop(columns=['Cluster', 'Kabupaten/Kota'] if 'Kabupaten/Kota' in df.columns else ['Cluster'])
        y = df['Cluster']
        
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        
        feat_importance = pd.DataFrame({
            'Fitur': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        st.dataframe(feat_importance)
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x='Importance', y='Fitur', data=feat_importance, ax=ax)
        ax.set_title('Feature Importance - Random Forest')
        st.pyplot(fig)
        
        st.subheader("Visualisasi Persebaran Cluster")
        if 'U_opt' in st.session_state:
            U_opt = st.session_state.U_opt
            fig, ax = plt.subplots(figsize=(10,8))
            scatter = ax.scatter(U_opt[:,0], U_opt[:,1], c=df['Cluster'], cmap='viridis', s=100)
            
            if 'Kabupaten/Kota' in df.columns:
                for i, row in df.iterrows():
                    ax.text(U_opt[i,0]+0.01, U_opt[i,1]+0.01, row['Kabupaten/Kota'], fontsize=8, alpha=0.7)
            
            ax.set_title("Visualisasi Persebaran Hasil Clustering")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
        
        st.subheader("Kabupaten/Kota dengan Kemiskinan Tertinggi dan Terendah")
        if 'Persentase Penduduk Miskin (%)' in df.columns:
            top3 = df.sort_values(by='Persentase Penduduk Miskin (%)', ascending=False).head(3)
            bottom3 = df.sort_values(by='Persentase Penduduk Miskin (%)', ascending=True).head(3)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**3 Kabupaten/Kota dengan Kemiskinan Tertinggi:**")
                st.dataframe(top3[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']])
            
            with col2:
                st.write("**3 Kabupaten/Kota dengan Kemiskinan Terendah:**")
                st.dataframe(bottom3[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']])
        else:
            st.warning("Kolom 'Persentase Penduduk Miskin (%)' tidak ditemukan dalam data.")
    else:
        st.warning("⚠️ Silakan lakukan clustering terlebih dahulu.")
