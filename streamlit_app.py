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
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

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
            .stRadio > div {
                display: flex;
                justify-content: center;
            }
            .stRadio > div > label {
                margin: 0 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu = st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Upload Data", "EDA", "Clustering", "Hasil & Interpretasi"),
    horizontal=True
)

# === HOME ===
if menu == "Home":
    st.markdown(""" 
    # 👋 Selamat Datang di Aplikasi Spectral Clustering dengan PSO 📊
    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data dengan Robust Scaling
    - 🤖 Menerapkan metode **Spectral Clustering** dengan optimasi PSO
    - 🔍 Menentukan parameter gamma optimal menggunakan Particle Swarm Optimization
    - 📈 Mengevaluasi hasil pengelompokan dengan Silhouette Score dan Davies-Bouldin Index
    """)

# === UPLOAD DATA ===
elif menu == "Upload Data":
    st.header("📤 Upload Data Excel")
    st.markdown("""
    **Petunjuk:**
    1. File yang diunggah harus berupa file **Excel** dengan ekstensi `.xlsx`.
    2. Data harus memuat kolom 'Kabupaten/Kota' dan variabel numerik lainnya.
    """)

    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        
        with st.expander("Lihat Data"):
            st.write(df)

# === EDA ===
elif menu == "EDA":
    st.header("🔍 Exploratory Data Analysis")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            st.write(df.info())
            
        with col2:
            st.subheader("Descriptive Statistics")
            st.write(df.describe())
        
        st.subheader("Distribusi Variabel")
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        
        selected_col = st.selectbox("Pilih variabel untuk dilihat distribusinya:", numeric_columns)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[selected_col], kde=True, bins=30, color='skyblue', ax=ax)
        ax.set_title(f'Distribusi: {selected_col}')
        st.pyplot(fig)
        
        st.subheader("Korelasi Antar Variabel")
        numerical_df = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === CLUSTERING ===
elif menu == "Clustering":
    st.header("🤖 Spectral Clustering dengan PSO")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        X = df.drop(columns=['Kabupaten/Kota'])
        
        # Preprocessing
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled = X_scaled
        
        st.subheader("Evaluasi Jumlah Cluster Optimal")
        
        k_range = range(2, 11)
        silhouette_scores = []
        db_scores = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, k in enumerate(k_range):
            status_text.text(f"Menghitung untuk k={k}...")
            model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
            labels = model.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            db_scores.append(davies_bouldin_score(X_scaled, labels))
            progress_bar.progress((i + 1) / len(k_range))
        
        status_text.text("Perhitungan selesai!")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
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
        
        k_final = st.number_input("Pilih jumlah cluster (k):", min_value=2, max_value=10, value=optimal_k, step=1)
        
        if st.button("Optimasi Gamma dengan PSO"):
            st.subheader("Optimasi Gamma dengan Particle Swarm Optimization")
            
            def evaluate_gamma_robust(gamma_array):
                scores = []
                data_for_kernel = X_scaled
                n_runs = 3
                
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
                            
                            kmeans = KMeans(n_clusters=k_final, random_state=SEED, n_init=10).fit(U)
                            labels = kmeans.labels_
                            
                            if len(set(labels)) < 2:
                                raise ValueError("Only one cluster.")
                            
                            sil = silhouette_score(U, labels)
                            dbi = davies_bouldin_score(U, labels)
                            
                            sil_list.append(sil)
                            dbi_list.append(dbi)
                            
                        except Exception:
                            sil_list.append(0.0)
                            dbi_list.append(10.0)
                    
                    mean_sil = np.mean(sil_list)
                    mean_dbi = np.mean(dbi_list)
                    fitness_score = -mean_sil + mean_dbi
                    scores.append(fitness_score)
                
                return np.array(scores)
            
            options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
            bounds = (np.array([0.001]), np.array([5.0]))
            
            with st.spinner('Menjalankan PSO...'):
                optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=bounds)
                best_cost, best_pos = optimizer.optimize(evaluate_gamma_robust, iters=100)
                best_gamma = best_pos[0]
                
            st.success(f"Gamma optimal (robust): {best_gamma:.4f}")
            st.session_state.best_gamma = best_gamma
            
            # Final clustering with optimized gamma
            W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
            L_opt = laplacian(W_opt, normed=True)
            eigvals_opt, eigvecs_opt = eigsh(L_opt, k=k_final, which='SM', tol=1e-6)
            U_opt = normalize(eigvecs_opt, norm='l2')
            kmeans_opt = KMeans(n_clusters=k_final, random_state=SEED, n_init=10).fit(U_opt)
            labels_opt = kmeans_opt.labels_
            
            st.session_state.U_opt = U_opt
            st.session_state.labels_opt = labels_opt
            
            silhouette = silhouette_score(U_opt, labels_opt)
            dbi = davies_bouldin_score(U_opt, labels_opt)
            
            st.subheader("Hasil Clustering dengan Gamma Optimal")
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", f"{silhouette:.4f}")
            col2.metric("Davies-Bouldin Index", f"{dbi:.4f}")
            
            # Visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(U_opt)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_opt, cmap='viridis', edgecolor='k')
            ax.set_title(f"Clustering Visualisasi (k={k_final}, γ={best_gamma:.4f})")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
            
            # Save results to dataframe
            df['Cluster'] = labels_opt
            st.session_state.df_with_cluster = df
            
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === HASIL & INTERPRETASI ===
elif menu == "Hasil & Interpretasi":
    st.header("📊 Hasil & Interpretasi")
    
    if 'df_with_cluster' in st.session_state:
        df = st.session_state.df_with_cluster
        
        st.subheader("Distribusi Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)
        
        st.subheader("Data dengan Label Cluster")
        st.dataframe(df.sort_values(by='Cluster'))
        
        st.subheader("Analisis Fitur Penting")
        X = df.drop(columns=['Kabupaten/Kota', 'Cluster'], errors='ignore')
        y = df['Cluster']
        
        rf = RandomForestClassifier(random_state=SEED)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Tingkat Pengaruh")
        st.pyplot(fig)
        
        st.subheader("Karakteristik Cluster")
        cluster_means = df.groupby('Cluster').mean(numeric_only=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cluster_means, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Rata-rata Fitur per Cluster')
        st.pyplot(fig)
        
        st.subheader("Wilayah dengan Kemiskinan Tertinggi & Terendah")
        kemiskinan_col = "Persentase Penduduk Miskin (%)"
        
        if kemiskinan_col in df.columns:
            top3 = df.sort_values(by=kemiskinan_col, ascending=False)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(3)
            bottom3 = df.sort_values(by=kemiskinan_col, ascending=True)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚨 3 Wilayah dengan Tingkat Kemiskinan Tertinggi")
                st.table(top3.reset_index(drop=True))
            
            with col2:
                st.markdown("#### 🟢 3 Wilayah dengan Tingkat Kemiskinan Terendah")
                st.table(bottom3.reset_index(drop=True))
        
        st.subheader("Kesimpulan")
        st.markdown("""
        - Cluster dengan rata-rata **persentase penduduk miskin paling rendah** bisa dianggap sebagai kategori **kinerja baik**.
        - Cluster dengan nilai indikator pendidikan dan kesehatan yang rendah mungkin termasuk **kategori rentan/tinggi kemiskinan**.
        - Hasil clustering ini dapat digunakan untuk menentukan kebijakan yang tepat sasaran untuk setiap kelompok wilayah.
        """)
    else:
        st.warning("⚠️ Silakan lakukan clustering terlebih dahulu.")
