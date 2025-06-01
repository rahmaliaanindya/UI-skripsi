import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import eigh
from scipy.stats import zscore
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from collections import Counter
import warnings
import random
import os
from io import BytesIO
from PIL import Image

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Page configuration
st.set_page_config(
    page_title="Poverty Indicator Clustering",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'labels_opt' not in st.session_state:
    st.session_state.labels_opt = None
if 'U_opt' not in st.session_state:
    st.session_state.U_opt = None
if 'best_gamma' not in st.session_state:
    st.session_state.best_gamma = None

# Sidebar menu
st.sidebar.title("Menu")
menu_options = ["Beranda", "Upload Data", "EDA", "Preprocessing", "Clustering", "Result"]
selected_menu = st.sidebar.radio("Pilih Menu", menu_options)

def home_page():
    st.title("Analisis Klaster Indikator Kemiskinan")
    st.write("""
    Aplikasi ini digunakan untuk melakukan analisis klaster pada data indikator kemiskinan 
    menggunakan Spectral Clustering yang dioptimasi dengan Particle Swarm Optimization (PSO).
    """)
    
    st.subheader("Alur Kerja Aplikasi:")
    st.write("""
    1. **Upload Data**: Unggah dataset dalam format Excel
    2. **EDA**: Exploratory Data Analysis untuk memahami data
    3. **Preprocessing**: Penskalakan data dengan RobustScaler
    4. **Clustering**: Spectral Clustering dengan optimasi PSO
    5. **Result**: Visualisasi dan interpretasi hasil clustering
    """)
    
    st.subheader("Metode yang Digunakan:")
    st.write("""
    - **Spectral Clustering**: Metode clustering berbasis graf yang bekerja dengan matriks similarity
    - **PSO (Particle Swarm Optimization)**: Algoritma optimasi untuk mencari parameter gamma terbaik
    - **RobustScaler**: Teknik penskalakan data yang robust terhadap outlier
    """)

def upload_data():
    st.title("Upload Data")
    st.write("Unggah dataset dalam format Excel (.xlsx)")
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            
            st.success("Data berhasil diunggah!")
            st.write("Preview data:")
            st.dataframe(df.head())
            
            # Show basic info
            st.subheader("Informasi Dataset")
            buffer = BytesIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue().decode('utf-8'))
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

def eda_page():
    st.title("Exploratory Data Analysis (EDA)")
    
    if st.session_state.df is None:
        st.warning("Silakan unggah data terlebih dahulu di menu Upload Data")
        return
    
    df = st.session_state.df
    
    # Show statistics
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())
    
    # Check for missing values
    st.subheader("Cek Nilai Kosong")
    st.dataframe(df.isnull().sum().to_frame(name="Jumlah Nilai Kosong"))
    
    # Select numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Distribution plots
    st.subheader("Distribusi Variabel")
    selected_col = st.selectbox("Pilih variabel untuk dilihat distribusinya", numeric_columns)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[selected_col], kde=True, bins=30, color='skyblue', ax=ax)
    ax.set_title(f'Distribusi Variabel: {selected_col}', fontsize=14)
    ax.set_xlabel(selected_col)
    ax.set_ylabel('Frekuensi')
    st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Heatmap Korelasi")
    numerical_df = df.select_dtypes(include=['number'])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Heatmap Korelasi Antar Fitur")
    st.pyplot(fig)

def preprocessing_page():
    st.title("Preprocessing Data")
    
    if st.session_state.df is None:
        st.warning("Silakan unggah data terlebih dahulu di menu Upload Data")
        return
    
    df = st.session_state.df
    
    # Drop non-numeric columns
    X = df.drop(columns=['Kabupaten/Kota']) if 'Kabupaten/Kota' in df.columns else df
    
    # Scaling options
    st.subheader("Penskalaan Data")
    scaling_method = st.radio("Pilih metode penskalakan", 
                             ["RobustScaler", "StandardScaler"], 
                             index=0)
    
    if st.button("Lakukan Preprocessing"):
        with st.spinner("Sedang memproses data..."):
            if scaling_method == "RobustScaler":
                scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
            else:
                scaler = StandardScaler()
            
            X_scaled = scaler.fit_transform(X)
            st.session_state.X_scaled = X_scaled
            
            st.success("Preprocessing selesai!")
            
            # Show scaled data
            st.subheader("Data setelah penskalakan")
            scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
            st.dataframe(scaled_df.head())
            
            # Show scaling explanation
            st.subheader("Penjelasan Penskalaan")
            if scaling_method == "RobustScaler":
                st.write("""
                **RobustScaler** melakukan penskalakan berdasarkan kuartil data (biasanya IQR antara Q1 dan Q3), 
                sehingga lebih robust terhadap outlier dibandingkan StandardScaler.
                
                Rumus:
                ```
                scaled_value = (x - median) / (Q3 - Q1)
                ```
                """)
            else:
                st.write("""
                **StandardScaler** melakukan penskalakan dengan menghilangkan mean dan menskalakan ke varians unit.
                
                Rumus:
                ```
                scaled_value = (x - mean) / std_dev
                ```
                """)

def clustering_page():
    st.title("Clustering dengan Spectral Clustering dan PSO")
    
    if st.session_state.X_scaled is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu di menu Preprocessing")
        return
    
    X_scaled = st.session_state.X_scaled
    
    st.subheader("Evaluasi Jumlah Cluster Optimal")
    
    # Determine optimal k
    if st.button("Tentukan Jumlah Cluster Optimal"):
        with st.spinner("Menghitung jumlah cluster optimal..."):
            silhouette_scores = []
            db_scores = []
            k_range = range(2, 11)

            for k in k_range:
                model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
                labels = model.fit_predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                db_scores.append(davies_bouldin_score(X_scaled, labels))

            optimal_k = k_range[np.argmax(silhouette_scores)]
            st.session_state.optimal_k = optimal_k
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
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
            
            st.success(f"Jumlah cluster optimal berdasarkan Silhouette Score: {optimal_k}")
    
    st.subheader("Spectral Clustering dengan Gamma Default")
    
    if st.button("Jalankan Spectral Clustering (Gamma Default)"):
        if st.session_state.optimal_k is None:
            st.warning("Silakan tentukan jumlah cluster optimal terlebih dahulu")
            return
            
        with st.spinner("Menjalankan Spectral Clustering..."):
            gamma = 0.1
            W = rbf_kernel(X_scaled, gamma=gamma)
            
            # Threshold to reduce noise
            threshold = 0.01
            W[W < threshold] = 0
            
            # Degree matrix and normalized Laplacian
            D = np.diag(W.sum(axis=1))
            D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
            L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
            
            # Eigen decomposition
            eigvals, eigvecs = eigh(L_sym)
            
            # Select k eigenvectors
            k = st.session_state.optimal_k
            U = eigvecs[:, :k]
            
            # Normalize rows
            U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)
            
            # KMeans in embedding space
            kmeans = KMeans(n_clusters=k, random_state=SEED)
            labels = kmeans.fit_predict(U_norm)
            st.session_state.labels = labels
            
            # Evaluation
            silhouette = silhouette_score(U_norm, labels)
            dbi = davies_bouldin_score(U_norm, labels)
            
            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            ax1.scatter(U_norm[:,0], U_norm[:,1], c=labels, cmap='viridis')
            ax1.set_title('Spectral Clustering dengan Handling Outlier')
            
            # Plot cluster centers
            centers = kmeans.cluster_centers_
            ax1.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')
            
            # Bar plot of cluster distribution
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
            ax2.set_title('Distribusi Cluster')
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Jumlah Data')
            
            st.pyplot(fig)
            
            st.success("Clustering selesai!")
            st.write(f"Silhouette Score: {silhouette:.4f}")
            st.write(f"Davies-Bouldin Index: {dbi:.4f}")
    
    st.subheader("Optimasi Gamma dengan PSO")
    
    if st.button("Jalankan Optimasi PSO"):
        if st.session_state.optimal_k is None:
            st.warning("Silakan tentukan jumlah cluster optimal terlebih dahulu")
            return
            
        with st.spinner("Menjalankan optimasi PSO..."):
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

                            eigvals, eigvecs = eigsh(L, k=st.session_state.optimal_k, which='SM', tol=1e-6)
                            U = normalize(eigvecs, norm='l2')

                            if np.isnan(U).any() or np.isinf(U).any():
                                raise ValueError("Invalid U.")

                            kmeans = KMeans(n_clusters=st.session_state.optimal_k, random_state=SEED, n_init=10).fit(U)
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

            best_cost, best_pos = optimizer.optimize(evaluate_gamma_robust, iters=50)
            best_gamma = best_pos[0]
            st.session_state.best_gamma = best_gamma
            
            # Final Evaluation with optimized gamma
            W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
            
            if not (np.allclose(W_opt, 0) or np.any(np.isnan(W_opt)) or np.any(np.isinf(W_opt))):
                L_opt = laplacian(W_opt, normed=True)
                if not (np.any(np.isnan(L_opt.data)) or np.any(np.isinf(L_opt.data))):
                    eigvals_opt, eigvecs_opt = eigsh(L_opt, k=st.session_state.optimal_k, which='SM', tol=1e-6)
                    U_opt = normalize(eigvecs_opt, norm='l2')
                    st.session_state.U_opt = U_opt

                    if not (np.isnan(U_opt).any() or np.isinf(U_opt).any()):
                        kmeans_opt = KMeans(n_clusters=st.session_state.optimal_k, random_state=42, n_init=10).fit(U_opt)
                        labels_opt = kmeans_opt.labels_
                        st.session_state.labels_opt = labels_opt

                        if len(set(labels_opt)) > 1:
                            silhouette = silhouette_score(U_opt, labels_opt)
                            dbi = davies_bouldin_score(U_opt, labels_opt)
                            
                            # Visualization
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                            
                            ax1.scatter(U_opt[:,0], U_opt[:,1], c=labels_opt, cmap='viridis')
                            ax1.set_title(f'Spectral Clustering dengan Gamma Optimal {best_gamma:.4f}')
                            
                            # Plot cluster centers
                            centers = kmeans_opt.cluster_centers_
                            ax1.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='X')
                            
                            # Bar plot of cluster distribution
                            cluster_counts = pd.Series(labels_opt).value_counts().sort_index()
                            ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
                            ax2.set_title('Distribusi Cluster Setelah Optimasi')
                            ax2.set_xlabel('Cluster')
                            ax2.set_ylabel('Jumlah Data')
                            
                            st.pyplot(fig)
                            
                            st.success("Optimasi PSO selesai!")
                            st.write(f"Gamma optimal: {best_gamma:.4f}")
                            st.write(f"Silhouette Score: {silhouette:.4f}")
                            st.write(f"Davies-Bouldin Index: {dbi:.4f}")
                            st.write(f"Distribusi Cluster: {Counter(labels_opt)}")
                        else:
                            st.error("Hanya 1 cluster yang terbentuk, evaluasi gagal.")
            
            # Show comparison if both methods have been run
            if st.session_state.labels is not None and st.session_state.labels_opt is not None:
                st.subheader("Perbandingan Sebelum dan Sesudah Optimasi")
                
                # Get U_norm from standard spectral clustering
                gamma = 0.1
                W = rbf_kernel(X_scaled, gamma=gamma)
                threshold = 0.01
                W[W < threshold] = 0
                D = np.diag(W.sum(axis=1))
                D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
                L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
                eigvals, eigvecs = eigh(L_sym)
                U_norm = eigvecs[:, :st.session_state.optimal_k]
                U_norm = U_norm / np.linalg.norm(U_norm, axis=1, keepdims=True)
                
                # Evaluation before PSO
                silhouette_before = silhouette_score(U_norm, st.session_state.labels)
                dbi_before = davies_bouldin_score(U_norm, st.session_state.labels)
                
                # Evaluation after PSO
                silhouette_after = silhouette_score(st.session_state.U_opt, st.session_state.labels_opt)
                dbi_after = davies_bouldin_score(st.session_state.U_opt, st.session_state.labels_opt)
                
                # Comparison table
                comparison_df = pd.DataFrame({
                    'Metrik': ['Silhouette Score', 'Davies-Bouldin Index'],
                    'Sebelum PSO': [silhouette_before, dbi_before],
                    'Sesudah PSO': [silhouette_after, dbi_after],
                    'Perubahan': [
                        f"{(silhouette_after - silhouette_before)/silhouette_before*100:.2f}%",
                        f"{(dbi_after - dbi_before)/dbi_before*100:.2f}%"
                    ]
                })
                
                st.dataframe(comparison_df)
                
                # Visualization comparison
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # Before PSO
                axes[0].scatter(U_norm[:,0], U_norm[:,1], c=st.session_state.labels, cmap='viridis')
                axes[0].set_title('Sebelum PSO')
                axes[0].set_xlabel('Eigenvector 1')
                axes[0].set_ylabel('Eigenvector 2')
                
                # After PSO
                axes[1].scatter(st.session_state.U_opt[:,0], st.session_state.U_opt[:,1], 
                               c=st.session_state.labels_opt, cmap='viridis')
                axes[1].set_title(f'Sesudah PSO (gamma={st.session_state.best_gamma:.4f})')
                axes[1].set_xlabel('Eigenvector 1')
                axes[1].set_ylabel('Eigenvector 2')
                
                st.pyplot(fig)

def result_page():
    st.title("Hasil Clustering dan Interpretasi")
    
    if st.session_state.df is None or st.session_state.labels_opt is None:
        st.warning("Silakan jalankan clustering terlebih dahulu di menu Clustering")
        return
    
    df = st.session_state.df
    labels_opt = st.session_state.labels_opt
    
    # Add cluster labels to original dataframe
    df['Cluster'] = labels_opt
    
    st.subheader("Tabel Hasil Clustering")
    
    # Show cluster results with region names
    if 'Kabupaten/Kota' in df.columns:
        result_df = df[['Kabupaten/Kota', 'Cluster']].sort_values(by='Cluster')
        st.dataframe(result_df)
        
        # Download button
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Hasil Clustering (CSV)",
            data=csv,
            file_name='hasil_clustering.csv',
            mime='text/csv'
        )
    else:
        st.warning("Kolom 'Kabupaten/Kota' tidak ditemukan dalam dataset")
    
    st.subheader("Analisis Cluster")
    
    # Cluster means
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if 'Cluster' in numeric_cols:
        numeric_cols = numeric_cols.drop('Cluster')
    
    cluster_means = df.groupby('Cluster')[numeric_cols].mean()
    st.write("Rata-rata nilai tiap variabel per cluster:")
    st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
    
    # Feature importance
    st.subheader("Feature Importance dengan Random Forest")
    
    if st.button("Hitung Feature Importance"):
        with st.spinner("Menghitung feature importance..."):
            X = df[numeric_cols]
            y = df['Cluster']
            
            rf = RandomForestClassifier(random_state=42)
            rf.fit(X, y)
            
            importances = rf.feature_importances_
            feat_importance = pd.DataFrame({
                'Fitur': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Fitur', data=feat_importance, palette='Blues_r', ax=ax)
            ax.set_title('Feature Importance - Random Forest')
            st.pyplot(fig)
    
    st.subheader("Visualisasi Cluster")
    
    # Select variables for scatter plot
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Pilih variabel untuk sumbu X", numeric_cols, index=0)
    with col2:
        y_var = st.selectbox("Pilih variabel untuk sumbu Y", numeric_cols, index=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=df,
        x=x_var,
        y=y_var,
        hue='Cluster',
        palette='viridis',
        s=100,
        ax=ax
    )
    
    if 'Kabupaten/Kota' in df.columns:
        # Add labels for some points to avoid clutter
        sample_df = df.sample(min(10, len(df)), random_state=42)
        for i, row in sample_df.iterrows():
            ax.text(row[x_var] + 0.02, row[y_var], row['Kabupaten/Kota'], fontsize=8, alpha=0.7)
    
    ax.set_title(f"Visualisasi Cluster berdasarkan {x_var} dan {y_var}")
    st.pyplot(fig)
    
    # Boxplot for each variable by cluster
    st.subheader("Distribusi Variabel per Cluster")
    selected_var = st.selectbox("Pilih variabel untuk dilihat distribusinya per cluster", numeric_cols)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='Cluster', y=selected_var, palette='viridis', ax=ax)
    ax.set_title(f'Distribusi {selected_var} per Cluster')
    st.pyplot(fig)

# Main app logic
if selected_menu == "Beranda":
    home_page()
elif selected_menu == "Upload Data":
    upload_data()
elif selected_menu == "EDA":
    eda_page()
elif selected_menu == "Preprocessing":
    preprocessing_page()
elif selected_menu == "Clustering":
    clustering_page()
elif selected_menu == "Result":
    result_page()
