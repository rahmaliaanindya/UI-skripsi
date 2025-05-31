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
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import warnings
from io import StringIO
import sys
import random
import os
import traceback

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
# CLUSTERING ANALYSIS FUNCTION (REVISED)
# ======================

def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO (Optimized)")
    
    if 'X_scaled' not in st.session_state or st.session_state.X_scaled is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # 1. Evaluasi Jumlah Cluster Optimal
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
    
    # 2. Pilih Cluster Optimal
    st.subheader("2. Pilih Cluster Optimal")
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
    
    # 3. Spectral Clustering Manual dengan Gamma=0.1
    st.subheader("3. Spectral Clustering Manual (γ=0.1)")

    gamma = 0.1
    W = rbf_kernel(X_scaled, gamma=gamma)
    threshold = 0.01
    W[W < threshold] = 0

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    # Tambahkan regularisasi kecil
    L_sym = L_sym + 1e-6 * np.eye(L_sym.shape[0])

    try:
        # Coba dengan eigsh dulu
        eigvals, eigvecs = eigsh(L_sym, k=best_cluster, which='SM', maxiter=1000, tol=1e-4)
    except:
        # Jika gagal, gunakan eigh sebagai fallback
        eigvals, eigvecs = eigh(L_sym)
        eigvecs = eigvecs[:, :best_cluster]

    U = eigvecs[:, :best_cluster]
    U_norm = U / np.linalg.norm(U, axis=1, keepdims=True)

    kmeans = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=10)
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

    # 4. Optimasi Gamma dengan PSO
    st.subheader("4. Optimasi Gamma dengan PSO")

    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        with st.spinner("Menjalankan optimasi PSO..."):
            try:
                def evaluate_gamma(gamma_array):
                    scores = []
                    for gamma_val in gamma_array:
                        try:
                            # Hitung kernel matrix
                            pairwise_dists = squareform(pdist(X_scaled, 'sqeuclidean'))
                            W = np.exp(-gamma_val * pairwise_dists)
                            
                            # Laplacian dengan regularisasi
                            D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
                            L_sym = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
                            L_sym = L_sym + 1e-6 * np.eye(L_sym.shape[0])
                            
                            # Eigen decomposition dengan fallback
                            try:
                                eigvals, eigvecs = eigsh(L_sym, k=best_cluster, which='SM', maxiter=1000, tol=1e-4)
                            except:
                                eigvals, eigvecs = eigh(L_sym)
                                eigvecs = eigvecs[:, :best_cluster]
                            
                            U = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
                            
                            # Clustering
                            kmeans = KMeans(n_clusters=best_cluster, random_state=SEED, n_init='auto')
                            labels = kmeans.fit_predict(U)
                            
                            # Hitung metrik
                            if len(np.unique(labels)) < 2:
                                sil = 0
                                dbi = 10
                            else:
                                sil = silhouette_score(U, labels)
                                dbi = davies_bouldin_score(U, labels)
                            
                            fitness = -sil + dbi
                            
                        except Exception as e:
                            fitness = 10
                        
                        scores.append(fitness)
                        
                    return np.array(scores)
                
                # Batasi rentang gamma yang lebih masuk akal
                options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
                bounds = (np.array([0.01]), np.array([1.0]))  # Lebih sempit dari sebelumnya
                
                optimizer = GlobalBestPSO(
                    n_particles=20,
                    dimensions=1,
                    options=options,
                    bounds=bounds
                )
                
                best_cost, best_pos = optimizer.optimize(
                    evaluate_gamma,
                    iters=50,
                    verbose=False
                )
                
                best_gamma = float(best_pos)
                st.session_state.best_gamma = best_gamma
                
                st.success(f"Optimasi PSO selesai! Gamma optimal: {best_gamma:.4f}")
                
                # 5. Clustering dengan Gamma Optimal
                st.subheader("5. Hasil Clustering dengan Gamma Optimal")
                
                pairwise_dists = squareform(pdist(X_scaled, 'sqeuclidean'))
                W_opt = np.exp(-best_gamma * pairwise_dists)
                D_inv_sqrt = np.diag(1.0 / np.sqrt(W_opt.sum(axis=1)))
                L_sym_opt = np.eye(W_opt.shape[0]) - D_inv_sqrt @ W_opt @ D_inv_sqrt
                L_sym_opt = L_sym_opt + 1e-6 * np.eye(L_sym_opt.shape[0])  # Regularisasi
                
                try:
                    eigvals_opt, eigvecs_opt = eigsh(L_sym_opt, k=best_cluster, which='SM', maxiter=1000, tol=1e-4)
                except:
                    eigvals_opt, eigvecs_opt = eigh(L_sym_opt)
                    eigvecs_opt = eigvecs_opt[:, :best_cluster]
                
                U_opt = eigvecs_opt / np.linalg.norm(eigvecs_opt, axis=1, keepdims=True)
                kmeans_opt = KMeans(n_clusters=best_cluster, random_state=SEED, n_init='auto')
                labels_opt = kmeans_opt.fit_predict(U_opt)
                
                st.session_state.U_opt = U_opt
                st.session_state.labels_opt = labels_opt
                
                sil_opt = silhouette_score(U_opt, labels_opt)
                dbi_opt = davies_bouldin_score(U_opt, labels_opt)
                
                col1, col2 = st.columns(2)
                col1.metric("Silhouette Score", f"{sil_opt:.4f}", 
                           f"{(sil_opt - sil_score):.4f} vs baseline")
                col2.metric("Davies-Bouldin Index", f"{dbi_opt:.4f}", 
                           f"{(dbi_score - dbi_opt):.4f} vs baseline")
                
                # 6. Visualisasi Hasil
                st.subheader("6. Visualisasi Hasil")
                
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
                
                # 7. Simpan Hasil ke DataFrame
                st.subheader("7. Simpan Hasil ke DataFrame")
                
                try:
                    if 'df_cleaned' in st.session_state:
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
                
            except Exception as e:
                st.error(f"Error dalam optimasi PSO: {str(e)}")
                st.error(traceback.format_exc())

# ======================
# MAIN APP LAYOUT
# ======================

menu_options = {
    "Beranda": landing_page,
    "Upload Data": upload_data,
    "EDA": exploratory_data_analysis,
    "Preprocessing": data_preprocessing,
    "Clustering": clustering_analysis,
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
