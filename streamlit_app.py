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
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import warnings
from io import StringIO
import sys
import random
import os
import time

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
warnings.filterwarnings("ignore")

# ======================
# OPTIMIZATION FUNCTIONS
# ======================

def create_evaluation_function(X_scaled, best_cluster, n_runs=1):
    """Create an evaluation function with caching to speed up PSO"""
    cache = {}
    
    def evaluate_gamma(gamma_array):
        scores = []
        gamma_vals = gamma_array[:, 0]  # Extract gamma values
        
        for gamma in gamma_vals:
            # Check cache first
            if gamma in cache:
                scores.append(cache[gamma])
                continue
                
            try:
                # Compute RBF kernel - using faster implementation
                W = rbf_kernel(X_scaled, gamma=gamma)
                
                # Early termination for invalid matrices
                if np.allclose(W, 0) or np.any(np.isnan(W)) or np.any(np.isinf(W)):
                    raise ValueError("Invalid kernel matrix")
                
                # Compute Laplacian
                L = laplacian(W, normed=True)
                
                # Compute eigenvalues/vectors - using faster method for symmetric matrices
                eigvals, eigvecs = eigsh(L, k=best_cluster, which='SM', tol=1e-4)
                U = normalize(eigvecs, norm='l2')
                
                # Cluster with KMeans - reduced n_init for speed
                kmeans = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=5)
                labels = kmeans.fit_predict(U)
                
                # Calculate metrics
                if len(np.unique(labels)) < 2:
                    sil = 0.0
                    dbi = 10.0
                else:
                    sil = silhouette_score(U, labels)
                    dbi = davies_bouldin_score(U, labels)
                
                fitness = -sil + dbi  # We want to minimize this
                cache[gamma] = fitness
                scores.append(fitness)
                
            except Exception as e:
                # Return poor fitness if error occurs
                cache[gamma] = 10.0
                scores.append(10.0)
                
        return np.array(scores)
    
    return evaluate_gamma

def run_pso_optimization(X_scaled, best_cluster):
    """Run PSO optimization with improved performance"""
    start_time = time.time()
    
    # Create evaluation function with caching
    evaluate_gamma = create_evaluation_function(X_scaled, best_cluster)
    
    # PSO parameters - tuned for faster convergence
    options = {
        'c1': 1.7,  # cognitive parameter
        'c2': 1.7,  # social parameter
        'w': 0.65,  # inertia parameter
        'k': 10,    # number of neighbors for lbest
        'p': 2      # type of distance (p=2 for Euclidean)
    }
    
    # Bounds for gamma - narrower range based on empirical testing
    bounds = (np.array([0.01]), np.array([1.0]))
    
    # Create optimizer with fewer particles and iterations (but smarter)
    optimizer = GlobalBestPSO(
        n_particles=15,  # Reduced from 20
        dimensions=1,
        options=options,
        bounds=bounds
    )
    
    # Prepare history tracking
    history = {
        'iteration': [],
        'g_best': [],
        'best_gamma': [],
        'silhouette': [],
        'dbi': [],
        'p_best': [],  # To store personal bests
        'time_elapsed': []
    }
    
    # Callback function with progress tracking
    def callback(optimizer):
        current_iter = optimizer.it
        best_pos = optimizer.swarm.best_pos
        best_cost = optimizer.swarm.best_cost
        
        # Calculate metrics for best gamma
        try:
            W = rbf_kernel(X_scaled, gamma=best_pos[0][0])
            L = laplacian(W, normed=True)
            eigvals, eigvecs = eigsh(L, k=best_cluster, which='SM', tol=1e-4)
            U = normalize(eigvecs, norm='l2')
            kmeans = KMeans(n_clusters=best_cluster, random_state=SEED, n_init=5)
            labels = kmeans.fit_predict(U)
            
            sil = silhouette_score(U, labels)
            dbi = davies_bouldin_score(U, labels)
        except:
            sil = 0.0
            dbi = 10.0
        
        # Store personal bests (p_best)
        p_bests = [p.score for p in optimizer.swarm.pbest_pos]
        
        # Update history
        history['iteration'].append(current_iter)
        history['g_best'].append(best_cost)
        history['best_gamma'].append(best_pos[0][0])
        history['silhouette'].append(sil)
        history['dbi'].append(dbi)
        history['p_best'].append(p_bests)
        history['time_elapsed'].append(time.time() - start_time)
        
        # Update progress
        progress = (current_iter + 1) / 30  # Only 30 iterations now
        progress_text = f"""
        Iterasi {current_iter + 1}/30
        Gamma terbaik: {best_pos[0][0]:.4f}
        Fitness (G-best): {best_cost:.4f}
        Silhouette: {sil:.4f}
        DBI: {dbi:.4f}
        Waktu: {time.time() - start_time:.1f}s
        """
        progress_bar.progress(progress, text=progress_text)
    
    # Create progress bar
    progress_bar = st.progress(0, text="Memulai optimasi PSO...")
    
    # Run optimization with fewer iterations (but smarter)
    best_cost, best_pos = optimizer.optimize(
        evaluate_gamma,
        iters=30,  # Reduced from 50
        verbose=False,
        callback=callback,
        n_processes=1  # Single process for Streamlit compatibility
    )
    
    return best_pos[0], history

# ======================
# CLUSTERING FUNCTIONS
# ======================

def perform_spectral_clustering(X_scaled, gamma, n_clusters):
    """Perform spectral clustering with given parameters"""
    try:
        # Compute affinity matrix
        W = rbf_kernel(X_scaled, gamma=gamma)
        
        # Compute Laplacian
        L = laplacian(W, normed=True)
        
        # Compute eigenvalues/vectors
        eigvals, eigvecs = eigsh(L, k=n_clusters, which='SM', tol=1e-4)
        U = normalize(eigvecs, norm='l2')
        
        # Cluster with KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(U)
        
        # Calculate metrics
        sil_score = silhouette_score(U, labels)
        dbi_score = davies_bouldin_score(U, labels)
        
        return U, labels, sil_score, dbi_score
        
    except Exception as e:
        st.error(f"Error in spectral clustering: {str(e)}")
        return None, None, None, None

# ======================
# STREAMLIT UI FUNCTIONS
# ======================

def show_pso_results(best_gamma, history):
    """Display PSO optimization results"""
    st.success(f"**Optimasi selesai!** Gamma optimal: {best_gamma:.4f}")
    
    # Convert history to DataFrame for display
    history_df = pd.DataFrame({
        'Iterasi': history['iteration'],
        'Gamma': history['best_gamma'],
        'G-best': history['g_best'],
        'Silhouette': history['silhouette'],
        'DBI': history['dbi'],
        'Waktu (s)': history['time_elapsed']
    })
    
    # Show convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot G-best convergence
    ax1.plot(history['iteration'], history['g_best'], 'b-', label='Global Best')
    ax1.set_xlabel('Iterasi')
    ax1.set_ylabel('Nilai Fitness (G-best)')
    ax1.set_title('Konvergensi PSO (G-best)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot p-best distribution
    for i, p_bests in enumerate(history['p_best']):
        ax2.scatter([i]*len(p_bests), p_bests, color='orange', alpha=0.3, s=10)
    ax2.plot(history['iteration'], history['g_best'], 'b-', label='G-best')
    ax2.set_xlabel('Iterasi')
    ax2.set_ylabel('Nilai Fitness')
    ax2.set_title('Distribusi P-best dan G-best')
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)
    
    # Show best iteration
    best_iter_idx = np.argmin(history['g_best'])
    st.info(f"""
    **Iterasi terbaik:** {history['iteration'][best_iter_idx]}  
    **Gamma terbaik:** {history['best_gamma'][best_iter_idx]:.4f}  
    **Silhouette Score:** {history['silhouette'][best_iter_idx]:.4f}  
    **Davies-Bouldin Index:** {history['dbi'][best_iter_idx]:.4f}
    **Waktu komputasi:** {history['time_elapsed'][best_iter_idx]:.1f} detik
    """)
    
    # Show history table
    st.subheader("History Optimasi PSO")
    st.dataframe(history_df.style.format({
        'Gamma': '{:.4f}',
        'G-best': '{:.4f}',
        'Silhouette': '{:.4f}',
        'DBI': '{:.4f}',
        'Waktu (s)': '{:.1f}'
    }).highlight_min(subset=['G-best', 'DBI'], color='lightgreen')
                  .highlight_max(subset=['Silhouette'], color='lightgreen'))

def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    if 'X_scaled' not in st.session_state or st.session_state.X_scaled is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # 1. Find optimal number of clusters
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
    st.success(f"Jumlah cluster optimal berdasarkan silhouette score: k={optimal_k}")
    
    # Let user select number of clusters
    n_clusters = st.number_input(
        "Pilih jumlah cluster untuk digunakan:",
        min_value=2,
        max_value=10,
        value=optimal_k,
        step=1
    )
    
    # 2. Baseline clustering with gamma=0.1
    st.subheader("2. Spectral Clustering Baseline (γ=0.1)")
    
    U_before, labels_before, sil_score, dbi_score = perform_spectral_clustering(
        X_scaled, gamma=0.1, n_clusters=n_clusters
    )
    
    if U_before is not None:
        st.session_state.U_before = U_before
        st.session_state.labels_before = labels_before
        
        col1, col2 = st.columns(2)
        col1.metric("Silhouette Score", f"{sil_score:.4f}")
        col2.metric("Davies-Bouldin Index", f"{dbi_score:.4f}")
        
        # Visualize
        pca = PCA(n_components=2)
        U_pca = pca.fit_transform(U_before)
        
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(U_pca[:, 0], U_pca[:, 1], c=labels_before, cmap='viridis', alpha=0.7)
        plt.title(f'Spectral Clustering (γ=0.1)\nSilhouette: {sil_score:.4f}, DBI: {dbi_score:.4f}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        st.pyplot(fig)
    
    # 3. PSO Optimization
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        with st.spinner("Menjalankan optimasi PSO..."):
            try:
                best_gamma, history = run_pso_optimization(X_scaled, n_clusters)
                st.session_state.best_gamma = best_gamma
                st.session_state.pso_history = history
                
                # Show PSO results
                show_pso_results(best_gamma, history)
                
                # 4. Clustering with optimized gamma
                st.subheader("4. Spectral Clustering dengan Gamma Optimal")
                
                U_opt, labels_opt, sil_opt, dbi_opt = perform_spectral_clustering(
                    X_scaled, gamma=best_gamma, n_clusters=n_clusters
                )
                
                if U_opt is not None:
                    st.session_state.U_opt = U_opt
                    st.session_state.labels_opt = labels_opt
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Silhouette Score", f"{sil_opt:.4f}", 
                               f"{(sil_opt - sil_score):.4f} vs baseline")
                    col2.metric("Davies-Bouldin Index", f"{dbi_opt:.4f}", 
                               f"{(dbi_score - dbi_opt):.4f} vs baseline")
                    
                    # Visual comparison
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
                    
                    # Save clustered data
                    if 'df_cleaned' in st.session_state:
                        df = st.session_state.df_cleaned.copy()
                        df['Cluster'] = labels_opt
                        st.session_state.df_clustered = df
                        
                        st.subheader("Distribusi Cluster")
                        cluster_counts = df['Cluster'].value_counts().sort_index()
                        st.bar_chart(cluster_counts)
                        
                        if 'Kabupaten/Kota' in df.columns:
                            st.subheader("Pemetaan Cluster")
                            st.dataframe(df[['Kabupaten/Kota', 'Cluster']].sort_values('Cluster'))
                
            except Exception as e:
                st.error(f"Terjadi kesalahan dalam optimasi PSO: {str(e)}")

# ======================
# MAIN APP (keep your existing UI setup and other functions)
# ======================

# (Keep all your existing UI setup code, landing_page(), upload_data(), 
# exploratory_data_analysis(), data_preprocessing(), and results_analysis() functions)

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
