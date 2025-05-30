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
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import warnings
from io import StringIO
import sys
import random
import os
import time
from functools import partial

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
# FAST PSO OPTIMIZATION
# ======================

class FastPSOOptimizer:
    def __init__(self, X_scaled, n_clusters):
        self.X_scaled = X_scaled
        self.n_clusters = n_clusters
        self.cache = {}
        
    def evaluate(self, gamma_array):
        scores = []
        for gamma in gamma_array[:, 0]:
            if gamma in self.cache:
                scores.append(self.cache[gamma])
                continue
                
            try:
                # Fast RBF kernel computation
                W = rbf_kernel(self.X_scaled, gamma=gamma)
                
                # Fast Laplacian computation
                L = laplacian(W, normed=True)
                
                # Approximate eigen decomposition (faster but less accurate)
                eigvals, eigvecs = eigsh(L, k=self.n_clusters, which='SM', tol=1e-2)
                U = normalize(eigvecs, norm='l2')
                
                # Faster KMeans with fewer iterations
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=SEED, n_init=3)
                labels = kmeans.fit_predict(U)
                
                if len(np.unique(labels)) < 2:
                    sil = 0.0
                    dbi = 10.0
                else:
                    sil = silhouette_score(U, labels)
                    dbi = davies_bouldin_score(U, labels)
                
                fitness = -sil + dbi
                self.cache[gamma] = fitness
                scores.append(fitness)
                
            except:
                scores.append(10.0)
                
        return np.array(scores)

def run_fast_pso(X_scaled, n_clusters, progress_bar):
    """Optimized PSO that runs quickly for demonstrations"""
    optimizer = FastPSOOptimizer(X_scaled, n_clusters)
    
    # Fast PSO parameters (optimized for speed)
    options = {
        'c1': 1.5,
        'c2': 1.5,
        'w': 0.7,
        'k': 5,
        'p': 1
    }
    
    # Narrower bounds for faster convergence
    bounds = (np.array([0.05]), np.array([0.5]))
    
    # Fewer particles and iterations
    pso = GlobalBestPSO(
        n_particles=10,
        dimensions=1,
        options=options,
        bounds=bounds
    )
    
    history = {
        'iteration': [],
        'best_gamma': [],
        'best_score': [],
        'silhouette': [],
        'dbi': []
    }
    
    def callback(optimizer):
        current_iter = optimizer.it
        best_pos = optimizer.swarm.best_pos[0][0]
        best_score = optimizer.swarm.best_cost
        
        # Fast evaluation without full metrics to save time
        try:
            W = rbf_kernel(X_scaled, gamma=best_pos)
            L = laplacian(W, normed=True)
            eigvals, eigvecs = eigsh(L, k=n_clusters, which='SM', tol=1e-2)
            U = normalize(eigvecs, norm='l2')
            kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=3)
            labels = kmeans.fit_predict(U)
            
            sil = silhouette_score(U, labels)
            dbi = davies_bouldin_score(U, labels)
        except:
            sil = 0.0
            dbi = 10.0
        
        history['iteration'].append(current_iter)
        history['best_gamma'].append(best_pos)
        history['best_score'].append(best_score)
        history['silhouette'].append(sil)
        history['dbi'].append(dbi)
        
        # Update progress (only 15 iterations for speed)
        progress = (current_iter + 1) / 15
        progress_text = f"Iterasi {current_iter+1}/15 - Gamma: {best_pos:.4f} - Skor: {best_score:.4f}"
        progress_bar.progress(progress, text=progress_text)
    
    # Run optimization with fewer iterations
    best_cost, best_pos = pso.optimize(
        optimizer.evaluate,
        iters=15,
        verbose=False,
        callback=callback
    )
    
    return best_pos[0], history

# ======================
# MAIN APP FUNCTIONS
# ======================

def landing_page():
    st.markdown('<div class="landing-header">', unsafe_allow_html=True)
    st.title("🔍 Spectral Clustering dengan Optimasi PSO (Versi Cepat)")
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3>Aplikasi Clustering untuk Analisis Kemiskinan</h3>
        <p>Versi optimasi cepat untuk presentasi dengan runtime ≤ 1 menit</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Fitur Utama</h4>
            <ul>
                <li>Analisis data kemiskinan</li>
                <li>Clustering dengan Spectral Clustering</li>
                <li>Optimasi parameter dengan PSO</li>
                <li>Visualisasi interaktif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚡ Versi Cepat</h4>
            <ul>
                <li>Proses PSO hanya 15 iterasi</li>
                <li>10 partikel (biasanya 20-30)</li>
                <li>Rentang gamma lebih sempit</li>
                <li>Perhitungan eigen lebih cepat</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def upload_data():
    st.header("📤 Upload Data Excel")
    st.info("Pastikan file memiliki kolom: Kabupaten/Kota, Persentase Penduduk Miskin (%), dll")
    
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil dimuat!")
        
        with st.expander("Lihat Data"):
            st.dataframe(df)

def exploratory_data_analysis():
    st.header("🔍 Exploratory Data Analysis (EDA)")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        return
    
    df = st.session_state.df
    
    # Basic info
    st.subheader("Informasi Dataset")
    buffer = StringIO()
    sys.stdout = buffer
    df.info()
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())
    
    # Quick stats
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe().style.format("{:.2f}"))
    
    # Fast correlation matrix
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 1:
        st.subheader("Korelasi Antar Variabel")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)

def data_preprocessing():
    st.header("⚙️ Data Preprocessing")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        return
    
    df = st.session_state.df.copy()
    
    # Drop non-numeric columns
    X = df.drop(columns=['Kabupaten/Kota'])
    
    # Fast scaling
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.session_state.X_scaled = X_scaled
    st.session_state.feature_names = X.columns.tolist()
    st.session_state.df_cleaned = df
    
    st.success("Preprocessing selesai! Data telah di-scaling.")

def clustering_analysis():
    st.header("⚡ Spectral Clustering dengan PSO Cepat")
    
    if 'X_scaled' not in st.session_state:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # Fast cluster evaluation
    st.subheader("Evaluasi Jumlah Cluster")
    k_range = range(2, 6)  # Only evaluate 2-5 clusters for speed
    
    silhouette_scores = []
    for k in k_range:
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
        labels = model.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, silhouette_scores, 'bo-')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Evaluasi Cluster Optimal')
    st.pyplot(fig)
    
    st.success(f"Jumlah cluster optimal: {optimal_k}")
    
    # Manual clustering with fixed gamma
    st.subheader("Clustering Awal (γ=0.1)")
    gamma = 0.1
    W = rbf_kernel(X_scaled, gamma=gamma)
    L = laplacian(W, normed=True)
    eigvals, eigvecs = eigsh(L, k=optimal_k, which='SM', tol=1e-2)
    U = normalize(eigvecs, norm='l2')
    kmeans = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=3)
    labels = kmeans.fit_predict(U)
    
    st.session_state.U_before = U
    st.session_state.labels_before = labels
    
    sil_score = silhouette_score(U, labels)
    dbi_score = davies_bouldin_score(U, labels)
    
    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", f"{sil_score:.4f}")
    col2.metric("Davies-Bouldin Index", f"{dbi_score:.4f}")
    
    # Fast PSO Optimization
    st.subheader("Optimasi Gamma dengan PSO (Versi Cepat)")
    
    if st.button("🚀 Jalankan PSO Cepat (15 Iterasi)", type="primary"):
        progress_bar = st.progress(0, text="Memulai optimasi cepat...")
        
        with st.spinner("Optimasi berjalan - selesai dalam ≤ 1 menit..."):
            try:
                best_gamma, history = run_fast_pso(
                    X_scaled,
                    optimal_k,
                    progress_bar
                )
                
                st.session_state.best_gamma = best_gamma
                st.session_state.pso_history = history
                
                # Show results
                st.success(f"Optimasi selesai! Gamma optimal: {best_gamma:.4f}")
                
                # Plot convergence
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(history['iteration'], history['best_score'], 'b-')
                ax.set_xlabel('Iterasi')
                ax.set_ylabel('Nilai Fitness')
                ax.set_title('Konvergensi PSO Cepat')
                st.pyplot(fig)
                
                # Cluster with optimized gamma
                st.subheader("Hasil Clustering dengan Gamma Optimal")
                W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
                L_opt = laplacian(W_opt, normed=True)
                eigvals_opt, eigvecs_opt = eigsh(L_opt, k=optimal_k, which='SM', tol=1e-2)
                U_opt = normalize(eigvecs_opt, norm='l2')
                kmeans_opt = KMeans(n_clusters=optimal_k, random_state=SEED, n_init=3)
                labels_opt = kmeans_opt.fit_predict(U_opt)
                
                st.session_state.U_opt = U_opt
                st.session_state.labels_opt = labels_opt
                
                sil_opt = silhouette_score(U_opt, labels_opt)
                dbi_opt = davies_bouldin_score(U_opt, labels_opt)
                
                col1, col2 = st.columns(2)
                col1.metric("Silhouette Score", f"{sil_opt:.4f}", f"{sil_opt-sil_score:+.4f}")
                col2.metric("Davies-Bouldin Index", f"{dbi_opt:.4f}", f"{dbi_score-dbi_opt:+.4f}")
                
                # Save to dataframe
                df = st.session_state.df_cleaned.copy()
                df['Cluster'] = labels_opt
                st.session_state.df_clustered = df
                
                # Show cluster distribution
                st.subheader("Distribusi Cluster")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                st.bar_chart(cluster_counts)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

def results_analysis():
    st.header("📊 Hasil Analisis Cluster")
    
    if 'df_clustered' not in st.session_state:
        st.warning("Silakan jalankan clustering terlebih dahulu")
        return
    
    df = st.session_state.df_clustered
    
    # Cluster characteristics
    st.subheader("Karakteristik Cluster")
    numeric_cols = df.select_dtypes(include=['number']).columns
    cluster_means = df.groupby('Cluster')[numeric_cols].mean()
    st.dataframe(cluster_means.style.format("{:.2f}").background_gradient(axis=0))
    
    # Feature importance
    st.subheader("Faktor Paling Berpengaruh")
    X = df[numeric_cols].drop(columns=['Cluster'], errors='ignore')
    y = df['Cluster']
    
    rf = RandomForestClassifier(random_state=SEED, n_estimators=50)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
    
    # Cluster mapping
    if 'Kabupaten/Kota' in df.columns:
        st.subheader("Pemetaan Kabupaten/Kota per Cluster")
        st.dataframe(df[['Kabupaten/Kota', 'Cluster']].sort_values('Cluster'))
    
    # Comparison before/after
    if all(k in st.session_state for k in ['U_before', 'U_opt', 'labels_before', 'labels_opt']):
        st.subheader("Perbandingan Sebelum/Sesudah PSO")
        
        pca = PCA(n_components=2)
        U_before_pca = pca.fit_transform(st.session_state.U_before)
        U_opt_pca = pca.transform(st.session_state.U_opt)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.scatter(U_before_pca[:,0], U_before_pca[:,1], 
                   c=st.session_state.labels_before, cmap='viridis')
        ax1.set_title("Sebelum PSO (γ=0.1)")
        
        ax2.scatter(U_opt_pca[:,0], U_opt_pca[:,1], 
                   c=st.session_state.labels_opt, cmap='viridis')
        ax2.set_title(f"Sesudah PSO (γ={st.session_state.best_gamma:.4f})")
        
        st.pyplot(fig)

# ======================
# APP LAYOUT
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
