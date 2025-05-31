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
from numba import njit
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
warnings.filterwarnings("ignore")

# ======================
# OPTIMIZED PSO FUNCTIONS
# ======================

@njit(fastmath=True, parallel=True)
def rbf_kernel_fast(X, gamma):
    """Optimized RBF kernel calculation using Numba"""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            diff = X[i] - X[j]
            K[i,j] = np.exp(-gamma * np.dot(diff, diff))
    return K

def evaluate_particle(gamma, X_scaled, k):
    """Evaluate a single particle (for parallel processing)"""
    try:
        # Fast RBF kernel calculation
        W = rbf_kernel_fast(X_scaled, gamma[0])
        W[W < 0.01] = 0
        
        # Sparse matrix operations
        W_sparse = csr_matrix(W)
        L = laplacian(W_sparse, normed=True)
        
        # Eigen decomposition
        eigvals, eigvecs = eigsh(L, k=k, which='SM', tol=1e-6)
        U = normalize(eigvecs, norm='l2')
        
        # Faster KMeans with auto init
        labels = KMeans(n_clusters=k, random_state=SEED, n_init='auto').fit_predict(U)
        
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(U, labels)
            dbi_score = davies_bouldin_score(U, labels)
            return -(sil_score - (dbi_score / 10))  # Combined metric
        return 100
    except:
        return 100

def evaluate_gamma_robust_fast(gammas):
    """Optimized evaluation function with parallel processing"""
    X_scaled = st.session_state.X_scaled
    k = st.session_state.get('optimal_k', 3)
    
    # Parallel evaluation of particles
    scores = Parallel(n_jobs=4)(
        delayed(evaluate_particle)(gamma, X_scaled, k) for gamma in gammas
    )
    
    return np.array(scores)

class FastPSO(GlobalBestPSO):
    """Optimized PSO with better progress tracking"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = {
            'iteration': [],
            'best_cost': [],
            'best_pos': []
        }
    
    def optimize(self, objective_func, iters, progress_bar=None):
        for i in range(iters):
            # Run one iteration
            super()._optimize(objective_func, 1)
            
            # Store history
            self.history['iteration'].append(i+1)
            self.history['best_cost'].append(self.swarm.best_cost)
            self.history['best_pos'].append(self.swarm.best_pos[0][0])
            
            # Update progress
            if progress_bar:
                progress = (i+1)/iters
                remaining = (iters - i - 1) * 0.5  # Estimated time
                progress_bar.progress(
                    progress,
                    text=f"Iter {i+1}/{iters} - Best: {self.swarm.best_cost:.4f} - Est: {remaining:.1f}s"
                )
        
        return self.swarm.best_cost, self.swarm.best_pos

# ======================
# STREAMLIT UI SETUP
# ======================
st.set_page_config(page_title="Fast Spectral Clustering", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f5f5f5; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
    .plot-container { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .metric-box { background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# ======================
# MAIN APP FUNCTIONS
# ======================

def landing_page():
    st.title("⚡ Fast Spectral Clustering with PSO")
    st.markdown("""
    <div class='metric-box'>
        <h3>Optimized Version for Faster Processing</h3>
        <p>This application uses parallel processing and optimized algorithms to speed up PSO optimization.</p>
    </div>
    """, unsafe_allow_html=True)

def upload_data():
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Choose Excel file", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data loaded successfully!")
        st.dataframe(df.head())

def exploratory_data_analysis():
    if 'df' not in st.session_state:
        st.warning("Please upload data first")
        return
    
    df = st.session_state.df
    st.header("🔍 Data Exploration")
    
    # Show basic info
    with st.expander("Dataset Info"):
        buffer = StringIO()
        sys.stdout = buffer
        df.info()
        sys.stdout = sys.__stdout__
        st.text(buffer.getvalue())
    
    # Show correlations
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

def data_preprocessing():
    if 'df' not in st.session_state:
        st.warning("Please upload data first")
        return
    
    df = st.session_state.df.copy()
    st.header("⚙️ Data Preprocessing")
    
    # Drop non-numeric columns and scale
    X = df.select_dtypes(include=['number'])
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    st.session_state.X_scaled = X_scaled
    st.session_state.feature_names = X.columns.tolist()
    st.session_state.df_cleaned = df
    
    st.success("Data scaled successfully!")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

def clustering_analysis():
    if 'X_scaled' not in st.session_state:
        st.warning("Please preprocess data first")
        return
    
    X_scaled = st.session_state.X_scaled
    st.header("🚀 Fast Spectral Clustering")
    
    # Determine optimal k (cached)
    @st.cache_data
    def find_optimal_k(X):
        scores = []
        for k in range(2, 11):
            model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', 
                                     random_state=SEED, n_init='auto')
            labels = model.fit_predict(X)
            scores.append(silhouette_score(X, labels))
        return range(2, 11)[np.argmax(scores)]
    
    optimal_k = find_optimal_k(X_scaled)
    st.session_state.optimal_k = optimal_k
    st.subheader(f"Optimal Clusters: {optimal_k}")
    
    # Baseline clustering
    st.subheader("Baseline Clustering (γ=0.1)")
    gamma = 0.1
    W = rbf_kernel_fast(X_scaled, gamma)
    W[W < 0.01] = 0
    L = laplacian(csr_matrix(W), normed=True)
    eigvals, eigvecs = eigsh(L, k=optimal_k, which='SM', tol=1e-6)
    U = normalize(eigvecs, norm='l2')
    labels = KMeans(n_clusters=optimal_k, random_state=SEED, n_init='auto').fit_predict(U)
    
    st.session_state.U_before = U
    st.session_state.labels_before = labels
    
    # Show baseline metrics
    col1, col2 = st.columns(2)
    col1.metric("Silhouette Score", f"{silhouette_score(U, labels):.4f}")
    col2.metric("Davies-Bouldin Index", f"{davies_bouldin_score(U, labels):.4f}")
    
    # PSO Optimization
    st.subheader("PSO Optimization")
    if st.button("🚀 Run Optimized PSO", type="primary"):
        with st.spinner("Optimizing gamma (faster with parallel processing)..."):
            progress_bar = st.progress(0)
            
            # Setup optimized PSO
            optimizer = FastPSO(
                n_particles=20,
                dimensions=1,
                options={'c1': 1.5, 'c2': 1.5, 'w': 0.7},
                bounds=([0.001], [5.0]),
                n_processes=4
            )
            
            # Run optimization
            cost, pos = optimizer.optimize(
                evaluate_gamma_robust_fast,
                iters=50,
                progress_bar=progress_bar
            )
            
            best_gamma = pos[0][0]
            st.session_state.best_gamma = best_gamma
            
            # Show results
            st.success(f"Optimization complete! Best gamma: {best_gamma:.4f}")
            
            # Visualize convergence
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(optimizer.history['iteration'], optimizer.history['best_cost'], 'b-')
            ax.set_title("PSO Convergence")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Cost")
            st.pyplot(fig)
            
            # Show optimized clustering
            st.subheader("Optimized Clustering Results")
            W_opt = rbf_kernel_fast(X_scaled, best_gamma)
            W_opt[W_opt < 0.01] = 0
            L_opt = laplacian(csr_matrix(W_opt), normed=True)
            eigvals_opt, eigvecs_opt = eigsh(L_opt, k=optimal_k, which='SM', tol=1e-6)
            U_opt = normalize(eigvecs_opt, norm='l2')
            labels_opt = KMeans(n_clusters=optimal_k, random_state=SEED, n_init='auto').fit_predict(U_opt)
            
            st.session_state.U_opt = U_opt
            st.session_state.labels_opt = labels_opt
            
            # Compare results
            col1, col2 = st.columns(2)
            col1.metric("Optimized Silhouette", 
                       f"{silhouette_score(U_opt, labels_opt):.4f}",
                       f"{(silhouette_score(U_opt, labels_opt) - silhouette_score(U, labels)):.4f}")
            col2.metric("Optimized DBI", 
                       f"{davies_bouldin_score(U_opt, labels_opt):.4f}",
                       f"{(davies_bouldin_score(U, labels) - davies_bouldin_score(U_opt, labels_opt)):.4f}")

def results_analysis():
    if 'df_cleaned' not in st.session_state or 'labels_opt' not in st.session_state:
        st.warning("Please run clustering first")
        return
    
    st.header("📊 Results Analysis")
    df = st.session_state.df_cleaned.copy()
    df['Cluster'] = st.session_state.labels_opt
    
    # Cluster distribution
    st.subheader("Cluster Distribution")
    st.bar_chart(df['Cluster'].value_counts())
    
    # Cluster characteristics
    st.subheader("Cluster Characteristics")
    numeric_cols = df.select_dtypes(include=['number']).columns
    cluster_means = df.groupby('Cluster')[numeric_cols].mean()
    st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
    
    # Feature importance
    st.subheader("Feature Importance")
    X = df[numeric_cols]
    y = df['Cluster']
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, ax=ax)
    ax.set_title("Feature Importance for Clustering")
    st.pyplot(fig)
    
    # Download results
    st.download_button(
        label="📥 Download Results",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='clustering_results.csv',
        mime='text/csv'
    )

# ======================
# APP LAYOUT
# ======================
menu = {
    "Home": landing_page,
    "Upload": upload_data,
    "EDA": exploratory_data_analysis,
    "Preprocess": data_preprocessing,
    "Clustering": clustering_analysis,
    "Results": results_analysis
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(menu.keys()))
menu[selection]()
