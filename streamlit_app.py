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
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO
import warnings
from io import StringIO
import sys
import random
import os

# Konfigurasi
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Spectral Clustering with PSO", layout="wide", page_icon="📊")

# Set random seed untuk reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ======================
# FUNGSI UTAMA
# ======================

def landing_page():
    st.title("🔍 Spectral Clustering with PSO Optimization")
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h3>Selamat Datang di Aplikasi Spectral Clustering dengan Optimasi PSO</h3>
        <p>Aplikasi ini membantu melakukan analisis clustering menggunakan Spectral Clustering yang dioptimasi dengan Particle Swarm Optimization (PSO).</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='padding: 20px; background-color: #f5f5f5; border-radius: 10px;'>
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
        <div style='padding: 20px; background-color: #f5f5f5; border-radius: 10px;'>
            <h4>🔧 Cara Menggunakan</h4>
            <ol>
                <li>Upload dataset Anda</li>
                <li>Lakukan eksplorasi data</li>
                <li>Bersihkan dan standarisasi data</li>
                <li>Tentukan parameter clustering</li>
                <li>Jalankan optimasi PSO</li>
                <li>Analisis hasil clustering</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

def upload_data():
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")

        with st.expander("📄 Lihat Data Mentah"):
            st.dataframe(df)

def exploratory_data_analysis():
    st.header("🔍 Exploratory Data Analysis (EDA)")
    
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        return
    
    df = st.session_state.df
    
    st.subheader("Informasi Dataset")
    buffer = StringIO()
    sys.stdout = buffer
    df.info()
    sys.stdout = sys.__stdout__
    st.text(buffer.getvalue())
    
    st.subheader("Statistik Deskriptif")
    st.dataframe(df.describe())
    
    st.subheader("Distribusi Variabel Numerik")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Pilih variabel:", numeric_cols)
    
    fig, ax = plt.subplots()
    sns.histplot(df[selected_col], kde=True, bins=30, color='skyblue')
    ax.set_title(f'Distribusi {selected_col}')
    st.pyplot(fig)
    
    st.subheader("Matriks Korelasi")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig)

def data_preprocessing():
    st.header("⚙️ Data Preprocessing")

    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu")
        return
    
    df = st.session_state.df.copy()
    X = df.drop(columns=['Kabupaten/Kota'])
    
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    X_scaled = scaler.fit_transform(X)
    
    st.session_state.X_scaled = X_scaled
    st.session_state.feature_names = X.columns.tolist()
    
    st.subheader("Data setelah Scaling")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns))

def clustering_analysis():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    if 'X_scaled' not in st.session_state:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    # Evaluasi jumlah cluster
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
    ax1.plot(k_range, silhouette_scores, 'bo-')
    ax1.set_title('Silhouette Score')
    ax2.plot(k_range, db_scores, 'ro-')
    ax2.set_title('Davies-Bouldin Index')
    st.pyplot(fig)
    
    # Spectral Clustering manual
    st.subheader("2. Spectral Clustering Manual (γ=0.1)")
    gamma = 0.1
    W = rbf_kernel(X_scaled, gamma=gamma)
    W[W < 0.01] = 0
    
    L = laplacian(csr_matrix(W), normed=True)
    eigvals, eigvecs = eigsh(L, k=2, which='SM', tol=1e-6)
    U = normalize(eigvecs, norm='l2')
    
    kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
    labels = kmeans.fit_predict(U)
    
    st.session_state.U_before = U
    st.session_state.labels_before = labels
    
    # Visualisasi
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(U[:, 0], U[:, 1], c=labels, cmap='viridis')
    plt.title("Clustering Manual (γ=0.1)")
    st.pyplot(fig)
    
    # PSO Optimization
    st.subheader("3. Optimasi Gamma dengan PSO")
    
    if st.button("🚀 Jalankan Optimasi PSO", type="primary"):
        with st.spinner("Menjalankan optimasi..."):
            optimizer = GlobalBestPSO(
                n_particles=20,
                dimensions=1,
                options={'c1': 1.5, 'c2': 1.5, 'w': 0.7},
                bounds=([0.001], [5.0])
            )
            
            cost, pos = optimizer.optimize(
                lambda gamma: np.array([evaluate_particle(g, X_scaled) for g in gamma]),
                iters=50
            )
            
            best_gamma = float(pos[0])
            st.session_state.best_gamma = best_gamma
            
            # Clustering dengan gamma optimal
            W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
            W_opt[W_opt < 0.01] = 0
            L_opt = laplacian(csr_matrix(W_opt), normed=True)
            eigvals_opt, eigvecs_opt = eigsh(L_opt, k=2, which='SM', tol=1e-6)
            U_opt = normalize(eigvecs_opt, norm='l2')
            labels_opt = KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(U_opt)
            
            st.session_state.U_opt = U_opt
            st.session_state.labels_opt = labels_opt
            
            # Hasil
            st.success(f"Gamma optimal: {best_gamma:.4f}")
            col1, col2 = st.columns(2)
            col1.metric("Silhouette Score", f"{silhouette_score(U_opt, labels_opt):.4f}")
            col2.metric("Davies-Bouldin Index", f"{davies_bouldin_score(U_opt, labels_opt):.4f}")
            
            # Visualisasi
            fig = plt.figure(figsize=(8, 6))
            plt.scatter(U_opt[:, 0], U_opt[:, 1], c=labels_opt, cmap='viridis')
            plt.title(f"Clustering Optimal (γ={best_gamma:.4f})")
            st.pyplot(fig)
            
            # Simpan hasil ke dataframe
            df = st.session_state.df.copy()
            df['Cluster'] = labels_opt
            st.session_state.df_clustered = df

def results_analysis():
    st.header("📊 Hasil Analisis Cluster")
    
    if 'df_clustered' not in st.session_state:
        st.warning("Silakan jalankan clustering terlebih dahulu")
        return
        
    df = st.session_state.df_clustered
    
    # Distribusi Cluster
    st.subheader("1. Distribusi Cluster")
    st.bar_chart(df['Cluster'].value_counts())
    
    # Karakteristik Cluster
    st.subheader("2. Karakteristik per Cluster")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    st.dataframe(df.groupby('Cluster')[numeric_cols].mean().style.format("{:.2f}"))
    
    # Feature Importance
    st.subheader("3. Feature Importance")
    X = df[numeric_cols]
    y = df['Cluster']
    
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index)
    st.pyplot(fig)
    
    # Perbandingan sebelum-sesudah
    if all(k in st.session_state for k in ['U_before', 'labels_before', 'U_opt', 'labels_opt']):
        st.subheader("4. Perbandingan Hasil")
        
        col1, col2 = st.columns(2)
        col1.metric("Silhouette (Before)", f"{silhouette_score(st.session_state.U_before, st.session_state.labels_before):.4f}")
        col2.metric("Silhouette (After)", f"{silhouette_score(st.session_state.U_opt, st.session_state.labels_opt):.4f}")
        
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.scatter(st.session_state.U_before[:,0], st.session_state.U_before[:,1], 
                    c=st.session_state.labels_before, cmap='viridis')
        plt.title("Sebelum Optimasi")
        
        plt.subplot(122)
        plt.scatter(st.session_state.U_opt[:,0], st.session_state.U_opt[:,1], 
                    c=st.session_state.labels_opt, cmap='viridis')
        plt.title("Sesudah Optimasi")
        
        st.pyplot(fig)
    
    # Download hasil
    st.download_button(
        label="📥 Download Hasil Clustering",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name='hasil_clustering.csv',
        mime='text/csv'
    )

def evaluate_particle(gamma, X_scaled):
    """Fungsi evaluasi untuk PSO"""
    gamma_val = gamma[0]
    
    try:
        W = rbf_kernel(X_scaled, gamma=gamma_val)
        W[W < 0.01] = 0
        
        L = laplacian(csr_matrix(W), normed=True)
        eigvals, eigvecs = eigsh(L, k=2, which='SM', tol=1e-6)
        U = normalize(eigvecs, norm='l2')
        
        kmeans = KMeans(n_clusters=2, random_state=SEED, n_init=10)
        labels = kmeans.fit_predict(U)
        
        sil_score = silhouette_score(U, labels)
        dbi_score = davies_bouldin_score(U, labels)
        
        return -sil_score + dbi_score
        
    except Exception:
        return 10.0

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
