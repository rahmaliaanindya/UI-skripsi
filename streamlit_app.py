import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import warnings
from io import StringIO
import sys
import random
import os
from pyswarms.single.global_best import GlobalBestPSO

# Set random seed for reproducibility and suppress warnings
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
warnings.filterwarnings("ignore")

# === PROFESSIONAL CSS STYLING ===
def local_css():
    st.markdown("""
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #34495e;
            --accent: #3498db;
            --background: #ffffff;
            --text: #2c3e50;
            --light-text: #7f8c8d;
        }
        
        body {
            background-color: var(--background);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .main {
            background-color: var(--background);
        }
        
        .block-container {
            padding-top: 1.5rem;
            background-color: var(--background);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary) !important;
            font-weight: 600;
        }
        
        .stRadio > div {
            display: flex;
            justify-content: center;
            background-color: var(--background);
            padding: 0.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .stButton>button {
            background-color: var(--accent);
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .metric {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# === MAIN APP STRUCTURE ===
def main():
    # Navigation menu at top using horizontal radio buttons
    st.title("📊 Data Analysis App")
    menu = st.radio(
        "Pilih Tahapan Analisis:",
        ("Upload Data", "EDA", "Preprocessing", "Clustering", "Results"),
        horizontal=True,
        label_visibility="collapsed"
    )

    if menu == "Upload Data":
        show_upload_data()
    elif menu == "EDA":
        show_eda()
    elif menu == "Preprocessing":
        show_preprocessing()
    elif menu == "Clustering":
        show_clustering()
    elif menu == "Results":
        show_results()

def show_upload_data():
    st.header("📤 Upload Data")
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        
        with st.expander("Lihat Data Mentah"):
            st.dataframe(df)

def show_eda():
    st.header("🔍 Exploratory Data Analysis (EDA)")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di menu Upload Data")
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
    st.dataframe(df.describe().style.format("{:.2f}"))
    
    # Missing Values
    st.subheader("Pengecekan Nilai Kosong")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if len(missing_values) == 0:
        st.info("Tidak ada nilai kosong pada dataset.")
    else:
        st.dataframe(missing_values.to_frame("Jumlah Nilai Kosong"))
    
    # Data Distribution
    st.subheader("Distribusi Data")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) == 0:
        st.info("Tidak ada kolom numerik untuk ditampilkan.")
        return
    selected_col = st.selectbox("Pilih variabel:", numeric_columns)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[selected_col], kde=True, bins=30, color='#3498db', ax=ax)
    ax.set_title(f'Distribusi {selected_col}')
    st.pyplot(fig)
    
    # Correlation Matrix
    st.subheader("Matriks Korelasi")
    numerical_df = df.select_dtypes(include=['number'])
    if numerical_df.shape[1] < 2:
        st.info("Tidak ada cukup kolom numerik untuk menghitung korelasi.")
        return
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0, ax=ax)
    st.pyplot(fig)

def show_preprocessing():
    st.header("⚙️ Preprocessing Data")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di menu Upload Data")
        return
    
    df = st.session_state.df.copy()
    
    st.subheader("Pembersihan Data")
    
    # Handle missing values
    if st.checkbox("Tampilkan nilai kosong sebelum pembersihan"):
        st.dataframe(df.isnull().sum().to_frame("Jumlah Nilai Kosong"))
    
    if st.button("Bersihkan Data"):
        # Fill missing values with median for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        st.session_state.df_cleaned = df
        st.success("✅ Data berhasil dibersihkan!")
    
    if 'df_cleaned' in st.session_state:
        st.subheader("Scaling Data")
        st.write("Menggunakan RobustScaler untuk menangani outlier:")
        df_cleaned = st.session_state.df_cleaned.copy()
        
        # Check if 'Kabupaten/Kota' column exists and exclude it from scaling
        if 'Kabupaten/Kota' in df_cleaned.columns:
            X = df_cleaned.drop(columns=['Kabupaten/Kota'])
        else:
            X = df_cleaned
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        st.session_state.X_scaled = X_scaled
        st.session_state.feature_names = X.columns.tolist()
        
        st.success("✅ Data berhasil di-scaling!")
        
        st.subheader("Contoh Data setelah Scaling")
        st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

def show_clustering():
    st.header("🤖 Spectral Clustering dengan PSO")
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu")
        return
    
    X_scaled = st.session_state.X_scaled
    
    st.subheader("Evaluasi Jumlah Cluster Optimal")
    
    k_range = range(2, 11)
    silhouette_scores = []
    db_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, k in enumerate(k_range):
        status_text.text(f"Menghitung untuk k = {k}...")
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
        labels = model.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        db_scores.append(davies_bouldin_score(X_scaled, labels))
        progress_bar.progress((i + 1) / len(k_range))
    
    status_text.text("Perhitungan selesai!")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.plot(k_range, silhouette_scores, marker='o', color='#3498db')
    ax1.set_xlabel('Jumlah Cluster')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Evaluasi Silhouette Score')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(k_range, db_scores, marker='o', color='#e74c3c')
    ax2.set_xlabel('Jumlah Cluster')
    ax2.set_ylabel('Davies-Bouldin Index')
    ax2.set_title('Evaluasi Davies-Bouldin Index')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    st.pyplot(fig)
    
    # Optimal cluster based on silhouette (max) and DBI (min)
    optimal_k_sil = k_range[np.argmax(silhouette_scores)]
    optimal_k_dbi = k_range[np.argmin(db_scores)]
    
    st.success(f"Jumlah cluster optimal berdasarkan Silhouette Score: {optimal_k_sil}")
    st.success(f"Jumlah cluster optimal berdasarkan Davies-Bouldin Index: {optimal_k_dbi}")
    
    k_final = st.number_input("Pilih jumlah cluster (k):", min_value=2, max_value=10, value=optimal_k_sil, step=1)
    
    if st.button("Optimasi Gamma dengan PSO"):
        with st.spinner('Menjalankan PSO...'):
            def evaluate_gamma_robust(gamma_array):
                scores = []
                data_for_kernel = X_scaled
                n_runs = 3  # number of runs to average results
                
                for gamma in gamma_array:
                    gamma_val = gamma[0]
                    sil_list, dbi_list = [], []
                    
                    for _ in range(n_runs):
                        try:
                            W = rbf_kernel(data_for_kernel, gamma=gamma_val)
                            L = laplacian(W, normed=True)
                            eigvals, eigvecs = eigsh(L, k=k_final, which='SM', tol=1e-6)
                            U = normalize(eigvecs, norm='l2')
                            kmeans = KMeans(n_clusters=k_final, random_state=SEED, n_init=10).fit(U)
                            labels = kmeans.labels_
                            
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
            
            optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=bounds)
            best_cost, best_pos = optimizer.optimize(evaluate_gamma_robust, iters=100, verbose=False)
            best_gamma = best_pos[0]
        
        st.success(f"Gamma optimal: {best_gamma:.4f}")
        st.session_state.best_gamma = best_gamma
        
        # Final clustering using optimal gamma
        W_opt = rbf_kernel(X_scaled, gamma=best_gamma)
        L_opt = laplacian(W_opt, normed=True)
        eigvals_opt, eigvecs_opt = eigsh(L_opt, k=k_final, which='SM', tol=1e-6)
        U_opt = normalize(eigvecs_opt, norm='l2')
        kmeans_opt = KMeans(n_clusters=k_final, random_state=SEED, n_init=10).fit(U_opt)
        labels_opt = kmeans_opt.labels_
        
        st.session_state.U_opt = U_opt
        st.session_state.labels_opt = labels_opt
        
        # Visualize clusters with PCA projection
        pca = PCA(n_components=2, random_state=SEED)
        X_pca = pca.fit_transform(U_opt)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_opt, cmap='viridis', s=80)
        ax.set_title(f"Visualisasi Cluster (k={k_final})")
        plt.colorbar(scatter, label='Cluster', ax=ax)
        st.pyplot(fig)
        
        # Save results to dataframe
        df = st.session_state.df_cleaned.copy()
        df['Cluster'] = labels_opt
        st.session_state.df_with_cluster = df

def show_results():
    st.header("📊 Hasil Analisis")
    
    if 'df_with_cluster' not in st.session_state:
        st.warning("⚠️ Silakan lakukan clustering terlebih dahulu")
        return
    
    df = st.session_state.df_with_cluster
    
    st.subheader("Distribusi Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    st.subheader("Karakteristik Cluster")
    cluster_means = df.groupby('Cluster').mean(numeric_only=True)
    st.dataframe(cluster_means.style.format("{:.2f}"))
    
    st.subheader("Kabupaten/Kota per Cluster")
    # Show sorted with cluster grouping
    if 'Kabupaten/Kota' in df.columns:
        st.dataframe(df[['Kabupaten/Kota', 'Cluster']].sort_values(['Cluster', 'Kabupaten/Kota']))
    else:
        st.info("Kolom 'Kabupaten/Kota' tidak ditemukan di dataset.")
    
    st.subheader("Feature Importance")
    X = df.drop(columns=['Cluster'])
    if 'Kabupaten/Kota' in X.columns:
        X = X.drop(columns=['Kabupaten/Kota'])
    y = df['Cluster']
    
    rf = RandomForestClassifier(random_state=SEED)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

if __name__ == "__main__":
    main()
