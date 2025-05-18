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

# === PROFESSIONAL CSS STYLING ===
def local_css():
    st.markdown(
        """
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
            
            .title {
                font-family: 'Helvetica Neue', sans-serif;
                color: var(--primary);
                font-size: 2.5rem;
                font-weight: 700;
                text-align: center;
                padding: 2rem 0 1rem 0;
                border-bottom: 1px solid #ecf0f1;
                margin-bottom: 1.5rem;
            }
            
            .sidebar .sidebar-content {
                background-color: var(--background);
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }
            
            .stRadio > div {
                display: flex;
                justify-content: center;
                background-color: var(--background);
                padding: 0.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            .stRadio > div > label {
                margin: 0 0.5rem;
                color: var(--text);
                font-weight: 500;
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
            
            .stSelectbox, .stNumberInput, .stFileUploader {
                margin-bottom: 1rem;
            }
            
            .metric {
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            .stProgress > div > div > div {
                background-color: var(--accent) !important;
            }
            
            .stAlert {
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu = st.radio(
    "NAVIGASI APLIKASI",
    ("Home", "Upload Data", "EDA", "Clustering", "Results"),
    horizontal=True
)

# === HOME ===
if menu == "Home":
    st.markdown(""" 
    <div style="text-align:center">
        <h1 style="color:#2c3e50; font-size:2.5rem; margin-bottom:1.5rem">
            Spectral Clustering Analysis with PSO Optimization
        </h1>
        <p style="font-size:1.1rem; color:#34495e; margin-bottom:2rem">
            Professional tool for poverty indicator analysis using advanced machine learning techniques
        </p>
    </div>
    
    <div style="background-color:#f8f9fa; border-radius:12px; padding:2rem; margin-bottom:2rem">
        <h3 style="color:#2c3e50; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
            Application Features
        </h3>
        <ul style="color:#34495e; line-height:2">
            <li>📊 Interactive data exploration and visualization</li>
            <li>⚙️ Robust data preprocessing pipeline</li>
            <li>🤖 Advanced Spectral Clustering algorithm</li>
            <li>🔬 Particle Swarm Optimization for parameter tuning</li>
            <li>📈 Comprehensive cluster evaluation metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === UPLOAD DATA ===
elif menu == "Upload Data":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Data Upload
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color:#f8f9fa; border-radius:8px; padding:1.5rem; margin-bottom:1.5rem">
        <h3 style="color:#2c3e50">Instructions</h3>
        <p style="color:#34495e">
            Please upload an Excel file (.xlsx) containing your dataset. The file should include:
        </p>
        <ul style="color:#34495e">
            <li>A 'Kabupaten/Kota' column for region names</li>
            <li>Numerical columns with poverty indicators</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose Excel file", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        
        st.success("✅ Data successfully loaded!")
        
        with st.expander("View Raw Data", expanded=False):
            st.dataframe(df.style.set_properties(**{
                'background-color': '#f8f9fa',
                'color': '#2c3e50',
                'border': '1px solid #ecf0f1'
            }))

# === EDA ===
elif menu == "EDA":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Exploratory Data Analysis
    </h1>
    """, unsafe_allow_html=True)
    
    if 'df' in st.session_state:
        df = st.session_state.df
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin-bottom:1rem">
                <h3 style="color:#2c3e50">Dataset Information</h3>
            </div>
            """, unsafe_allow_html=True)
            st.write(df.info())
            
        with col2:
            st.markdown("""
            <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin-bottom:1rem">
                <h3 style="color:#2c3e50">Descriptive Statistics</h3>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(df.describe().style.format("{:.2f}"))
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Variable Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Select variable:", numeric_columns)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, bins=30, color='#3498db')
        ax.set_title(f'Distribution of {selected_col}', fontsize=14)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Feature Correlation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        numerical_df = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                   center=0, ax=ax, annot_kws={"size": 10})
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please upload data first.")

# === CLUSTERING ===
elif menu == "Clustering":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Spectral Clustering with PSO Optimization
    </h1>
    """, unsafe_allow_html=True)
    
    if 'df' in st.session_state:
        df = st.session_state.df
        X = df.drop(columns=['Kabupaten/Kota'])
        
        # Preprocessing
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled = X_scaled
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Optimal Cluster Evaluation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        k_range = range(2, 11)
        silhouette_scores = []
        db_scores = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, k in enumerate(k_range):
            status_text.text(f"Calculating for k={k}...")
            model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=SEED)
            labels = model.fit_predict(X_scaled)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
            db_scores.append(davies_bouldin_score(X_scaled, labels))
            progress_bar.progress((i + 1) / len(k_range))
        
        status_text.text("Calculation complete!")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.plot(k_range, silhouette_scores, 'bo-', color='#3498db')
        ax1.set_xlabel('Number of Clusters', fontsize=12)
        ax1.set_ylabel('Silhouette Score', fontsize=12)
        ax1.set_title('Silhouette Score Evaluation', fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_facecolor('#f8f9fa')
        
        ax2.plot(k_range, db_scores, 'ro-', color='#e74c3c')
        ax2.set_xlabel('Number of Clusters', fontsize=12)
        ax2.set_ylabel('Davies-Bouldin Index', fontsize=12)
        ax2.set_title('Davies-Bouldin Index Evaluation', fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_facecolor('#f8f9fa')
        
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        st.success(f"Optimal number of clusters based on Silhouette Score: **{optimal_k}**")
        
        k_final = st.number_input("Select number of clusters (k):", 
                                 min_value=2, max_value=10, 
                                 value=optimal_k, step=1)
        
        if st.button("Optimize Gamma with PSO"):
            st.markdown("""
            <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
                <h3 style="color:#2c3e50">Particle Swarm Optimization</h3>
            </div>
            """, unsafe_allow_html=True)
            
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
                            
                            eigvals, eigvecs = eigsh(L, k=k_final, which='SM', tol=1e-6)
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
            
            with st.spinner('Running PSO optimization...'):
                optimizer = GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=bounds)
                best_cost, best_pos = optimizer.optimize(evaluate_gamma_robust, iters=100)
                best_gamma = best_pos[0]
                
            st.success(f"Optimal gamma parameter: **{best_gamma:.4f}**")
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
            
            st.markdown("""
            <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
                <h3 style="color:#2c3e50">Clustering Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            col1.markdown(f"""
            <div class="metric">
                <h4 style="color:#2c3e50">Silhouette Score</h4>
                <p style="font-size:1.5rem; color:#3498db; font-weight:bold">{silhouette:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col2.markdown(f"""
            <div class="metric">
                <h4 style="color:#2c3e50">Davies-Bouldin Index</h4>
                <p style="font-size:1.5rem; color:#e74c3c; font-weight:bold">{dbi:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(U_opt)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_opt, 
                               cmap='viridis', edgecolor='k', s=80, alpha=0.8)
            ax.set_title(f"Cluster Visualization (k={k_final}, γ={best_gamma:.4f})", fontsize=14)
            ax.set_xlabel("Principal Component 1", fontsize=12)
            ax.set_ylabel("Principal Component 2", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#f8f9fa')
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig)
            
            # Save results to dataframe
            df['Cluster'] = labels_opt
            st.session_state.df_with_cluster = df
            
    else:
        st.warning("⚠️ Please upload data first.")

# === RESULTS ===
elif menu == "Results":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Analysis Results
    </h1>
    """, unsafe_allow_html=True)
    
    if 'df_with_cluster' in st.session_state:
        df = st.session_state.df_with_cluster
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Cluster Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, 
                   palette="viridis", ax=ax)
        ax.set_title('Number of Regions per Cluster', fontsize=14)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Cluster Members</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(df.sort_values(by='Cluster').style.set_properties(**{
            'background-color': '#f8f9fa',
            'color': '#2c3e50',
            'border': '1px solid #ecf0f1'
        }))
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Feature Importance</h3>
        </div>
        """, unsafe_allow_html=True)
        
        X = df.drop(columns=['Kabupaten/Kota', 'Cluster'], errors='ignore')
        y = df['Cluster']
        
        rf = RandomForestClassifier(random_state=SEED)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=importances.values, y=importances.index, 
                   palette="viridis", ax=ax)
        ax.set_title("Feature Importance Analysis", fontsize=14)
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Cluster Characteristics</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_means = df.groupby('Cluster').mean(numeric_only=True)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(cluster_means, annot=True, cmap='coolwarm', 
                   fmt=".2f", center=0, ax=ax)
        ax.set_title('Average Feature Values per Cluster', fontsize=14)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Regional Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        kemiskinan_col = "Persentase Penduduk Miskin (%)"
        
        if kemiskinan_col in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin-bottom:1rem">
                    <h4 style="color:#2c3e50">Highest Poverty Rates</h4>
                </div>
                """, unsafe_allow_html=True)
                top3 = df.sort_values(by=kemiskinan_col, ascending=False)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(3)
                st.dataframe(top3.style.format({
                    kemiskinan_col: "{:.2f}"
                }))
            
            with col2:
                st.markdown("""
                <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin-bottom:1rem">
                    <h4 style="color:#2c3e50">Lowest Poverty Rates</h4>
                </div>
                """, unsafe_allow_html=True)
                bottom3 = df.sort_values(by=kemiskinan_col, ascending=True)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(3)
                st.dataframe(bottom3.style.format({
                    kemiskinan_col: "{:.2f}"
                }))
        
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1.5rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
                Conclusions
            </h3>
            <div style="color:#34495e; line-height:1.6">
                <p><strong>Key Findings:</strong></p>
                <ul>
                    <li>Clusters with <strong>lower average poverty rates</strong> typically show better performance across multiple indicators</li>
                    <li>Regions with <strong>higher poverty rates</strong> often correlate with lower education and health indicators</li>
                    <li>The clustering results provide clear segmentation for <strong>targeted policy interventions</strong></li>
                </ul>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    <li>Prioritize resource allocation to clusters with highest poverty indicators</li>
                    <li>Develop cluster-specific development programs based on characteristic patterns</li>
                    <li>Monitor progress using the same indicators to measure intervention effectiveness</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please complete clustering analysis first.")
