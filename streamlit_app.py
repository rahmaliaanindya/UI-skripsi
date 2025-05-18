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
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS Styling ===
def local_css():
    st.markdown(
        """
        <style>
            body {
                background-color: #f8f9fa;
            }
            .main {
                background-color: #ffffff;
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
                background-color: #ffffff;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50 !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .title {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #1f3a93;
                font-size: 38px;
                font-weight: bold;
                text-align: center;
                padding: 20px 0 15px 0;
                border-bottom: 2px solid #e9ecef;
                margin-bottom: 25px;
            }
            .stRadio > div {
                flex-direction: row !important;
                justify-content: center !important;
                gap: 15px !important;
            }
            .stButton button {
                background-color: #4a6baf !important;
                color: white !important;
                border-radius: 5px !important;
                padding: 8px 20px !important;
                font-weight: 500 !important;
                transition: all 0.3s ease;
            }
            .stButton button:hover {
                background-color: #3a5683 !important;
                transform: translateY(-2px);
            }
            .next-button {
                display: flex;
                justify-content: flex-end;
                margin-top: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu_items = {
    "Home": "🏠 Home",
    "Upload Data": "📤 Upload Data",
    "EDA": "🔍 EDA",
    "Preprocessing": "⚙️ Preprocessing",
    "Clustering": "🧩 Clustering",
    "Hasil & Analisis": "📊 Hasil & Analisis"
}

menu = st.radio(
    "Navigasi Aplikasi:",
    list(menu_items.values()),
    horizontal=True
)

# Fungsi untuk tombol next
def next_button(next_page):
    st.markdown(
        f"""
        <div class="next-button">
            <button onclick="window.location.href='?page={list(menu_items.keys()).index(next_page)}'" style="background-color: #4a6baf; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                Next: {menu_items[next_page]} →
            </button>
        </div>
        """,
        unsafe_allow_html=True
    )

# === HOME ===
if menu == menu_items["Home"]:
    st.markdown(""" 
    <div class="title">👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur</div>
    
    <div style="text-align: center; margin-bottom: 30px;">
        <p style="font-size: 18px; color: #555;">
            Aplikasi ini dirancang untuk analisis clustering data kemiskinan di Jawa Timur menggunakan metode Spectral Clustering dengan optimasi PSO
        </p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 30px;">
        <div style="background: #f1f5fd; padding: 20px; border-radius: 10px; border-left: 4px solid #4a6baf;">
            <h3 style="color: #2c3e50; margin-top: 0;">📁 Upload Data</h3>
            <p>Unggah dataset indikator kemiskinan dalam format Excel</p>
        </div>
        <div style="background: #f1f5fd; padding: 20px; border-radius: 10px; border-left: 4px solid #4a6baf;">
            <h3 style="color: #2c3e50; margin-top: 0;">🔍 EDA</h3>
            <p>Exploratory Data Analysis untuk memahami karakteristik data</p>
        </div>
        <div style="background: #f1f5fd; padding: 20px; border-radius: 10px; border-left: 4px solid #4a6baf;">
            <h3 style="color: #2c3e50; margin-top: 0;">⚙️ Preprocessing</h3>
            <p>Pemrosesan data sebelum analisis clustering</p>
        </div>
        <div style="background: #f1f5fd; padding: 20px; border-radius: 10px; border-left: 4px solid #4a6baf;">
            <h3 style="color: #2c3e50; margin-top: 0;">🧩 Clustering</h3>
            <p>Analisis clustering dengan Spectral Clustering dan optimasi PSO</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    next_button("Upload Data")

# === UPLOAD DATA ===
elif menu == menu_items["Upload Data"]:
    st.markdown('<div class="title">📤 Upload Data Excel</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="margin-top: 0;">Petunjuk:</h4>
            <ol>
                <li>File yang diunggah harus berupa file <strong>Excel</strong> dengan ekstensi <code>.xlsx</code></li>
                <li>Data harus memuat variabel-variabel terkait indikator kemiskinan</li>
                <li>Pastikan data sudah bersih dan siap diproses</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx", key="file_uploader")
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.session_state.df = df
                st.success("✅ Data berhasil dimuat!")
                
                with st.expander("Lihat Data"):
                    st.dataframe(df)
                
                # Informasi dasar tentang data
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Jumlah Baris", df.shape[0])
                with cols[1]:
                    st.metric("Jumlah Kolom", df.shape[1])
                
                next_button("EDA")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memuat file: {str(e)}")

# === EDA ===
elif menu == menu_items["EDA"]:
    st.markdown('<div class="title">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di halaman Upload Data")
    else:
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Distribusi", "🔗 Korelasi"])
        
        with tab1:
            st.subheader("Informasi Dataset")
            st.write(df.info())
            
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe())
            
            st.subheader("Missing Values")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=["Jumlah Missing"])
            st.dataframe(missing_df)
        
        with tab2:
            st.subheader("Distribusi Variabel Numerik")
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            selected_col = st.selectbox("Pilih variabel:", numeric_columns)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(df[selected_col], kde=True, bins=30, color='#4a6baf', ax=ax)
            ax.set_title(f'Distribusi {selected_col}', fontsize=14)
            st.pyplot(fig)
        
        with tab3:
            st.subheader("Heatmap Korelasi")
            numerical_df = df.select_dtypes(include=['number'])
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
            st.pyplot(fig)
        
        next_button("Preprocessing")

# === PREPROCESSING ===
elif menu == menu_items["Preprocessing"]:
    st.markdown('<div class="title">⚙️ Preprocessing Data</div>', unsafe_allow_html=True)
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di halaman Upload Data")
    else:
        df = st.session_state.df
        
        st.subheader("Pemilihan Variabel")
        all_columns = list(df.columns)
        if 'Kabupaten/Kota' in all_columns:
            default_cols = [col for col in all_columns if col != 'Kabupaten/Kota']
        else:
            default_cols = all_columns
        
        selected_cols = st.multiselect(
            "Pilih variabel yang akan digunakan:",
            options=all_columns,
            default=default_cols
        )
        
        st.subheader("Scaling Data")
        scaler_type = st.radio(
            "Pilih metode scaling:",
            ("Standard Scaler", "Robust Scaler"),
            horizontal=True
        )
        
        if st.button("Proses Data"):
            try:
                X = df[selected_cols]
                
                if scaler_type == "Standard Scaler":
                    scaler = StandardScaler()
                else:
                    scaler = RobustScaler()
                
                X_scaled = scaler.fit_transform(X)
                st.session_state.X_scaled = X_scaled
                st.session_state.X = X
                st.session_state.selected_cols = selected_cols
                
                st.success("✅ Preprocessing data berhasil!")
                
                # Tampilkan hasil scaling
                st.subheader("Hasil Scaling")
                scaled_df = pd.DataFrame(X_scaled, columns=selected_cols)
                st.dataframe(scaled_df.head())
                
                # Visualisasi distribusi setelah scaling
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.boxplot(data=scaled_df, palette="Blues", ax=ax)
                ax.set_title("Distribusi Data Setelah Scaling")
                st.pyplot(fig)
                
                next_button("Clustering")
                
            except Exception as e:
                st.error(f"Terjadi kesalahan saat preprocessing: {str(e)}")

# === CLUSTERING ===
elif menu == menu_items["Clustering"]:
    st.markdown('<div class="title">🧩 Spectral Clustering dengan Optimasi PSO</div>', unsafe_allow_html=True)
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu")
    else:
        X_scaled = st.session_state.X_scaled
        
        st.subheader("Evaluasi Jumlah Cluster Optimal")
        clusters_range = range(2, 11)
        silhouette_scores = []
        db_scores = []

        with st.spinner("Menghitung metrik evaluasi..."):
            for k in clusters_range:
                model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
                labels = model.fit_predict(X_scaled)
                silhouette_scores.append(silhouette_score(X_scaled, labels))
                db_scores.append(davies_bouldin_score(X_scaled, labels))

        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': db_scores
        }, index=clusters_range)
        
        # Visualisasi metrik evaluasi
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(clusters_range, silhouette_scores, 'bo-', label='Silhouette Score')
        ax.set_xlabel('Jumlah Cluster')
        ax.set_ylabel('Silhouette Score', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax.twinx()
        ax2.plot(clusters_range, db_scores, 'ro-', label='Davies-Bouldin Index')
        ax2.set_ylabel('DB Index', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('Evaluasi Jumlah Cluster Optimal')
        fig.tight_layout()
        st.pyplot(fig)
        
        best_k_silhouette = max(zip(clusters_range, silhouette_scores), key=lambda x: x[1])[0]
        best_k_dbi = min(zip(clusters_range, db_scores), key=lambda x: x[1])[0]
        
        cols = st.columns(2)
        with cols[0]:
            st.metric("Jumlah Cluster Optimal (Silhouette)", best_k_silhouette)
        with cols[1]:
            st.metric("Jumlah Cluster Optimal (DBI)", best_k_dbi)
        
        st.subheader("Parameter Clustering")
        k_final = st.number_input(
            "Pilih jumlah cluster (k):",
            min_value=2,
            max_value=10,
            value=best_k_silhouette,
            step=1
        )
        
        if st.button("Lakukan Clustering"):
            with st.spinner("Menjalankan Spectral Clustering..."):
                try:
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
                                eigvals, eigvecs = eigsh(L, k=k_final, which='SM')
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
                        ax1.scatter(U_norm[:, 0], U_norm[:, 1], c=labels, cmap='viridis')
                        ax1.set_title(f'Sebelum Optimasi PSO (gamma={gamma})')
                        ax1.set_xlabel('Komponen 1')
                        ax1.set_ylabel('Komponen 2')
                        
                        # Sesudah optimasi
                        ax2.scatter(U_opt[:, 0], U_opt[:, 1], c=labels_opt, cmap='viridis')
                        ax2.set_title(f'Sesudah Optimasi PSO (gamma={best_gamma:.4f})')
                        ax2.set_xlabel('Komponen 1')
                        ax2.set_ylabel('Komponen 2')
                        
                        st.pyplot(fig)
                        
                        # Perbandingan evaluasi
                        st.subheader("Perbandingan Evaluasi Clustering")
                        
                        silhouette_before = silhouette_score(U_norm, labels)
                        dbi_before = davies_bouldin_score(U_norm, labels)
                        silhouette_after = silhouette_score(U_opt, labels_opt)
                        dbi_after = davies_bouldin_score(U_opt, labels_opt)
                        
                        eval_df = pd.DataFrame({
                            'Metrik': ['Silhouette Score', 'Davies-Bouldin Index'],
                            'Sebelum Optimasi': [silhouette_before, dbi_before],
                            'Sesudah Optimasi': [silhouette_after, dbi_after],
                            'Perubahan': [
                                f"{(silhouette_after-silhouette_before)/silhouette_before*100:.1f}%" if silhouette_before != 0 else "N/A",
                                f"{(dbi_after-dbi_before)/dbi_before*100:.1f}%" if dbi_before != 0 else "N/A"
                            ]
                        })
                        
                        st.dataframe(eval_df)
                        
                        next_button("Hasil & Analisis")
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat clustering: {str(e)}")

# === HASIL & ANALISIS ===
elif menu == menu_items["Hasil & Analisis"]:
    st.markdown('<div class="title">📊 Hasil & Analisis Clustering</div>', unsafe_allow_html=True)
    
    if 'labels' not in st.session_state or 'df' not in st.session_state:
        st.warning("⚠️ Silakan lakukan clustering terlebih dahulu")
    else:
        df = st.session_state.df.copy()
        labels = st.session_state.labels
        df['Cluster'] = labels
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Hasil Cluster", "📊 Visualisasi", "🔍 Analisis", "📌 Rekomendasi"])
        
        with tab1:
            st.subheader("Distribusi Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            
            cols = st.columns(2)
            with cols[0]:
                st.dataframe(cluster_counts)
            with cols[1]:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Blues", ax=ax)
                ax.set_title('Jumlah Data per Cluster')
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Jumlah')
                st.pyplot(fig)
            
            st.subheader("Data dengan Label Cluster")
            st.dataframe(df.sort_values(by='Cluster'))
        
        with tab2:
            st.subheader("Visualisasi Persebaran Cluster")
            
            if 'U_opt' in st.session_state:
                U_opt = st.session_state.U_opt
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(U_opt[:,0], U_opt[:,1], c=df['Cluster'], cmap='viridis', s=100)
                
                if 'Kabupaten/Kota' in df.columns:
                    for i, row in df.iterrows():
                        ax.text(U_opt[i,0]+0.01, U_opt[i,1]+0.01, row['Kabupaten/Kota'], fontsize=8, alpha=0.7)
                
                ax.set_title("Visualisasi Persebaran Hasil Clustering")
                ax.set_xlabel('Komponen 1')
                ax.set_ylabel('Komponen 2')
                plt.colorbar(scatter, label='Cluster')
                st.pyplot(fig)
            
            st.subheader("Karakteristik Cluster")
            if 'selected_cols' in st.session_state:
                numeric_cols = [col for col in st.session_state.selected_cols if col in df.select_dtypes(include=['float64', 'int64']).columns]
                if numeric_cols:
                    cluster_means = df.groupby('Cluster')[numeric_cols].mean()
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                    ax.set_title('Rata-rata Nilai per Cluster')
                    st.pyplot(fig)
        
        with tab3:
            st.subheader("Feature Importance")
            
            if 'selected_cols' in st.session_state:
                X = df[st.session_state.selected_cols]
                y = df['Cluster']
                
                rf = RandomForestClassifier(random_state=42)
                rf.fit(X, y)
                importances = rf.feature_importances_
                
                feat_importance = pd.DataFrame({
                    'Fitur': X.columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)
                
                cols = st.columns(2)
                with cols[0]:
                    st.dataframe(feat_importance)
                with cols[1]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.barplot(x='Importance', y='Fitur', data=feat_importance, palette="Blues", ax=ax)
                    ax.set_title('Feature Importance - Random Forest')
                    st.pyplot(fig)
            
            st.subheader("Kabupaten/Kota dengan Kemiskinan Tertinggi dan Terendah")
            if 'Persentase Penduduk Miskin (%)' in df.columns:
                top3 = df.sort_values(by='Persentase Penduduk Miskin (%)', ascending=False).head(3)
                bottom3 = df.sort_values(by='Persentase Penduduk Miskin (%)', ascending=True).head(3)
                
                st.write("**3 Kabupaten/Kota dengan Kemiskinan Tertinggi:**")
                st.dataframe(top3[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']])
                
                st.write("**3 Kabupaten/Kota dengan Kemiskinan Terendah:**")
                st.dataframe(bottom3[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']])
            else:
                st.warning("Kolom 'Persentase Penduduk Miskin (%)' tidak ditemukan dalam data.")
        
        with tab4:
            st.subheader("Rekomendasi Kebijakan Berdasarkan Cluster")
            
            if 'selected_cols' in st.session_state and 'cluster_means' in locals():
                st.write("""
                **Analisis Cluster:**
                - **Cluster 0:** Wilayah dengan tingkat kemiskinan tinggi, membutuhkan intervensi khusus
                - **Cluster 1:** Wilayah dengan tingkat kemiskinan sedang, perlu peningkatan program sosial
                - **Cluster 2:** Wilayah dengan tingkat kemiskinan rendah, bisa menjadi model untuk cluster lain
                """)
                
                st.write("""
                **Rekomendasi:**
                1. Fokuskan program pengentasan kemiskinan pada wilayah di Cluster 0
                2. Tingkatkan akses pendidikan dan kesehatan di Cluster 1
                3. Pelajari best practice dari Cluster 2 untuk diterapkan di cluster lain
                4. Monitor perkembangan tiap cluster secara berkala
                """)
