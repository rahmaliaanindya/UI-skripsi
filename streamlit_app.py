# === IMPORT LIBRARY ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings("ignore")

# === KONFIGURASI HALAMAN ===
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS Styling Modern ===
def local_css():
    st.markdown(
        """
        <style>
            /* Font dan warna dasar */
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f8f9fa;
                color: #333;
            }
            
            /* Container utama */
            .main {
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            /* Judul */
            .title {
                color: #2c3e50;
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid #e9ecef;
            }
            
            /* Subjudul */
            h2 {
                color: #3498db;
                font-size: 1.5rem;
                margin-top: 1.5rem;
            }
            
            /* Sidebar */
            .sidebar .sidebar-content {
                background-color: #2c3e50;
                color: white;
            }
            
            /* Tombol */
            .stButton>button {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.3s;
            }
            
            .stButton>button:hover {
                background-color: #2980b9;
                transform: translateY(-2px);
            }
            
            /* Dataframe */
            .dataframe {
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            /* Visualisasi */
            .stPlot {
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            /* Card */
            .card {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                border-left: 4px solid #3498db;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu = st.sidebar.radio(
    "NAVIGASI APLIKASI",
    ("🏠 Home", "📤 Upload Data", "🔍 EDA", "⚙️ Preprocessing", "🧩 Clustering", "📊 Hasil Analisis"),
    index=0
)

# === FUNGSI UTAMA ===
def main():
    try:
        # === HOME ===
        if menu == "🏠 Home":
            st.markdown("""
            <div class="title">ANALISIS KLUSTER KEMISKINAN JAWA TIMUR</div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Tentang Aplikasi</h3>
                <p>Aplikasi ini membantu analisis pola kemiskinan di Jawa Timur menggunakan metode Spectral Clustering.</p>
                <p>Gunakan menu sidebar untuk mulai menganalisis data Anda.</p>
            </div>
            
            <div class="card">
                <h3>Langkah-langkah Analisis</h3>
                <ol>
                    <li>Upload data Excel</li>
                    <li>Exploratory Data Analysis</li>
                    <li>Preprocessing data</li>
                    <li>Clustering dengan Spectral Clustering</li>
                    <li>Analisis hasil clustering</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        # === UPLOAD DATA ===
        elif menu == "📤 Upload Data":
            st.markdown('<div class="title">UPLOAD DATA</div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Persyaratan Data</h3>
                <p>File Excel harus mengandung kolom-kolom berikut:</p>
                <ul>
                    <li>Kabupaten/Kota</li>
                    <li>Persentase Penduduk Miskin (%)</li>
                    <li>Jumlah Penduduk Miskin (ribu jiwa)</li>
                    <li>Indikator kemiskinan lainnya</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx", "xls"])
            
            if uploaded_file:
                try:
                    df = pd.read_excel(uploaded_file)
                    st.session_state.df = df
                    
                    st.success("✅ Data berhasil dimuat!")
                    
                    with st.expander("LIHAT DATA", expanded=True):
                        st.dataframe(df.style.highlight_max(axis=0, color='#e6f3ff'))
                        
                        cols = st.columns(2)
                        with cols[0]:
                            st.metric("Jumlah Kabupaten/Kota", df.shape[0])
                        with cols[1]:
                            st.metric("Jumlah Variabel", df.shape[1])
                
                except Exception as e:
                    st.error(f"Error membaca file: {str(e)}")

        # === EDA ===
        elif menu == "🔍 EDA":
            st.markdown('<div class="title">EXPLORATORY DATA ANALYSIS</div>', unsafe_allow_html=True)
            
            if 'df' not in st.session_state:
                st.warning("⚠️ Silakan upload data terlebih dahulu")
            else:
                df = st.session_state.df
                
                tab1, tab2, tab3 = st.tabs(["📋 Data Overview", "📈 Distribusi", "🔗 Korelasi"])
                
                with tab1:
                    st.subheader("Data Lengkap")
                    st.dataframe(df)
                    
                    st.subheader("Statistik Deskriptif")
                    st.dataframe(df.describe().style.background_gradient(cmap='Blues'))
                
                with tab2:
                    st.subheader("Distribusi Variabel")
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                    selected_col = st.selectbox("Pilih variabel:", numeric_cols)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.histplot(df[selected_col], kde=True, color='#3498db', bins=30, ax=ax)
                    ax.set_title(f'Distribusi {selected_col}', fontweight='bold')
                    st.pyplot(fig)
                
                with tab3:
                    st.subheader("Matriks Korelasi")
                    numeric_df = df.select_dtypes(include=['number'])
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, 
                               fmt='.2f', linewidths=.5, ax=ax)
                    ax.set_title('Korelasi Antar Variabel', fontweight='bold')
                    st.pyplot(fig)

        # === PREPROCESSING ===
        elif menu == "⚙️ Preprocessing":
            st.markdown('<div class="title">PREPROCESSING DATA</div>', unsafe_allow_html=True)
            
            if 'df' not in st.session_state:
                st.warning("⚠️ Silakan upload data terlebih dahulu")
            else:
                df = st.session_state.df
                
                st.markdown("""
                <div class="card">
                    <h3>Metode Preprocessing</h3>
                    <p>Data akan diproses menggunakan <strong>RobustScaler</strong> untuk menangani outlier.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("PROSES DATA", key="preprocess_btn"):
                    with st.spinner("Sedang memproses data..."):
                        try:
                            X = df.select_dtypes(include=['float64', 'int64'])
                            
                            scaler = RobustScaler()
                            X_scaled = scaler.fit_transform(X)
                            st.session_state.X_scaled = X_scaled
                            
                            st.success("✅ Preprocessing data selesai!")
                            
                            scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
                            
                            st.subheader("Hasil Scaling")
                            st.dataframe(scaled_df)
                            
                            # Visualisasi perbandingan sebelum/sesudah scaling
                            st.subheader("Perbandingan Distribusi")
                            selected_col = st.selectbox("Pilih variabel untuk visualisasi:", X.columns)
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                            
                            # Sebelum scaling
                            sns.histplot(X[selected_col], kde=True, color='#3498db', ax=ax1)
                            ax1.set_title(f'Sebelum Scaling\n({selected_col})')
                            
                            # Sesudah scaling
                            sns.histplot(scaled_df[selected_col], kde=True, color='#2ecc71', ax=ax2)
                            ax2.set_title(f'Sesudah Scaling\n({selected_col})')
                            
                            st.pyplot(fig)
                        
                        except Exception as e:
                            st.error(f"Error saat preprocessing: {str(e)}")

        # === CLUSTERING ===
        elif menu == "🧩 Clustering":
            st.markdown('<div class="title">SPECTRAL CLUSTERING</div>', unsafe_allow_html=True)
            
            if 'X_scaled' not in st.session_state:
                st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu")
            else:
                X_scaled = st.session_state.X_scaled
                
                st.markdown("""
                <div class="card">
                    <h3>Clustering dengan Spectral Clustering</h3>
                    <p>Analisis clustering akan dilakukan dengan 2 cluster berdasarkan evaluasi DBI.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("PROSES CLUSTERING", key="cluster_btn"):
                    with st.spinner("Sedang melakukan clustering..."):
                        try:
                            # Spectral Clustering
                            sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', 
                                                  random_state=42)
                            labels = sc.fit_predict(X_scaled)
                            st.session_state.labels = labels
                            
                            st.success("✅ Clustering selesai!")
                            
                            # Visualisasi dengan PCA
                            pca = PCA(n_components=2)
                            X_pca = pca.fit_transform(X_scaled)
                            
                            fig, ax = plt.subplots(figsize=(10, 7))
                            scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', s=100)
                            
                            # Tambahkan label jika ada kolom Kabupaten/Kota
                            if 'df' in st.session_state and 'Kabupaten/Kota' in st.session_state.df.columns:
                                df = st.session_state.df
                                for i, txt in enumerate(df['Kabupaten/Kota']):
                                    ax.annotate(txt, (X_pca[i,0], X_pca[i,1]), 
                                               fontsize=8, alpha=0.7)
                            
                            ax.set_title('Visualisasi Cluster (PCA)', fontweight='bold')
                            ax.set_xlabel('Principal Component 1')
                            ax.set_ylabel('Principal Component 2')
                            plt.colorbar(scatter, label='Cluster')
                            st.pyplot(fig)
                            
                            # Evaluasi clustering
                            st.subheader("Evaluasi Clustering")
                            
                            eval_df = pd.DataFrame({
                                'Metric': ['Silhouette Score', 'Davies-Bouldin Index'],
                                'Value': [
                                    silhouette_score(X_scaled, labels),
                                    davies_bouldin_score(X_scaled, labels)
                                ]
                            })
                            
                            st.dataframe(eval_df.style.highlight_max(subset=['Value'], color='#e6f3ff', axis=0))
                        
                        except Exception as e:
                            st.error(f"Error saat clustering: {str(e)}")

        # === HASIL ANALISIS ===
        elif menu == "📊 Hasil Analisis":
            st.markdown('<div class="title">HASIL ANALISIS CLUSTERING</div>', unsafe_allow_html=True)
            
            if 'labels' not in st.session_state or 'df' not in st.session_state:
                st.warning("⚠️ Silakan lakukan clustering terlebih dahulu")
            else:
                df = st.session_state.df.copy()
                df['Cluster'] = st.session_state.labels
                
                st.markdown("""
                <div class="card">
                    <h3>Hasil Clustering</h3>
                    <p>Berikut adalah hasil pengelompokan wilayah berdasarkan indikator kemiskinan.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan data dengan cluster
                st.subheader("Data dengan Label Cluster")
                st.dataframe(df.style.background_gradient(subset=['Cluster'], cmap='viridis'))
                
                # Analisis cluster
                st.subheader("Karakteristik Cluster")
                
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                cluster_means = df.groupby('Cluster')[numeric_cols].mean()
                
                st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))
                
                # Visualisasi heatmap
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.heatmap(cluster_means.T, annot=True, cmap='coolwarm', fmt='.2f', 
                           linewidths=.5, ax=ax)
                ax.set_title('Perbandingan Rata-rata Indikator per Cluster', fontweight='bold')
                st.pyplot(fig)
                
                # Wilayah dengan kemiskinan tertinggi/terendah
                if 'Persentase Penduduk Miskin (%)' in df.columns:
                    st.subheader("Wilayah dengan Tingkat Kemiskinan Ekstrim")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**5 Wilayah dengan Kemiskinan Tertinggi**")
                        top5 = df.nlargest(5, 'Persentase Penduduk Miskin (%)')
                        st.dataframe(top5[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']]
                                   .style.background_gradient(subset=['Persentase Penduduk Miskin (%)'], cmap='Reds'))
                    
                    with cols[1]:
                        st.markdown("**5 Wilayah dengan Kemiskinan Terendah**")
                        bottom5 = df.nsmallest(5, 'Persentase Penduduk Miskin (%)')
                        st.dataframe(bottom5[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)', 'Cluster']]
                                   .style.background_gradient(subset=['Persentase Penduduk Miskin (%)'], cmap='Greens'))
                
                # Rekomendasi kebijakan
                st.subheader("Rekomendasi Kebijakan")
                
                st.markdown("""
                <div class="card">
                    <h4>Cluster 0 (Kemiskinan Tinggi):</h4>
                    <ul>
                        <li>Prioritaskan program bantuan sosial</li>
                        <li>Tingkatkan akses pendidikan dan kesehatan</li>
                        <li>Program pelatihan keterampilan</li>
                    </ul>
                    
                    <h4>Cluster 1 (Kemiskinan Rendah):</h4>
                    <ul>
                        <li>Pertahankan program yang sudah berjalan</li>
                        <li>Fokus pada peningkatan kualitas hidup</li>
                        <li>Pengembangan ekonomi lokal</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan sistem: {str(e)}")

if __name__ == "__main__":
    main()
