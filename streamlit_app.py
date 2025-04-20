import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide"
)

# CSS Styling
def local_css():
    st.markdown(
        """
        <style>
            body {
                background-color: #fdf0ed;
            }
            .main {
                background: linear-gradient(to bottom right, #e74c3c, #f39c12, #f8c471);
            }
            .block-container {
                padding-top: 1rem;
                background-color: transparent;
            }
            h1, h2, h3, h4, h5, h6, p, div, span {
                color: #2c3e50 !important;
            }
            .title {
                font-family: 'Helvetica', sans-serif;
                color: #1f3a93;
                font-size: 38px;
                font-weight: bold;
                text-align: center;
                padding: 30px 0 10px 0;
            }
            .sidebar .sidebar-content {
                background-color: #fef5e7;
            }
            .legend-box {
                padding: 15px;
                border-radius: 10px;
                background-color: #ffffffdd;
                box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
                margin-top: 20px;
            }
            .info-card {
                background-color: #ffffffaa;
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 25px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            }
            .nav-button {
                display: flex;
                justify-content: space-between;
                margin-top: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Terapkan CSS
local_css()

# === Navigasi Menu di Atas ===
menu_options = ["Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", 
                "Step 3: Visualisasi Data", "Step 4: Hasil Clustering", "Step 5: Analisis Hasil"]
menu = st.radio(
    "Navigasi Aplikasi:",
    menu_options,
    horizontal=True
)

# Fungsi navigasi
def create_nav_buttons(current_step):
    if current_step == "Home":
        if st.button("Mulai Analisis →", key="home_next"):
            st.session_state.menu = "Step 1: Upload Data"
            st.experimental_rerun()
    
    elif current_step in menu_options[1:-1]:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Kembali", key=f"{current_step}_back"):
                prev_index = menu_options.index(current_step) - 1
                st.session_state.menu = menu_options[prev_index]
                st.experimental_rerun()
        with col2:
            if st.button("Lanjut →", key=f"{current_step}_next"):
                next_index = menu_options.index(current_step) + 1
                st.session_state.menu = menu_options[next_index]
                st.experimental_rerun()
    
    elif current_step == "Step 5: Analisis Hasil":
        if st.button("← Kembali ke Clustering", key="analysis_back"):
            st.session_state.menu = "Step 4: Hasil Clustering"
            st.experimental_rerun()

# === Konten berdasarkan Menu ===
if menu == "Home":
    st.markdown("""
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊

    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode **Spectral Clustering**
    - 📈 Mengevaluasi hasil pengelompokan
    - 🔍 Menganalisis karakteristik cluster

    📌 Silakan pilih menu di atas atau klik tombol di bawah untuk memulai analisis.
    """)
    create_nav_buttons(menu)

# 2. UPLOAD DATA
elif menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")

    # Deskripsi tentang data yang harus diunggah
    st.markdown("""
    ### Ketentuan Data:
    - Data berupa file **Excel (.xlsx)**.
    - Data mencakup kolom-kolom berikut:
        1. **Persentase Penduduk Miskin (%)**
        2. **Jumlah Penduduk Miskin (ribu jiwa)**
        3. **Harapan Lama Sekolah (Tahun)**
        4. **Rata-Rata Lama Sekolah (Tahun)**
        5. **Tingkat Pengangguran Terbuka (%)**
        6. **Tingkat Partisipasi Angkatan Kerja (%)**
        7. **Angka Harapan Hidup (Tahun)**
        8. **Garis Kemiskinan (Rupiah/Bulan/Kapita)**
        9. **Indeks Pembangunan Manusia**
        10. **Rata-rata Upah/Gaji Bersih Pekerja Informal Berdasarkan Lapangan Pekerjaan Utama (Rp)**
        11. **Rata-rata Pendapatan Bersih Sebulan Pekerja Informal berdasarkan Pendidikan Tertinggi - Jumlah (Rp)**
    """)

    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
    
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil dimuat!")
        st.write(df)
    
    create_nav_buttons(menu)

# 3. PREPROCESSING
elif menu == "Step 2: Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    if 'df' in st.session_state:
        df = st.session_state.df
        st.subheader("Cek Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Cek Duplikat")
        st.write(f"Jumlah duplikat: {df.duplicated().sum()}")

        st.subheader("Statistik Deskriptif")
        st.write(df.describe())

        st.subheader("Normalisasi dan Seleksi Fitur")
        features = df.select_dtypes(include=['float64', 'int64']).columns
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.session_state.X_scaled = X_scaled
        st.session_state.features = features
        st.success("Fitur telah dinormalisasi dan disimpan.")
    else:
        st.warning("Silakan upload data terlebih dahulu.")
    
    create_nav_buttons(menu)

# 4. VISUALISASI DATA
elif menu == "Step 3: Visualisasi Data":
    st.header("📊 Visualisasi Data")
    if 'df' in st.session_state:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])

        st.subheader("Heatmap Korelasi")
        plt.figure(figsize=(10, 5))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()

    else:
        st.warning("Silakan upload data terlebih dahulu.")
    
    create_nav_buttons(menu)

# 5. HASIL CLUSTERING
elif menu == "Step 4: Hasil Clustering":
    st.header("🧩 Hasil Clustering")
    
    if 'X_scaled' in st.session_state:
        X_scaled = st.session_state.X_scaled
        st.subheader("Evaluasi Jumlah Cluster (Silhouette & DBI)")

        clusters_range = range(2, 10)
        silhouette_scores = {}
        dbi_scores = {}

        for k in clusters_range:
            clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
            labels = clustering.fit_predict(X_scaled)
            silhouette_scores[k] = silhouette_score(X_scaled, labels)
            dbi_scores[k] = davies_bouldin_score(X_scaled, labels)

        # Tampilkan grafik evaluasi
        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': dbi_scores
        })
        st.line_chart(score_df)

        # Menentukan cluster terbaik dari dua metrik
        best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
        best_k_dbi = min(dbi_scores, key=dbi_scores.get)

        st.success(f"🔹 Jumlah cluster optimal berdasarkan **Silhouette Score**: {best_k_silhouette}")
        st.success(f"🔸 Jumlah cluster optimal berdasarkan **Davies-Bouldin Index**: {best_k_dbi}")

        # Pilihan manual untuk k_final atau default ke Silhouette
        st.subheader("Pilih Jumlah Cluster untuk Clustering Final")
        k_final = st.number_input("Jumlah Cluster (k):", min_value=2, max_value=10, value=best_k_silhouette, step=1)

        # Final Clustering
        if st.button("Lakukan Clustering"):
            final_cluster = SpectralClustering(n_clusters=k_final, affinity='nearest_neighbors', random_state=42)
            labels = final_cluster.fit_predict(X_scaled)
            st.session_state.labels = labels
            st.session_state.k_final = k_final
            st.success("Clustering berhasil dilakukan!")

        if 'labels' in st.session_state:
            # Visualisasi 2D menggunakan PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            st.subheader("Visualisasi Clustering (PCA)")
            plt.figure(figsize=(8, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=st.session_state.labels, cmap='viridis', edgecolor='k')
            plt.title("Visualisasi Clustering dengan Spectral Clustering")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            st.pyplot(plt.gcf())
            plt.clf()

            # Menampilkan hasil clustering
            if 'df' in st.session_state:
                df = st.session_state.df.copy()
                df['Cluster'] = st.session_state.labels

                st.subheader("📄 Hasil Cluster pada Data")

                # Urutkan data berdasarkan 'Cluster'
                df_sorted = df.sort_values(by='Cluster')

                # Tampilkan DataFrame yang sudah diurutkan
                st.dataframe(df_sorted)

                # Tampilkan jumlah anggota tiap cluster
                st.subheader("📊 Jumlah Anggota per Cluster")
                cluster_counts = df['Cluster'].value_counts().sort_index()
                st.bar_chart(cluster_counts)

    else:
        st.warning("⚠️ Data belum diproses. Silakan lakukan preprocessing terlebih dahulu.")
    
    create_nav_buttons(menu)

# 6. ANALISIS HASIL
elif menu == "Step 5: Analisis Hasil":
    st.header("🔍 Analisis Hasil Clustering")

    if 'labels' in st.session_state and 'df' in st.session_state:
        df = st.session_state.df.copy()
        labels = st.session_state.labels
        k_final = st.session_state.k_final
        df['Cluster'] = labels

        # Hanya ambil kolom numerik untuk analisis
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_numeric = df[numeric_cols]

        # 1. Analisis distribusi cluster
        st.subheader("📊 Analisis Distribusi Cluster")
        with st.expander("Lihat Rata-rata Nilai per Cluster"):
            cluster_summary = df_numeric.groupby('Cluster').mean().T
            st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))
            
            # Visualisasi heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(cluster_summary, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Perbandingan Rata-rata Nilai per Cluster")
            st.pyplot(plt)
            plt.clf()

        # 2. Visualisasi distribusi cluster
        st.subheader("📈 Distribusi Cluster")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            cluster_counts = df['Cluster'].value_counts().sort_index()
            fig1, ax1 = plt.subplots()
            ax1.pie(cluster_counts, labels=cluster_counts.index, 
                   autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis"))
            ax1.axis('equal')
            ax1.set_title("Persentase Distribusi Cluster")
            st.pyplot(fig1)
        
        with col2:
            # Bar chart
            fig2, ax2 = plt.subplots()
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, 
                        palette="viridis", ax=ax2)
            ax2.set_title("Jumlah Kabupaten/Kota per Cluster")
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Jumlah")
            st.pyplot(fig2)

        # 3. Insight berdasarkan cluster
        st.subheader("🔎 Insight per Cluster")
        tab1, tab2, tab3 = st.tabs(["Statistik Deskriptif", "Karakteristik", "Perbandingan"])
        
        with tab1:
            for cluster_num in sorted(df['Cluster'].unique()):
                st.markdown(f"### Cluster {cluster_num}")
                cluster_data = df[df['Cluster'] == cluster_num]
                st.dataframe(cluster_data[numeric_cols].describe().style.background_gradient(cmap='Greens'))
        
        with tab2:
            for cluster_num in sorted(df['Cluster'].unique()):
                st.markdown(f"### Karakteristik Cluster {cluster_num}")
                cluster_data = df[df['Cluster'] == cluster_num]
                
                # Ambil 3 indikator tertinggi dan terendah
                mean_values = cluster_data[numeric_cols].mean().sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Indikator Tertinggi:**")
                    for i, (ind, val) in enumerate(mean_values.head(3).items()):
                        st.write(f"{i+1}. {ind}: {val:.2f}")
                
                with col2:
                    st.markdown("**Indikator Terendah:**")
                    for i, (ind, val) in enumerate(mean_values.tail(3).items()):
                        st.write(f"{i+1}. {ind}: {val:.2f}")
                
                st.markdown("**Contoh Kabupaten/Kota:**")
                st.write(cluster_data['Kabupaten/Kota'].head(5).tolist())
        
        with tab3:
            st.markdown("### Perbandingan Antar Cluster")
            selected_feature = st.selectbox("Pilih indikator untuk dibandingkan:", numeric_cols)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=selected_feature, data=df, palette="viridis", ax=ax3)
            ax3.set_title(f"Distribusi {selected_feature} per Cluster")
            st.pyplot(fig3)

        # 4. Feature Importance
        st.subheader("📌 Feature Importance")
        try:
            X = df[numeric_cols].drop(columns=['Cluster'], errors='ignore')
            y = df['Cluster']
            
            # Hapus kolom dengan variansi rendah
            selector = VarianceThreshold(threshold=0.1)
            X_selected = selector.fit_transform(X)
            selected_features = X.columns[selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features)
            
            # Latih model RandomForest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Tampilkan feature importance
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            
            fig4, ax4 = plt.subplots(figsize=(10,6))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax4)
            ax4.set_title("Indikator Paling Berpengaruh dalam Clustering")
            ax4.set_xlabel("Tingkat Kepentingan")
            st.pyplot(fig4)
            
            # Tampilkan penjelasan
            st.markdown(f"**Indikator paling penting:** {importances.index[0]} ({importances.values[0]:.2f})")
            st.markdown(f"**Indikator paling tidak penting:** {importances.index[-1]} ({importances.values[-1]:.2f})")
            
        except Exception as e:
            st.error(f"Error dalam menghitung feature importance: {str(e)}")

        # 5. Analisis PCA
        st.subheader("🔄 Analisis PCA")
        try:
            # Standarisasi data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Lakukan PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            
            # Visualisasi hasil PCA
            fig5, ax5 = plt.subplots(figsize=(10,8))
            scatter = ax5.scatter(principal_components[:,0], principal_components[:,1], 
                                 c=df['Cluster'], cmap='viridis', alpha=0.7)
            ax5.set_title("Visualisasi Cluster dalam Ruang PCA 2D")
            ax5.set_xlabel(f"PC1 (Variansi: {pca.explained_variance_ratio_[0]:.2f})")
            ax5.set_ylabel(f"PC2 (Variansi: {pca.explained_variance_ratio_[1]:.2f})")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig5)
            
            # Tampilkan kontribusi fitur
            st.markdown("**Kontribusi Fitur terhadap Komponen Utama:**")
            pca_loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=X.columns)
            st.dataframe(pca_loadings.style.background_gradient(cmap='RdBu', axis=0))
            
        except Exception as e:
            st.error(f"Error dalam analisis PCA: {str(e)}")

        # 6. Contoh Wilayah Ekstrim
        st.subheader("🏙️ Contoh Wilayah Ekstrim")
        
        if k_final == 2:
            # Untuk kasus 2 cluster
            poor_cluster = df[df['Cluster'] == df['Cluster'].min()]
            rich_cluster = df[df['Cluster'] == df['Cluster'].max()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 3 Wilayah dengan Tingkat Kemiskinan Tertinggi")
                top_poor = poor_cluster.nlargest(3, 'Persentase Penduduk Miskin (%)')
                st.dataframe(top_poor[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)']])
            
            with col2:
                st.markdown("### 3 Wilayah dengan Tingkat Kemiskinan Terendah")
                top_rich = rich_cluster.nsmallest(3, 'Persentase Penduduk Miskin (%)')
                st.dataframe(top_rich[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)']])
        else:
            st.warning("Analisis wilayah ekstrim hanya tersedia untuk clustering dengan 2 cluster")

    else:
        st.warning("⚠️ Hasil clustering belum ada. Silakan lakukan clustering terlebih dahulu.")
    
    create_nav_buttons(menu)
