import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

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
        </style>
        """,
        unsafe_allow_html=True
    )

# Terapkan CSS
local_css()

# === Navigasi Menu di Atas ===
menu = st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"),
    horizontal=True
)

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

    📌 Silakan pilih menu di atas untuk memulai analisis.
    """)

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
        st.write("Fitur telah dinormalisasi dan disimpan.")
    else:
        st.warning("Silakan upload data terlebih dahulu.")

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
        final_cluster = SpectralClustering(n_clusters=k_final, affinity='nearest_neighbors', random_state=42)
        labels = final_cluster.fit_predict(X_scaled)
        st.session_state.labels = labels

        # Visualisasi 2D menggunakan PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.subheader("Visualisasi Clustering (PCA)")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
        plt.title("Visualisasi Clustering dengan Spectral Clustering")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        st.pyplot(plt.gcf())
        plt.clf()

        # Menampilkan hasil clustering
        if 'df' in st.session_state:
            df = st.session_state.df.copy()
            df['Cluster'] = labels

            st.subheader("📄 Hasil Cluster pada Data")

            # Urutkan data berdasarkan 'Cluster'
            df_sorted = df.sort_values(by='Cluster')

            # Tampilkan DataFrame yang sudah diurutkan
            st.dataframe(df_sorted)

            # Tampilkan jumlah anggota tiap cluster
            st.subheader("📊 Jumlah Anggota per Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)

            # Menambahkan evaluasi variabel yang paling berpengaruh menggunakan RandomForest
            X = df.drop(columns=["Kabupaten/Kota", "Cluster"], errors="ignore")
            y = df["Cluster"]

            # Inisialisasi dan pelatihan model RandomForest
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Menampilkan feature importance
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

            # Visualisasi Feature Importance
            plt.figure(figsize=(10,5))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis")
            plt.title("Feature Importance (Indikator Paling Berpengaruh)")
            plt.xlabel("Tingkat Pengaruh")
            st.pyplot(plt.gcf())
            plt.clf()

            # Visualisasi kontribusi terhadap komponen utama dengan PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)

            plt.figure(figsize=(8,6))
            plt.bar(range(len(X.columns)), pca.components_[0], tick_label=X.columns)
            plt.xticks(rotation=90)
            plt.title("Kontribusi Indikator terhadap Komponen Utama")
            plt.ylabel("Kontribusi")
            st.pyplot(plt.gcf())
            plt.clf()

            # ===========================
# Kesimpulan
# ===========================
elif selected_step == "Kesimpulan":
    st.title("📊 Kesimpulan Clustering Kemiskinan")

    # Menampilkan jumlah kabupaten/kota per cluster
    st.subheader("Jumlah Kabupaten/Kota per Cluster")
    cluster_counts = df['Cluster'].value_counts().sort_index()
    cluster_counts_df = pd.DataFrame({
        'Cluster': cluster_counts.index,
        'Jumlah Kabupaten/Kota': cluster_counts.values
    })
    st.dataframe(cluster_counts_df, use_container_width=True)

    # Menghitung rata-rata persentase kemiskinan tiap cluster
    st.subheader("Rata-rata Persentase Kemiskinan per Cluster")
    cluster_means = df.groupby('Cluster')['Persentase Penduduk Miskin (%)'].mean().sort_values()
    cluster_means_df = cluster_means.reset_index().rename(columns={'Persentase Penduduk Miskin (%)': 'Rata-rata (%)'})
    st.dataframe(cluster_means_df, use_container_width=True)

    # Buat ranking berdasarkan kemiskinan (0 = terendah)
    cluster_order = cluster_means.index
    cluster_rank = {cluster: rank for rank, cluster in enumerate(cluster_order)}
    df['Kemiskinan_Rank'] = df['Cluster'].map(cluster_rank)

    # 3 terendah
    st.subheader("🟢 3 Kabupaten/Kota dengan Tingkat Kemiskinan Terendah")
    top_3_lowest = df[df['Kemiskinan_Rank'] == 0][['Kabupaten/Kota', 'Cluster', 'Persentase Penduduk Miskin (%)']].sort_values(by='Persentase Penduduk Miskin (%)').head(3)
    st.dataframe(top_3_lowest, use_container_width=True)

    # 3 tertinggi
    st.subheader("🔴 3 Kabupaten/Kota dengan Tingkat Kemiskinan Tertinggi")
    highest_rank = df['Kemiskinan_Rank'].max()
    top_3_highest = df[df['Kemiskinan_Rank'] == highest_rank][['Kabupaten/Kota', 'Cluster', 'Persentase Penduduk Miskin (%)']].sort_values(by='Persentase Penduduk Miskin (%)', ascending=False).head(3)
    st.dataframe(top_3_highest, use_container_width=True)

    # Tambahan ringkasan visual (jika mau)
    st.markdown("---")
    st.markdown("✅ **Cluster dengan tingkat kemiskinan terendah adalah Cluster {}**".format(cluster_order[0]))
    st.markdown("❌ **Cluster dengan tingkat kemiskinan tertinggi adalah Cluster {}**".format(cluster_order[-1]))
