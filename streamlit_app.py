import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from PIL import Image

# === Konfigurasi halaman ===
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide"
)

# === CSS Styling ===
def local_css():
    st.markdown(
        """
        <style>
            body { background-color: #fdf0ed; }
            .main {
                background: linear-gradient(to bottom right, #e74c3c, #f39c12, #f8c471);
            }
            .block-container { padding-top: 1rem; background-color: transparent; }
            h1, h2, h3, h4, h5, h6, p, div, span { color: #2c3e50 !important; }
            .title {
                font-family: 'Helvetica', sans-serif;
                color: #1f3a93;
                font-size: 38px;
                font-weight: bold;
                text-align: center;
                padding: 30px 0 10px 0;
            }
            .sidebar .sidebar-content { background-color: #fef5e7; }
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

local_css()

# === Inisialisasi step ===
if "step" not in st.session_state:
    st.session_state.step = 0

def next_step():
    if st.session_state.step < 4:
        st.session_state.step += 1

def prev_step():
    if st.session_state.step > 0:
        st.session_state.step -= 1

steps = [
    "Home",
    "Step 1: Upload Data",
    "Step 2: Preprocessing Data",
    "Step 3: Visualisasi Data",
    "Step 4: Hasil Clustering"
]

st.markdown(f"### 🧭 Tahap: {steps[st.session_state.step]}")

# === Navigasi tombol ===
col1, col2 = st.columns([1, 5])
with col1:
    if st.session_state.step > 0:
        st.button("⬅️ Previous", on_click=prev_step)
with col2:
    if st.session_state.step < len(steps) - 1:
        st.button("Next ➡️", on_click=next_step)

# === Konten berdasarkan Step ===
step = st.session_state.step

# STEP 0: HOME
if step == 0:
    st.markdown("""
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊

    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode **Spectral Clustering**
    - 📈 Mengevaluasi hasil pengelompokan

    📌 Silakan klik tombol **Next ➡️** untuk memulai analisis.
    """)

# STEP 1: Upload Data
elif step == 1:
    st.header("📤 Upload Data Excel")
    st.markdown("""
    ### Ketentuan Data:
    - File **Excel (.xlsx)** berisi kolom-kolom berikut:
        - Persentase Penduduk Miskin (%)
        - Jumlah Penduduk Miskin (ribu jiwa)
        - Harapan Lama Sekolah (Tahun)
        - Rata-Rata Lama Sekolah (Tahun)
        - Tingkat Pengangguran Terbuka (%)
        - Tingkat Partisipasi Angkatan Kerja (%)
        - Angka Harapan Hidup (Tahun)
        - Garis Kemiskinan (Rupiah/Bulan/Kapita)
        - Indeks Pembangunan Manusia
        - Rata-rata Upah/Gaji Pekerja Informal
        - Rata-rata Pendapatan Bersih Sebulan
    """)

    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        st.dataframe(df)

# STEP 2: Preprocessing
elif step == 2:
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
        st.success("✅ Data telah dinormalisasi dan disimpan.")
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# STEP 3: Visualisasi
elif step == 3:
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
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# STEP 4: Clustering
elif step == 4:
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

        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': dbi_scores
        })
        st.line_chart(score_df)

        best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
        best_k_dbi = min(dbi_scores, key=dbi_scores.get)

        st.success(f"🔹 Cluster terbaik (Silhouette): {best_k_silhouette}")
        st.success(f"🔸 Cluster terbaik (DBI): {best_k_dbi}")

        k_final = st.number_input("Pilih Jumlah Cluster:", min_value=2, max_value=10, value=best_k_silhouette, step=1)
        final_cluster = SpectralClustering(n_clusters=k_final, affinity='nearest_neighbors', random_state=42)
        labels = final_cluster.fit_predict(X_scaled)
        st.session_state.labels = labels

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.subheader("Visualisasi Clustering (PCA)")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
        plt.title("Visualisasi Clustering")
        st.pyplot(plt.gcf())
        plt.clf()

        if 'df' in st.session_state:
            df = st.session_state.df.copy()
            df['Cluster'] = labels
            st.subheader("📄 Data dengan Cluster")
            st.dataframe(df.sort_values(by='Cluster'))

            st.subheader("📊 Jumlah per Cluster")
            st.bar_chart(df['Cluster'].value_counts().sort_index())
    else:
        st.warning("⚠️ Data belum diproses. Silakan ke step Preprocessing.")
