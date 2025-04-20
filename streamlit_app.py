# === IMPORT LIBRARY ===
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

# === KONFIGURASI HALAMAN ===
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
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu = st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"),
    horizontal=True
)

# === HOME ===
if menu == "Home":
    st.markdown("""
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊
    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode **Spectral Clustering**
    - 📈 Mengevaluasi hasil pengelompokan
    """)

# === UPLOAD DATA ===
elif menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        st.write(df)

# === PREPROCESSING DATA ===
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
        st.write("✅ Fitur telah dinormalisasi dan disimpan.")
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === VISUALISASI ===
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
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# === CLUSTERING ===
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

        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': dbi_scores
        })
        st.line_chart(score_df)

        best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
        best_k_dbi = min(dbi_scores, key=dbi_scores.get)

        st.success(f"🔹 Jumlah cluster optimal (Silhouette Score): {best_k_silhouette}")
        st.success(f"🔸 Jumlah cluster optimal (Davies-Bouldin Index): {best_k_dbi}")

        k_final = st.number_input("Jumlah Cluster (k):", min_value=2, max_value=10, value=best_k_silhouette, step=1)

        final_cluster = SpectralClustering(n_clusters=k_final, affinity='nearest_neighbors', random_state=42)
        labels = final_cluster.fit_predict(X_scaled)
        st.session_state.labels = labels

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        st.subheader("Visualisasi Clustering (PCA)")
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolor='k')
        plt.title("Clustering Visualisasi (PCA)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        st.pyplot(plt.gcf())
        plt.clf()

        # === HASIL CLUSTER ===
        if 'df' in st.session_state:
            df = st.session_state.df.copy()
            df['Cluster'] = labels
            st.subheader("📄 Hasil Cluster pada Data")
            st.dataframe(df.sort_values(by='Cluster'))

            st.subheader("📊 Jumlah Anggota per Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            st.bar_chart(cluster_counts)

            # === FEATURE IMPORTANCE ===
            X = df.drop(columns=["Kabupaten/Kota", "Cluster"], errors="ignore")
            y = df["Cluster"]

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

            st.subheader("🔍 Indikator Paling Berpengaruh (Random Forest)")
            plt.figure(figsize=(10,5))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis")
            plt.title("Feature Importance")
            plt.xlabel("Tingkat Pengaruh")
            st.pyplot(plt.gcf())
            plt.clf()

            # === PCA CONTRIBUTION ===
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2)
            pca.fit(X_scaled)

            st.subheader("🧭 Kontribusi Variabel terhadap PCA")
            plt.figure(figsize=(10,6))
            plt.bar(range(len(X.columns)), pca.components_[0], tick_label=X.columns)
            plt.xticks(rotation=90)
            plt.ylabel("Kontribusi terhadap Komponen Utama")
            st.pyplot(plt.gcf())
            plt.clf()

            # === INTERPRETASI HASIL ===
            st.subheader("💡 Kesimpulan & Interpretasi Hasil")
            cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
            st.write("Rata-rata nilai indikator untuk masing-masing cluster:")
            st.dataframe(cluster_summary)

            st.markdown("""
            ### Interpretasi Awal:
            - Cluster dengan rata-rata **persentase penduduk miskin paling rendah** bisa dianggap sebagai kategori **kinerja baik**.
            - Cluster dengan nilai indikator pendidikan dan kesehatan yang rendah mungkin termasuk **kategori rentan/tinggi kemiskinan**.
            - Data ini bisa menjadi dasar rekomendasi kebijakan untuk setiap kelompok wilayah.
            """)
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")
