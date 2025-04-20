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

def local_css():
    st.markdown(
        """
        <style>
            /* Base Styles */
            html, body, [class*="css"] {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #2c3e50;
            }
            
            /* Main Container */
            .main {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
                padding: 2rem;
            }
            
            /* Header Styles */
            h1 {
                color: #1f3a93;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
                border-bottom: 3px solid #3498db;
                padding-bottom: 0.5rem;
                text-align: center;
                background: linear-gradient(to right, #1f3a93, #3498db);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            h2 {
                color: #2980b9;
                font-size: 1.8rem;
                margin-top: 2rem;
                margin-bottom: 1rem;
                position: relative;
                padding-left: 1rem;
            }
            
            h2:before {
                content: "";
                position: absolute;
                left: 0;
                top: 0;
                height: 100%;
                width: 5px;
                background: linear-gradient(to bottom, #3498db, #2ecc71);
                border-radius: 3px;
            }
            
            h3 {
                color: #16a085;
                font-size: 1.4rem;
                margin-top: 1.5rem;
            }
            
            /* Cards and Containers */
            .block-container {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid rgba(0, 0, 0, 0.05);
            }
            
            .info-card {
                background: white;
                border-radius: 10px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
                border-left: 4px solid #3498db;
            }
            
            /* Buttons and Interactive Elements */
            .stButton>button {
                border-radius: 8px;
                border: none;
                background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%);
                color: white;
                font-weight: 600;
                padding: 0.5rem 1.5rem;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                background: linear-gradient(135deg, #2980b9 0%, #27ae60 100%);
            }
            
            /* Radio Buttons */
            .stRadio>div {
                flex-direction: row !important;
                align-items: center;
                background: white;
                padding: 0.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            }
            
            .stRadio>div>label {
                margin-right: 15px;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                transition: all 0.2s ease;
            }
            
            .stRadio>div>label:hover {
                background-color: #f8f9fa;
            }
            
            /* Dataframes */
            .dataframe {
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }
            
            /* Navigation Buttons */
            .nav-button {
                display: flex;
                justify-content: space-between;
                margin-top: 2rem;
            }
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(#3498db, #2ecc71);
                border-radius: 10px;
            }
            
            /* Tooltips */
            .tooltip {
                position: relative;
                display: inline-block;
            }
            
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 200px;
                background-color: #2c3e50;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -100px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .main {
                    padding: 1rem;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .block-container {
                    padding: 1rem;
                }
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

            # === INTERPRETASI TAMBAHAN: Wilayah Miskin Tinggi & Rendah ===
            st.subheader("📌 Tabel Wilayah dengan Kemiskinan Tertinggi dan Terendah")

            # Pastikan nama kolom kemiskinan benar
            kemiskinan_col = "Persentase Penduduk Miskin (%)"

            if kemiskinan_col in df.columns:
                top3 = df.sort_values(by=kemiskinan_col, ascending=False)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(3)
                bottom3 = df.sort_values(by=kemiskinan_col, ascending=True)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(3)

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### 🚨 3 Wilayah dengan Tingkat Kemiskinan Tertinggi")
                    st.table(top3.reset_index(drop=True))

                with col2:
                    st.markdown("#### 🟢 3 Wilayah dengan Tingkat Kemiskinan Terendah")
                    st.table(bottom3.reset_index(drop=True))
            else:
                st.warning(f"Kolom '{kemiskinan_col}' tidak ditemukan dalam data.")

            st.markdown("""
            ### Interpretasi Awal:
            - Cluster dengan rata-rata **persentase penduduk miskin paling rendah** bisa dianggap sebagai kategori **kinerja baik**.
            - Cluster dengan nilai indikator pendidikan dan kesehatan yang rendah mungkin termasuk **kategori rentan/tinggi kemiskinan**.
            - Data ini bisa menjadi dasar rekomendasi kebijakan untuk setiap kelompok wilayah.
            """)
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")
