import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.stats import zscore
import io

# ========== PAGE CONFIG & STYLE ==========
st.set_page_config(page_title="Analisis Kemiskinan", layout="wide")

def local_css():
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                background-color: #cdd4b1;
            }
            [data-testid="stAppViewContainer"] {
                background-color: #feecd0;
            }
            h1, h2, h3, h4, h5, h6, p, div, span {
                color: #4a4a4a !important;
            }
        </style>
        """, unsafe_allow_html=True
    )
local_css()

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    return pd.read_excel('/content/dataset indikator kemiskinan.xlsx')

df = load_data()

# ========== SIDEBAR MENU ==========
menu = st.sidebar.radio("📌 Navigasi", (
    "Data Awal",
    "Preprocessing",
    "Clustering",
    "Evaluasi & Optimasi",
    "Hasil Akhir",
    "Visualisasi Khusus"
))

# ========== MENU: DATA AWAL ==========
if menu == "Data Awal":
    st.header("📂 Data Awal dan Eksplorasi")
    st.dataframe(df.head())
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("📌 Missing Values:")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={'index': 'Kolom', 0: 'Jumlah'}))

    st.write(f"📌 Jumlah Data Duplikat: {df.duplicated().sum()}")
    st.write("📊 Statistik Deskriptif")
    st.dataframe(df.describe())

# ========== MENU: PREPROCESSING ==========
elif menu == "Preprocessing":
    st.header("🔧 Preprocessing dan Korelasi Fitur")
    numerical_df = df.select_dtypes(include=['number'])

    st.subheader("📌 Korelasi Fitur")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Standarisasi")
    features = numerical_df.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numerical_df)
    st.write("Data berhasil dinormalisasi.")

# ========== MENU: CLUSTERING ==========
elif menu == "Clustering":
    st.header("🔗 Clustering: Menentukan Cluster Optimal")

    X = df.select_dtypes(include=['float64', 'int64']).values
    clusters_range = range(2, 11)
    silhouette_scores = {}
    dbi_scores = {}

    for k in clusters_range:
        spectral = SpectralClustering(n_clusters=k, affinity='rbf', random_state=42)
        labels = spectral.fit_predict(X)
        silhouette_scores[k] = silhouette_score(X, labels)
        dbi_scores[k] = davies_bouldin_score(X, labels)

    st.subheader("📈 Silhouette Score dan DBI")
    fig, ax = plt.subplots()
    ax.plot(clusters_range, list(silhouette_scores.values()), marker='o', label='Silhouette Score')
    ax.plot(clusters_range, list(dbi_scores.values()), marker='s', label='DB Index')
    ax.set_xlabel("Jumlah Cluster")
    ax.legend()
    st.pyplot(fig)

# ========== MENU: EVALUASI & OPTIMASI ==========
elif menu == "Evaluasi & Optimasi":
    st.header("⚙️ Evaluasi & Optimasi Clustering (PSO)")
    data = StandardScaler().fit_transform(df.select_dtypes(include=['float64', 'int64']))
    
    # Clustering awal
    clustering = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42)
    labels = clustering.fit_predict(data)
    dbi_awal = davies_bouldin_score(data, labels)
    silhouette_awal = silhouette_score(data, labels)

    st.write(f"DBI Awal: {dbi_awal:.4f}")
    st.write(f"Silhouette Score Awal: {silhouette_awal:.4f}")

    # Optimasi PSO
    try:
        from pyswarm import pso
    except ImportError:
        st.warning("🔧 Modul `pyswarm` belum terinstal. Silakan install dengan `pip install pyswarm` di terminal.")
        st.stop()

    def evaluate_fitness(weights):
        weighted_data = data * weights
        labels = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42).fit_predict(weighted_data)
        return -silhouette_score(weighted_data, labels) + davies_bouldin_score(weighted_data, labels)

    lb, ub = [0.1]*data.shape[1], [2.0]*data.shape[1]
    optimal_weights, _ = pso(evaluate_fitness, lb, ub, swarmsize=20, maxiter=50)
    st.success(f"Bobot Fitur Optimal: {optimal_weights}")

    # Clustering ulang
    data_opt = data * optimal_weights
    labels_opt = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42).fit_predict(data_opt)
    dbi_opt = davies_bouldin_score(data_opt, labels_opt)
    silhouette_opt = silhouette_score(data_opt, labels_opt)

    st.write(f"DBI Setelah Optimasi: {dbi_opt:.4f}")
    st.write(f"Silhouette Setelah Optimasi: {silhouette_opt:.4f}")

# ========== MENU: HASIL AKHIR ==========
elif menu == "Hasil Akhir":
    st.header("📊 Hasil Akhir Clustering dan Ranking")
    X = df.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    labels = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=42).fit_predict(X_scaled)
    df['Cluster'] = labels

    centroid = df.groupby('Cluster').mean(numeric_only=True)
    st.subheader("📌 Centroid Tiap Cluster")
    st.dataframe(centroid)

    # Skor Kemiskinan
    scaler = MinMaxScaler()
    scaled_numeric = scaler.fit_transform(X)
    df['Skor Kemiskinan'] = scaled_numeric.mean(axis=1)
    df_ranked = df[['Kabupaten/Kota', 'Skor Kemiskinan']].sort_values(by='Skor Kemiskinan', ascending=False)

    st.subheader("📌 Ranking Skor Kemiskinan")
    st.dataframe(df_ranked)

# ========== MENU: VISUALISASI KHUSUS ==========
elif menu == "Visualisasi Khusus":
    st.header("📍 Visualisasi Cluster dan Kota Surabaya")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    z_scores = np.abs(zscore(df[numerical_columns]))
    outlier_mask = (z_scores > 3).any(axis=1)
    df['Outlier'] = np.where(outlier_mask, 'Yes', 'No')

    surabaya_data = df[df['Kabupaten/Kota'] == 'Kota Surabaya']
    if not surabaya_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df, x='Persentase Penduduk Miskin (%)',
            y='Garis Kemiskinan (Rupiah/Bulan/Kapita)',
            hue='Cluster', palette='viridis', ax=ax, alpha=0.6
        )
        plt.scatter(
            surabaya_data['Persentase Penduduk Miskin (%)'],
            surabaya_data['Garis Kemiskinan (Rupiah/Bulan/Kapita)'],
            s=300, color='red', label='Kota Surabaya', edgecolor='black'
        )
        plt.legend()
        plt.title("Cluster dan Posisi Kota Surabaya")
        st.pyplot(fig)
    else:
        st.warning("Kota Surabaya tidak ditemukan.")
