import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Menu pilihan
menu_options = [
    "Home", 
    "Step 1: Upload Data", 
    "Step 2: Preprocessing Data", 
    "Step 3: Visualisasi Data", 
    "Step 4: Hasil Clustering", 
    "Step 5: Analisis Hasil"
]

# Fungsi navigasi untuk setiap langkah
def create_nav_buttons(current_step):
    if current_step == "Home":
        if st.button("Mulai Analisis →", key="home_next"):
            st.session_state.menu = "Step 1: Upload Data"
    
    elif current_step in menu_options[1:-1]:
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("← Kembali", key=f"{current_step}_back"):
                prev_index = menu_options.index(current_step) - 1
                st.session_state.menu = menu_options[prev_index]
        with col2:
            if st.button("Lanjut →", key=f"{current_step}_next"):
                next_index = menu_options.index(current_step) + 1
                st.session_state.menu = menu_options[next_index]
    
    elif current_step == "Step 5: Analisis Hasil":
        if st.button("← Kembali ke Clustering", key="analysis_back"):
            st.session_state.menu = "Step 4: Hasil Clustering"

# Menentukan menu awal di session_state jika tidak ada
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"

# Logika Menu Berdasarkan Menu yang Dipilih
if st.session_state.menu == "Home":
    st.markdown("""
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊
    """)
    create_nav_buttons("Home")

elif st.session_state.menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil dimuat!")
        st.write(df)
    create_nav_buttons("Step 1: Upload Data")

elif st.session_state.menu == "Step 2: Preprocessing Data":
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
    create_nav_buttons("Step 2: Preprocessing Data")

elif st.session_state.menu == "Step 3: Visualisasi Data":
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
    create_nav_buttons("Step 3: Visualisasi Data")

elif st.session_state.menu == "Step 4: Hasil Clustering":
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
        st.success(f"🔹 Jumlah cluster optimal berdasarkan **Silhouette Score**: {best_k_silhouette}")
        st.success(f"🔸 Jumlah cluster optimal berdasarkan **Davies-Bouldin Index**: {best_k_dbi}")
        optimal_k = best_k_silhouette
        clustering = SpectralClustering(n_clusters=optimal_k, affinity='nearest_neighbors', random_state=42)
        labels = clustering.fit_predict(X_scaled)

        # Memastikan df telah terdefinisi sebelum menambahkan kolom 'Cluster'
        if 'df' not in st.session_state:
            st.session_state.df = pd.DataFrame(X_scaled, columns=st.session_state.features)
        
        df = st.session_state.df
        df['Cluster'] = labels
        st.write(df)
        st.session_state.df_clustered = df
    else:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu.")
    create_nav_buttons("Step 4: Hasil Clustering")

elif st.session_state.menu == "Step 5: Analisis Hasil":
    st.header("🔍 Analisis Hasil Clustering")
    if 'df_clustered' in st.session_state:
        df = st.session_state.df_clustered
        st.subheader("Visualisasi Hasil Clustering")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="Harapan Lama Sekolah (Tahun)", y="Tingkat Pengangguran Terbuka (%)", hue="Cluster", palette="viridis", ax=ax)
        st.pyplot(fig)
        st.subheader("Karakteristik Cluster")
        for cluster_num in df['Cluster'].unique():
            st.markdown(f"### Cluster {cluster_num}")
            cluster_data = df[df['Cluster'] == cluster_num]
            mean_values = cluster_data.mean()
            st.write(f"Rata-rata nilai fitur pada Cluster {cluster_num}:")
            st.write(mean_values)
            st.markdown("**Fitur tertinggi:**")
            for i, (ind, val) in enumerate(mean_values.head(3).items()):
                st.write(f"{i+1}. {ind}: {val:.2f}")
            st.markdown("**Fitur terendah:**")
            for i, (ind, val) in enumerate(mean_values.tail(3).items()):
                st.write(f"{i+1}. {ind}: {val:.2f}")
    else:
        st.warning("⚠️ Hasil clustering belum tersedia. Silakan lakukan clustering terlebih dahulu.")
    create_nav_buttons("Step 5: Analisis Hasil")
