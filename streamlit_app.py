import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering

# Inisialisasi session_state
if 'step' not in st.session_state:
    st.session_state.step = 1

st.set_page_config(page_title="Analisis Clustering Kemiskinan", layout="wide")

st.title("Aplikasi Analisis Clustering Tingkat Kemiskinan Kabupaten/Kota di Jawa Timur")
st.sidebar.title("Navigasi")
step = st.sidebar.radio("Pilih Tahapan Analisis", ["1. Upload Data", "2. Preprocessing & Clustering", "3. Evaluasi", "4. Visualisasi & Kesimpulan"])

# Langkah 1: Upload Data
if step == "1. Upload Data":
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type="xlsx")

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.write("Preview Data:")
        st.dataframe(df)
        st.success("File berhasil diupload. Lanjut ke tahapan berikutnya.")

# Langkah 2: Preprocessing & Clustering
elif step == "2. Preprocessing & Clustering":
    st.header("2. Preprocessing dan Clustering")

    if 'df' in st.session_state:
        df = st.session_state.df.copy()
        kolom_kota = df.columns[0]
        X = df.drop(columns=[kolom_kota])

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Pilih jumlah cluster
        n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=3)

        # Clustering
        clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=42, affinity='nearest_neighbors')
        labels = clustering.fit_predict(X_scaled)

        df['Cluster'] = labels
        st.session_state.df_clustered = df
        st.session_state.labels = labels
        st.session_state.X_scaled = X_scaled
        st.success(f"Clustering selesai dengan {n_clusters} cluster.")
        st.dataframe(df)
    else:
        st.warning("Silakan upload data terlebih dahulu.")

# Langkah 3: Evaluasi
elif step == "3. Evaluasi":
    st.header("3. Evaluasi Clustering")

    if 'X_scaled' in st.session_state and 'labels' in st.session_state:
        X_scaled = st.session_state.X_scaled
        labels = st.session_state.labels

        silhouette = silhouette_score(X_scaled, labels)
        db_index = davies_bouldin_score(X_scaled, labels)

        st.metric("Silhouette Score", f"{silhouette:.3f}")
        st.metric("Davies-Bouldin Index", f"{db_index:.3f}")

        with st.expander("Interpretasi Skor"):
            st.markdown("""
            - **Silhouette Score**: Semakin mendekati 1 maka cluster semakin baik (terpisah dan kompak).
            - **Davies-Bouldin Index**: Semakin kecil nilainya maka cluster semakin baik.
            """)
    else:
        st.warning("Silakan lakukan clustering terlebih dahulu.")

# Langkah 4: Visualisasi & Kesimpulan
elif step == "4. Visualisasi & Kesimpulan":
    st.header("4. Visualisasi & Kesimpulan")

    if 'df_clustered' in st.session_state:
        df = st.session_state.df_clustered
        kolom_kota = df.columns[0]

        st.subheader("Tabel Hasil Clustering")
        st.dataframe(df[[kolom_kota, 'Cluster']])

        st.subheader("Distribusi Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        st.subheader("Boxplot per Cluster")
        variabel = st.selectbox("Pilih variabel untuk boxplot", df.columns[1:-1])
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Cluster', y=variabel, ax=ax)
        st.pyplot(fig)

        st.subheader("Kesimpulan per Cluster")
        means = df.groupby('Cluster').mean(numeric_only=True)
        st.dataframe(means)

        with st.expander("Interpretasi Sederhana Tiap Cluster"):
            for cluster in means.index:
                st.markdown(f"### Cluster {cluster}")
                insight = means.loc[cluster].sort_values(ascending=False)
                st.markdown(insight.to_frame("Rata-rata").to_markdown())
    else:
        st.warning("Silakan lakukan clustering terlebih dahulu.")
