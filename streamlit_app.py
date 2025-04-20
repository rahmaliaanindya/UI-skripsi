import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Inisialisasi session_state
if 'step' not in st.session_state:
    st.session_state.step = 0

steps = [
    "Home",
    "Upload Data",
    "Preprocessing Data",
    "Visualisasi Data",
    "Hasil Clustering"
]

current_step = st.session_state.step
menu = steps[current_step]

# Variabel Global untuk Simpan Data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

st.title("📊 Aplikasi Analisis Clustering Kemiskinan di Jawa Timur")
st.markdown(f"**Tahap Saat Ini:** `{menu}`")

# STEP 1: Home
if menu == "Home":
    st.markdown("""
    Selamat datang di aplikasi analisis clustering kemiskinan berbasis **Spectral Clustering**.  
    Aplikasi ini akan membantumu menganalisis dan memetakan tingkat kemiskinan berdasarkan indikator sosial ekonomi.
    """)
    
# STEP 2: Upload Data
elif menu == "Upload Data":
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.session_state.uploaded_data = df
        st.success("✅ Data berhasil diunggah!")
        st.dataframe(df)

# STEP 3: Preprocessing Data
elif menu == "Preprocessing Data":
    st.header("🔍 Preprocessing Data")
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data.copy()
        if 'Kabupaten/Kota' in df.columns:
            data_numeric = df.drop(columns=['Kabupaten/Kota'])
        else:
            data_numeric = df

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data_numeric)

        st.session_state.scaled_data = data_scaled
        st.session_state.labels = df['Kabupaten/Kota'] if 'Kabupaten/Kota' in df.columns else None

        st.success("✅ Data berhasil dinormalisasi!")
        st.dataframe(pd.DataFrame(data_scaled, columns=data_numeric.columns))
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# STEP 4: Visualisasi Data
elif menu == "Visualisasi Data":
    st.header("📈 Visualisasi Data")
    if st.session_state.uploaded_data is not None:
        df = st.session_state.uploaded_data
        selected_columns = st.multiselect("Pilih Kolom untuk Visualisasi", df.columns, default=df.columns[:2])
        if len(selected_columns) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=selected_columns[0], y=selected_columns[1], ax=ax)
            st.pyplot(fig)
        else:
            st.info("Pilih minimal 2 kolom untuk visualisasi scatterplot.")
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")

# STEP 5: Hasil Clustering
elif menu == "Hasil Clustering":
    st.header("🔮 Hasil Clustering")
    if 'scaled_data' in st.session_state:
        X = st.session_state.scaled_data
        k = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=10, value=3)
        clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
        cluster_labels = clustering.fit_predict(X)

        st.session_state.cluster_labels = cluster_labels

        # Evaluasi
        silhouette = silhouette_score(X, cluster_labels)
        davies = davies_bouldin_score(X, cluster_labels)

        st.metric("Silhouette Score", f"{silhouette:.3f}")
        st.metric("Davies-Bouldin Index", f"{davies:.3f}")

        # Tampilkan hasil cluster
        if st.session_state.labels is not None:
            result_df = pd.DataFrame({
                "Kabupaten/Kota": st.session_state.labels,
                "Cluster": cluster_labels
            })
        else:
            result_df = pd.DataFrame({
                "Index": list(range(len(cluster_labels))),
                "Cluster": cluster_labels
            })

        st.dataframe(result_df)
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")

# TOMBOL NAVIGASI BAWAH
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    if current_step > 0:
        if st.button("⬅️ Kembali"):
            st.session_state.step -= 1
with col2:
    if current_step < len(steps) - 1:
        if st.button("➡️ Selanjutnya"):
            st.session_state.step += 1
