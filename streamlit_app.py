import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import SpectralClustering
from io import BytesIO

# Konfigurasi halaman
st.set_page_config(page_title="Clustering Kemiskinan Jatim", layout="wide")

# Inisialisasi step navigasi
if 'step' not in st.session_state:
    st.session_state.step = "Home"

def go_to_step(step_name):
    st.session_state.step = step_name
    st.experimental_rerun()

# Optional: Navigasi dengan radio (shortcut)
st.radio(
    "Navigasi Aplikasi (opsional):",
    ("Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"),
    index=["Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"].index(st.session_state.step),
    key="menu_radio",
    on_change=lambda: go_to_step(st.session_state.menu_radio)
)

# ======================== HOME ========================
if st.session_state.step == "Home":
    st.title("📊 Aplikasi Clustering Kemiskinan Jawa Timur")
    st.markdown("""
    Selamat datang di aplikasi analisis clustering tingkat kemiskinan di Jawa Timur menggunakan Spectral Clustering.  
    Aplikasi ini akan membantu Anda memahami pola kemiskinan melalui beberapa tahapan analisis.
    """)
    st.button("➡️ Mulai", on_click=lambda: go_to_step("Step 1: Upload Data"))

# ======================== STEP 1: UPLOAD ========================
elif st.session_state.step == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("Data berhasil dimuat!")
        st.write(df)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("⬅️ Kembali", on_click=lambda: go_to_step("Home"))
        with col2:
            st.button("➡️ Lanjut", on_click=lambda: go_to_step("Step 2: Preprocessing Data"))

# ======================== STEP 2: PREPROCESSING ========================
elif st.session_state.step == "Step 2: Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    if 'df' in st.session_state:
        df = st.session_state.df.copy()
        df_cleaned = df.dropna()

        scaler = StandardScaler()
        numeric_df = df_cleaned.select_dtypes(include=[np.number])
        df_scaled = scaler.fit_transform(numeric_df)
        df_scaled_df = pd.DataFrame(df_scaled, columns=numeric_df.columns)
        df_scaled_df.index = df_cleaned.index

        st.session_state.processed_df = df_scaled_df
        st.session_state.original_df_cleaned = df_cleaned

        st.success("Preprocessing selesai. Data telah dinormalisasi.")
        st.write(df_scaled_df)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("⬅️ Kembali", on_click=lambda: go_to_step("Step 1: Upload Data"))
        with col2:
            st.button("➡️ Lanjut", on_click=lambda: go_to_step("Step 3: Visualisasi Data"))
    else:
        st.warning("Mohon upload data terlebih dahulu.")

# ======================== STEP 3: VISUALISASI ========================
elif st.session_state.step == "Step 3: Visualisasi Data":
    st.header("📈 Visualisasi Data")
    if 'processed_df' in st.session_state:
        df_scaled_df = st.session_state.processed_df

        # Korelasi
        st.subheader("📊 Korelasi Antarfaktor")
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_scaled_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt.gcf())
        plt.clf()

        # Boxplot
        st.subheader("📦 Boxplot Setiap Fitur")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df_scaled_df, orient="h", palette="Set2")
        st.pyplot(fig)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("⬅️ Kembali", on_click=lambda: go_to_step("Step 2: Preprocessing Data"))
        with col2:
            st.button("➡️ Lanjut", on_click=lambda: go_to_step("Step 4: Hasil Clustering"))
    else:
        st.warning("Mohon lakukan preprocessing terlebih dahulu.")

# ======================== STEP 4: HASIL CLUSTERING ========================
elif st.session_state.step == "Step 4: Hasil Clustering":
    st.header("📌 Hasil Clustering")
    if 'processed_df' in st.session_state:
        df_scaled_df = st.session_state.processed_df
        df_cleaned = st.session_state.original_df_cleaned

        st.subheader("📍 Tentukan Jumlah Cluster")
        num_clusters = st.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)

        clustering = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
        cluster_labels = clustering.fit_predict(df_scaled_df)

        df_cleaned['Cluster'] = cluster_labels
        clustered_df = df_cleaned.copy()
        st.session_state.clustered_df = clustered_df

        st.success("Clustering selesai!")
        st.write(clustered_df)

        # Evaluasi
        silhouette_avg = silhouette_score(df_scaled_df, cluster_labels)
        db_index = davies_bouldin_score(df_scaled_df, cluster_labels)

        st.markdown(f"📈 **Silhouette Score:** {silhouette_avg:.3f}")
        st.markdown(f"📉 **Davies-Bouldin Index:** {db_index:.3f}")

        # Visualisasi bar plot jumlah anggota per cluster
        cluster_counts = clustered_df['Cluster'].value_counts().sort_index()
        st.subheader("📊 Jumlah Anggota per Cluster")
        fig2, ax2 = plt.subplots()
        ax2.bar(cluster_counts.index.astype(str), cluster_counts.values, color='skyblue')
        ax2.set_xlabel("Cluster")
        ax2.set_ylabel("Jumlah Anggota")
        st.pyplot(fig2)

        # Download hasil clustering
        st.markdown("📥 **Download Data dengan Cluster**")

        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Clustered Data')
            return output.getvalue()

        st.download_button(
            label="Download Excel",
            data=to_excel(clustered_df),
            file_name="hasil_clustering.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.button("⬅️ Kembali", on_click=lambda: go_to_step("Step 3: Visualisasi Data"))
    else:
        st.warning("Mohon lakukan preprocessing dan visualisasi terlebih dahulu.")
