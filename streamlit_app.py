import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === MENYIAPKAN STATE ===
if 'menu' not in st.session_state:
    st.session_state.menu = "Home"
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_scaled' not in st.session_state:
    st.session_state.X_scaled = None

# === MENU NAVIGASI ===
menu = st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"),
    horizontal=True
)

# === HALAMAN HOME ===
if menu == "Home":
    st.title("Aplikasi Analisis Clustering Kemiskinan di Jawa Timur")
    st.write("""
        Selamat datang di aplikasi analisis clustering kemiskinan kabupaten/kota di Jawa Timur.
        Aplikasi ini menggunakan metode **Spectral Clustering** untuk mengelompokkan kabupaten/kota berdasarkan variabel kemiskinan.
    """)

    # Menambahkan tombol Next untuk menuju Step 1
    if st.button("Next"):
        st.session_state.menu = "Step 1: Upload Data"

# === STEP 1: UPLOAD DATA ===
elif menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")
    st.markdown("""
    **Petunjuk:**
    1. File yang diunggah harus berupa file **Excel** dengan ekstensi `.xlsx`.
    2. Data yang diunggah harus memuat variabel berikut:
        - **Persentase Penduduk Miskin (%)**
        - **Jumlah Penduduk Miskin (ribu jiwa)**
        - **Harapan Lama Sekolah (Tahun)**
        - **Rata-Rata Lama Sekolah (Tahun)**
        - **Tingkat Pengangguran Terbuka (%)**
        - **Tingkat Partisipasi Angkatan Kerja (%)**
        - **Angka Harapan Hidup (Tahun)**
        - **Garis Kemiskinan (Rupiah/Bulan/Kapita)**
        - **Indeks Pembangunan Manusia**
        - **Rata-rata Upah/Gaji Bersih Pekerja Informal Berdasarkan Lapangan Pekerjaan Utama (Rp)**
        - **Rata-rata Pendapatan Bersih Sebulan Pekerja Informal berdasarkan Pendidikan Tertinggi - Jumlah (Rp)**
    """)

    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        st.write(df)

    # Menambahkan tombol Next untuk menuju Step 2
    if st.button("Next"):
        st.session_state.menu = "Step 2: Preprocessing Data"

# === STEP 2: PREPROCESSING DATA ===
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
    
    # Menambahkan tombol Next untuk menuju Step 3
    if st.button("Next"):
        st.session_state.menu = "Step 3: Visualisasi Data"

# === STEP 3: VISUALISASI DATA ===
elif menu == "Step 3: Visualisasi Data":
    st.header("📊 Visualisasi Data")
    if 'X_scaled' in st.session_state:
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_scaled = pd.DataFrame(st.session_state.X_scaled, columns=df.select_dtypes(include=['float64', 'int64']).columns)
        
        st.subheader("Pairplot")
        fig, ax = plt.subplots()
        sns.pairplot(df_scaled)
        st.pyplot(fig)

        st.subheader("Heatmap Korelasi")
        corr = df_scaled.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")

    # Menambahkan tombol Next untuk menuju Step 4
    if st.button("Next"):
        st.session_state.menu = "Step 4: Hasil Clustering"

# === STEP 4: HASIL CLUSTERING ===
elif menu == "Step 4: Hasil Clustering":
    st.header("🔍 Hasil Clustering")
    if 'X_scaled' in st.session_state:
        from sklearn.cluster import SpectralClustering
        import numpy as np

        # Menggunakan Spectral Clustering untuk mengelompokkan data
        clustering = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
        labels = clustering.fit_predict(st.session_state.X_scaled)

        df_cluster = pd.DataFrame(st.session_state.X_scaled, columns=df.select_dtypes(include=['float64', 'int64']).columns)
        df_cluster['Cluster'] = labels

        st.subheader("Hasil Pengelompokan")
        st.write(df_cluster)

        st.subheader("Visualisasi Hasil Clustering")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df_cluster.iloc[:, 0], y=df_cluster.iloc[:, 1], hue=df_cluster['Cluster'], palette='Set1')
        st.pyplot(fig)

    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")

# === MENAMBAHKAN NAVIGASI TOMBOL NEXT UNTUK MASING-MASING LANGKAH ===
