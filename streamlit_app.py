import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Insight Predict",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk memuat CSS kustom
def local_css():
    st.markdown(
        """
        <style>
            /* Mengatur sidebar */
            [data-testid="stSidebar"] {
                background-color: #cdd4b1;
            }
            
            /* Mengatur background main content */
            [data-testid="stAppViewContainer"] {
                background-color: #feecd0;
            }

            /* Mengatur logo agar lebih kecil dan di tengah */
            .logo-container {
                text-align: center;
                margin-top: -20px;
            }

            .logo-container img {
                width: 100px;
            }

            /* Styling untuk teks agar lebih kontras */
            h1, h2, h3, h4, h5, h6, p, div, span {
                color: #4a4a4a !important;
            }

            /* Styling untuk teks sambutan */
            .welcome-text {
                font-size: 22px;
                font-weight: bold;
                color: #4a4a4a;
                text-align: center;
                background-color: #f5deb3;
                padding: 15px;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

# Panggil fungsi CSS
local_css()

# Menu Navigasi di Sidebar
menu = st.sidebar.radio(
    "Navigasi",
    ("Beranda", "Dashboard", "Visualisasi", "Tentang")
)

# Fungsi untuk memuat data
@st.cache_data
def load_data(file):
    # Membaca file yang diupload
    if file is not None:
        return pd.read_excel(file)
    else:
        return None

# Konten berdasarkan menu yang dipilih
if menu == "Beranda":
    st.markdown('<div class="welcome-text">Selamat datang di Insight Predict 📊</div>', unsafe_allow_html=True)

elif menu == "Dashboard":
    st.header("📈 Dashboard")
    st.write("Di sini kamu bisa menampilkan metrik, grafik, dan insight lainnya.")
    # Contoh metrik
    col1, col2, col3 = st.columns(3)
    col1.metric("Pengguna Aktif", "1,024", "+5%")
    col2.metric("Pendapatan", "$12.3K", "+2.3%")
    col3.metric("Retensi", "82%", "-1.1%")

elif menu == "Visualisasi":
    st.header("📊 Visualisasi Data")
    st.write("Contoh visualisasi:")

    # Upload file Excel untuk memuat data
    uploaded_file = st.file_uploader("Upload file Excel", type="xlsx")
    
    # Load data jika file diupload
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("Data berhasil dimuat!")
            st.dataframe(df.head())
            
            # Contoh visualisasi
            fig, ax = plt.subplots()
            data = df[['Kategori', 'Nilai']]  # Sesuaikan dengan kolom yang ada dalam dataset
            ax.bar(data['Kategori'], data['Nilai'], color='#6c8c4c')
            st.pyplot(fig)
        else:
            st.write("Gagal memuat data.")
    else:
        st.write("Silakan unggah file Excel terlebih dahulu.")

elif menu == "Tentang":
    st.header("ℹ️ Tentang Aplikasi")
    st.write("""
        Insight Predict adalah aplikasi visualisasi dan analisis data yang 
        membantu memahami pola dan tren penting dari dataset yang kamu miliki.
    """)
