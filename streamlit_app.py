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
    data = pd.DataFrame({
        'Kategori': ['A', 'B', 'C', 'D'],
        'Nilai': [23, 45, 56, 78]
    })
    fig, ax = plt.subplots()
    ax.bar(data['Kategori'], data['Nilai'], color='#6c8c4c')
    st.pyplot(fig)

elif menu == "Tentang":
    st.header("ℹ️ Tentang Aplikasi")
    st.write("""
        Insight Predict adalah aplikasi visualisasi dan analisis data yang 
        membantu memahami pola dan tren penting dari dataset yang kamu miliki.
    """)
