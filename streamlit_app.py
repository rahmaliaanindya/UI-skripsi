import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from streamlit_option_menu import option_menu

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

# Fungsi utama
def main():
    local_css()  # Memanggil CSS

    # Sidebar dengan menu navigasi
    with st.sidebar:
        st.markdown(
            """
            <div class='logo-container'>
                <img src="https://raw.githubusercontent.com/amqhis/skripsi_balqhis/main/Logo%20Insight%20Predict.png" 
                     alt="Logo">
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Menu navigasi
        selected = option_menu(
            None,
            ['🏠 Home', '📖 Tentang Aplikasi', '📂 Upload Data', 
             '⚙️ Preprocessing Data', '📊 Visualisasi Data Historis', '🔮 Prediksi Masa Depan'],
            menu_icon='cast',
            icons=['house', 'info-circle', 'cloud-upload', 'filter', 'bar-chart', 'line-chart'],
            default_index=0,
            styles={
                "container": {
                    "padding": "0px",
                    "background-color": "#cdd4b1"
                },
                "icon": {
                    "color": "#4a4a4a",
                    "font-size": "17px"
                },
                "nav-link": {
                    "font-size": "15px",
                    "text-align": "left",
                    "margin": "5px",
                    "color": "#4a4a4a",
                    "--hover-color": "#b5c19a"
                },
                "nav-link-selected": {
                    "background-color": "#6b8e23",
                    "color": "white"
                },
            }
        )

    # **📌 Kata Sambutan Muncul di Homepage**
    if selected == '🏠 Home':
        st.markdown("<div class='welcome-text'>🎉 Hai, Selamat Datang di Insight Predict! 🎉</div>", unsafe_allow_html=True)
        st.write("""
        Insight Predict adalah aplikasi yang dirancang untuk membantu Anda menganalisis dan memprediksi tren data menggunakan metode berbasis **Machine Learning**.  
        Dengan fitur interaktif yang mudah digunakan, aplikasi ini memungkinkan Anda memahami pola data historis dan melakukan prediksi masa depan dengan lebih akurat.  
        📊 **Ayo mulai jelajahi fitur yang tersedia!** 🚀
        """)

    # **📌 Tentang Aplikasi**
    elif selected == '📖 Tentang Aplikasi':
        st.title('📊 Insight Predict')
        st.write("""
        Insight Predict adalah platform analisis berbasis data yang dirancang untuk membantu pengguna dalam memahami tren data dan membuat prediksi berdasarkan data historis.  
        Aplikasi ini menggunakan **model Machine Learning canggih** untuk memberikan hasil prediksi yang lebih akurat dan dapat diandalkan. Dengan visualisasi interaktif, pengguna dapat dengan mudah menginterpretasikan data, mengevaluasi hasil analisis, dan mengambil keputusan berbasis data dengan lebih baik.
        """)

        # **📌 Terms & Conditions**
        with st.expander("📜 Syarat & Ketentuan Penggunaan"):
            st.markdown("""
            **Jenis Data yang Dapat Digunakan:**  
            - Format **Excel (.xlsx)**
            - Harus memiliki kolom berikut:  
                - **Tanggal**  
                - **Jenis Produk**  
                - **Quantity**   
            """)

    # **📌 Fitur Upload Data**
    elif selected == '📂 Upload Data':
        st.title('📂 Upload Data Anda')
        uploaded_file = st.file_uploader("Pilih file Excel (.xlsx) untuk dianalisis", type=['xlsx'])

        if uploaded_file is not None:
            # Membaca file Excel
            df = pd.read_excel(uploaded_file)
            st.write("### 📊 Data yang Diupload")
            st.dataframe(df)
            # Menyimpan ke session state
            st.session_state['original_data'] = df
            st.success('✅ Data berhasil diunggah!')


    elif selected == '⚙️ Preprocessing Data':
        st.title("⚙️ Preprocessing Data")
        
        if 'original_data' in st.session_state:
            df = st.session_state['original_data'].copy()
            
            st.write("### 📌 Data Sebelum Preprocessing")
            st.dataframe(df)
    
            # **1️⃣ Validasi Kolom yang Diperlukan**
            required_columns = ['Tanggal Pembelian', 'Quantity']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"⚠️ Kolom berikut tidak ditemukan dalam data: {', '.join(missing_cols)}")
            else:
                # **2️⃣ Konversi 'Tanggal' ke Datetime**
                df['Tanggal Pembelian'] = pd.to_datetime(df['Tanggal Pembelian'], errors='coerce')
                df.dropna(subset=['Tanggal Pembelian'], inplace=True)
    
                # **3️⃣ Ekstrak Tahun & Bulan**
                df['Year'] = df['Tanggal Pembelian'].dt.year
                df['Month'] = df['Tanggal Pembelian'].dt.month
    
                # **4️⃣ Menghapus Nilai Kosong**
                df.dropna(inplace=True)
    
                # **5️⃣ Normalisasi dengan MinMaxScaler**
                scaler = MinMaxScaler()
                numeric_cols = ['Quantity']  # Pastikan hanya 'Quantity' yang dinormalisasi
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
                st.write("### ✅ Data Setelah Preprocessing")
                st.dataframe(df)
    
                # **6️⃣ Simpan Hasil Preprocessing ke Session State**
                st.session_state['processed_data'] = df
    
                # **7️⃣ Visualisasi Data Setelah Normalisasi**
                st.write("### 📊 Visualisasi Distribusi Data Setelah Normalisasi")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['Tanggal Pembelian'], df['Quantity'], label="Quantity Normalized", color="blue")
                ax.set_title("Distribusi Data Setelah Normalisasi")
                ax.set_xlabel("Tanggal Pembelian")
                ax.set_ylabel("Quantity (Scaled)")
                ax.legend()
                st.pyplot(fig)
    
        else:
            st.warning("⚠️ Harap unggah data terlebih dahulu di bagian '📂 Upload Data'.")

    
    elif selected == '📊 Visualisasi Data Historis':
        st.title("Visualisasi Data Historis")
        if 'original_data' in st.session_state:
            df = st.session_state['original_data']
            df['Month'] = df['Tanggal Pembelian'].dt.to_period('M').astype(str)
            df_monthly = df.groupby('Month')['Quantity'].sum().reset_index()
            max_month = df_monthly.loc[df_monthly['Quantity'].idxmax()]
            min_month = df_monthly.loc[df_monthly['Quantity'].idxmin()]
            st.write(f"Penjualan tertinggi terjadi pada {max_month['Month']} sebanyak {max_month['Quantity']} unit")
            st.write(f"Penjualan terendah terjadi pada {min_month['Month']} sebanyak {min_month['Quantity']} unit")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(df_monthly['Month'], df_monthly['Quantity'], marker='o', linestyle='-', color='b')
            plt.xticks(rotation=45)
            plt.title('Tren Penjualan Per Bulan')
            st.pyplot(fig)
            
            df_yearly = df.groupby(df['Tanggal Pembelian'].dt.year)['Quantity'].sum()
            fig, ax = plt.subplots()
            df_yearly.plot(kind='bar', color='skyblue', ax=ax)
            plt.title('Total Penjualan per Tahun')
            st.pyplot(fig)
            
            sales_by_type = df.groupby('Jenis Strapping Band')['Quantity'].sum()
            fig, ax = plt.subplots()
            sales_by_type.plot(kind='bar', color='green', ax=ax)
            plt.title('Penjualan Berdasarkan Jenis Strapping Band')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.warning("Upload data terlebih dahulu!")

    
    elif selected == '🔮 Prediksi Masa Depan':
        st.title("🔮 Prediksi Masa Depan")
    
        if 'processed_data' in st.session_state and not st.session_state['processed_data'].empty:
            df = st.session_state['processed_data']
    
            # Pastikan ada data historis yang benar
            if df.empty:
                st.warning("⚠️ Data historis kosong. Pastikan Anda telah melakukan preprocessing.")
            else:
                # Agregasi Data Bulanan
                df_monthly = df.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
    
                # Pastikan ada data setelah preprocessing
                if df_monthly.empty:
                    st.warning("⚠️ Data setelah preprocessing kosong. Coba ulangi preprocessing.")
                else:
                    # Buat kolom 'Date' agar bisa dipakai di visualisasi
                    df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(day=1))
    
                    # Normalisasi Data
                    scaler = MinMaxScaler()
                    df_monthly['Quantity_Scaled'] = scaler.fit_transform(df_monthly[['Quantity']])
    
                    # Persiapan Data untuk Model
                    X = df_monthly[['Year', 'Month']]
                    y = df_monthly['Quantity_Scaled']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
                    # Training Model XGBoost
                    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
                    model.fit(X_train, y_train)
    
                    # Prediksi Masa Depan
                    future_dates = pd.DataFrame({
                        'Year': np.repeat(range(2024, 2026), 12),
                        'Month': list(range(1, 13)) * 2
                    })
    
                    future_dates['Date'] = pd.to_datetime(future_dates[['Year', 'Month']].assign(day=1))
    
                    # Prediksi
                    future_pred_scaled = model.predict(future_dates[['Year', 'Month']])
                    future_pred_actual = scaler.inverse_transform(future_pred_scaled.reshape(-1, 1)).flatten()
    
                    future_results = pd.DataFrame({
                        'Year': future_dates['Year'].values,
                        'Month': future_dates['Month'].values,
                        'Predicted_Quantity': future_pred_actual,
                        'Date': future_dates['Date']
                    })
    
                    # **Visualisasi Prediksi**
                    st.subheader("📈 Prediksi Penjualan 2024-2025")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(df_monthly['Date'], df_monthly['Quantity'], label="Data Historis", marker='o', color='blue')
                    ax.plot(future_results['Date'], future_results['Predicted_Quantity'], label="Prediksi 2024-2025", marker='s', color='red')
                    ax.set_xlabel("Bulan")
                    ax.set_ylabel("Total Quantity")
                    ax.set_title("Prediksi Kuantitas Januari 2024 - Desember 2025")
                    ax.legend()
                    ax.grid()
                    st.pyplot(fig)
    
                    # **Tampilkan Tabel Hasil Prediksi**
                    st.write("### 📋 Tabel Hasil Prediksi")
                    st.dataframe(future_results)
    
        else:
            st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu!")

  

# Menjalankan aplikasi
if __name__ == "__main__":
    main()
