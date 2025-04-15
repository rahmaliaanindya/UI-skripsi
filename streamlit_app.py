import streamlit as st

# -------------------- KONFIGURASI HALAMAN --------------------
st.set_page_config(
    page_title="Dashboard Clustering Kemiskinan Jatim",
    page_icon=":bar_chart:",
    layout="wide"
)

# -------------------- HEADER --------------------
st.title(":bar_chart: Dashboard Clustering Kemiskinan - Jawa Timur")

st.markdown("""
Dashboard ini menampilkan hasil klasterisasi kabupaten/kota di **Provinsi Jawa Timur** berdasarkan indikator sosial-ekonomi yang relevan terhadap kemiskinan.  
Gunakan panel di sebelah kiri untuk memilih indikator dan menentukan jumlah klaster.
""")

st.markdown("---")

# -------------------- SIDEBAR (Filter Panel) --------------------
with st.sidebar:
    st.header("🎛️ Panel Pengaturan")

    indikator_list = [
        "Persentase Penduduk Miskin (%)",
        "Jumlah Penduduk Miskin (ribu jiwa)",
        "Harapan Lama Sekolah (Tahun)",
        "Rata-Rata Lama Sekolah (Tahun)",
        "Tingkat Pengangguran Terbuka (%)",
        "Tingkat Partisipasi Angkatan Kerja (%)",
        "Angka Harapan Hidup (Tahun)",
        "Garis Kemiskinan (Rupiah/Bulan/Kapita)",
        "Indeks Pembangunan Manusia",
        "Rata-rata Upah/Gaji Bersih Pekerja Informal Berdasarkan Lapangan Pekerjaan Utama (Rp)",
        "Rata-rata Pendapatan Bersih Sebulan Pekerja Informal berdasarkan Pendidikan Tertinggi - Jumlah (Rp)"
    ]

    indikator = st.multiselect(
        "Pilih Indikator Sosial Ekonomi",
        options=indikator_list,
        default=indikator_list[:5]  # Default: 5 indikator pertama
    )

    num_clusters = st.slider(
        "Jumlah Klaster",
        min_value=2,
        max_value=6,
        value=3
    )

    st.markdown("Klik tombol di bawah untuk memulai proses clustering.")
    start = st.button("🔍 Jalankan Clustering")

# -------------------- KONTEN UTAMA --------------------
if not start:
    st.info("Gunakan panel di sebelah kiri untuk memilih indikator dan jalankan clustering.")

else:
    # Placeholder konten visualisasi dan output
    st.subheader("📍 Visualisasi Hasil Clustering")
    st.info("Hasil scatter plot atau peta akan ditampilkan di sini.")

    st.subheader("📊 Rangkuman Statistik Tiap Klaster")
    st.info("Tabel statistik ringkasan per klaster akan muncul di sini.")

    st.subheader("📋 Tabel Data per Klaster")
    st.info("Tabel detail kabupaten/kota per klaster akan ditampilkan di sini.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("© 2025 – Dashboard oleh [Nama Kamu] | Data dari BPS dan instansi terkait.")
