# === IMPORT LIBRARY ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# === KONFIGURASI HALAMAN ===
st.set_page_config(
    page_title="Analisis Kemiskinan Jatim",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === CSS Styling ===
def local_css():
    st.markdown(
        """
        <style>
            /* Main Background Gradient */
            .main {
                background: linear-gradient(135deg, #fdf0ed 0%, #f8f9fa 100%);
            }
            
            /* Title Styling */
            .title {
                font-family: 'Helvetica', sans-serif;
                color: #1f3a93;
                font-size: 2.8rem;
                font-weight: 800;
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
            }
            
            /* Header Styling */
            h1, h2, h3, h4, h5, h6 {
                color: #2c3e50;
                font-family: 'Helvetica', sans-serif;
                margin-top: 1.5rem;
            }
            
            h1 {
                border-bottom: 3px solid #e74c3c;
                padding-bottom: 10px;
            }
            
            h2 {
                background-color: #fef5e7;
                padding: 12px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            
            /* Card Styling */
            .block-container {
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                padding: 25px;
                margin-bottom: 25px;
                border-left: 4px solid #e74c3c;
            }
            
            /* Button Styling */
            .stButton>button {
                border-radius: 8px;
                border: none;
                background: linear-gradient(135deg, #e74c3c 0%, #f39c12 100%);
                color: white;
                font-weight: 600;
                padding: 0.7rem 1.8rem;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 10px rgba(0,0,0,0.2);
            }
            
            /* Navigation Radio Buttons */
            .stRadio > div {
                flex-direction: row;
                gap: 10px;
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 30px;
            }
            
            .stRadio > div > label {
                margin-right: 15px;
                font-weight: 600;
                color: #2c3e50;
            }
            
            /* Dataframe Styling */
            .dataframe {
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }
            
            /* Success/Warning Boxes */
            .stSuccess {
                background-color: #d5f5e3;
                border-left: 4px solid #2ecc71;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
            
            .stWarning {
                background-color: #fadbd8;
                border-left: 4px solid #e74c3c;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
            
            /* Tab Styling */
            .stTabs > div > div > div > button {
                border-radius: 8px 8px 0 0 !important;
                padding: 10px 20px !important;
                font-weight: 600 !important;
                background-color: #f8f9fa !important;
            }
            
            .stTabs > div > div > div > button[aria-selected="true"] {
                background-color: #e74c3c !important;
                color: white !important;
            }
            
            /* Progress Bar */
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #e74c3c 0%, #f39c12 100%);
            }
            
            /* File Uploader */
            .stFileUploader > div > div {
                border: 2px dashed #e74c3c;
                border-radius: 8px;
                padding: 30px;
                background-color: #fef5e7;
            }
            
            /* Sidebar */
            .sidebar .sidebar-content {
                background-color: #fef5e7;
                border-right: 1px solid #e74c3c;
            }
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 10px;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #e74c3c;
                border-radius: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
st.markdown('<div class="title">📊 Analisis Kemiskinan Jawa Timur</div>', unsafe_allow_html=True)
menu = st.radio(
    "Navigasi Aplikasi:",
    ("Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"),
    horizontal=True,
    label_visibility="visible"
)

# === NAVIGATION BUTTONS ===
def navigation_buttons(current_step):
    cols = st.columns(5)
    steps = ["Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"]
    current_index = steps.index(current_step)
    
    if current_index > 0:
        with cols[0]:
            if st.button("← Kembali", key=f"{current_step}_back"):
                st.session_state.menu = steps[current_index - 1]
                st.experimental_rerun()
    
    if current_index < len(steps) - 1:
        with cols[4]:
            if st.button("Lanjut →", key=f"{current_step}_next"):
                st.session_state.menu = steps[current_index + 1]
                st.experimental_rerun()

# === HOME ===
if menu == "Home":
    st.markdown("""
    <div style='background-color: white; border-radius: 12px; padding: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 30px;'>
        <h2 style='color: #e74c3c; text-align: center;'>👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur</h2>
        <p style='font-size: 1.1rem; text-align: center; margin-bottom: 30px;'>
            Aplikasi ini membantu Anda menganalisis data kemiskinan di Jawa Timur menggunakan metode Spectral Clustering.
        </p>
        
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 30px;'>
            <div style='background-color: #fef5e7; border-radius: 10px; padding: 20px; border-left: 4px solid #f39c12;'>
                <h4 style='color: #e74c3c;'>📁 Upload Data</h4>
                <p>Unggah data Excel berisi indikator kemiskinan untuk dianalisis</p>
            </div>
            
            <div style='background-color: #fef5e7; border-radius: 10px; padding: 20px; border-left: 4px solid #f39c12;'>
                <h4 style='color: #e74c3c;'>🧹 Preprocessing</h4>
                <p>Lakukan pembersihan dan normalisasi data sebelum analisis</p>
            </div>
            
            <div style='background-color: #fef5e7; border-radius: 10px; padding: 20px; border-left: 4px solid #f39c12;'>
                <h4 style='color: #e74c3c;'>📊 Visualisasi</h4>
                <p>Eksplorasi data melalui berbagai visualisasi interaktif</p>
            </div>
            
            <div style='background-color: #fef5e7; border-radius: 10px; padding: 20px; border-left: 4px solid #f39c12;'>
                <h4 style='color: #e74c3c;'>🤖 Clustering</h4>
                <p>Gunakan Spectral Clustering untuk pengelompokan wilayah</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #fef5e7; border-radius: 12px; padding: 20px; margin-top: 20px;'>
        <h4 style='color: #e74c3c;'>📌 Petunjuk Penggunaan:</h4>
        <ol>
            <li>Mulai dengan mengunggah data di menu <b>Step 1: Upload Data</b></li>
            <li>Lakukan preprocessing data di menu <b>Step 2: Preprocessing Data</b></li>
            <li>Eksplorasi data melalui visualisasi di menu <b>Step 3: Visualisasi Data</b></li>
            <li>Lihat hasil clustering di menu <b>Step 4: Hasil Clustering</b></li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# === UPLOAD DATA ===
elif menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")
    
    with st.container():
        st.markdown("""
        <div style='background-color: #fef5e7; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
            <h4 style='color: #e74c3c;'>📋 Format Data yang Didukung:</h4>
            <ul>
                <li>File Excel (.xlsx) dengan kolom-kolom indikator kemiskinan</li>
                <li>Minimal mengandung kolom:
                    <ul>
                        <li>Nama Kabupaten/Kota</li>
                        <li>Persentase Penduduk Miskin (%)</li>
                        <li>Indikator pendidikan dan kesehatan</li>
                    </ul>
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file Excel untuk diunggah", type="xlsx", 
                                   help="Unggah file data dalam format Excel (.xlsx)")
    
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("✅ Data berhasil dimuat!")
            
            with st.expander("🔍 Lihat Data", expanded=True):
                st.dataframe(df.style.background_gradient(cmap='Oranges'))
                
            # Show basic info
            cols = st.columns(3)
            with cols[0]:
                st.metric("Jumlah Kabupaten/Kota", len(df))
            with cols[1]:
                st.metric("Jumlah Kolom", len(df.columns))
            with cols[2]:
                st.metric("Jumlah Nilai Kosong", df.isnull().sum().sum())
                
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")

# === PREPROCESSING DATA ===
elif menu == "Step 2: Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        
        with st.container():
            st.subheader("1. Pengecekan Data Awal")
            tab1, tab2, tab3 = st.tabs(["Missing Values", "Duplikat", "Statistik Deskriptif"])
            
            with tab1:
                st.write("**Jumlah Missing Values per Kolom:**")
                missing_df = df.isnull().sum().to_frame("Jumlah Missing")
                st.dataframe(missing_df.style.background_gradient(cmap='Reds'))
                
            with tab2:
                dup_count = df.duplicated().sum()
                if dup_count > 0:
                    st.warning(f"⚠️ Ditemukan {dup_count} data duplikat!")
                    st.write("Data duplikat:")
                    st.write(df[df.duplicated(keep=False)])
                else:
                    st.success("✅ Tidak ditemukan data duplikat")
                
            with tab3:
                st.write("**Statistik Deskriptif Data Numerik:**")
                st.dataframe(df.describe().style.background_gradient(cmap='Greens'))
        
        with st.container():
            st.subheader("2. Normalisasi Data")
            st.info("Normalisasi akan mengubah skala data numerik ke skala yang sama menggunakan StandardScaler")
            
            if st.button("🚀 Lakukan Normalisasi", help="Klik untuk melakukan normalisasi data"):
                with st.spinner("Sedang memproses normalisasi..."):
                    features = df.select_dtypes(include=['float64', 'int64']).columns
                    X = df[features].dropna()
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.X_scaled = X_scaled
                    st.session_state.features = features
                    st.session_state.scaler = scaler
                    st.success("✅ Normalisasi berhasil dilakukan!")
                    
                    # Show sample of scaled data
                    st.write("**Contoh Data yang Telah Dinormalisasi (5 baris pertama):**")
                    st.dataframe(pd.DataFrame(X_scaled, columns=features).head())
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu di menu Step 1: Upload Data")

# === VISUALISASI ===
elif menu == "Step 3: Visualisasi Data":
    st.header("📊 Visualisasi Data")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        
        with st.container():
            st.subheader("1. Heatmap Korelasi")
            st.info("Heatmap menunjukkan hubungan korelasi antar variabel")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax, center=0)
            plt.title("Korelasi Antar Variabel", pad=20)
            st.pyplot(fig)
            plt.clf()
            
        with st.container():
            st.subheader("2. Distribusi Variabel")
            selected_col = st.selectbox("Pilih variabel untuk dilihat distribusinya:", numerical_df.columns)
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram
            sns.histplot(numerical_df[selected_col], kde=True, ax=ax[0], color='#e74c3c')
            ax[0].set_title(f"Distribusi {selected_col}")
            
            # Boxplot
            sns.boxplot(x=numerical_df[selected_col], ax=ax[1], color='#f39c12')
            ax[1].set_title(f"Boxplot {selected_col}")
            
            st.pyplot(fig)
            plt.clf()
            
        with st.container():
            st.subheader("3. Pair Plot (Sampel Data)")
            st.warning("Note: Pair plot mungkin memakan waktu untuk dataset besar")
            
            if st.button("🔄 Generate Pair Plot"):
                with st.spinner("Membuat pair plot..."):
                    sample_df = numerical_df.sample(min(100, len(numerical_df)))
                    fig = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.6})
                    st.pyplot(fig)
                    plt.clf()
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu di menu Step 1: Upload Data")

# === CLUSTERING ===
elif menu == "Step 4: Hasil Clustering":
    st.header("🧩 Hasil Clustering")
    
    if 'X_scaled' in st.session_state:
        X_scaled = st.session_state.X_scaled
        
        with st.container():
            st.subheader("1. Evaluasi Jumlah Cluster Optimal")
            st.info("Gunakan Silhouette Score dan Davies-Bouldin Index untuk menentukan jumlah cluster terbaik")
            
            with st.spinner("Menghitung skor untuk berbagai jumlah cluster..."):
                clusters_range = range(2, 10)
                silhouette_scores = {}
                dbi_scores = {}

                progress_bar = st.progress(0)
                for i, k in enumerate(clusters_range):
                    clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
                    labels = clustering.fit_predict(X_scaled)
                    silhouette_scores[k] = silhouette_score(X_scaled, labels)
                    dbi_scores[k] = davies_bouldin_score(X_scaled, labels)
                    progress_bar.progress((i+1)/len(clusters_range))

            # Create score dataframe
            score_df = pd.DataFrame({
                'Silhouette Score': silhouette_scores,
                'Davies-Bouldin Index': dbi_scores
            })
            
            # Plot scores
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=score_df, markers=True, ax=ax)
            plt.title("Evaluasi Jumlah Cluster Optimal", pad=15)
            plt.xlabel("Jumlah Cluster")
            plt.ylabel("Skor")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.clf()

            best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
            best_k_dbi = min(dbi_scores, key=dbi_scores.get)

            cols = st.columns(2)
            with cols[0]:
                st.metric("Jumlah Cluster Optimal (Silhouette)", best_k_silhouette, 
                         help="Semakin tinggi Silhouette Score semakin baik")
            with cols[1]:
                st.metric("Jumlah Cluster Optimal (DBI)", best_k_dbi,
                         help="Semakin rendah Davies-Bouldin Index semakin baik")

        with st.container():
            st.subheader("2. Clustering Final")
            k_final = st.number_input("Pilih jumlah cluster untuk analisis:", 
                                    min_value=2, max_value=10, 
                                    value=best_k_silhouette, step=1)
            
            if st.button("🔮 Jalankan Spectral Clustering", key="final_cluster_btn"):
                with st.spinner(f"Melakukan clustering dengan {k_final} cluster..."):
                    final_cluster = SpectralClustering(n_clusters=k_final, 
                                                     affinity='nearest_neighbors', 
                                                     random_state=42)
                    labels = final_cluster.fit_predict(X_scaled)
                    st.session_state.labels = labels
                    st.session_state.k_final = k_final
                    st.success(f"✅ Clustering dengan {k_final} cluster berhasil!")

        if 'labels' in st.session_state:
            with st.container():
                st.subheader("3. Visualisasi Cluster (PCA)")
                st.info("Principal Component Analysis (PCA) untuk visualisasi 2D")
                
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                                   c=st.session_state.labels, 
                                   cmap='viridis', s=100, alpha=0.7)
                plt.title(f"Visualisasi Cluster (2D PCA) - {st.session_state.k_final} Cluster", pad=15)
                plt.xlabel("Komponen Utama 1")
                plt.ylabel("Komponen Utama 2")
                plt.colorbar(scatter, label='Cluster')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.clf()

            with st.container():
                st.subheader("4. Hasil Clustering")
                df = st.session_state.df.copy()
                df['Cluster'] = st.session_state.labels
                
                tab1, tab2 = st.tabs(["Data dengan Cluster", "Distribusi Cluster"])
                
                with tab1:
                    st.write(f"**Data dengan Label Cluster ({st.session_state.k_final} cluster):**")
                    st.dataframe(df.sort_values(by='Cluster').style.background_gradient(subset=['Cluster'], cmap='viridis'))
                
                with tab2:
                    cluster_counts = df['Cluster'].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, 
                               palette="viridis", ax=ax)
                    plt.title("Jumlah Wilayah per Cluster", pad=10)
                    plt.xlabel("Cluster")
                    plt.ylabel("Jumlah Wilayah")
                    st.pyplot(fig)
                    plt.clf()

            with st.container():
                st.subheader("5. Analisis Fitur Penting")
                st.info("Random Forest digunakan untuk mengetahui indikator paling berpengaruh")
                
                X = df.drop(columns=["Kabupaten/Kota", "Cluster"], errors="ignore")
                y = df["Cluster"]
                
                with st.spinner("Menghitung feature importance..."):
                    rf = RandomForestClassifier(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    importances = pd.Series(rf.feature_importances_, 
                                           index=X.columns).sort_values(ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=importances.values, y=importances.index, 
                               palette="rocket", ax=ax)
                    plt.title("Feature Importance untuk Clustering", pad=15)
                    plt.xlabel("Tingkat Kepentingan")
                    st.pyplot(fig)
                    plt.clf()

            with st.container():
                st.subheader("6. Interpretasi Hasil")
                
                cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
                st.write("**Rata-rata indikator per cluster:**")
                st.dataframe(cluster_summary.style.background_gradient(axis=0, cmap='YlOrBr'))
                
                # Check if poverty column exists
                kemiskinan_col = "Persentase Penduduk Miskin (%)"
                if kemiskinan_col in df.columns:
                    st.write("**Wilayah dengan Tingkat Kemiskinan Ekstrem:**")
                    
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("##### 🚨 3 Wilayah dengan Kemiskinan Tertinggi")
                        top3 = df.sort_values(by=kemiskinan_col, ascending=False).head(3)
                        st.dataframe(top3[["Kabupaten/Kota", kemiskinan_col, "Cluster"]])
                    
                    with cols[1]:
                        st.markdown("##### 🟢 3 Wilayah dengan Kemiskinan Terendah")
                        bottom3 = df.sort_values(by=kemiskinan_col, ascending=True).head(3)
                        st.dataframe(bottom3[["Kabupaten/Kota", kemiskinan_col, "Cluster"]])
                
                st.markdown("""
                **📝 Kesimpulan:**
                - Cluster dengan **nilai kemiskinan rendah** menunjukkan kinerja pembangunan yang baik
                - Cluster dengan **nilai pendidikan/kesehatan rendah** membutuhkan intervensi khusus
                - Hasil clustering dapat menjadi dasar rekomendasi kebijakan yang tepat sasaran
                
                **💡 Rekomendasi:**
                - Fokuskan program pengentasan kemiskinan pada cluster dengan nilai tinggi
                - Tingkatkan akses pendidikan dan kesehatan di cluster yang membutuhkan
                - Gunakan analisis ini untuk monitoring evaluasi program
                """)
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu di menu Step 2: Preprocessing Data")

# Add navigation buttons at the bottom of each page
navigation_buttons(menu)
