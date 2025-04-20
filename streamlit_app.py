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
            /* Main Background */
            .main {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
            }
            
            /* Titles and Headers */
            h1 {
                color: #1f3a93;
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
                border-bottom: 3px solid #3498db;
                padding-bottom: 0.5rem;
                text-align: center;
            }
            
            h2 {
                color: #2980b9;
                font-size: 1.8rem;
                margin-top: 2rem;
                margin-bottom: 1rem;
                padding-left: 1rem;
                border-left: 4px solid #3498db;
            }
            
            h3 {
                color: #16a085;
                font-size: 1.4rem;
                margin-top: 1.5rem;
            }
            
            /* Cards and Containers */
            .block-container {
                background-color: white;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
                padding: 2rem;
                margin-bottom: 2rem;
            }
            
            /* Buttons */
            .stButton>button {
                border-radius: 8px;
                border: none;
                background: linear-gradient(135deg, #3498db 0%, #2ecc71 100%);
                color: white;
                font-weight: 600;
                padding: 0.5rem 1.5rem;
                transition: all 0.3s ease;
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
            }
            
            /* Navigation Buttons */
            .nav-button {
                background: linear-gradient(135deg, #9b59b6 0%, #3498db 100%) !important;
                margin-top: 2rem;
            }
            
            /* Dataframes */
            .dataframe {
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }
            
            /* Radio Buttons - Hidden Label */
            .stRadio > label {
                display: none !important;
            }
            
            .stRadio>div {
                flex-direction: row !important;
                background: white;
                padding: 0.5rem;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
                justify-content: center;
                margin-bottom: 1.5rem;
            }
            
            /* Success/Warning Messages */
            .stSuccess {
                border-left: 4px solid #2ecc71;
                padding-left: 1rem;
            }
            
            .stWarning {
                border-left: 4px solid #e74c3c;
                padding-left: 1rem;
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
                background: #3498db;
                border-radius: 10px;
            }
            
            /* Progress Bar */
            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #3498db 0%, #2ecc71 100%);
            }
            
            /* Tab Styling */
            .stTabs > div > div > button {
                border-radius: 8px 8px 0 0 !important;
                padding: 0.5rem 1rem !important;
                font-weight: 600 !important;
            }
            
            .stTabs > div > div > button[aria-selected="true"] {
                background-color: #3498db !important;
                color: white !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu = st.radio(
    "",
    ("Home", "Step 1: Upload Data", "Step 2: Preprocessing Data", "Step 3: Visualisasi Data", "Step 4: Hasil Clustering"),
    horizontal=True
)

# === HOME ===
if menu == "Home":
    st.markdown(""" 
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊
    
    <div style="background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 20px;">
    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode <b>Spectral Clustering</b>
    - 📈 Mengevaluasi hasil pengelompokan
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Mulai Analisis →", key="home_next", help="Klik untuk mulai analisis"):
        menu = "Step 1: Upload Data"

# === UPLOAD DATA ===
elif menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")
    
    with st.container():
        st.markdown("""
        <div style="background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #2980b9;">Ketentuan Data:</h3>
        <ul>
            <li>Data berupa file <b>Excel (.xlsx)</b></li>
            <li>Mengandung kolom indikator kemiskinan seperti:
                <ul>
                    <li>Persentase Penduduk Miskin (%)</li>
                    <li>Jumlah Penduduk Miskin (ribu jiwa)</li>
                    <li>Indikator pendidikan dan kesehatan</li>
                </ul>
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        
        with st.expander("Lihat Data", expanded=True):
            st.dataframe(df.style.background_gradient(cmap='Blues'))
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Kembali ke Home", key="upload_back"):
            menu = "Home"
    with col2:
        if uploaded_file and st.button("Lanjut ke Preprocessing →", key="upload_next", help="Klik untuk lanjut ke step preprocessing"):
            menu = "Step 2: Preprocessing Data"

# === PREPROCESSING DATA ===
elif menu == "Step 2: Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        
        with st.container():
            st.subheader("1. Pengecekan Data")
            tab1, tab2, tab3 = st.tabs(["Missing Values", "Duplikat", "Statistik Deskriptif"])
            
            with tab1:
                st.write("**Jumlah Missing Values per Kolom:**")
                st.dataframe(df.isnull().sum().to_frame("Jumlah Missing"))
                
            with tab2:
                st.write(f"**Jumlah Data Duplikat:** {df.duplicated().sum()}")
                
            with tab3:
                st.write("**Statistik Deskriptif:**")
                st.dataframe(df.describe().style.background_gradient(cmap='Greens'))
        
        with st.container():
            st.subheader("2. Normalisasi Data")
            if st.button("Lakukan Normalisasi", key="normalize_btn"):
                features = df.select_dtypes(include=['float64', 'int64']).columns
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state.X_scaled = X_scaled
                st.session_state.features = features
                st.success("✅ Fitur telah dinormalisasi dan disimpan!")
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Kembali ke Upload Data", key="preprocess_back"):
            menu = "Step 1: Upload Data"
    with col3:
        if 'X_scaled' in st.session_state and st.button("Lanjut ke Visualisasi →", key="preprocess_next"):
            menu = "Step 3: Visualisasi Data"

# === VISUALISASI ===
elif menu == "Step 3: Visualisasi Data":
    st.header("📊 Visualisasi Data")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
        
        with st.container():
            st.subheader("Heatmap Korelasi")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
            
        with st.container():
            st.subheader("Distribusi Data")
            selected_column = st.selectbox("Pilih kolom untuk dilihat distribusinya:", numerical_df.columns)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(numerical_df[selected_column], kde=True, ax=ax)
            plt.title(f"Distribusi {selected_column}")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Kembali ke Preprocessing", key="visual_back"):
            menu = "Step 2: Preprocessing Data"
    with col3:
        if 'df' in st.session_state and st.button("Lanjut ke Clustering →", key="visual_next"):
            menu = "Step 4: Hasil Clustering"

# === CLUSTERING ===
elif menu == "Step 4: Hasil Clustering":
    st.header("🧩 Hasil Clustering")
    
    if 'X_scaled' in st.session_state:
        X_scaled = st.session_state.X_scaled
        
        with st.container():
            st.subheader("Evaluasi Jumlah Cluster")
            
            clusters_range = range(2, 10)
            silhouette_scores = {}
            dbi_scores = {}

            with st.spinner("Menghitung skor clustering..."):
                for k in clusters_range:
                    clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
                    labels = clustering.fit_predict(X_scaled)
                    silhouette_scores[k] = silhouette_score(X_scaled, labels)
                    dbi_scores[k] = davies_bouldin_score(X_scaled, labels)

            score_df = pd.DataFrame({
                'Silhouette Score': silhouette_scores,
                'Davies-Bouldin Index': dbi_scores
            })
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=score_df, markers=True, ax=ax)
            plt.title("Evaluasi Jumlah Cluster Optimal")
            plt.xlabel("Jumlah Cluster")
            plt.ylabel("Skor")
            st.pyplot(fig)

            best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
            best_k_dbi = min(dbi_scores, key=dbi_scores.get)

            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🔹 Optimal (Silhouette): {best_k_silhouette} cluster")
            with col2:
                st.success(f"🔸 Optimal (Davies-Bouldin): {best_k_dbi} cluster")

        with st.container():
            st.subheader("Clustering Final")
            k_final = st.number_input("Pilih jumlah cluster:", min_value=2, max_value=10, value=best_k_silhouette, step=1)
            
            if st.button("Lakukan Clustering Final", key="final_cluster_btn"):
                final_cluster = SpectralClustering(n_clusters=k_final, affinity='nearest_neighbors', random_state=42)
                labels = final_cluster.fit_predict(X_scaled)
                st.session_state.labels = labels
                st.session_state.k_final = k_final
                st.success(f"✅ Clustering dengan {k_final} cluster berhasil!")

        if 'labels' in st.session_state:
            with st.container():
                st.subheader("Visualisasi Cluster (PCA)")
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=st.session_state.labels, cmap='viridis', s=100, alpha=0.7)
                plt.title("Visualisasi Cluster (2D PCA)")
                plt.xlabel("Komponen Utama 1")
                plt.ylabel("Komponen Utama 2")
                plt.colorbar(scatter, label='Cluster')
                st.pyplot(fig)

            with st.container():
                st.subheader("Hasil Clustering")
                df = st.session_state.df.copy()
                df['Cluster'] = st.session_state.labels
                
                tab1, tab2 = st.tabs(["Data", "Distribusi"])
                
                with tab1:
                    st.dataframe(df.sort_values(by='Cluster').style.background_gradient(subset=['Cluster'], cmap='viridis'))
                
                with tab2:
                    cluster_counts = df['Cluster'].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis", ax=ax)
                    plt.title("Jumlah Wilayah per Cluster")
                    plt.xlabel("Cluster")
                    plt.ylabel("Jumlah")
                    st.pyplot(fig)

            with st.container():
                st.subheader("Analisis Fitur Penting")
                
                X = df.drop(columns=["Kabupaten/Kota", "Cluster"], errors="ignore")
                y = df["Cluster"]
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=importances.values, y=importances.index, palette="rocket", ax=ax)
                plt.title("Feature Importance")
                plt.xlabel("Tingkat Kepentingan")
                st.pyplot(fig)

            with st.container():
                st.subheader("Interpretasi Hasil")
                
                cluster_summary = df.groupby("Cluster").mean(numeric_only=True)
                st.write("Rata-rata indikator per cluster:")
                st.dataframe(cluster_summary.style.background_gradient(axis=0, cmap='YlOrBr'))
                
                if "Persentase Penduduk Miskin (%)" in df.columns:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Wilayah dengan Kemiskinan Tertinggi**")
                        top3 = df.sort_values(by="Persentase Penduduk Miskin (%)", ascending=False).head(3)
                        st.dataframe(top3[["Kabupaten/Kota", "Persentase Penduduk Miskin (%)", "Cluster"]])
                    
                    with col2:
                        st.markdown("**Wilayah dengan Kemiskinan Terendah**")
                        bottom3 = df.sort_values(by="Persentase Penduduk Miskin (%)", ascending=True).head(3)
                        st.dataframe(bottom3[["Kabupaten/Kota", "Persentase Penduduk Miskin (%)", "Cluster"]])
                
                st.markdown("""
                **Kesimpulan:**
                - Cluster dengan nilai kemiskinan rendah menunjukkan kinerja pembangunan yang baik
                - Cluster dengan nilai pendidikan/kesehatan rendah membutuhkan intervensi khusus
                - Hasil ini dapat menjadi dasar rekomendasi kebijakan yang tepat sasaran
                """)
    else:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("← Kembali ke Visualisasi", key="cluster_back"):
            menu = "Step 3: Visualisasi Data"
    with col2:
        if st.button("Selesai 🎉", key="cluster_finish"):
            menu = "Home"
