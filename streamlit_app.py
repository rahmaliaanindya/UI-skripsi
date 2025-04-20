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
    layout="wide"
)

# === CSS Styling ===
def local_css():
    st.markdown(
        """
        <style>
            /* Main App Styling */
            .main {
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
            }
            .block-container {
                padding-top: 1rem;
                background-color: transparent;
                max-width: 90%;
            }
            
            /* Header Styling */
            h1 {
                font-family: 'Poppins', sans-serif;
                color: #2c3e50;
                font-size: 2.8rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 1.5rem;
                background: linear-gradient(to right, #3498db, #9b59b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            h2 {
                font-family: 'Poppins', sans-serif;
                color: #34495e;
                font-size: 1.8rem;
                font-weight: 600;
                border-bottom: 2px solid #3498db;
                padding-bottom: 0.5rem;
                margin-top: 1.5rem;
            }
            h3 {
                font-family: 'Poppins', sans-serif;
                color: #34495e;
                font-size: 1.4rem;
                font-weight: 500;
            }
            
            /* Text Styling */
            p, div, span {
                font-family: 'Open Sans', sans-serif;
                color: #2c3e50 !important;
            }
            
            /* Sidebar Styling */
            .sidebar .sidebar-content {
                background-color: #ffffff;
                box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            }
            
            /* Button Styling */
            .stButton>button {
                border-radius: 20px;
                border: none;
                background: linear-gradient(to right, #3498db, #9b59b6);
                color: white;
                font-weight: 600;
                padding: 0.5rem 1.5rem;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            /* Radio Button Styling */
            .stRadio>div {
                flex-direction: row;
                align-items: center;
                background: #ffffff;
                padding: 0.5rem;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .stRadio>div>label {
                margin: 0 0.5rem;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            .stRadio>div>label:hover {
                background: #f0f0f0;
            }
            .stRadio>div>label[data-baseweb="radio"]>div:first-child {
                margin-right: 0.5rem;
            }
            
            /* Dataframe Styling */
            .dataframe {
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            /* Card Styling */
            .stAlert {
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            /* Navigation Button Container */
            .nav-container {
                display: flex;
                justify-content: space-between;
                margin-top: 2rem;
                padding: 1rem 0;
                border-top: 1px solid #e0e0e0;
            }
            .nav-button {
                background: linear-gradient(to right, #3498db, #9b59b6);
                color: white;
                border: none;
                padding: 0.5rem 1.5rem;
                border-radius: 20px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .nav-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .nav-button:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
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
                background: linear-gradient(to bottom, #3498db, #9b59b6);
                border-radius: 10px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: #2980b9;
            }
            
            /* Success/Warning Messages */
            .stSuccess {
                background-color: #d4edda;
                color: #155724;
                border-radius: 8px;
            }
            .stWarning {
                background-color: #fff3cd;
                color: #856404;
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# === MENU NAVIGASI ===
menu_options = {
    "Home": "🏠 Home",
    "Step 1: Upload Data": "📤 Step 1: Upload Data",
    "Step 2: Preprocessing Data": "⚙️ Step 2: Preprocessing",
    "Step 3: Visualisasi Data": "📊 Step 3: Visualisasi",
    "Step 4: Hasil Clustering": "🧩 Step 4: Hasil"
}

# Create horizontal radio buttons for navigation
menu = st.radio(
    "Navigasi Aplikasi:",
    list(menu_options.values()),
    horizontal=True,
    label_visibility="hidden"
)

# Get the key from the displayed value
current_menu = [k for k, v in menu_options.items() if v == menu][0]

# === HOME ===
if current_menu == "Home":
    st.markdown(""" 
    # 👋 Selamat Datang di Aplikasi Analisis Cluster Kemiskinan Jawa Timur 📊
    
    Aplikasi ini dirancang untuk:
    - 📁 Mengunggah dan mengeksplorasi data indikator kemiskinan
    - 🧹 Melakukan preprocessing data
    - 📊 Menampilkan visualisasi
    - 🤖 Menerapkan metode **Spectral Clustering**
    - 📈 Mengevaluasi hasil pengelompokan
    
    ### 🚀 Mulai Analisis Anda
    Gunakan menu navigasi di atas atau tombol di bawah untuk memulai analisis.
    """)
    
    # Navigation buttons
    cols = st.columns([3, 1, 1])
    with cols[2]:
        if st.button("Next ➡️ Step 1: Upload Data"):
            current_menu = "Step 1: Upload Data"

# === UPLOAD DATA ===
elif current_menu == "Step 1: Upload Data":
    st.header("📤 Upload Data Excel")
    uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("✅ Data berhasil dimuat!")
        
        # Show basic info
        st.subheader("Preview Data")
        st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
        st.dataframe(df.head())
    
    # Navigation buttons
    cols = st.columns([3, 1, 1])
    with cols[2]:
        if 'df' in st.session_state and st.button("Next ➡️ Step 2: Preprocessing"):
            current_menu = "Step 2: Preprocessing Data"

# === PREPROCESSING DATA ===
elif current_menu == "Step 2: Preprocessing Data":
    st.header("⚙️ Preprocessing Data")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di Step 1.")
    else:
        df = st.session_state.df
        
        st.subheader("Cek Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values)
        
        if missing_values.sum() > 0:
            st.warning("Terdapat missing values dalam data!")
            handle_missing = st.selectbox(
                "Penanganan Missing Values:",
                ["Hapus baris dengan missing values", "Isi dengan nilai rata-rata"]
            )
            
            if st.button("Terapkan Penanganan"):
                if handle_missing == "Hapus baris dengan missing values":
                    df = df.dropna()
                else:
                    for col in df.select_dtypes(include=['float64', 'int64']):
                        df[col] = df[col].fillna(df[col].mean())
                st.session_state.df = df
                st.success("Missing values berhasil ditangani!")
                st.experimental_rerun()

        st.subheader("Cek Duplikat")
        duplicates = df.duplicated().sum()
        st.write(f"Jumlah duplikat: {duplicates}")
        
        if duplicates > 0 and st.button("Hapus Duplikat"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success("Duplikat berhasil dihapus!")
            st.experimental_rerun()

        st.subheader("Statistik Deskriptif")
        st.write(df.describe())

        st.subheader("Normalisasi dan Seleksi Fitur")
        features = df.select_dtypes(include=['float64', 'int64']).columns
        X = df[features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled = X_scaled
        st.success("✅ Fitur telah dinormalisasi dan disimpan.")
        
        st.subheader("Fitur yang Digunakan:")
        st.write(features.tolist())
    
    # Navigation buttons
    cols = st.columns([3, 1, 1])
    with cols[0]:
        if st.button("⬅️ Kembali ke Upload Data"):
            current_menu = "Step 1: Upload Data"
    with cols[2]:
        if 'X_scaled' in st.session_state and st.button("Next ➡️ Step 3: Visualisasi"):
            current_menu = "Step 3: Visualisasi Data"

# === VISUALISASI ===
elif current_menu == "Step 3: Visualisasi Data":
    st.header("📊 Visualisasi Data")
    
    if 'df' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df
        numerical_df = df.select_dtypes(include=['float64', 'int64'])

        st.subheader("Distribusi Data")
        selected_col = st.selectbox("Pilih kolom untuk dilihat distribusinya:", numerical_df.columns)
        plt.figure(figsize=(10, 5))
        sns.histplot(numerical_df[selected_col], kde=True)
        plt.title(f"Distribusi {selected_col}")
        st.pyplot(plt)
        plt.clf()

        st.subheader("Heatmap Korelasi")
        plt.figure(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
        plt.title("Matriks Korelasi Antar Variabel")
        st.pyplot(plt)
        plt.clf()

        st.subheader("Pair Plot (Sampel Data)")
        sample_size = min(100, len(df))
        sample_df = numerical_df.sample(sample_size)
        fig = sns.pairplot(sample_df)
        st.pyplot(fig)
        plt.clf()
    
    # Navigation buttons
    cols = st.columns([3, 1, 1])
    with cols[0]:
        if st.button("⬅️ Kembali ke Preprocessing"):
            current_menu = "Step 2: Preprocessing Data"
    with cols[2]:
        if st.button("Next ➡️ Step 4: Clustering"):
            current_menu = "Step 4: Hasil Clustering"

# === CLUSTERING ===
elif current_menu == "Step 4: Hasil Clustering":
    st.header("🧩 Hasil Clustering")
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Silakan lakukan preprocessing data terlebih dahulu.")
    else:
        X_scaled = st.session_state.X_scaled
        df = st.session_state.df.copy()

        st.subheader("Evaluasi Jumlah Cluster Optimal")
        st.markdown("""
        Kita akan mengevaluasi jumlah cluster optimal menggunakan dua metrik:
        - **Silhouette Score**: Semakin tinggi semakin baik (range -1 sampai 1)
        - **Davies-Bouldin Index**: Semakin rendah semakin baik
        """)

        clusters_range = range(2, 10)
        silhouette_scores = {}
        dbi_scores = {}

        with st.spinner("Menghitung skor untuk berbagai jumlah cluster..."):
            for k in clusters_range:
                clustering = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
                labels = clustering.fit_predict(X_scaled)
                silhouette_scores[k] = silhouette_score(X_scaled, labels)
                dbi_scores[k] = davies_bouldin_score(X_scaled, labels)

        score_df = pd.DataFrame({
            'Silhouette Score': silhouette_scores,
            'Davies-Bouldin Index': dbi_scores
        })

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Silhouette Score")
            st.line_chart(score_df['Silhouette Score'])
        with col2:
            st.subheader("Davies-Bouldin Index")
            st.line_chart(score_df['Davies-Bouldin Index'])

        best_k_silhouette = max(silhouette_scores, key=silhouette_scores.get)
        best_k_dbi = min(dbi_scores, key=dbi_scores.get)

        st.success(f"🔹 Jumlah cluster optimal (Silhouette Score): **{best_k_silhouette}** cluster")
        st.success(f"🔸 Jumlah cluster optimal (Davies-Bouldin Index): **{best_k_dbi}** cluster")

        k_final = st.number_input(
            "Pilih jumlah cluster (k):",
            min_value=2, 
            max_value=10, 
            value=best_k_silhouette, 
            step=1
        )

        if st.button("Jalankan Clustering"):
            with st.spinner("Sedang melakukan clustering..."):
                final_cluster = SpectralClustering(
                    n_clusters=k_final, 
                    affinity='nearest_neighbors', 
                    random_state=42
                )
                labels = final_cluster.fit_predict(X_scaled)
                st.session_state.labels = labels
                st.session_state.k_final = k_final
                st.success("Clustering berhasil dilakukan!")

        if 'labels' in st.session_state:
            st.subheader("Visualisasi Cluster (PCA)")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)

            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                X_pca[:, 0], 
                X_pca[:, 1], 
                c=st.session_state.labels, 
                cmap='viridis', 
                edgecolor='k',
                alpha=0.7,
                s=100
            )
            plt.title(f"Visualisasi Cluster (k={st.session_state.k_final})", fontsize=16)
            plt.xlabel("Principal Component 1", fontsize=12)
            plt.ylabel("Principal Component 2", fontsize=12)
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, linestyle='--', alpha=0.3)
            st.pyplot(plt)
            plt.clf()

            # Add cluster labels to dataframe
            df['Cluster'] = st.session_state.labels

            st.subheader("Distribusi Cluster")
            cluster_counts = df['Cluster'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(
                cluster_counts.index.astype(str), 
                cluster_counts.values,
                color=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
            ax.set_xlabel('Cluster', fontsize=12)
            ax.set_ylabel('Jumlah Wilayah', fontsize=12)
            ax.set_title('Distribusi Wilayah per Cluster', fontsize=16)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom')
            
            st.pyplot(fig)
            plt.clf()

            st.subheader("Karakteristik Cluster")
            cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
            st.dataframe(cluster_summary.style.background_gradient(cmap='viridis'))

            st.subheader("Wilayah per Cluster")
            for cluster in sorted(df['Cluster'].unique()):
                st.markdown(f"**Cluster {cluster}**")
                st.write(df[df['Cluster'] == cluster]['Kabupaten/Kota'].reset_index(drop=True))

            st.subheader("Analisis Fitur Penting")
            X = df.select_dtypes(include=['float64', 'int64']).drop('Cluster', axis=1, errors='ignore')
            y = df['Cluster']

            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax)
            ax.set_title("Feature Importance untuk Pembentukan Cluster", fontsize=16)
            ax.set_xlabel("Tingkat Kepentingan", fontsize=12)
            ax.set_ylabel("Variabel", fontsize=12)
            st.pyplot(fig)
            plt.clf()

            st.subheader("Interpretasi Hasil")
            st.markdown("""
            ### Panduan Interpretasi:
            1. **Cluster dengan nilai rendah** pada indikator kemiskinan menunjukkan wilayah dengan **kinerja baik**
            2. **Cluster dengan nilai tinggi** pada indikator kemiskinan memerlukan **perhatian khusus**
            3. Analisis feature importance menunjukkan variabel yang paling berpengaruh dalam pembentukan cluster
            """)

            # Show top and bottom regions
            kemiskinan_col = "Persentase Penduduk Miskin (%)"
            if kemiskinan_col in df.columns:
                st.subheader("Wilayah dengan Tingkat Kemiskinan Ekstrim")
                top5 = df.sort_values(by=kemiskinan_col, ascending=False)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(5)
                bottom5 = df.sort_values(by=kemiskinan_col, ascending=True)[["Kabupaten/Kota", kemiskinan_col, "Cluster"]].head(5)

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🚨 5 Wilayah dengan Kemiskinan Tertinggi")
                    st.dataframe(top5.style.highlight_max(axis=0, color='#ffcccc'))
                with col2:
                    st.markdown("#### 🟢 5 Wilayah dengan Kemiskinan Terendah")
                    st.dataframe(bottom5.style.highlight_min(axis=0, color='#ccffcc'))
    
    # Navigation buttons
    cols = st.columns([3, 1, 1])
    with cols[0]:
        if st.button("⬅️ Kembali ke Visualisasi"):
            current_menu = "Step 3: Visualisasi Data"
