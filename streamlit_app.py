elif menu == "Step 5: Analisis Hasil":
    st.header("🔍 Analisis Hasil Clustering")

    if 'labels' in st.session_state and 'df' in st.session_state:
        df = st.session_state.df.copy()
        labels = st.session_state.labels
        df['Cluster'] = labels

        # Hanya ambil kolom numerik untuk analisis
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df_numeric = df[numeric_cols]

        # 1. Analisis distribusi cluster
        st.subheader("📊 Analisis Distribusi Cluster")
        with st.expander("Lihat Rata-rata Nilai per Cluster"):
            cluster_summary = df_numeric.groupby('Cluster').mean().T
            st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))
            
            # Visualisasi heatmap
            plt.figure(figsize=(10, 6))
            sns.heatmap(cluster_summary, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Perbandingan Rata-rata Nilai per Cluster")
            st.pyplot(plt)
            plt.clf()

        # 2. Visualisasi distribusi cluster
        st.subheader("📈 Distribusi Cluster")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            cluster_counts = df['Cluster'].value_counts().sort_index()
            fig1, ax1 = plt.subplots()
            ax1.pie(cluster_counts, labels=cluster_counts.index, 
                   autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis"))
            ax1.axis('equal')
            ax1.set_title("Persentase Distribusi Cluster")
            st.pyplot(fig1)
        
        with col2:
            # Bar chart
            fig2, ax2 = plt.subplots()
            sns.barplot(x=cluster_counts.index, y=cluster_counts.values, 
                        palette="viridis", ax=ax2)
            ax2.set_title("Jumlah Kabupaten/Kota per Cluster")
            ax2.set_xlabel("Cluster")
            ax2.set_ylabel("Jumlah")
            st.pyplot(fig2)

        # 3. Insight berdasarkan cluster
        st.subheader("🔎 Insight per Cluster")
        tab1, tab2, tab3 = st.tabs(["Statistik Deskriptif", "Karakteristik", "Perbandingan"])
        
        with tab1:
            for cluster_num in sorted(df['Cluster'].unique()):
                st.markdown(f"### Cluster {cluster_num}")
                cluster_data = df[df['Cluster'] == cluster_num]
                st.dataframe(cluster_data[numeric_cols].describe().style.background_gradient(cmap='Greens'))
        
        with tab2:
            for cluster_num in sorted(df['Cluster'].unique()):
                st.markdown(f"### Karakteristik Cluster {cluster_num}")
                cluster_data = df[df['Cluster'] == cluster_num]
                
                # Ambil 3 indikator tertinggi dan terendah
                mean_values = cluster_data[numeric_cols].mean().sort_values(ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Indikator Tertinggi:**")
                    for i, (ind, val) in enumerate(mean_values.head(3).items()):
                        st.write(f"{i+1}. {ind}: {val:.2f}")
                
                with col2:
                    st.markdown("**Indikator Terendah:**")
                    for i, (ind, val) in enumerate(mean_values.tail(3).items()):
                        st.write(f"{i+1}. {ind}: {val:.2f}")
                
                st.markdown("**Contoh Kabupaten/Kota:**")
                st.write(cluster_data['Kabupaten/Kota'].head(5).tolist())
        
        with tab3:
            st.markdown("### Perbandingan Antar Cluster")
            selected_feature = st.selectbox("Pilih indikator untuk dibandingkan:", numeric_cols)
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.boxplot(x='Cluster', y=selected_feature, data=df, palette="viridis", ax=ax3)
            ax3.set_title(f"Distribusi {selected_feature} per Cluster")
            st.pyplot(fig3)

        # 4. Feature Importance
        st.subheader("📌 Feature Importance")
        try:
            X = df[numeric_cols].drop(columns=['Cluster'], errors='ignore')
            y = df['Cluster']
            
            # Hapus kolom dengan variansi rendah
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.1)
            X_selected = selector.fit_transform(X)
            selected_features = X.columns[selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features)
            
            # Latih model RandomForest
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)

            # Tampilkan feature importance
            importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            
            fig4, ax4 = plt.subplots(figsize=(10,6))
            sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax4)
            ax4.set_title("Indikator Paling Berpengaruh dalam Clustering")
            ax4.set_xlabel("Tingkat Kepentingan")
            st.pyplot(fig4)
            
            # Tampilkan penjelasan
            st.markdown(f"**Indikator paling penting:** {importances.index[0]} ({importances.values[0]:.2f})")
            st.markdown(f"**Indikator paling tidak penting:** {importances.index[-1]} ({importances.values[-1]:.2f})")
            
        except Exception as e:
            st.error(f"Error dalam menghitung feature importance: {str(e)}")

        # 5. Analisis PCA
        st.subheader("🔄 Analisis PCA")
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standarisasi data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Lakukan PCA
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(X_scaled)
            
            # Visualisasi hasil PCA
            fig5, ax5 = plt.subplots(figsize=(10,8))
            scatter = ax5.scatter(principal_components[:,0], principal_components[:,1], 
                                 c=df['Cluster'], cmap='viridis', alpha=0.7)
            ax5.set_title("Visualisasi Cluster dalam Ruang PCA 2D")
            ax5.set_xlabel(f"PC1 (Variansi: {pca.explained_variance_ratio_[0]:.2f})")
            ax5.set_ylabel(f"PC2 (Variansi: {pca.explained_variance_ratio_[1]:.2f})")
            plt.colorbar(scatter, label='Cluster')
            st.pyplot(fig5)
            
            # Tampilkan kontribusi fitur
            st.markdown("**Kontribusi Fitur terhadap Komponen Utama:**")
            pca_loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=X.columns)
            st.dataframe(pca_loadings.style.background_gradient(cmap='RdBu', axis=0))
            
        except Exception as e:
            st.error(f"Error dalam analisis PCA: {str(e)}")

        # 6. Contoh Wilayah Ekstrim
        st.subheader("🏙️ Contoh Wilayah Ekstrim")
        
        if len(df['Cluster'].unique()) == 2:
            # Untuk kasus 2 cluster
            poor_cluster = df[df['Cluster'] == df['Cluster'].min()]
            rich_cluster = df[df['Cluster'] == df['Cluster'].max()]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 3 Wilayah dengan Tingkat Kemiskinan Tertinggi")
                top_poor = poor_cluster.nlargest(3, 'Persentase Penduduk Miskin (%)')
                st.dataframe(top_poor[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)']])
            
            with col2:
                st.markdown("### 3 Wilayah dengan Tingkat Kemiskinan Terendah")
                top_rich = rich_cluster.nsmallest(3, 'Persentase Penduduk Miskin (%)')
                st.dataframe(top_rich[['Kabupaten/Kota', 'Persentase Penduduk Miskin (%)']])
        else:
            st.warning("Analisis wilayah ekstrim hanya tersedia untuk clustering dengan 2 cluster")

    else:
        st.warning("⚠️ Hasil clustering belum ada. Silakan lakukan clustering terlebih dahulu.")
