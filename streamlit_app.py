# ... (keep all previous imports and CSS styling)

# === EDA ===
elif menu == "EDA":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Exploratory Data Analysis
    </h1>
    """, unsafe_allow_html=True)
    
    if 'df' in st.session_state:
        df = st.session_state.df
        
        # === PREPROCESSING SECTION ===
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Data Preprocessing</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show missing values
        st.subheader("Missing Values Check")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values[missing_values > 0].to_frame("Missing Values").style.background_gradient(cmap='Reds'))
        
        # Show duplicates
        st.subheader("Duplicate Check")
        st.write(f"Number of duplicate rows: {df.duplicated().sum()}")
        
        # Data cleaning options
        if st.checkbox("Perform Data Cleaning"):
            # Drop duplicates
            df = df.drop_duplicates()
            
            # Handle missing values (you can customize this based on your needs)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            st.success("Data cleaning completed!")
            st.session_state.df = df
        
        # === DATASET INFO ===
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Dataset Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a buffer to capture info output
        from io import StringIO
        import sys
        
        buffer = StringIO()
        sys.stdout = buffer
        df.info()
        sys.stdout = sys.__stdout__
        info_output = buffer.getvalue()
        
        # Display the info output
        st.text(info_output)
        
        # Show descriptive statistics
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe().style.format("{:.2f}").background_gradient(cmap='Blues'))
        
        # === DATA DISTRIBUTION ===
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Variable Distribution</h3>
        </div>
        """, unsafe_allow_html=True)
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Select variable:", numeric_columns)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[selected_col], kde=True, bins=30, color='#3498db')
        ax.set_title(f'Distribution of {selected_col}', fontsize=14)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
        # === CORRELATION ANALYSIS ===
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Feature Correlation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        numerical_df = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                   center=0, ax=ax, annot_kws={"size": 10})
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
        # Store cleaned data for clustering
        st.session_state.df_cleaned = df
        
    else:
        st.warning("⚠️ Please upload data first.")

# === CLUSTERING ===
elif menu == "Clustering":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Spectral Clustering with PSO Optimization
    </h1>
    """, unsafe_allow_html=True)
    
    if 'df_cleaned' in st.session_state:
        df = st.session_state.df_cleaned
        X = df.drop(columns=['Kabupaten/Kota'])
        
        # === PREPROCESSING FOR CLUSTERING ===
        st.markdown("""
        <div style="background-color:#f8f9fa; border-radius:8px; padding:1rem; margin:1.5rem 0">
            <h3 style="color:#2c3e50">Data Scaling</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("Using RobustScaler to handle potential outliers:")
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        st.session_state.X_scaled = X_scaled
        
        # Show sample of scaled data
        st.write("Sample of scaled data (first 5 rows):")
        st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())
        
        # Rest of your clustering code...
        # ... (keep all the existing clustering code from before)

# ... (keep the rest of the code the same)
