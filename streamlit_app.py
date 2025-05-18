# === MENU NAVIGASI ===
menu = st.radio(
    "NAVIGASI APLIKASI",
    ("Home", "Upload Data", "EDA", "Clustering", "Results"),
    horizontal=True
)

# === HOME ===
if menu == "Home":
    st.markdown(""" 
    <div style="text-align:center">
        <h1 style="color:#2c3e50; font-size:2.5rem; margin-bottom:1.5rem">
            Spectral Clustering Analysis with PSO Optimization
        </h1>
        <p style="font-size:1.1rem; color:#34495e; margin-bottom:2rem">
            Professional tool for poverty indicator analysis using advanced machine learning techniques
        </p>
    </div>
    
    <div style="background-color:#f8f9fa; border-radius:12px; padding:2rem; margin-bottom:2rem">
        <h3 style="color:#2c3e50; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
            Application Features
        </h3>
        <ul style="color:#34495e; line-height:2">
            <li>📊 Interactive data exploration and visualization</li>
            <li>⚙️ Robust data preprocessing pipeline</li>
            <li>🤖 Advanced Spectral Clustering algorithm</li>
            <li>🔬 Particle Swarm Optimization for parameter tuning</li>
            <li>📈 Comprehensive cluster evaluation metrics</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# === UPLOAD DATA ===
elif menu == "Upload Data":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Data Upload
    </h1>
    """, unsafe_allow_html=True)
    
    # Rest of your Upload Data section...

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
        st.dataframe(missing_values[missing_values > 0].to_frame("Missing Values"))
        
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
        st.dataframe(df.describe().style.format("{:.2f}"))
        
        # Rest of your EDA section...

    else:
        st.warning("⚠️ Please upload data first.")

# === CLUSTERING ===
elif menu == "Clustering":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Spectral Clustering with PSO Optimization
    </h1>
    """, unsafe_allow_html=True)
    
    if 'df' in st.session_state:
        df = st.session_state.df
        X = df.drop(columns=['Kabupaten/Kota'])
        
        # Rest of your Clustering section...

    else:
        st.warning("⚠️ Please upload data first.")

# === RESULTS ===
elif menu == "Results":
    st.markdown("""
    <h1 style="color:#2c3e50; font-size:2rem; border-bottom:1px solid #ecf0f1; padding-bottom:0.5rem">
        Analysis Results
    </h1>
    """, unsafe_allow_html=True)
    
    if 'df_with_cluster' in st.session_state:
        df = st.session_state.df_with_cluster
        
        # Rest of your Results section...
        
    else:
        st.warning("⚠️ Please complete clustering analysis first.")
