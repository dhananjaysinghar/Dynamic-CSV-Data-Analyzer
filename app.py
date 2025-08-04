import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno
import io

st.set_page_config(page_title="📊 Dynamic CSV Analyzer", layout="wide")
st.title("📊 Dynamic CSV Data Analyzer")

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])


# Caching expensive operations
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Convert datetime-like columns
    for col in df.select_dtypes(include='object').columns:
        try:
            if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0.9 * len(df):
                df[col] = pd.to_datetime(df[col])
        except:
            continue
    return df


if uploaded_file:
    try:
        df = load_data(uploaded_file)

        if df.empty:
            st.error("❌ Uploaded CSV is empty.")
            st.stop()

        st.subheader("📄 Dataset Overview")
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", list(df.columns))
        st.write("**Data Types:**")
        st.write(df.dtypes)
        st.write("**Missing Values Count:**")
        st.write(df.isnull().sum())
        st.write("**Descriptive Statistics:**")
        st.write(df.describe(include='all'))

        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(num_cols) >= 2:
            st.subheader("🔗 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        if df.isnull().values.any():
            st.subheader("🚨 Missing Values Heatmap")
            fig, ax = plt.subplots(figsize=(10, 4))
            msno.heatmap(df, ax=ax)
            st.pyplot(fig)

        # Expanders for optional visualizations
        with st.expander("📈 Distribution Plots (Numerical Columns)"):
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {col}')
                st.pyplot(fig)

        with st.expander("📦 Box Plots (Numerical Columns)"):
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f'Box Plot of {col}')
                st.pyplot(fig)

        with st.expander("🎻 Violin Plots (Numerical Columns)"):
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.violinplot(x=df[col], ax=ax)
                ax.set_title(f'Violin Plot of {col}')
                st.pyplot(fig)

        with st.expander("🧮 Count Plots (Top 10 Categories per Column)"):
            for col in cat_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().head(10).plot(kind='bar', ax=ax)
                ax.set_title(f'Count Plot of {col}')
                plt.xticks(rotation=45)
                st.pyplot(fig)

        if len(num_cols) >= 2 and len(num_cols) <= 10:
            st.subheader("🌐 Interactive Scatter Matrix (Plotly)")
            fig = px.scatter_matrix(df[num_cols])
            st.plotly_chart(fig, use_container_width=True)

        if len(num_cols) >= 2 and df.shape[0] < 5000:
            with st.expander("🔍 Pairplot (First 4 Numerical Columns)"):
                st.info("Pairplot may take a few seconds to load.")
                fig = sns.pairplot(df[num_cols[:4]].dropna())
                st.pyplot(fig)
        elif df.shape[0] >= 5000:
            st.warning("Skipping pairplot – dataset too large for performance.")

        st.success("✅ Analysis Completed Successfully!")

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("Please upload a CSV file to begin analysis.")
