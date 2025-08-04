import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import missingno as msno

st.set_page_config(page_title="📊 Dynamic Data Analyzer", layout="wide")
st.title("📊 Dynamic CSV / Parquet Data Analyzer")

uploaded_file = st.file_uploader("📂 Upload a CSV or Parquet file", type=["csv", "parquet"])

if uploaded_file:
    try:
        # Load the file based on extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.parquet'):
            df = pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format.")
            st.stop()

        # Try converting datetime-like columns
        for col in df.select_dtypes(include='object').columns:
            try:
                if pd.to_datetime(df[col], errors='coerce').notna().sum() > 0.9 * len(df):
                    df[col] = pd.to_datetime(df[col])
            except:
                continue

        if df.empty:
            st.error("❌ Uploaded file is empty.")
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

        if st.checkbox("🔗 Show Correlation Heatmap"):
            if len(num_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            else:
                st.info("Not enough numerical columns to compute correlation.")

        if st.checkbox("🚨 Show Missing Values Heatmap"):
            if df.isnull().values.any():
                fig, ax = plt.subplots(figsize=(10, 4))
                msno.heatmap(df, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No missing values to display.")

        if st.checkbox("📈 Show Distribution Plots"):
            for col in num_cols:
                st.markdown(f"**Distribution of `{col}`**")
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)

        if st.checkbox("📦 Show Box Plots"):
            for col in num_cols:
                st.markdown(f"**Box Plot of `{col}`**")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)

        if st.checkbox("🎻 Show Violin Plots"):
            for col in num_cols:
                st.markdown(f"**Violin Plot of `{col}`**")
                fig, ax = plt.subplots()
                sns.violinplot(x=df[col], ax=ax)
                st.pyplot(fig)

        if st.checkbox("🧮 Show Count Plots (Top Categories)"):
            for col in cat_cols:
                st.markdown(f"**Top Categories in `{col}`**")
                fig, ax = plt.subplots()
                df[col].value_counts().head(10).plot(kind='bar', ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        if st.checkbox("🌐 Show Interactive Scatter Matrix"):
            if 2 <= len(num_cols) <= 10:
                fig = px.scatter_matrix(df[num_cols])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need 2–10 numerical columns for scatter matrix.")

        if st.checkbox("🔍 Show Pairplot (first 4 numerical columns)"):
            if len(num_cols) >= 2 and df.shape[0] <= 5000:
                fig = sns.pairplot(df[num_cols[:4]].dropna())
                st.pyplot(fig)
            elif df.shape[0] > 5000:
                st.warning("Dataset too large for pairplot (limit: 5000 rows).")
            else:
                st.info("Not enough numerical columns.")

        st.success("✅ Analysis Completed Successfully!")

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("Please upload a CSV or Parquet file to begin analysis.")
