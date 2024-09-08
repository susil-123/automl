import streamlit as st
import pandas as pd
import os
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.components.data_injestion import DataInjestion

df = None


if os.path.exists('./source_data/source_data.csv'):
    df = pd.read_csv('./source_data/source_data.csv')

with st.sidebar:
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profile", "ML", "Download"])

if choice == "Upload":
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.to_csv('./source_data/source_data.csv', index=False)
        DataInjestion(df)
        st.dataframe(df)
        st.success("File uploaded successfully!")

elif choice == "Profile":
    profile_button = st.button("Profiling")
    if profile_button:
        profile = ProfileReport(df, title="Profiling Report")
        st_profile_report(profile)

# ML and Download sections (placeholders)
elif choice == "ML":
    button = st.button("ML")
    if button:
        if df is not None:
            st.dataframe(df)
            di_obj = DataInjestion(df)
            train_data_path, train_data_path = di_obj.initiate_data_injestion()
        else:
            st.write("No data frame found")

    
elif choice == "Download":
    st.write("Download feature coming soon...")
