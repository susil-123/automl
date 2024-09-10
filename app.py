import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.components.data_injestion import DataInjestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

df = None

# Check if the data already exists
if os.path.exists('./source_data/source_data.csv'):
    df = pd.read_csv('./source_data/source_data.csv')

# Streamlit sidebar
with st.sidebar:
    st.title("AutoML")
    choice = st.radio("Navigation", ["Upload", "Profile", "ML", "Download"])

# File Upload Section
if choice == "Upload":
    uploaded_file = st.file_uploader("Choose a file")
    upload = st.button("upload? ")
    if upload:
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df.to_csv('./source_data/source_data.csv', index=False)
            di_obj = DataInjestion(df)
            train_data_path, test_data_path = di_obj.initiate_data_injestion()
            st.session_state.train_data_path = train_data_path
            st.session_state.test_data_path = test_data_path
            st.dataframe(df)
            st.success("File uploaded successfully!")
        else:
            st.write("No file uploaded...")

# Profile Section
elif choice == "Profile":
    profile_button = st.button("Profiling")
    if profile_button and df is not None:
        profile = ProfileReport(df, title="Profiling Report")
        st_profile_report(profile)

# Machine Learning Section
elif choice == "ML":
    prob_type = ['classification', 'Regression']

    if "problem" not in st.session_state:
            st.session_state.problem = None

    st.session_state.problem = st.selectbox('Problem Type ?', prob_type)

    if df is not None:
        target_list = df.columns
        
        st.dataframe(df)

        if "target" not in st.session_state:
            st.session_state.target = None

        st.session_state.target = st.selectbox('Target variable?', target_list)

        train_button = st.button("Train?")

        if train_button:
            problem = st.session_state.problem
            target = st.session_state.target
            train_data_path = st.session_state.train_data_path
            test_data_path = st.session_state.test_data_path
            
            data_transformation = DataTransformation(df,train_data_path,test_data_path,target)
            train_data_preprocessed, test_data_preprocessed, preprocessor_path = data_transformation.initiate_data_tranformation()
            
            st.warning('Preprocessing Completed, Model Training Started', icon="🚀")
            
            start_time = datetime.now()
            progress_bar = st.progress(0)  # Create progress bar
            time_placeholder = st.empty()  # Placeholder for the time

            model_training = ModelTraining()

            try:
                best_param,best_model,training_score,testing_score,metrics = model_training.initiate_model_training(train_data_preprocessed, test_data_preprocessed, problem)
                end_time = datetime.now()
                elapsed_time = (end_time - start_time).total_seconds()
                progress_bar.progress(100)
                time_placeholder.text(f"Total Time elapsed: {elapsed_time:.2f} seconds")

                st.code(best_param)

                st.success(f"Training completed with target variable: {st.session_state.target} with a {metrics} training score of : {training_score} and testing score : {testing_score} on {best_model} model")
            except Exception as e:
                st.error(f"Error during training: {e}")
                progress_bar.progress(0)


# Download Section
elif choice == "Download":
    st.write("Download feature coming soon...")
