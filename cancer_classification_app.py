import streamlit as st
import pandas as pd
from seaborn import countplot, histplot
import matplotlib.pyplot as plt



# Set page configuration
st.set_page_config(page_title="Cancer Patient Classification", layout="wide")

# Title of the app
st.title("Cancer Patient Classification (2024)")

# Description of the app
st.markdown("""
This web app allows you to explore synthetic data about cancer patients, including their **Age**, **Gender**, **Cancer Type**, **Duration of Survival**, and **Survival Status**.

You can visualize different aspects of the data and gain insights based on various categories like **Age Group**, **Gender**, **Cancer Type**, and more.
""")

# Sidebar for uploading data
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV
    data = pd.read_csv(uploaded_file)

    # Show the first few rows of the dataset
    st.write("### Dataset Preview:")
    st.dataframe(data.head())

    # Age Group Classification
    data['Age_Group'] = pd.cut(data['Age'], bins=[20, 35, 50, 65, 80, 100],
                               labels=['20-35', '36-50', '51-65', '66-80', '81+'])

    # Group data by Age Group, Gender, and Status
    st.write("### Data Classification Summary:")
    grouped_classification = data.groupby(['Age_Group', 'Gender', 'Status']).size().unstack().fillna(0)
    st.write(grouped_classification)

    # Plot 1: Survival by Gender
    st.write("### Survival vs Death by Gender")
    fig, ax = plt.subplots(figsize=(8, 6))
    countplot(data=data, x='Gender', hue='Status', ax=ax)
    ax.set_title("Survival vs Death by Gender")
    st.pyplot(fig)

    # Plot 2: Survival by Cancer Type
    st.write("### Survival vs Death by Cancer Type")
    fig, ax = plt.subplots(figsize=(8, 6))
    countplot(data=data, x='Cancer_Type', hue='Status', ax=ax)
    ax.set_title("Survival vs Death by Cancer Type")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Plot 3: Duration of Survival
    st.write("### Survival Duration Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    histplot(data=data, x='Duration_Months', hue='Status', bins=30, kde=True, ax=ax)
    ax.set_title("Duration of Survival After Diagnosis")
    st.pyplot(fig)

else:
    st.write("cancer_patient_data_2024.csv")

