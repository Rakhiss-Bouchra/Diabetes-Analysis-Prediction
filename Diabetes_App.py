import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained random forest model
model = joblib.load('random_forest_diabetes_model.joblib')

data = pd.read_csv('diabetes.csv')

# Set Streamlit page title and description
st.set_page_config(
    page_title='Diabetes Prediction & Visualization',
    page_icon=":bar_chart:",
    layout="wide"
)

# Create a sidebar to select the mode
mode = st.sidebar.selectbox("Select Mode", ("Visualization", "Prediction"))

# Main content
st.title("Diabetes Prediction and Visualization Dashboard")

if mode == "Visualization":
    st.subheader("Data Visualization Dashboard")

    st.markdown("This dashboard provides insightful visualizations about the diabetes dataset.")

    # Visualization 1: Glucose, Blood Pressure, BMI Histograms Side by Side
    st.subheader("Glucose | Blood Pressure | BMI Histograms")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(data['Glucose'], kde=True, ax=axes[0])
    axes[0].set_title('Glucose Histogram')
    sns.histplot(data['BloodPressure'], kde=True, ax=axes[1])
    axes[1].set_title('Blood Pressure Histogram')
    sns.histplot(data['BMI'], kde=True, ax=axes[2])
    axes[2].set_title('BMI Histogram')
    st.pyplot(fig)

    # Visualization 2: Age and Pregnancies Histograms
    st.subheader("Age and Pregnancies Histograms")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Age Histogram
    sns.histplot(data['Age'], kde=True, ax=ax[0])
    ax[0].set_title('Age Histogram')

    # Pregnancies Histogram
    sns.histplot(data['Pregnancies'], kde=True, ax=ax[1])
    ax[1].set_title('Pregnancies Histogram')

    st.pyplot(fig)

    # Visualization 3: Box Plots for Numerical Features
    st.subheader("Box Plots for Numerical Features: Blood Pressure | BMI | Age")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.boxplot(x='Outcome', y='BloodPressure', data=data, ax=axes[0])
    axes[0].set_title('Blood Pressure Box Plot')
    sns.boxplot(x='Outcome', y='BMI', data=data, ax=axes[1])
    axes[1].set_title('BMI Box Plot')
    sns.boxplot(x='Outcome', y='Age', data=data, ax=axes[2])
    axes[2].set_title('Age Box Plot')
    st.pyplot(fig)

    # Visualization 4: Correlation Heatmap
    st.subheader("Correlation Heatmap")
    st.write("This heatmap shows the correlation between different features.")

    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', linewidths=.5)
    st.pyplot(plt.gcf())

elif mode == "Prediction":
    st.subheader("Diabetes Prediction")

    st.write("Input the values for prediction:")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.0, step=0.01)
    age = st.number_input("Age", min_value=0, max_value=150)

    # Make a prediction
    
    input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]]
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    if st.button("Predict"):
        if prediction[0] == 0:
            st.success("Good news! The patient does not suffer from diabetes.")
            st.markdown("<style>div.stButton button {background-color: green;}</style>", unsafe_allow_html=True)
        else:
            st.error("Unfortunately, the patient suffers from diabetes!")
            st.markdown("<style>div.stButton button {background-color: red;}</style>", unsafe_allow_html=True)
                
        # Display the prediction percentage
        st.write(f"Diabetes Prediction Percentage: {prediction_proba[0] * 100:.2f}%")
