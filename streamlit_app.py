import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.preprocessing import StandardScaler

# Loading the saved models from the main directory
breast_cancer_model = pickle.load(open('breast_cancer_model.sav', 'rb'))
diabetes_model = pickle.load(open('diabetic_model.sav', 'rb'))
heart_failure_model = pickle.load(open('heart_failure_model.sav', 'rb'))

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Breast Cancer Prediction'],
                           icons=['capsule', 'heart-pulse', 'activity'],
                           default_index=0)

# Diabetes prediction page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose level')
    with col3:
        SkinThickness = st.text_input('Skin thickness value')
    with col1:
        BMI = st.text_input('BMI level')
    with col2:
        Age = st.text_input('Age of the person')

    # Code for prediction
    diab_diagnosis = ''
    if st.button('Diabetes test results'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, SkinThickness, BMI, Age]])
        if diab_prediction[0] == 1:
            diab_diagnosis = 'This person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'
    st.success(diab_diagnosis)

# Heart disease prediction page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    # Getting the input data from the user
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        Age = st.text_input('Age of person')
    with col2:
        Anaemia = st.text_input('Anaemia status')
    with col3:
        CreatinePhosphokinase = st.text_input('Creatine Phosphokinase value')
    with col4:
        Diabetes = st.text_input('Diabetes status')
    with col1:
        EjectionFraction = st.text_input('Ejection fraction')
    with col2:
        HighBloodPressure = st.text_input('High blood pressure status')
    with col3:
        Platelets = st.text_input('Platelets value')
    with col4:
        SerumCreatine = st.text_input('Serum Creatine level')
    with col1:
        SerumSodium = st.text_input('Serum sodium level')
    with col2:
        Sex = st.text_input('Sex of person')
    with col3:
        Smoking = st.text_input('Smoking status')
    with col4:
        Time = st.text_input('Time')

    # Code for prediction
    heart_disease_diagnosis = ''
    if st.button('Heart disease test results'):
        input_data = [Age, Anaemia, CreatinePhosphokinase, Diabetes, EjectionFraction, HighBloodPressure, Platelets, SerumCreatine, SerumSodium, Sex, Smoking, Time]
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1, -1)
        heart_disease_prediction = heart_failure_model.predict(input_data_reshaped)
        if heart_disease_prediction[0] == 1:
            heart_disease_diagnosis = 'This person is suffering from heart disease'
        else:
            heart_disease_diagnosis = 'The person is not suffering from heart disease'
    st.success(heart_disease_diagnosis)

# Breast cancer disease prediction page
if selected == 'Breast Cancer Prediction':
    st.title('Breast Cancer Prediction using ML')

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        MeanRadius = st.text_input('Mean radius of breast')
    with col2:
        MeanTexture = st.text_input('Mean texture of breast')
    with col3:
        MeanPerimeter = st.text_input('Mean perimeter of breast')
    with col1:
        MeanArea = st.text_input('Mean area of breast')
    with col2:
        MeanSmoothness = st.text_input('Mean smoothness of breast')

    # Code for prediction
    breast_cancer_diagnosis = ''
    if st.button('Breast cancer test results'):
        input_data = [MeanRadius, MeanTexture, MeanPerimeter, MeanArea, MeanSmoothness]
        input_data_as_np_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_np_array.reshape(1, -1)
        scaler = StandardScaler()
        scaler.fit(input_data_reshaped)
        std_data = scaler.transform(input_data_reshaped)
        breast_cancer_prediction = breast_cancer_model.predict(std_data)
        if breast_cancer_prediction[0] == 1:
            breast_cancer_diagnosis = 'Patient has a malignant (cancerous) tumor'
        else:
            breast_cancer_diagnosis = 'Patient is not suffering from breast cancer'
    st.success(breast_cancer_diagnosis)
