import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('diabetes_model.pkl')

# Page config
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🏥")

st.title("🏥 Diabetes Risk Predictor")
st.write("Enter the patient's medical details below:")

# Input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 25)

# Predict button
if st.button("🔍 Check Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes ({probability:.1f}% probability)")
        st.write("Please consult a doctor immediately for proper diagnosis.")
    else:
        st.success(f"✅ Low Risk of Diabetes ({probability:.1f}% probability)")
        st.write("Maintain a healthy lifestyle and get regular checkups.")
