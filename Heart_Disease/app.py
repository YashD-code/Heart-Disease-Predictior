import streamlit as st
import pickle
import json

with open("Heart_Disease_model.pkl","rb")as f:
    model=pickle.load(f)
with open("scaler.pkl","rb")as f:
    scaler=pickle.load(f)
with open("feature_columns.json","r")as f:
    feature_cols=json.load(f)

st.set_page_config(page_title="Heart Disease Prediction")
st.title("Heart Disease Prediction Model")
with st.form("classification_form"):
    st.subheader("Enter input fields:")
    age=st.number_input("Age (in years)")
    sex=st.selectbox("Sex (Male(0),Female(1))",(0,1))
    chest_pain=st.selectbox("Chest Pain Type",(0,1,2,3))
    rest_bp=st.number_input("Resting Blood Pressure")
    cholesterol=st.number_input("Cholesterol Level:")
    fast_bp=st.selectbox("Fasting Blood Pressure",(0,1))
    rest_ecg=st.selectbox("Resting ECG Value",(0,1,2))
    max_heartRate=st.number_input("Max Heart Rate")
    angina=st.selectbox("Exercise Induced Angina",(0,1))
    st_dep=st.number_input("ST Depression")
    st_slope=st.selectbox("ST Slope",(0,1,2))
    num_vessels=st.selectbox("Number of Major Vessels",(0,1,2,3))
    thalassemia=st.selectbox("Thalassemia Type",(0,1,2,3))

    submitted=st.form_submit_button("Output")

if submitted:
    try:
        input_data={
            "age":age,
            "sex":sex,
            "chest_pain_type":chest_pain,
            "resting_blood_pressure":rest_bp,
            "cholesterol":cholesterol,
            "fasting_blood_sugar":fast_bp,
            "resting_ecg":rest_ecg,
            "max_heart_rate":max_heartRate,
            "exercise_induced_angina":angina,
            "st_depression":st_dep,
            "st_slope":st_slope,
            "num_major_vessels":num_vessels,
            "thalassemia":thalassemia
        }
        # Match feature order
        input_list = [input_data[col] for col in feature_cols]

        # Scale and predict
        scaled_input = scaler.transform([input_list])
        classification = model.predict(scaled_input)
        if classification[0]==0:
            st.success("Heart Disease:No")
        else:
            st.success("Hear Disease:Yes")
    except Exception as e:
        st.error(f"Error:{e}")