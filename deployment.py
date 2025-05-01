import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

#when open pickle model use this commmand
with open("XGB_Kidney_Disease.pkl","rb") as file:
    model = pickle.load(file)

st.title("Chronic Kidney Disease Predictor - XGB Model")
st.write("This model predicts whether you have 'Chronic Kidney Disease' or not by taking in your information.")
st.subheader("Enter your information")

age = st.number_input("Age",min_value=0,max_value=100)
Blood_Pressure = st.number_input("Blood Pressure(mg)",min_value=0,max_value=200)
Specific_Gravity = st.number_input("Specific Gravity",min_value=1.005000,max_value=1.030)
Albumin = st.number_input("Albumin",min_value=0,max_value=5)
Sugar = st.number_input("Sugar Level",min_value=0,max_value=5)
Red_blood_cell = st.selectbox("Red Blood Cells",['Select','Normal','Abnormal'])
Pus_Cell = st.selectbox("Pus Cell",['Select','Normal','Abnormal'])
Pus_cell_clumps =  st.selectbox("Pus Cell Clumps",['Select','Present','Not Present'])
Bacteria = st.selectbox("Bacteria",['Select','Present','Not Present'])
Blood_glucose_random = st.number_input("Blood Glucose Random",min_value=22,max_value=490)
Blood_urea = st.number_input("Blood Urea",min_value=1.5,max_value=391.0)
serum_creatinine = st.number_input("Serum Creatinine",min_value=0.4,max_value=76.0)
sodium = st.number_input("Sodium",min_value=4.5,max_value=163.0)
potassium = st.number_input("Pottasium",min_value=2.5,max_value=47.0)
haemoglobin = st.number_input("Heamoglobin",min_value=3.1,max_value=17.8)
packed_cell_volume = st.number_input("Packed Cell Volumne",min_value=9,max_value=54)
white_blood_cell_count = st.number_input("WBC Count",min_value=2200,max_value=26400)
red_blood_cell_count = st.number_input("RBC Count",min_value=2.1,max_value=8.0)
hypertension = st.selectbox("Hypertension",['Select','Yes','No'])
diabetes_mellitus = st.selectbox("Diabetes",['Select','Yes','No'])
coronary_artery_disease = st.selectbox("Coronary Artinerey Disease",['Select','Yes','No'])
appetite = st.selectbox("Appetitte",['Select','Good','Poor'])
peda_edema = st.selectbox("Pedal Adema",['Select','Yes','No'])
aanemia = st.selectbox("Anemia",['Select','Yes','No'])

# in input data we will put all the variables that were in x.test
input_data = pd.DataFrame({'Age':[age],
                           "Blood Pressure(mg)":[Blood_Pressure],
                           "Specific Gravity":[Specific_Gravity],
                           "Albumin":[Albumin],
                           "Sugar Level":[Sugar],
                           "Red Blood Cells":[0 if Red_blood_cell== 'Abnormal' else 1],
                           "Pus Cell": [0 if Pus_Cell == 'Abnormal' else 1],
                           "Pus Cell Clumps": [1 if Pus_cell_clumps == 'Present' else 0],
                           "Bacteria": [1 if Bacteria == 'Present' else 0],
                           "Blood Glucose Random": [Blood_glucose_random],
                           "Blood Urea": [Blood_urea],
                           "Serum Creatinine": [serum_creatinine],
                           "Sodium": [sodium],
                           "Potassium": [potassium],
                           "Hemoglobin": [haemoglobin],
                           "Packed Cell Volume": [packed_cell_volume],
                           "WBC Count": [white_blood_cell_count],
                           "RBC Count": [red_blood_cell_count],
                           "Hypertension": [1 if hypertension == 'Yes' else 0],
                           "Diabetes Mellitus": [1 if diabetes_mellitus == 'Yes' else 0],
                           "Coronary Artery Disease": [1 if coronary_artery_disease == 'Yes' else 0],
                           "Appetite": [0 if appetite == 'Good' else 1],
                           "Pedal Edema": [1 if peda_edema == 'Yes' else 0],
                           "Anemia": [1 if aanemia == 'Yes' else 0]})

expected_order = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar',
       'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria',
       'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
       'potassium', 'haemoglobin', 'packed_cell_volume',
       'white_blood_cell_count', 'red_blood_cell_count', 'hypertension',
       'diabetes_mellitus', 'coronary_artery_disease', 'appetite',
       'peda_edema', 'aanemia']

input_data = input_data.reindex(columns = expected_order)
if st.button("Predict"):
    prediction = model.predict(input_data)
    result_mapping = {1:"Not Chronic kidney Disease",0:"Chronic Kidney Disease"}
    result = result_mapping[prediction[0]]
    st.write(f"The Predicted outcome is: **{result}**")