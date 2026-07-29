import streamlit as st
import joblib
import pandas as pd

model = joblib.load("random_forest.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🚨 Crime Type Prediction")

# 5 inputs
age = st.number_input("Victim Age", 1, 100, 25)
hour = st.number_input("Crime Hour", 0, 23, 12)
area = st.number_input("Area Code", 1, 30, 1)
victim_sex = st.number_input("Victim Sex", 0, 3, 0)
premise = st.number_input("Premise Code", 0, 999, 0)

if st.button("🔮 Predict Crime Type"):

    input_data = pd.DataFrame([[
        age,
        hour,
        area,
        victim_sex,
        premise
    ]])

    prediction = model.predict(input_data)

    crime_type = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Crime Type: **{crime_type}**")