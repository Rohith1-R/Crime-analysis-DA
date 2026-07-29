import streamlit as st
import joblib
import pandas as pd

# Load model and target encoder
model = joblib.load("random_forest.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🚨 Crime Type Prediction")

# Enter the feature values
input_data = []

for feature in model.feature_names_in_:
    value = st.number_input(feature, value=0.0)
    input_data.append(value)

# Prediction
if st.button("Predict Crime Type"):

    X_input = pd.DataFrame(
        [input_data],
        columns=model.feature_names_in_
    )

    prediction = model.predict(X_input)

    crime_type = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Crime Type: {crime_type}")