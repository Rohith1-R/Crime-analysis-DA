import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Crime Type Prediction",
    page_icon="🚨"
)

# Load trained model and label encoder
model = joblib.load("random_forest.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🚨 Crime Type Prediction")

st.write("Enter the crime details to predict the crime type.")

# Model features
features = model.feature_names_in_

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        feature,
        value=0.0
    )

if st.button("🔮 Predict Crime Type"):

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    prediction = model.predict(input_df)

    crime_type = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Crime Type: {crime_type}")