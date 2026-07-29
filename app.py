import streamlit as st
import joblib
import pandas as pd

# Load model and encoder
rf = joblib.load("random_forest.pkl")
le = joblib.load("label_encoder.pkl")

st.title("🚨 LAPD Crime Prediction")

st.subheader("Enter Crime Details")

# Enter the features that your model was trained on
# Replace/add fields according to your X columns

input_data = {}

for col in rf.feature_names_in_:

    if col in ["Vict Age", "Hour", "Year"]:
        input_data[col] = st.number_input(
            f"{col}",
            value=0
        )

    else:
        input_data[col] = st.number_input(
            f"{col}",
            value=0
        )

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("🔮 Predict Crime"):

    prediction = rf.predict(input_df)

    crime = le.inverse_transform(prediction)[0]

    st.success(f"Predicted Crime: **{crime}**")