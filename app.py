import streamlit as st
import joblib
import pandas as pd

# Load PKL files
model = joblib.load("random_forest.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="LAPD Crime Prediction",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 LAPD CRIME PREDICTION")
st.markdown("### Predict Crime Category Using Random Forest")

st.divider()

# Get model feature names
features = model.feature_names_in_

st.subheader("Enter Crime Details")

input_data = {}

cols = st.columns(3)

for i, feature in enumerate(features):

    with cols[i % 3]:

        input_data[feature] = st.number_input(
            feature,
            value=0.0
        )

# Prediction
if st.button("🔮 Predict Crime", use_container_width=True):

    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    prediction = model.predict(input_df)

    predicted_crime = le.inverse_transform(prediction)[0]

    st.success(
        f"### Predicted Crime: {predicted_crime}"
    )

st.divider()

st.caption("LAPD Crime Analysis | Random Forest | Streamlit")