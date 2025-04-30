
import streamlit as st
import mlflow.pyfunc
import numpy as np

st.title("🩸 Blood Donation Prediction App")

run_id = "daab4ca0d0d74ba99eab0cbb745579db"
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

st.markdown("Enter your details:")

features = [
    st.number_input("Recency", min_value=0.0),
    st.number_input("Frequency", min_value=0.0),
    st.number_input("Monetary", min_value=0.0),
    st.number_input("Time", min_value=0.0),
    st.number_input("Recency / Time Ratio", min_value=0.0),
    st.number_input("Frequency × Monetary", min_value=0.0)
]

if st.button("Predict"):
    input_array = np.array(features).reshape(1, -1).astype(np.float32)
    prediction = model.predict(input_array)
    score = prediction.item()
    st.success(f"🧪 Predicted Donation Likelihood: {score:.4f}")
