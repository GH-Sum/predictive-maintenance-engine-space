import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Download model from HF Hub
model_path = hf_hub_download(
    repo_id="HF-Sum/predictive-maintenance-engine-model", # Corrected repo_id
    filename="best_model.pkl"
)

model = joblib.load(model_path)

st.title("Predictive Maintenance System")

st.write("Enter engine sensor values to predict engine condition.")

engine_rpm = st.number_input("Engine RPM")
lub_oil_pressure = st.number_input("Lub Oil Pressure")
fuel_pressure = st.number_input("Fuel Pressure")
coolant_pressure = st.number_input("Coolant Pressure")
lub_oil_temp = st.number_input("Lub Oil Temperature")
coolant_temp = st.number_input("Coolant Temperature")

if st.button("Predict"):

    input_df = pd.DataFrame({
        'engine_rpm': [engine_rpm],
        'lub_oil_pressure': [lub_oil_pressure],
        'fuel_pressure': [fuel_pressure],
        'coolant_pressure': [coolant_pressure],
        'lub_oil_temp': [lub_oil_temp],
        'coolant_temp': [coolant_temp]
    })

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Maintenance Required")
    else:
        st.success("Engine Operating Normally")

# Note: To run this as a Streamlit application, you would typically save
# this code to a file (e.g., app.py) using '%%writefile app.py' in a separate
# cell, and then run it from the terminal using '!streamlit run app.py &' 
# along with a port forwarding solution like ngrok or localtunnel.
