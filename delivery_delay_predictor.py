%%writefile delivery_delay_predictor.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.sav')

# Define the feature names, matching the order used during training
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops',
                 'Vehicle_Age', 'Road_Condition_Score', 'Package_Weight',
                 'Fuel_Efficiency', 'Warehouse_Processing_Time']

# Streamlit App Title
st.title('Delivery Delay Predictor')
st.write('Enter the details below to predict if a delivery will be delayed.')

# Input widgets for each feature
delivery_distance = st.slider('Delivery Distance (km)', 0.0, 100.0, 25.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', 1, 5, 3)
weather_condition = st.slider('Weather Condition (1-5)', 1, 5, 3)
delivery_slot = st.slider('Delivery Slot (1-3)', 1, 3, 2)
driver_experience = st.slider('Driver Experience (years)', 0, 20, 5)
num_stops = st.slider('Number of Stops', 0, 10, 3)
vehicle_age = st.slider('Vehicle Age (years)', 0, 10, 5)
road_condition_score = st.slider('Road Condition Score (1-5)', 1, 5, 3)
package_weight = st.slider('Package Weight (kg)', 0.0, 100.0, 10.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', 5.0, 25.0, 15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', 0, 120, 60)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([[delivery_distance, traffic_congestion, weather_condition,
                            delivery_slot, driver_experience, num_stops,
                            vehicle_age, road_condition_score, package_weight,
                            fuel_efficiency, warehouse_processing_time]],
                            columns=feature_names)

# Prediction button
if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader('Prediction Results:')
    if prediction == 1:
        st.error('**Delivery is likely to be delayed!**')
    else:
        st.success('**Delivery is likely to be on time.**')
    
    st.write(f"Probability of Delay: {prediction_proba[1]:.2f}")
    st.write(f"Probability of On-Time: {prediction_proba[0]:.2f}")

# Optional: Display the input data for verification
# st.subheader('Input Data:')
# st.write(input_data)
