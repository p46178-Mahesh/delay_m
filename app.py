%%writefile app.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('delivery_delay_model.pkl')

st.title('Delivery Delay Prediction')
st.write('Enter the features below to predict if there will be a delivery delay.')

# Create input widgets for each feature
delivery_distance = st.number_input('Delivery Distance', min_value=0.0, value=20.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1-3)', min_value=1, max_value=3, value=2)
delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=5)
num_stops = st.number_input('Number of Stops', min_value=0, value=2)
vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=3)
road_condition_score = st.slider('Road Condition Score (1-4)', min_value=1, max_value=4, value=3)
package_weight = st.number_input('Package Weight', min_value=0.0, value=10.0)
fuel_efficiency = st.number_input('Fuel Efficiency', min_value=0.0, value=15.0)
warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60)

# Create a DataFrame from the input values
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.error(f'Likely to have a **Delay** (Probability: {prediction_proba[0][1]:.2f})')
    else:
        st.success(f'Likely **No Delay** (Probability: {prediction_proba[0][0]:.2f})')

st.write("\n--- Make sure to upload the 'delivery_delay_model.pkl' file to the same directory as this app.py before running.")
