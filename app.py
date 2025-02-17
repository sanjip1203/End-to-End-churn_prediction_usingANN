import streamlit as st 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd 
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title with customized font
st.title("Customer Churn Prediction")
st.markdown("<h2 style='color:#FF6347;'>Predict whether a customer is likely to churn or not</h2>", unsafe_allow_html=True)

# Add a custom icon or logo (optional)
# st.image("your_logo.png", width=100)

# Adding a description section to guide the user
st.markdown("""
    This model predicts whether a customer is likely to churn based on their details.
    Fill out the following fields and hit the button to get a prediction.
    """, unsafe_allow_html=True)

# Add a separator line for better UI organization
st.markdown("---")

# Create an attractive user input section with a well-defined layout
st.sidebar.header("Customer Information")
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0], key='geo')
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_, key='gender')
age = st.sidebar.slider('Age', 18, 92, 30)
balance = st.sidebar.number_input('Balance', min_value=0, step=1000)
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0, step=1000)
tenure = st.sidebar.slider("Tenure (Number of Products)", 1, 4, 2)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850)

# Add an action button to trigger the prediction
predict_button = st.sidebar.button("Predict Churn")

# User input dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [tenure],  # Assuming you are using 'tenure' as 'NumOfProducts'
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction logic after the button is pressed
if predict_button:
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]
    
    # Display prediction probability
    st.subheader(f"Churn Probability: {prediction_proba:.2f}")
    
    # Display final prediction result
    if prediction_proba > 0.5:
        st.write("<h3 style='color:#FF6347;'>The customer is likely to churn.</h3>", unsafe_allow_html=True)
    else:
        st.write("<h3 style='color:#32CD32;'>The customer is not likely to churn.</h3>", unsafe_allow_html=True)
