import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Collect user inputs for the features
pclass = st.selectbox('Passenger Class', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 80, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=8, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=6, value=0)
fare = st.slider('Fare', 0.0, 500.0, 50.0)
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Encode categorical variables (Sex and Embarked)
label_encoder_sex = LabelEncoder()
label_encoder_sex.fit(['male', 'female'])
sex_encoded = label_encoder_sex.transform([sex])[0]

label_encoder_embarked = LabelEncoder()
label_encoder_embarked.fit(['C', 'Q', 'S'])
embarked_encoded = label_encoder_embarked.transform([embarked])[0]

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked_encoded]
})

# Scale the input data using the pre-fitted scaler
input_data_scaled = scaler.transform(input_data)

# Make prediction using the pre-trained model
prediction = model.predict(input_data_scaled)

# Display the result
if prediction[0] == 1:
    st.success('The passenger would have survived!')
else:
    st.error('The passenger would not have survived.')
