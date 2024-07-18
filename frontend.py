import streamlit as st
import pickle
import numpy as np

# Load the model
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define a function to make predictions
def predict(input_features):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

# Streamlit UI
st.title('Calories Burnt Predictor')

# Input features
st.header('Input Features')
feature1 = st.number_input('Gender[Female : 0 , Male : 1]: ', value=1)
feature2 = st.number_input('Age:', value=1)
feature3 = st.number_input('Height: ', value=1.0)
feature4 = st.number_input('Weight: ', value=1.0)
feature5 = st.number_input('Duration: ', value=1.0)
feature6 = st.number_input('Heart rate: ', value=1.0)
feature7 = st.number_input('Body temperature(C): ', value=1.0)

# Prediction
if st.button('Predict'):
    input_features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7]
    prediction = predict(input_features)
    st.success(f'Predicted Output: {prediction}')
