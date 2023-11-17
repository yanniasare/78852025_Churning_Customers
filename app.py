import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle as pkl
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the saved model
best_model = tf.keras.models.load_model('model.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    encoder = pkl.load(file)

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    sc = pkl.load(file)

# Streamlit app
def main():
    st.title('Customer Churn Prediction App')

    # Sidebar for user input
    st.sidebar.header("User Input")

    # Assuming the order of features in input_data matches the order of columns in this list
    feature_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                       'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                       'MonthlyCharges', 'TotalCharges']

    user_input = []

    for column in feature_columns:
        user_input.append(st.sidebar.text_input(column, 0))

    # Preprocess user input
    input_data = [int(value) if i not in [17, 18] else float(value) for i, value in enumerate(user_input)]
    input_scaled = preprocess_input(input_data)

    # Make predictions
    if st.button("Predict"):
        prediction = best_model.predict(input_scaled)
        churn_prob = prediction[0][0]
        churn_confidence = churn_prob * 100  # Convert probability to percentage

        st.subheader("Prediction:")
        if churn_prob > 0.5:
            st.write(
                f"The customer is likely to churn with a probability of {churn_prob:.2f} and a confidence of {churn_confidence:.2f}%.")
        else:
            st.write(
                f"The customer is not likely to churn with a probability of {1 - churn_prob:.2f} and a confidence of {100 - churn_confidence:.2f}%.")

# Function to preprocess input data
def preprocess_input(data):
    # Convert the list to a DataFrame
    input_df = pd.DataFrame([data], columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                                              'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                                              'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                                              'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                                              'MonthlyCharges', 'TotalCharges'])

    # Assuming 'sc' is the StandardScaler applied to the training data
    scaled_data = sc.transform(input_df)
    return scaled_data

# Run the Streamlit app
if __name__ == '__main__':
    main()
