# 78852025_Churning_Customers

https://drive.google.com/file/d/1qGhoj4gUoK7qjKSeHqKtv_DoM31Vh7e2/view?usp=sharing

Customer Churn Predictor

Overview

This project predicts if customers will leave using a model and turns it into a user-friendly web app with Streamlit. We use TensorFlow and Keras to build the model, and Streamlit for the app. The model estimates the chance of a customer leaving based on various details.

Project Steps

1. Data Setup
Load data from '/content/drive/My Drive/CustomerChurn_dataset.csv'.
Study data patterns through EDA (Exploratory Data Analysis).
Drop less important columns like 'customerID'.
Fill missing values in 'TotalCharges' with the average.
Convert categorical columns using Label Encoding.

2. Random Forest Model
Train a Random Forest Classifier to find crucial features.
Visualize feature importance, selecting the top 15.

3. Neural Network Model
Build a Neural Network using TensorFlow and Keras.
Use Keras Tuner for hyperparameter tuning.
Choose the best model based on validation accuracy.

4. Model Saving
Save the best Neural Network model as 'model.h5'.
Save Label Encoder and Standard Scaler as 'label_encoder.pkl' and 'scaler.pkl'.

5. Streamlit App
Develop a Streamlit app for user input.
Apply preprocessing to user data and predict using the model.
Display predictions and confidence scores.

Usage

Run the Streamlit app using:

streamlit run your_app_file.py
