# -*- coding: utf-8 -*-
"""78852025_Churning_Customers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FIDDS0LFhtwo3SY0svGZZUGqXBkZ6XZ2
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import pickle as pkl
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/CustomerChurn_dataset.csv')

df.head()

df.info()

df["Churn"].value_counts()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cols = ['gender','SeniorCitizen',"Partner","Dependents"]
numerical = cols

plt.figure(figsize=(20,4))

for i, col in enumerate(numerical):
    ax = plt.subplot(1, len(numerical), i+1)
    sns.countplot(x=str(col), data=df)
    ax.set_title(f"{col}")

sns.boxplot(x='Churn', y='MonthlyCharges', data=df)

cols = ['InternetService',"TechSupport","OnlineBackup","Contract"]

plt.figure(figsize=(14,4))

for i, col in enumerate(cols):
    ax = plt.subplot(1, len(cols), i+1)
    sns.countplot(x ="Churn", hue = str(col), data = df)
    ax.set_title(f"{col}")

columns_to_drop = ['customerID']
df = df.drop(columns_to_drop, axis=1)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encoder.fit_transform(df[column])

X = df.drop('Churn', axis=1)
y = df['Churn']

sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Create a DataFrame from the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_df

rf = RandomForestClassifier(random_state=46)
rf.fit(X_scaled_df, y)

feature_importance_rf = pd.Series(rf.feature_importances_, index=X.columns)
selected_features_rf = feature_importance_rf.nlargest(15).index
X_selected_rf = X[selected_features_rf]

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
input_layer = Input(shape=(X_train.shape[1],))
hidden_layer_1 = Dense(32, activation='relu')(input_layer)
hidden_layer_2 = Dense(24, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(12, activation='relu')(hidden_layer_2)
output_layer = Dense(1, activation='sigmoid')(hidden_layer_3)

model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

_, accuracy = model.evaluate(X_train, y_train)
accuracy*100
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy*100:.4f}')

!pip install keras-tuner

import keras_tuner
from tensorflow import keras

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Tune the number of hidden layers and units
    for i in range(hp.Int('num_hidden_layers', min_value=1, max_value=4)):
        model.add(keras.layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=96, step=32),
                             activation=hp.Choice(f'activation_{i}', values=['relu', 'tanh'])))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # Tune the learning rate
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return model

build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.Hyperband(
  hypermodel=build_model,
  objective='val_accuracy',
  max_epochs=100,
  factor=3,
  directory='tuning_dir',
  project_name='samples')

tuner.search(X_train, y_train, epochs=30 ,validation_data=(X_test, y_test))

tuner.search_space_summary()

tuner.results_summary()

model = tf.keras.Model(...)
checkpoint = tf.train.Checkpoint(model)

# Save a checkpoint to /tmp/training_checkpoints-{save_counter}. Every time
# checkpoint.save is called, the save counter is increased.
save_path = checkpoint.save('/tmp/training_checkpoints')

# Restore the checkpointed values to the `model` object.
checkpoint.restore(save_path)

best_model = tuner.get_best_models(num_models=2)[0]

best_model.summary()

test_accuracy = best_model.evaluate(X_test, y_test)[1]
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = best_model.predict(X_test)
y_pred_binary = (y_pred>0.5).astype(int)


# Calculate AUC
accuracy = accuracy_score(y_test,y_pred_binary )
auc_score = roc_auc_score(y_test,y_pred)
print("Initial Model Accuracy:", accuracy)
print("Initial Model Auc Score:", auc_score)

best_model.save('model.h5')

with open('/content/drive/MyDrive/label_encoder.pkl', 'wb') as file:
    pkl.dump(encoder, file)

with open('scaler.pkl','wb') as file:
  pkl.dump(sc,file)

with open('label_encoder.pkl', 'wb') as file:
    pkl.dump(encoder, file)

y.info()

X.info()

X_selected_rf.info()

X_selected_rf

