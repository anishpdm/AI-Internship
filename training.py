# training.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
from tensorflow.keras import Input


# Step 1: Load dataset
df = pd.read_csv("big.csv")
df.columns = ['age', 'gender', 'body_type', 'height', 'weight']

# Step 2: Encode categorical features
gender_encoder = LabelEncoder()
body_encoder = LabelEncoder()
df['gender_encoded'] = gender_encoder.fit_transform(df['gender'])
df['body_type_encoded'] = body_encoder.fit_transform(df['body_type'])

# Step 3: Prepare features and labels
X = df[['age', 'gender_encoded', 'body_type_encoded', 'height']]
Y = df['weight']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

# Step 4: Build model

model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=200, verbose=0)

# Step 5: Save model and encoders
model.save("weight_model.keras")

joblib.dump(gender_encoder, "gender_encoder.pkl")
joblib.dump(body_encoder, "body_encoder.pkl")

print("Training Completed")