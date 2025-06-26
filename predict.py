# predicted.py
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved model and encoders
model = load_model("weight_model.keras", compile=False)
gender_encoder = joblib.load("gender_encoder.pkl")
body_encoder = joblib.load("body_encoder.pkl")

# Example input
new_input = {
    "age": 29,
    "gender": "Male",
    "body_type": "Average",
    "height": 173
}

# Encode categorical values
gender_val = gender_encoder.transform([new_input["gender"]])[0]
body_val = body_encoder.transform([new_input["body_type"]])[0]

# Prepare input for model
input_array = np.array([[new_input["age"], gender_val, body_val, new_input["height"]]])

# Predict
predicted_weight = model.predict(input_array)
print("Predicted Weight:", round(predicted_weight[0][0], 2))
