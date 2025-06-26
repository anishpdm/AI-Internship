# api.py
from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = load_model("weight_model.keras", compile=False)
gender_encoder = joblib.load("gender_encoder.pkl")
body_encoder = joblib.load("body_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        # Convert to proper types
        age = int(data['age'])
        height = float(data['height'])
        gender = data['gender']
        body_type = data['body_type']

        # Encode categorical
        gender_val = gender_encoder.transform([gender])[0]
        body_val = body_encoder.transform([body_type])[0]

        # Make prediction
        input_array = np.array([[age, gender_val, body_val, height]])
        predicted_weight = model.predict(input_array)[0][0]

        # Convert float32 -> float to make it JSON serializable
        return jsonify({"predicted_weight": round(float(predicted_weight), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
