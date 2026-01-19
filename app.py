from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and helpers
model = joblib.load("model/churn_model.pkl")
features = joblib.load("model/features.pkl")
encoders = joblib.load("model/encoders.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = []

    for feature in features:
        value = float(request.form[feature])
        input_data.append(value)

    prediction = model.predict([input_data])[0]

    result = "Customer Will Churn ❌" if prediction == 1 else "Customer Will Not Churn ✅"
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)