import pickle
import flask
import numpy as np
from flask import Flask, render_template, request
import joblib
import sklearn.svm
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the machine learning model
model = joblib.load('diabetes_prediction_model.joblib')

@app.route("/")
def index():
    form = """
    <form action="/predict" method="post">
        <input type="number" name="Pregnancies" placeholder="Number of pregnancies">
        <input type="number" name="glucose" placeholder="Glucose level">
        <input type="number" name="blood_pressure" placeholder="Blood pressure">
        <input type="number" name="skin_thickness" placeholder="Skin thickness">
        <input type="number" name="insulin" placeholder="Insulin level">
        <input type="number" name="bmi" placeholder="BMI">
        <input type="number" name="dpf" placeholder="DiabetesPedigreeFunction">
        <input type="number" name="age" placeholder="Age">
        <input type="submit" value="Predict">
    </form>
    """
    return form

@app.route("/predict", methods=["POST"])
def predict():
    # Get the user input data
    pregnancies = flask.request.form.get("pregnancies")
    glucose = flask.request.form.get("glucose")
    blood_pressure = flask.request.form.get("blood_pressure")
    skin_thickness = flask.request.form.get("skin_thickness")
    insulin = flask.request.form.get("insulin")
    bmi = flask.request.form.get("bmi")
    dpf = flask.request.form.get("dpf")
    age = flask.request.form.get("age")

    # Standardize the input data
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age])
    scaler = StandardScaler()
    scaler.fit(input_data)
    scaled_data = scaler.transform(input_data)

    # Reshape the input data
    scaled_data = scaled_data.reshape(1, -1)

    # Predict whether the user is diabetic
    prediction = model.predict_proba(scaled_data)[0][1]

    # Return the prediction
    return "The probability of you being diabetic is {}%".format(round(prediction * 100, 2))

if __name__ == "__main__":
    app.run(debug=True)
