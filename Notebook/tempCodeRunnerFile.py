import flask
import pickle
import numpy as np
from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler

app = flask.Flask(__name__)

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

# Predict whether the user is diabetic
    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    # standardize the input data
    # Assuming you have a dataset 'X' that you want to scale
    X = input_data

    # Create a StandardScaler instance
    scaler = StandardScaler()

    # Fit the StandardScaler to your data
    scaler.fit(X)

    # Transform your data using the fitted scaler
    scaled_X = scaler.transform(X)
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(scaled_X)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)
    print(prediction)
    # Return the prediction
    if prediction == 1:
        return "You are diabetic."
    else:
        return "You are not diabetic."

if __name__ == "__main__":
    app.run(debug=True)
import requests

# Assuming you have some data in 'input_data'
data = {'data': input_data}

# Send a POST request to the Flask app running in Colab
response = requests.post('http://localhost:5000/predict', json=data)

# Get the predictions from the response
predictions = response.json()['predictions']
print(predictions)