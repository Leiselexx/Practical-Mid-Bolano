from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes.pkl')

# Create a StandardScaler instance and fit it to the training data
scaler = StandardScaler()

# Define the actual mean and standard deviation values from your training data
mean_age = 30.0  # Replace with your actual mean for 'age'
mean_hypertension = 0.1  # Replace with your actual mean for 'hypertension'
mean_heartDisease = 0.2  # Replace with your actual mean for 'heartDisease'
mean_bmi = 25.0  # Replace with your actual mean for 'bmi'
mean_HbA1c = 5.4  # Replace with your actual mean for 'HbA1c'
mean_glucose = 100.0  # Replace with your actual mean for 'glucose'

std_age = 5.0  # Replace with your actual standard deviation for 'age'
std_hypertension = 0.2  # Replace with your actual standard deviation for 'hypertension'
std_heartDisease = 0.4  # Replace with your actual standard deviation for 'heartDisease'
std_bmi = 4.0  # Replace with your actual standard deviation for 'bmi'
std_HbA1c = 0.5  # Replace with your actual standard deviation for 'HbA1c'
std_glucose = 20.0  # Replace with your actual standard deviation for 'glucose'

mean_values = np.array([mean_age, mean_hypertension, mean_heartDisease, mean_bmi, mean_HbA1c, mean_glucose])
std_values = np.array([std_age, std_hypertension, std_heartDisease, std_bmi, std_HbA1c, std_glucose])

scaler.mean_ = mean_values
scaler.scale_ = std_values

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        hypertension = float(request.form['hypertension'])
        heartDisease = float(request.form['heartDisease'])
        bmi = float(request.form['bmi'])
        HbA1c = float(request.form['HbA1c'])
        glucose = float(request.form['glucose'])

        # Scale the input features using the scaler
        input_data = scaler.transform([[age, hypertension, heartDisease, bmi, HbA1c, glucose]])

        # Make a prediction using the loaded model
        prediction = model.predict(input_data)[0]

        # Map the prediction to labels
        prediction_label = "Non-Diabetic" if prediction == 0 else "Diabetic"

        # Return the prediction label as JSON
        return jsonify({'prediction': prediction_label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
