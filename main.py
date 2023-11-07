from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('diabetes.pkl')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        HbA1c = float(request.form['HbA1c'])
        glucose = float(request.form['glucose'])


        # Make a prediction using the loaded model
        prediction = model.predict([[age, HbA1c, glucose]])[0]

        prediction = int(prediction)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
