from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_values = [float(x) for x in request.form.values()]
    input_array = np.array(input_values).reshape(1, -1)
    std_input = scaler.transform(input_array)
    prediction = model.predict(std_input)

    result = "The person has Parkinson's Disease." if prediction[0] == 1 else "The person does not have Parkinson's Disease."
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
