from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')

app = Flask(__name__)
cors=CORS(app)
# Load the Engine Health Analysis Model and Preprocessor
with open('rf_classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
Engine=pd.read_csv('Engine_Health_Dataset.csv')

def predict(Engine_rpm,Lub_oil_pressure,Fuel_pressure,Coolant_pressure,lub_oil_temp,Coolant_temp):
    # Prepare features array
    features = np.array([[Engine_rpm,Lub_oil_pressure,Fuel_pressure,Coolant_pressure,lub_oil_temp,Coolant_temp]],dtype = 'object')

    # transformed featured
    transformed_features = preprocessor.transform(features)

    # predict by model
    result = clf.predict(transformed_features).reshape(1, -1)

    return result[0]

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def analyze_engine_health():
    if request.method == 'POST':
        Engine_rpm = int(request.form.get('Engine_rpm'))
        Lub_oil_pressure = float(request.form.get('Lub_oil_pressure'))
        Fuel_pressure = float(request.form.get('Fuel_pressure'))
        Coolant_pressure = float(request.form.get('Coolant_pressure'))
        lub_oil_temp =float(request.form.get('lub_oil_temp'))
        Coolant_temp = float(request.form.get('Coolant_temp'))

        prediction = predict(Engine_rpm,Lub_oil_pressure,Fuel_pressure,Coolant_pressure,lub_oil_temp,Coolant_temp)

        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)