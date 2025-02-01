from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS
app = Flask(__name__)
allowed_origins = [
    "http://localhost:3000",  # Example of an allowed origin
    "http://127.0.0.1:5500"   #  domain
]

# Enable CORS for the app
CORS(app)

# Load the trained model
model = joblib.load('svm_model.pkl')

# Mean values for columns not provided by the user
mean_values = {
    'MonsoonIntensity': 1.5,
    'TopographyDrainage': 2.3,
    'RiverManagement': 1.1,
    'Deforestation': 0.8,
    'Urbanization': 0.6,
    'ClimateChange': 0.4,
    'DamsQuality': 0.5,
    'Siltation': 0.3,
    'AgriculturalPractices': 0.7,
    'Encroachments': 0.2,
    'IneffectiveDisasterPreparedness': 0.6,
    'DrainageSystems': 0.5,
    'CoastalVulnerability': 0.4,
    'Landslides': 0.3,
    'Watersheds': 0.5,
    'DeterioratingInfrastructure': 0.6,
    'PopulationScore': 0.7,
    'WetlandLoss': 0.5,
    'InadequatePlanning': 0.4,
    'PoliticalFactors': 0.2,
    'FloodProbability': 0.5,
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print(data)
    # Prepare input data for prediction, using means for columns not provided
    input_data = {
        'MonsoonIntensity': data.get('MonsoonIntensity', mean_values['MonsoonIntensity']),
        'TopographyDrainage': data.get('TopographyDrainage', mean_values['TopographyDrainage']),
        'RiverManagement': data.get('RiverManagement', mean_values['RiverManagement']),
        'Deforestation': data.get('Deforestation', mean_values['Deforestation']),
        'Urbanization': data.get('Urbanization', mean_values['Urbanization']),
        'ClimateChange': mean_values['ClimateChange'],
        'DamsQuality': mean_values['DamsQuality'],
        'Siltation': mean_values['Siltation'],
        'AgriculturalPractices': mean_values['AgriculturalPractices'],
        'Encroachments': mean_values['Encroachments'],
        'IneffectiveDisasterPreparedness': mean_values['IneffectiveDisasterPreparedness'],
        'DrainageSystems': mean_values['DrainageSystems'],
        'CoastalVulnerability': mean_values['CoastalVulnerability'],
        'Landslides': mean_values['Landslides'],
        'Watersheds': mean_values['Watersheds'],
        'DeterioratingInfrastructure': mean_values['DeterioratingInfrastructure'],
        'PopulationScore': mean_values['PopulationScore'],
        'WetlandLoss': mean_values['WetlandLoss'],
        'InadequatePlanning': mean_values['InadequatePlanning'],
        'PoliticalFactors': mean_values['PoliticalFactors'],
        'FloodProbability': mean_values['FloodProbability'],
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Make prediction based on input data
    prediction = model.predict(input_df)[0]


 
    

    # # Determine warning message based on prediction
    if prediction < 1:
        warning = "No flood risk."
    elif 1 <= prediction < 2:
        warning = "Chance of flood."
    elif 2 <= prediction < 3:
        warning = "Flood risk present."
    else:
        warning = "High flood risk!"
        

    # Return the results
    return jsonify({'Prediction': prediction, 'Warning': warning})


if __name__ == '__main__':
    app.run(debug=True)
