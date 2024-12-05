from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    try:
        # Extract features from the request
        area = float(data['area'])
        bedrooms = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        stories = int(data['stories'])
        parking = int(data['parking'])

        # Prepare the data for prediction
        features = np.array([[area, bedrooms, bathrooms, stories, parking]])

        # Make the prediction
        prediction = model.predict(features)[0]

        # Return the result
        return jsonify({"price": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
