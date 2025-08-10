
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        float(data['experience']),
        education_map[data['education']],
        float(data['evaluation'])
    ]
    prediction = model.predict([features])[0]
    return jsonify({'performance': prediction})

if __name__ == '__main__':
    app.run(port=5000)
