from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask_cors import CORS  # Importa CORS

# Inizializza Flask app
app = Flask(__name__)

# Aggiungi il supporto CORS
CORS(app)

# Carica il modello salvato
model = load_model('sentiment_analysis_model.h5')

# Funzione per fare previsioni
def predict(input_data):
    input_array = np.array(input_data).reshape(1, -1)  # Adatta al tuo modello
    prediction = model.predict(input_array)
    return prediction[0][0]

@app.route("/predict", methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        input_data = data['input']  # Adatta il nome della chiave
        result = predict(input_data)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
