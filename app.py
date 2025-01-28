from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask_cors import CORS


# Inizializza Flask app
app = Flask(__name__)
CORS(app)  # Rimuovi qualsiasi restrizione alle origini



# Carica il modello salvato
model = load_model('sentiment_analysis_model.h5')

# Funzione per fare previsioni
def predict(input_data):
    # Preprocessa il dato di input (dipende dal tuo dataset e modello)
    input_array = np.array(input_data).reshape(1, -1)  # Modifica questo per adattarlo
    prediction = model.predict(input_array)
    return prediction[0][0]  # Cambia questa parte per adattarla alla tua previsione

@app.route("/")
def home():
    return """
    <html>
        <head>
            <style>
                body {
                    background-color: white; /* Cambia il colore a quello che preferisci */
                    color: black;
                    font-family: Arial, sans-serif;
                    text-align: center;
                    margin-top: 20%;
                }
            </style>
        </head>
        <body>
            <h1>Server is running</h1>
        </body>
    </html>
    """


# Route per fare previsioni
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        input_data = data['input']
        result = predict(input_data)
        response = jsonify({'prediction': result})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Consente tutte le origini
        return response
    except Exception as e:
        response = jsonify({'error': str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Consente tutte le origini
        return response, 400

@app.route('/predict', methods=['OPTIONS'])
def options():
    response = jsonify({'message': 'Options allowed'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

