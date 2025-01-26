from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Inizializza Flask app
app = Flask(__name__)

# Carica il modello salvato
model = load_model('sentiment_analysis_model.h5')

# Funzione per fare previsioni
def predict(input_data):
    # Preprocessa il dato di input (dipende dal tuo dataset e modello)
    input_array = np.array(input_data).reshape(1, -1)  # Modifica questo per adattarlo
    prediction = model.predict(input_array)
    return prediction[0][0]  # Cambia questa parte per adattarla alla tua previsione

@app.route('/')
def home():
    return "Server is running!"

# Route per fare previsioni
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()  # Ricevi i dati inviati dal frontend
        input_data = data['input']  # Cambia il nome della chiave se necessario
        
        # Previsioni
        result = predict(input_data)
        
        return jsonify({'prediction': result})  # Restituisci la previsione come JSON
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

