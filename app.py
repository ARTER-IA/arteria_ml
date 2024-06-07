from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo
model = joblib.load("model.pkl")


@app.route("/")
def home():
    return "Modelo de predicción de enfermedad arterial coronaria"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Crear un DataFrame con los datos recibidos
    df = pd.DataFrame(data, index=[0])

    # Realizar la predicción
    prediction_proba = model.predict_proba(df)[:, 1]
    prediction = model.predict(df)

    response = {
        "predicted_class": int(prediction[0]),
        "prediction_probability": float(prediction_proba[0]),
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
