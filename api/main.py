import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FoodReviewAI")

modelo = joblib.load("models/foodreview_model.pkl")

class Avaliacao(BaseModel):
    texto: str

@app.post("/prever")
def prever(avaliacao: Avaliacao):
    resultado = int(modelo.predict([avaliacao.texto])[0])
    probabilidades = modelo.predict_proba([avaliacao.texto])[0]

    sentimento = "Positiva" if resultado == 1 else "Negativa"

    return {
        "sentimento": sentimento,
        "confianca": round(float(probabilidades[resultado]), 4)
    }