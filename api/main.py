import joblib
from functools import lru_cache
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

app = FastAPI(title="FoodReviewAI", version="2.0")

modelo_ml = joblib.load("models/foodreview_model.pkl")

class Avaliacao(BaseModel):
    texto: str
    modelo: Literal["logistic", "bert"] = "logistic"

@lru_cache(maxsize=1)
def carregar_bert():
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model="felipermtrindade/FoodReviewAI-BERTimbau",
        device="cpu"
    )

@app.post("/prever")
def prever(avaliacao: Avaliacao):
    if avaliacao.modelo == "bert":
        classificador = carregar_bert()

        resultado = classificador(
            avaliacao.texto,
            truncation=True,
            max_length=128
        )[0]

        sentimento = resultado["label"]
        confianca = resultado["score"]

    else:
        resultado = int(modelo_ml.predict([avaliacao.texto])[0])
        probabilidades = modelo_ml.predict_proba([avaliacao.texto])[0]

        sentimento = "Positiva" if resultado == 1 else "Negativa"
        confianca = probabilidades[resultado]

    return {
        "modelo": avaliacao.modelo,
        "sentimento": sentimento,
        "confianca": round(float(confianca), 4)
    }