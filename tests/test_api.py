
import importlib
import sys
from unittest.mock import Mock

import joblib
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def ambiente(monkeypatch):
    modelo = Mock()
    modelo.predict.return_value = [0]
    modelo.predict_proba.return_value = [[0.85, 0.15]]

    monkeypatch.setattr(joblib, "load", lambda caminho: modelo)

    sys.modules.pop("api.main", None)
    api = importlib.import_module("api.main")

    monkeypatch.setattr(
        api,
        "carregar_bert",
        lambda: lambda texto, **kwargs: [
            {"label": "Negativa", "score": 0.95}
        ]
    )

    yield TestClient(api.app), modelo

    sys.modules.pop("api.main", None)


def test_modelo_padrao(ambiente):
    cliente, modelo = ambiente

    resposta = cliente.post(
        "/prever",
        json={"texto": "Atendimento péssimo"}
    )

    assert resposta.status_code == 200
    assert resposta.json()["modelo"] == "logistic"
    assert resposta.json()["sentimento"] == "Negativa"


def test_classificacao_positiva(ambiente):
    cliente, modelo = ambiente

    modelo.predict.return_value = [1]
    modelo.predict_proba.return_value = [[0.10, 0.90]]

    resposta = cliente.post(
        "/prever",
        json={"texto": "Comida maravilhosa", "modelo": "logistic"}
    )

    assert resposta.status_code == 200
    assert resposta.json()["sentimento"] == "Positiva"
    assert resposta.json()["confianca"] == 0.90


def test_classificacao_bert(ambiente):
    cliente, modelo = ambiente

    resposta = cliente.post(
        "/prever",
        json={"texto": "Pedido atrasado", "modelo": "bert"}
    )

    assert resposta.status_code == 200
    assert resposta.json()["modelo"] == "bert"
    assert resposta.json()["sentimento"] == "Negativa"
    assert resposta.json()["confianca"] == 0.95


def test_texto_vazio(ambiente):
    cliente, modelo = ambiente

    resposta = cliente.post(
        "/prever",
        json={"texto": "   "}
    )

    assert resposta.status_code == 422


def test_modelo_invalido(ambiente):
    cliente, modelo = ambiente

    resposta = cliente.post(
        "/prever",
        json={"texto": "Comida boa", "modelo": "invalido"}
    )

    assert resposta.status_code == 422


def test_texto_obrigatorio(ambiente):
    cliente, modelo = ambiente

    resposta = cliente.post(
        "/prever",
        json={"modelo": "logistic"}
    )

    assert resposta.status_code == 422