import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

print("Modelo salvo com sucesso!")

df = pd.read_csv("data/avaliacoes_tratadas.csv")

X = df["texto"]
y = df["sentimento"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

modelo = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ("classificador", LogisticRegression(class_weight="balanced", max_iter=1000))
])

modelo.fit(X_treino, y_treino)

joblib.dump(modelo, "models/foodreview_model.pkl")
print("Modelo salvo com sucesso!")

previsoes = modelo.predict(X_teste)

print("\nRelatório de classificação:")
print(classification_report(y_teste, previsoes, target_names=["Negativa", "Positiva"]))

print("\nMatriz de confusão:")
print(confusion_matrix(y_teste, previsoes))
frases = [
    "A comida estava maravilhosa e o atendimento foi excelente!",
    "Péssima experiência, o pedido chegou frio e atrasado.",
    "Gostei da comida, mas o atendimento deixou a desejar.",
    "Nunca mais volto nesse restaurante.",
    "O preço é alto, porém a qualidade é excepcional."
]

resultados = modelo.predict(frases)
probabilidades = modelo.predict_proba(frases)

print("\nTestes com novas avaliações:")

for frase, resultado, probabilidade in zip(frases, resultados, probabilidades):
    sentimento = "Positiva" if resultado == 1 else "Negativa"
    confianca = max(probabilidade) * 100

    print(f"\nAvaliação: {frase}")
    print(f"Sentimento: {sentimento}")
    print(f"Confiança do modelo: {confianca:.2f}%")