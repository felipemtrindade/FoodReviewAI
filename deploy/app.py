
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download
from transformers import pipeline

st.set_page_config(page_title="FoodReviewAI", page_icon="🍽️")

@st.cache_resource
def carregar_logistic():
    caminho = hf_hub_download(
        repo_id="felipermtrindade/FoodReviewAI-Sentiment",
        filename="foodreview_model.pkl"
    )
    return joblib.load(caminho)

@st.cache_resource
def carregar_bert():
    return pipeline(
        "text-classification",
        model="felipermtrindade/FoodReviewAI-BERTimbau",
        device="cpu"
    )

st.title("🍽️ FoodReviewAI")
st.write("Análise de sentimentos de avaliações de restaurantes com Machine Learning.")

texto = st.text_area("Digite uma avaliação:")

modelo = st.selectbox(
    "Selecione o modelo:",
    ["Logistic Regression", "BERTimbau"]
)

if st.button("Analisar avaliação"):
    if not texto.strip():
        st.warning("Digite uma avaliação.")
    else:
        with st.spinner("Analisando..."):
            if modelo == "BERTimbau":
                resultado = carregar_bert()(
                    texto, truncation=True, max_length=128
                )[0]

                sentimento = resultado["label"]
                confianca = resultado["score"]
            else:
                classificador = carregar_logistic()
                previsao = int(classificador.predict([texto])[0])
                sentimento = "Positiva" if previsao == 1 else "Negativa"
                confianca = classificador.predict_proba([texto])[0][previsao]

        if sentimento == "Positiva":
            st.success("Avaliação positiva")
        else:
            st.error("Avaliação negativa")

        st.metric("Confiança estimada", f"{confianca * 100:.2f}%")
        st.caption(f"Modelo utilizado: {modelo}")

st.divider()
st.markdown(
    "[Código-fonte no GitHub](https://github.com/felipemtrindade/FoodReviewAI)"
)