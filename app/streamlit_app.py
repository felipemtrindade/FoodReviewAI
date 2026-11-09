
import streamlit as st
import requests

st.set_page_config(page_title="FoodReviewAI", page_icon="🍽️")

st.title("FoodReviewAI")
st.write("Análise de sentimentos de avaliações de restaurantes utilizando Machine Learning.")

texto = st.text_area(
    "Digite uma avaliação:",
    placeholder="Exemplo: A comida estava deliciosa e o atendimento foi excelente!"
)

modelo = st.selectbox(
    "Selecione o modelo:",
    ["Logistic Regression", "BERTimbau"]
)

if st.button("Analisar avaliação"):
    if not texto.strip():
        st.warning("Digite uma avaliação.")
    else:
        with st.spinner("Analisando avaliação..."):
            resposta = requests.post(
                "http://localhost:8000/prever",
                json={
                    "texto": texto,
                    "modelo": "bert" if modelo == "BERTimbau" else "logistic"
                },
                timeout=180
            )

        if resposta.ok:
            resultado = resposta.json()

            st.subheader("Resultado")

            if resultado["sentimento"] == "Positiva":
                st.success("Avaliação positiva")
            else:
                st.error("Avaliação negativa")

            st.metric("Confiança estimada", f'{resultado["confianca"] * 100:.2f}%')
            st.caption(f"Modelo utilizado: {modelo}")
        else:
            st.error("Não foi possível obter a classificação.")