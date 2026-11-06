# FoodReviewAI

Classificador de avaliações de restaurantes em português. O modelo recebe um comentário e indica se o sentimento é positivo ou negativo. As previsões podem ser consultadas por uma API feita com FastAPI.

O projeto reúne a preparação dos dados, o treinamento e a API em Python, usando Pandas, scikit-learn e Joblib.

## Como executar

Execute os comandos na raiz do projeto. Primeiro, instale as dependências:

```bash
pip install -r requirements.txt
```

Coloque o arquivo original do dataset em `data/comentarios.csv`. Depois, prepare os dados e treine o modelo:

```bash
python src/preparar_dados.py
python src/treinar_modelo.py
```

O treinamento salva o modelo em `models/foodreview_model.pkl` e exibe as métricas no terminal. Com o modelo salvo, inicie a API:

```bash
uvicorn api.main:app --reload
```

A documentação interativa fica em [localhost:8000/docs](http://localhost:8000/docs).

## Uso da API

Envie o texto da avaliação para `POST /prever`:

```json
{
  "texto": "A comida chegou fria e o atendimento foi péssimo."
}
```

Exemplo de resposta:

```json
{
  "sentimento": "Negativa",
  "confianca": 0.8586
}
```

O campo `confianca` contém a probabilidade atribuída pelo modelo à classe prevista.

## Dados e treinamento

As avaliações vêm do [StarRest](https://data.mendeley.com/datasets/r5th4s7vpv/2), um dataset de avaliações de restaurantes brasileiros.

Na preparação, são mantidos os comentários em português das plataformas com IDs 1, 2 e 3. Notas 1 e 2 são consideradas negativas; notas 4 e 5, positivas. Avaliações com nota 3, registros sem texto e textos duplicados são descartados.

Após esse tratamento, o conjunto utilizado ficou com 16.293 avaliações:

- 15.207 positivas;
- 1.086 negativas.

O modelo usa TF-IDF com palavras isoladas e pares de palavras, limitado a 10.000 atributos, seguido de regressão logística. O parâmetro `class_weight="balanced"` ajusta os pesos das classes para lidar com a diferença de quantidade entre avaliações positivas e negativas.

A divisão dos dados reserva 80% para treino e 20% para teste, mantendo a proporção das classes e usando `random_state=42`.

## Resultados

Resultados registrados no conjunto de teste:

| Métrica | Resultado |
| --- | --- |
| Acurácia | 95,6% |
| Recall da classe negativa | 91% |
| F1-score da classe negativa | 0,73 |
| F1-score da classe positiva | 0,98 |

Das 217 avaliações negativas no teste, 198 foram identificadas corretamente.

Como há muito mais avaliações positivas, a acurácia sozinha não descreve bem o desempenho. O recall e o F1-score da classe negativa ajudam a avaliar como o modelo se sai com a classe menos frequente.

## Comparação entre modelos

Foram desenvolvidos e avaliados dois modelos de classificação de sentimentos:

1. **Logistic Regression + TF-IDF:** modelo tradicional de Machine Learning, utilizado como referência inicial.
2. **BERTimbau:** modelo Transformer pré-treinado em português brasileiro, ajustado com fine-tuning para classificação de avaliações.

Ambos foram avaliados no mesmo conjunto de teste, contendo 3.259 avaliações.

| Métrica | Logistic Regression | BERTimbau |
|---|---:|---:|
| Acurácia | 95,58% | 97,64% |
| Precisão negativa | 61% | 80% |
| Recall negativo | 91% | 86% |
| F1-score negativo | 0,73 | 0,83 |
| F1-score positivo | 0,98 | 0,99 |

O BERTimbau apresentou melhor desempenho geral e maior F1-score na classe negativa.

Entretanto, o Logistic Regression obteve recall superior para avaliações negativas, identificando uma proporção maior dos comentários classificados como negativos.

Esses resultados demonstram a importância de avaliar diferentes métricas antes de escolher um modelo para uma aplicação real.

### Modelos publicados

- [FoodReviewAI - Logistic Regression](https://huggingface.co/felipermtrindade/FoodReviewAI-Sentiment)
- [FoodReviewAI - BERTimbau](https://huggingface.co/felipermtrindade/FoodReviewAI-BERTimbau)

### Notebook de treinamento

O processo de fine-tuning do BERTimbau, incluindo preparação dos dados, tokenização, treinamento e avaliação, está documentado em `notebooks/treinamento_bertimbau.ipynb`.

## Limitações e próximos passos

A classificação é binária: toda avaliação recebe uma das duas classes, mesmo quando mistura elogios e críticas. O desequilíbrio dos dados também é uma limitação do conjunto usado.

Os próximos passos previstos são analisar os erros, comparar o resultado com modelos Transformers, adicionar testes automatizados, preparar uma imagem Docker e publicar o modelo no Hugging Face.