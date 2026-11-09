# 🍽️ FoodReviewAI

### Análise de sentimentos de avaliações de restaurantes com Machine Learning e Transformers

O **FoodReviewAI** é um projeto de Engenharia de Inteligência Artificial desenvolvido para classificar avaliações de restaurantes em português como positivas ou negativas.

O projeto apresenta duas abordagens de Processamento de Linguagem Natural (NLP): um modelo tradicional utilizando **TF-IDF e Logistic Regression** e um modelo **BERTimbau**, ajustado por fine-tuning.

A solução contempla preparação de dados reais, treinamento, avaliação, comparação entre modelos, API REST e uma aplicação web disponível publicamente.

### 🔗 Links do projeto

- **[Experimentar o FoodReviewAI](https://foodreviewai.streamlit.app/)**
- [Modelo Logistic Regression - Hugging Face](https://huggingface.co/felipermtrindade/FoodReviewAI-Sentiment)
- [Modelo BERTimbau - Hugging Face](https://huggingface.co/felipermtrindade/FoodReviewAI-BERTimbau)

## Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Desenvolvimento principal |
| Pandas | Preparação e análise de dados |
| scikit-learn | TF-IDF, Logistic Regression e métricas |
| PyTorch | Treinamento do Transformer |
| Hugging Face Transformers | Fine-tuning e inferência do BERTimbau |
| Hugging Face Hub | Publicação e distribuição dos modelos |
| FastAPI | API REST para classificação |
| Streamlit | Interface web interativa |
| Google Colab | Treinamento do BERTimbau utilizando GPU |

## Dataset e preparação dos dados

Os dados utilizados são provenientes do [StarRest](https://data.mendeley.com/datasets/r5th4s7vpv/2), conjunto de avaliações de restaurantes brasileiros.

O dataset original possui 37.642 registros.

Durante a preparação, foram aplicados os seguintes critérios:

- Seleção de comentários em português.
- Utilização das plataformas identificadas pelos IDs 1, 2 e 3.
- Exclusão de avaliações com nota 3.
- Remoção de textos vazios ou duplicados.
- Classificação de notas 1 e 2 como negativas e notas 4 e 5 como positivas.

Após o tratamento, o dataset ficou com **16.293 avaliações**.

| Sentimento | Quantidade | Proporção |
|---|---:|---:|
| Positivo | 15.207 | 93,3% |
| Negativo | 1.086 | 6,7% |
| **Total** | **16.293** | **100%** |

O desequilíbrio entre as classes foi considerado durante o treinamento e a avaliação dos modelos.

Os rótulos de sentimento foram derivados das notas atribuídas pelos usuários, e não de uma classificação manual dos textos.

## Modelos desenvolvidos

### 1. Logistic Regression + TF-IDF

O primeiro modelo foi desenvolvido como referência inicial (*baseline*), utilizando técnicas tradicionais de Machine Learning.

**Configuração:**

- TF-IDF com unigramas e bigramas.
- Limite de 10.000 atributos.
- Logistic Regression com `class_weight="balanced"`.
- Divisão estratificada de 80% para treinamento e 20% para teste.
- `random_state=42`.

O modelo completo, incluindo o processamento dos textos, foi salvo utilizando Joblib.

Essa abordagem apresenta baixo consumo de recursos e permite realizar previsões rapidamente.

### 2. BERTimbau - Fine-tuning

O segundo modelo utiliza o [BERTimbau Base](https://huggingface.co/neuralmind/bert-base-portuguese-cased), um Transformer pré-treinado em português brasileiro.

O modelo foi ajustado para classificação binária de sentimentos utilizando PyTorch e Hugging Face Transformers.

**Configuração do treinamento:**

| Parâmetro | Valor |
|---|---|
| Modelo base | BERTimbau Base |
| Épocas | 2 |
| Learning rate | 2e-5 |
| Batch size de treinamento | 16 |
| Batch size de avaliação | 32 |
| Comprimento máximo | 128 tokens |
| Balanceamento | Weighted Cross-Entropy |
| Métrica para seleção | F1-score da classe negativa |
| Hardware | NVIDIA Tesla T4 |

O treinamento foi realizado no Google Colab.

Foram utilizados 11.730 exemplos para treinamento, 1.304 para validação e 3.259 para teste.

A seleção do melhor checkpoint considerou o F1-score da classe negativa no conjunto de validação.

O processo completo está documentado em:

`notebooks/treinamento_bertimbau.ipynb`

## Resultados e comparação

Ambos os modelos foram avaliados no mesmo conjunto de teste, contendo 3.259 avaliações que não participaram do treinamento.

| Métrica | Logistic Regression | BERTimbau |
|---|---:|---:|
| Acurácia | 95,58% | **97,64%** |
| Precisão negativa | 0,61 | **0,80** |
| Recall negativo | **0,91** | 0,86 |
| F1-score negativo | 0,73 | **0,83** |
| F1-score positivo | 0,98 | **0,99** |

### Matrizes de confusão

**Logistic Regression**

| | Prevista negativa | Prevista positiva |
|---|---:|---:|
| Real negativa | 198 | 19 |
| Real positiva | 125 | 2.917 |

**BERTimbau**

| | Prevista negativa | Prevista positiva |
|---|---:|---:|
| Real negativa | 186 | 31 |
| Real positiva | 46 | 2.996 |

### Análise dos resultados

O BERTimbau apresentou melhor acurácia, precisão e F1-score na identificação de avaliações negativas.

Entretanto, o Logistic Regression apresentou recall superior nessa classe, identificando 198 das 217 avaliações negativas, enquanto o BERTimbau identificou 186.

Isso demonstra que a escolha de um modelo depende das necessidades da aplicação.

Em um sistema voltado para identificar o maior número possível de clientes insatisfeitos, o recall negativo pode ser especialmente importante.

Já em aplicações nas quais falsos alarmes representam um custo maior, a precisão pode receber mais atenção.

A comparação evidencia a importância de avaliar múltiplas métricas, principalmente em conjuntos de dados desbalanceados.

## API REST - FastAPI

O projeto disponibiliza uma API REST para integração dos modelos com outras aplicações.

A API permite selecionar o modelo responsável pela classificação.

**Endpoint:** `POST /prever`

Exemplo de requisição:

```json
{
  "texto": "A comida chegou fria e o atendimento foi péssimo.",
  "modelo": "bert"
}
```

Exemplo de resposta obtida nos testes:

```json
{
  "modelo": "bert",
  "sentimento": "Negativa",
  "confianca": 0.999
}
```

O parâmetro `modelo` aceita os valores `logistic` e `bert`. Quando não informado, o modelo padrão é Logistic Regression.

O campo `confianca` representa a probabilidade estimada pelo modelo para a classe prevista. Esse valor não deve ser interpretado como uma garantia de acerto.

## Interface Web - Streamlit

O FoodReviewAI possui uma interface interativa que permite:

- Digitar avaliações de restaurantes em português.
- Selecionar entre Logistic Regression e BERTimbau.
- Visualizar a classificação positiva ou negativa.
- Consultar a confiança estimada pelo modelo.

A aplicação está publicada no Streamlit Community Cloud.

**[Acessar demonstração pública](https://foodreviewai.streamlit.app/)**

A versão publicada carrega os modelos diretamente do Hugging Face, sem precisar executar a API separadamente.

## Como executar localmente

### 1. Clonar o repositório

```bash
git clone https://github.com/felipemtrindade/FoodReviewAI.git
cd FoodReviewAI
```

### 2. Criar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

No Windows PowerShell, a ativação é feita com `.venv\Scripts\Activate.ps1`.

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

Recomenda-se utilizar Python 3.12.

### 4. Preparar os dados e treinar o modelo tradicional

Baixe o dataset StarRest e coloque o arquivo original em `data/comentarios.csv`.

Execute:

```bash
python src/preparar_dados.py
python src/treinar_modelo.py
```

O treinamento criará o arquivo:

`models/foodreview_model.pkl`

O BERTimbau não precisa ser treinado novamente para utilização, pois seus pesos ajustados estão disponíveis no Hugging Face.

### 5. Executar a API

```bash
uvicorn api.main:app --reload
```

Documentação interativa:

http://localhost:8000/docs

### 6. Executar a interface local

Com a API em execução, abra outro terminal e execute:

```bash
streamlit run app/streamlit_app.py
```

Acesse:

http://localhost:8501

A interface local utiliza a API FastAPI, enquanto a versão pública no Streamlit Cloud utiliza os modelos diretamente.

## Limitações

O FoodReviewAI ainda apresenta limitações importantes:

- O dataset contém aproximadamente 93% de avaliações positivas.
- Os rótulos foram derivados das notas, podendo existir diferenças entre a nota e o sentimento expresso no texto.
- A classificação é binária e pode apresentar dificuldades com avaliações neutras ou sentimentos mistos.
- O BERTimbau limita as entradas a 128 tokens, truncando textos maiores.
- Os resultados foram obtidos em um conjunto de teste separado, mas proveniente da mesma fonte de dados utilizada no treinamento.
- Os modelos ainda não foram avaliados com um dataset externo independente.
- As probabilidades apresentadas não passaram por calibração específica.
- O BERTimbau exige mais memória e processamento que o Logistic Regression.

## Melhorias planejadas

- Implementar testes automatizados.
- Realizar uma análise detalhada dos erros de classificação.
- Avaliar os modelos em outros datasets.
- Comparar tempo de inferência e consumo de memória.
- Melhorar o tratamento de avaliações ambíguas.
- Adicionar uma configuração Docker para execução reproduzível.

## Referências

- [StarRest - Mendeley Data](https://data.mendeley.com/datasets/r5th4s7vpv/2)
- [BERTimbau - NeuralMind](https://huggingface.co/neuralmind/bert-base-portuguese-cased)
- [scikit-learn](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)