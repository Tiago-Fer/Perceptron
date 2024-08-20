# Projeto Extra-acadêmico: Rede Neural Perceptron - Megatron

## Descrição do Projeto

Este projeto implementa um modelo de rede neural Perceptron, chamado de "Megatron", para a classificação de vinhos com base em características químicas. O objetivo principal do projeto é demonstrar o funcionamento de um Perceptron e explorar suas limitações em um problema de classificação multiclasses.

O projeto é parte de um esforço extra-acadêmico voltado para a área de ciência de dados, unindo alunos das instituições Unopar/Anhanguera. Este repositório contém o código-fonte do modelo, relatórios detalhados, e resultados das análises.

## Estrutura do Projeto

- `Megatron-V1_2.py`: Código-fonte principal que implementa o modelo Perceptron para a classificação de vinhos.
- `Relatorio sobre o Megatron -Perceptron-.docx`: Documento detalhado que discute o funcionamento do Perceptron, suas limitações, e as análises realizadas com base nos resultados obtidos.

## Funcionalidades

- **Carregamento de Dados**: O modelo carrega os dados de treino e teste de arquivos CSV.
- **Pré-processamento**: As características químicas dos vinhos são preparadas para serem utilizadas pelo modelo.
- **Treinamento do Perceptron**: O modelo é treinado usando uma função de ativação degrau, ajustando pesos e bias.
- **Avaliação de Desempenho**: Métricas como acurácia, precisão, recall e F1-score são calculadas para avaliar o modelo.
- **Análise dos Resultados**: O desempenho do modelo é analisado e discutido no relatório, destacando as limitações do Perceptron para problemas de classificação multiclasses.

## Como Executar

1. **Pré-requisitos**:
   - Python 3.x
   - Bibliotecas: `pandas`, `sklearn`
   
   Instale as bibliotecas necessárias usando:
   ```bash
   pip install pandas scikit-learn

2. Executar o Código:

    ·Para executar o modelo Perceptron, rode o script Megatron-V1_2.py:

    ```bash
    
    python Megatron-V1_2.py


Análise dos Resultados
Os resultados das execuções mostraram que o Perceptron, conforme implementado, teve dificuldades em classificar corretamente a qualidade dos vinhos. Todas as métricas de desempenho (acurácia, precisão, recall e F1-score) resultaram em zero, o que sugere que o modelo não conseguiu aprender com os dados de treino.

Para detalhes mais completos, consulte o relatório anexado (Relatorio sobre o Megatron -Perceptron-.docx), onde as limitações do modelo são discutidas e sugestões de melhorias são fornecidas.

Limitações do Perceptron
Classificação Binária: O Perceptron é ideal para problemas de classificação binária, mas enfrenta dificuldades com problemas de multiclasses, como a classificação de vinhos.
Linearidade: O modelo é limitado a fronteiras de decisão lineares, inadequado para problemas com relações complexas entre características.
Generalização: O Perceptron pode não generalizar bem em problemas complexos, como demonstrado nos resultados.
Recomendações
Para melhorar o desempenho em problemas semelhantes:

Considere usar redes neurais multicamadas (MLP) ou outros algoritmos de machine learning que possam capturar relações não lineares.
Explore técnicas de engenharia de características para melhorar a representação dos dados.
Utilize validação cruzada para avaliar melhor a capacidade de generalização do modelo.
Autores
Tiago Fernando Piveta
Orientadoras: Prof.ª. Drª Vanessa Matias Leite e Prof.ª Elisa Antolli
Licença
Este projeto é licenciado sob os termos da licença MIT. Veja o arquivo LICENSE para mais detalhes.