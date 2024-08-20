import pandas as pd
import random as rd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Função de carregar e visualizar od dados.
def carregar_dados():
    dataset_de_treino = pd.read_csv('dataset_QUINTO_TREINO.csv', sep=';')
    print(f'Dados de Treino: \n{dataset_de_treino.head()}')
    dataset_de_teste = pd.read_csv('UNIAO_DFS_TESTE.csv', sep=';')
    print(f'Dados de teste: \n {dataset_de_teste.head()}')
    return dataset_de_treino, dataset_de_teste 

# Função para pré-processar os dados.
def pre_processamento_dados(dataset_de_treino, dataset_de_teste):
    X_treino = dataset_de_treino.drop(columns='quality').values
    Y_treino = dataset_de_treino['quality'].values
    X_teste = dataset_de_teste.drop(columns='quality').values
    Y_teste = dataset_de_teste['quality'].values
    return X_treino, Y_treino, X_teste, Y_teste  

# Função de ativação.
def passo(z):
    return 1 if z >= 0 else 0

# Função para inicializar pesos e bias.
def inicializacao_pesos(X_treino):
    pesos = [rd.uniform(0, 1) for _ in range(X_treino.shape[1])]
    bias = rd.uniform(0, 1)
    return pesos, bias

# Função para treinar o Megatron "Perceptron".
def treinar_perceptron(X_treino, Y_treino, pesos, bias, taxa_de_aprendizagem, num_iter):
    for iter in range(num_iter):
        for i in range(len(X_treino)):
            z = sum(X_treino[i][j] * pesos[j] for j in range(len(pesos))) + bias
            output = passo(z)
            erro = Y_treino[i] - output
            pesos = [pesos[j] + taxa_de_aprendizagem * erro * X_treino[i][j] for j in range(len(pesos))]
            bias += taxa_de_aprendizagem * erro
    return pesos, bias

# Função para testar o Megatron "Perceptron".
def teste_perceptron(X_teste, Y_teste, pesos, bias):
    corretos = 0
    for i in range(len(X_teste)):
        z = sum(X_teste[i][j] * pesos[j] for j in range(len(pesos))) + bias
        output = passo(z)
        if output == Y_teste[i]:
            corretos += 1
    precisao = corretos / len(X_teste)
    return precisao

# Função para calcular metricas
def calcular_metricas(X_teste, Y_teste, pesos, bias):
    predicoes = []
    for i in range(len(X_teste)):
        z = sum(X_teste[i][j] * pesos[j] for j in range(len(pesos))) + bias
        predicoes.append(passo(z))
    acuracia = accuracy_score(Y_teste, predicoes)
    precisao = precision_score(Y_teste, predicoes, average='macro', zero_division=0)
    recall = recall_score(Y_teste, predicoes, average='macro', zero_division=0)
    f1 = f1_score(Y_teste, predicoes, average='macro', zero_division=0)
    return acuracia, precisao, recall, f1

# Função principal
def main():
    dataset_de_treino, dataset_de_teste = carregar_dados()
    X_teste, Y_teste, X_treino, Y_treino = pre_processamento_dados(dataset_de_treino, dataset_de_teste)
    pesos, bias = inicializacao_pesos(X_treino)
    taxa_de_aprendizagem = 0.01
    num_iter = 10000
    pesos, bias = treinar_perceptron(X_treino, Y_treino, pesos, bias, taxa_de_aprendizagem, num_iter)
    print('Pesos ajustados: ', pesos)
    print('Bias ajustado: ', bias)
    precisao = teste_perceptron(X_teste, Y_teste, pesos, bias)
    print('Precisão do Megatron: ', precisao)
    acuracia, precisao, recall, f1 = calcular_metricas(X_teste, Y_teste, pesos, bias)
    print(f'Acurácia: {acuracia}')
    print(f'Precisão: {precisao}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')

# Executando a função principal
if __name__ == "__main__":
    main()