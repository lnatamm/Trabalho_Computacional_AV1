import numpy as np
import matplotlib.pyplot as plt

plot_graphs = True

# Tarefa de Classificação

# Configuração padrão dos plots
def get_plot_configuration(file, n_figure, title):
    fig = plt.figure(n_figure)
    plot = fig.add_subplot(projection='3d')

    plot.scatter(
        file[:, 0],
        file[:, 1],
        file[:, 2],
        color='teal',
        edgecolors='k'
    )

    plot.set_xlabel("Temperatura")
    plot.set_ylabel("pH da solução")
    plot.set_zlabel("Nível de Atividade Enzimática")
    plot.set_title(title)

    return plot

#TODO AJUSTAR!!!!!
def acuracia(X, y, W, N):
    acertos = 0
    for i, x in enumerate(X):
        if np.argmax(x@W) == np.argmax(y[i]):
            acertos += 1
    return (acertos/N) * 100

def acuracia_naive(X, y, W, N):
    acertos = 0
    for i, x in enumerate(X):
        if np.argmin(x@W) == np.argmin(y[i]):
            acertos += 1
    return (acertos/N) * 100
figure_index = 1

def fdp(x, mu, sigma, sigma_inv):
    """
    Calcula a função de densidade de probabilidade de uma distribuição Gaussiana Multivariada.

    Parâmetros:
    x : ndarray
        Vetor de entrada (p dimensões).
    mu : ndarray
        Vetor de média (p dimensões).
    sigma : ndarray
        Matriz de covariância (p x p).

    Retorna:
    float
        Probabilidade P(x | y_i).
    """
    p = len(mu)  # Dimensão dos dados
    coef = 1 / ((2 * np.pi) ** (p / 2) * np.linalg.det(sigma) ** 0.5)
    exp_term = np.exp(-0.5 * (x - mu).T @ sigma_inv @ (x - mu))
    
    return coef * exp_term

# 1.

classes = [1, 2, 3, 4, 5]

EMGs = np.loadtxt("datasets\\EMGsDataset.csv", delimiter=',')
EMGs_MQO = EMGs.T

N, p = EMGs.T.shape[0], EMGs.shape[1] - 1 # Removendo y


# Todas as linhas da primeira até a segunda e todas as colunas onde a última linha é igual a cada classe respectiva.
# Isso organiza os dados para ser primeiro da classe 1, 2, 3... c
X_Gauss = np.hstack((
    EMGs[:2,EMGs[-1,:] == classes[0]],
    EMGs[:2,EMGs[-1,:] == classes[1]],
    EMGs[:2,EMGs[-1,:] == classes[2]],
    EMGs[:2,EMGs[-1,:] == classes[3]],
    EMGs[:2,EMGs[-1,:] == classes[4]],
))

y_Gauss = np.vstack((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (10000, 1)),
))

X_MQO = np.vstack((
    EMGs_MQO[EMGs_MQO[:,-1] == classes[0],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[1],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[2],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[3],:2],
    EMGs_MQO[EMGs_MQO[:,-1] == classes[4],:2],
))

X_MQO = np.hstack((
    np.ones((N, 1)), X_MQO
))

y_MQO = np.vstack((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (10000, 1)),
))

y_MQO = np.vstack((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (10000, 1)),
))

W_MQO = np.linalg.pinv(X_MQO.T@X_MQO)@X_MQO.T@y_MQO

x_novo = np.hstack((1, EMGs[np.random.randint(0, len(EMGs)), 0:2]))

y_predicao = x_novo@W_MQO

print(np.argmax(y_predicao))

rodadas = 500
particionamento = 0.8

desempenhos_MQO_tradicional = []
desempenhos_gaussiano_tradicional = []
desempenhos_gaussiano_cov_todo_conjunto = []
desempenhos_gaussiano_cov_agregada = []
desempenhos_bayes_ingenuo_naive_bayes_classifier = []
desempenhos_gaussiano_regularizado = []

for rodada in range(rodadas):
    # Indices permutados
    index = np.random.permutation(N)

    # LMQ (MQO Linear)
    X_MQO_embaralhado = X_MQO[index, :]
    y_MQO_embaralhado = y_MQO[index]

    X_treino_MQO = X_MQO_embaralhado[:int(N*particionamento),:]
    y_treino_MQO = y_MQO_embaralhado[:int(N*particionamento)]

    X_teste_MQO = X_MQO_embaralhado[int(N*particionamento):,:]
    y_teste_MQO = y_MQO_embaralhado[int(N*particionamento):]

    W_MQO_tradicional_MC = np.linalg.pinv(X_treino_MQO.T@X_treino_MQO)@X_treino_MQO.T@y_treino_MQO

    desempenhos_MQO_tradicional.append(acuracia(X_teste_MQO, y_teste_MQO, W_MQO_tradicional_MC, N*(1-particionamento)))

    # Gaussiano Bayesiano Tradicional
    dados = EMGs.copy()
    X_Gauss_embaralhado = X_Gauss[:, index]
    y_Gauss_embaralhado = y_Gauss[index]
    dados_embaralhado = dados[:, index]

    X_treino_Gauss = X_Gauss_embaralhado[:,:int(N*particionamento)]
    y_treino_Gauss = y_Gauss_embaralhado[:int(N*particionamento)]
    dados_treino = dados_embaralhado[:, :int(N*particionamento)]

    X_teste_Gauss = X_Gauss_embaralhado[:,int(N*particionamento):]
    y_teste_Gauss = y_Gauss_embaralhado[int(N*particionamento):]
    dados_teste = dados_embaralhado[:, int(N*particionamento):]

    dados_classes = [
        dados_treino[:2,dados_treino[-1,:] == classes[0]],
        dados_treino[:2,dados_treino[-1,:] == classes[1]],
        dados_treino[:2,dados_treino[-1,:] == classes[2]],
        dados_treino[:2,dados_treino[-1,:] == classes[3]],
        dados_treino[:2,dados_treino[-1,:] == classes[4]]
    ]
    

    medias = np.array([
        np.mean(dados_classes[0], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[1]], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[2]], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[3]], axis=1),
        np.mean(dados_treino[:2,dados_treino[-1,:] == classes[4]], axis=1),
    ])
    try:
        matrizes_de_covariancia = np.array([
            np.cov(dados_classes[0]),
            np.cov(dados_treino[:2,dados_treino[-1,:] == classes[1]]),
            np.cov(dados_treino[:2,dados_treino[-1,:] == classes[2]]),
            np.cov(dados_treino[:2,dados_treino[-1,:] == classes[3]]),
            np.cov(dados_treino[:2,dados_treino[-1,:] == classes[4]]),
        ])
        matrizes_de_covariancia_inversas = np.array([
            np.linalg.inv(matrizes_de_covariancia[0]),
            np.linalg.inv(matrizes_de_covariancia[1]),
            np.linalg.inv(matrizes_de_covariancia[2]),
            np.linalg.inv(matrizes_de_covariancia[3]),
            np.linalg.inv(matrizes_de_covariancia[4]),
        ])
        for x in X_teste_Gauss:
            y_Gauss_resultado = []
            for i, classe in enumerate(classes):
                y_Gauss_resultado.append(fdp(x, medias[i], matrizes_de_covariancia[i], matrizes_de_covariancia_inversas[i]))
    except Exception as e:
        print(e)

    
    
    # Gaussiano Bayesiano Covariancia Igual
    matriz_de_covariancia = np.cov(X_treino_Gauss)
    matriz_de_covariancia_inversa = np.linalg.inv(matriz_de_covariancia)
    g_covariancia_igual = []
    for x in X_teste_Gauss.T:
        y_Gauss_cov_igual_resultado = []
        for i, classe in enumerate(classes):
            y_Gauss_cov_igual_resultado.append(fdp(x, medias[i], matriz_de_covariancia, matriz_de_covariancia_inversa))
        g_covariancia_igual.append(np.argmin(y_Gauss_cov_igual_resultado) + 1)

    # Gaussiano Bayesiano Covariancia Agregada
    
    matrizes_de_covariancia = np.array([
        np.cov(dados_classes[0]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[1]]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[2]]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[3]]),
        np.cov(dados_treino[:2,dados_treino[-1,:] == classes[4]]),
    ])

    matriz_de_covariancia_agregada = 0
    soma_pesos = 0
    for i, classe in enumerate(classes):
        matriz_de_covariancia_agregada += (len(dados_classes[i]) / len(y_treino_Gauss))*matrizes_de_covariancia[i]
        soma_pesos = (len(dados_classes[i]) / len(y_treino_Gauss))
    
    matriz_de_covariancia_agregada = (len(dados_classes[0]) / len(y_treino_Gauss))*matrizes_de_covariancia[0] + (len(dados_classes[1]) / len(y_treino_Gauss))*matrizes_de_covariancia[1] + (len(dados_classes[2]) / len(y_treino_Gauss))*matrizes_de_covariancia[2] + (len(dados_classes[3]) / len(y_treino_Gauss))*matrizes_de_covariancia[3] + (len(dados_classes[4]) / len(y_treino_Gauss))*matrizes_de_covariancia[4]
    soma_pesos = (len(dados_treino[:2,dados_treino[-1,:] == classes[0]]) / len(y_treino_Gauss)) + (len(dados_treino[:2,dados_treino[-1,:] == classes[1]]) / len(y_treino_Gauss)) + (len(dados_treino[:2,dados_treino[-1,:] == classes[2]]) / len(y_treino_Gauss)) + (len(dados_treino[:2,dados_treino[-1,:] == classes[3]]) / len(y_treino_Gauss)) + (len(dados_treino[:2,dados_treino[-1,:] == classes[4]]) / len(y_treino_Gauss))
    matriz_de_covariancia_agregada = matriz_de_covariancia_agregada/soma_pesos
    matriz_de_covariancia_agregada_inversa = np.linalg.inv(matriz_de_covariancia_agregada)
    g_covariancia_agregada = []
    for x in X_teste_Gauss.T:
        y_Gauss_agregada_resultado = []
        for i, classe in enumerate(classes):
            y_Gauss_agregada_resultado.append(fdp(x, medias[i], matriz_de_covariancia_agregada, matriz_de_covariancia_agregada_inversa))
        g_covariancia_agregada.append(np.argmin(y_Gauss_agregada_resultado) + 1)
    bp = 1

    # Gaussiano Bayesiano Friedman
    lambdas = [0.25, 0.50, 0.75]

    matrizes_de_covariancia_025 = [
        ((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[0]]) + (lambdas[0] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[0]])) + (lambdas[0] * len(y_treino_Gauss))),
        ((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[1]]) + (lambdas[0] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[1]])) + (lambdas[0] * len(y_treino_Gauss))),
        ((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[2]]) + (lambdas[0] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[2]])) + (lambdas[0] * len(y_treino_Gauss))),
        ((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[3]]) + (lambdas[0] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[3]])) + (lambdas[0] * len(y_treino_Gauss))),
        ((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[4]]) + (lambdas[0] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[0]) * len(dados_treino[:2,dados_treino[-1,:] == classes[4]])) + (lambdas[0] * len(y_treino_Gauss))),
    ]
    matrizes_de_covariancia_50 = [
        ((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[0]]) + (lambdas[1] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[0]])) + (lambdas[1] * len(y_treino_Gauss))),
        ((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[1]]) + (lambdas[1] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[1]])) + (lambdas[1] * len(y_treino_Gauss))),
        ((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[2]]) + (lambdas[1] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[2]])) + (lambdas[1] * len(y_treino_Gauss))),
        ((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[3]]) + (lambdas[1] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[3]])) + (lambdas[1] * len(y_treino_Gauss))),
        ((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[4]]) + (lambdas[1] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[1]) * len(dados_treino[:2,dados_treino[-1,:] == classes[4]])) + (lambdas[1] * len(y_treino_Gauss))),
    ]
    matrizes_de_covariancia_075 = [
        ((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[0]]) + (lambdas[2] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[0]])) + (lambdas[2] * len(y_treino_Gauss))),
        ((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[1]]) + (lambdas[2] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[1]])) + (lambdas[2] * len(y_treino_Gauss))),
        ((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[2]]) + (lambdas[2] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[2]])) + (lambdas[2] * len(y_treino_Gauss))),
        ((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[3]]) + (lambdas[2] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[3]])) + (lambdas[2] * len(y_treino_Gauss))),
        ((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[4]]) + (lambdas[2] * len(y_treino_Gauss) * matriz_de_covariancia_agregada)) / (((1 - lambdas[2]) * len(dados_treino[:2,dados_treino[-1,:] == classes[4]])) + (lambdas[2] * len(y_treino_Gauss))),
    ]

    # Gaussiano Bayesiano Naive Bayes


metricas_MQO_tradicional = {
    'media': np.mean(desempenhos_MQO_tradicional),
    'desvio_padrao': np.std(desempenhos_MQO_tradicional),
    'maximo': np.max(desempenhos_MQO_tradicional),
    'minimo': np.min(desempenhos_MQO_tradicional)
}
print("MQO tradicional:")
print(f"Média: {metricas_MQO_tradicional['media']}")
print(f"Desvio Padrão: {metricas_MQO_tradicional['desvio_padrao']}")
print(f"Valor máximo: {metricas_MQO_tradicional['maximo']}")
print(f"Valor mínimo: {metricas_MQO_tradicional['minimo']}")
print("-------------------------------")

modelos = (
    "MQO tradicional",
)

metricas = {
    'Média': 
        (
            metricas_MQO_tradicional['media'],
        ),
    'Desvio Padrão': 
        (
            metricas_MQO_tradicional['desvio_padrao'],
        ),
    'Valor máximo': 
        (
            metricas_MQO_tradicional['maximo'],
        ),
    'Valor mínimo': 
        (
            metricas_MQO_tradicional['minimo'],
        ),
}

x = np.arange(len(modelos))  # the label locations
largura = 0.2  # the width of the bars
mult = 0

fig, ax = plt.subplots(layout='constrained')

for tipo, medida in metricas.items():
    offset = largura * mult
    rects = ax.bar(x + offset, medida, largura, label=tipo)
    ax.bar_label(rects, padding=3)
    mult += 1

ax.set_title('Métricas dos modelos')
ax.set_xticks(x + largura, modelos)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)


if plot_graphs:
    plt.show()

bp = 1