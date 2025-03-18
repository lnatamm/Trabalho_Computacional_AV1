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

def acuracia(X, y, W, N):
    acertos = 0
    for i, X in enumerate(X):
        if np.argmax(X@W) == np.argmax(y[i]):
            acertos += 1
    return acertos/N

figure_index = 1

# 1.

c1, c2, c3, c4, c5 = 1, 2, 3, 4, 5

EMGs = np.loadtxt("datasets\\EMGsDataset.csv", delimiter=',').T

N, p = EMGs.shape[0], EMGs.shape[1] - 1 # Removendo y

X = np.vstack((
    EMGs[EMGs[:,-1] == c1,:2],
    EMGs[EMGs[:,-1] == c2,:2],
    EMGs[EMGs[:,-1] == c3,:2],
    EMGs[EMGs[:,-1] == c4,:2],
    EMGs[EMGs[:,-1] == c5,:2],
))

X = np.hstack((
    np.ones((N, 1)), X
))

y = np.vstack((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (10000, 1)),
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (10000, 1)),
))

X_MQO = np.vstack((
    EMGs[EMGs[:,-1] == c1,:2],
    EMGs[EMGs[:,-1] == c2,:2],
    EMGs[EMGs[:,-1] == c3,:2],
    EMGs[EMGs[:,-1] == c4,:2],
    EMGs[EMGs[:,-1] == c5,:2],
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

print(acuracia(X_MQO, y_MQO, W_MQO, N))

for rodada in range(rodadas):
    index = np.random.permutation(EMGs.shape[0])
    X_embaralhado = X[index, :]
    y_embaralhado = y[index]

    X_treino = X_embaralhado[:int(N*particionamento),:]
    y_treino = y_embaralhado[:int(N*particionamento)]

    X_teste = X_embaralhado[int(N*particionamento):,:]
    y_teste = y_embaralhado[int(N*particionamento):]

    X_treino = X_embaralhado[:int(N*particionamento),:]
    y_treino = y_embaralhado[:int(N*particionamento)]

    X_teste = X_embaralhado[int(N*particionamento):,:]
    y_teste = y_embaralhado[int(N*particionamento):]

    W_MQO_tradicional_MC = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino

    desempenhos_MQO_tradicional.append(acuracia(X_teste, y_teste, W_MQO_tradicional_MC, N*(1-particionamento)))

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