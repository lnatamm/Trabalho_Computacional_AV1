import numpy as np
import matplotlib.pyplot as plt

plot_graphs = False

# Tarefa de Regressão

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

figure_index = 1

# 1.
atividade_enzimatica = np.loadtxt("atividade_enzimatica.csv", delimiter=",")

N, p = atividade_enzimatica.shape[0], atividade_enzimatica.shape[1] - 1 # Removendo y

plot_1 = get_plot_configuration(atividade_enzimatica, figure_index, "Atividade Enzimática")
figure_index+=1

# 2.
X = atividade_enzimatica[:,:2]

y = atividade_enzimatica[:,-1]

# 3.
# Constantes
n_linspace = 40
x_axis = np.linspace(np.min(X[:,0]), np.max(X[:,0]), n_linspace)
y_axis = np.linspace(np.min(X[:,1]), np.max(X[:,1]), n_linspace)
X3d, Y3d = np.meshgrid(x_axis, y_axis)

# MQO Tradicional
plot_MQO_tradicional = get_plot_configuration(atividade_enzimatica, figure_index, "Atividade Enzimática - MQO Tradicional")
figure_index+=1

X_MQO_tradicional = np.hstack((
    np.ones((X.shape[0], 1)), X
))
B_MQO_tradicional = np.linalg.pinv(X_MQO_tradicional.T@X_MQO_tradicional)@X_MQO_tradicional.T@y

Z_MQO_tradicional = B_MQO_tradicional[0] + B_MQO_tradicional[1]*X3d + B_MQO_tradicional[2]* Y3d

plot_MQO_tradicional.plot_surface(X3d, Y3d, Z_MQO_tradicional, cmap='gray')

# MQO Regularizado
X_MQO_regularizado = np.hstack((
    np.ones((X.shape[0], 1)), X
))

# 4.
lambdas_MQO_regularizado = [0, 0.25, 0.5, 0.75, 1]

for lambda_MQO_regularizado in lambdas_MQO_regularizado:
    plot_MQO_regularizado = get_plot_configuration(atividade_enzimatica, (figure_index), f"Atividade Enzimática - MQO Regularizado λ: {lambda_MQO_regularizado}")
    figure_index+=1
    B_MQO_regularizado = np.linalg.pinv(X_MQO_regularizado.T@X_MQO_regularizado + lambda_MQO_regularizado*np.identity(X_MQO_regularizado.shape[1]))@X_MQO_regularizado.T@y
    Z_MQO_regularizado = B_MQO_regularizado[0] + B_MQO_regularizado[1]*X3d + B_MQO_regularizado[2]*Y3d
    plot_MQO_regularizado.plot_surface(X3d, Y3d, Z_MQO_regularizado, cmap='gray')

# Média
plot_media = get_plot_configuration(atividade_enzimatica, figure_index, "Atividade Enzimática - Média")
figure_index+=1
media = np.mean(y)

B_media = [
    media,
    0,
    0
]

Z_media = B_media[0] + B_media[1]*X3d + B_media[2]*Y3d

plot_media.plot_surface(X3d, Y3d, Z_media, cmap='gray')

if plot_graphs:
    plt.show()

# 5.

# Simulações por Monte Carlo

rodadas = 500
particionamento = 0.8

desempenhos_MQO_tradicional = []
desempenhos_MQO_regularizado = []
desempenhos_media = []

for rodada in range(rodadas):
    index = np.random.permutation(atividade_enzimatica.shape[0])

    X_embaralhado = X[index, :]
    y_embaralhado = y[index]

    X_treino = X_embaralhado[:int(N*particionamento),:]
    y_treino = y_embaralhado[:int(N*particionamento)]

    X_teste = X_embaralhado[int(N*particionamento):,:]
    y_teste = y_embaralhado[int(N*particionamento):]

    X_treino = np.hstack((
        np.ones((X_treino.shape[0], 1)), X_treino
    ))

    X_teste = np.hstack((
        np.ones((X_teste.shape[0],1)), X_teste
    ))

    B_MQO_tradicional_MC = np.linalg.pinv(X_treino.T@X_treino)@X_treino.T@y_treino
    
    Bs_MQO_regularizado_MC = []

    for lambda_MQO_regularizado in lambdas_MQO_regularizado:
        B_MQO_regularizado_MC = np.linalg.pinv(X_treino.T@X_treino + lambda_MQO_regularizado*np.identity(X_treino.shape[1]))@X_treino.T@y_treino
        Bs_MQO_regularizado_MC.append(B_MQO_regularizado_MC)

    B_media_MC = [
        media,
        0,
        0
    ]

    y_predicao_MQO_tradicional = X_teste@B_MQO_regularizado_MC
    y_predicao_MQO_regularizado = [X_teste@Bs_MQO_regularizado_MC[i] for i in range(len(Bs_MQO_regularizado_MC))]
    y_predicao_media = X_teste@B_media_MC

    desempenhos_MQO_tradicional.append(np.sum((y_teste-y_predicao_MQO_tradicional)**2))
    desempenhos_MQO_regularizado.append(
        [
            np.sum((y_teste-y_predicao_MQO_regularizado[0])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[1])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[2])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[3])**2),
            np.sum((y_teste-y_predicao_MQO_regularizado[4])**2),
        ]
    )
    desempenhos_media.append(np.sum((y_teste-y_predicao_media)**2))

# 6.

# Média da variável dependente
print("Média da variável dependente:")
print(f"Média: {np.mean(desempenhos_media)}")
print(f"Desvio Padrão: {np.std(desempenhos_media)}")
print(f"Valor máximo: {np.max(desempenhos_media)}")
print(f"Valor mínimo: {np.min(desempenhos_media)}")
print("-------------------------------")

print("MQO tradicional:")
print(f"Média: {np.mean(desempenhos_MQO_tradicional)}")
print(f"Desvio Padrão: {np.std(desempenhos_MQO_tradicional)}")
print(f"Valor máximo: {np.max(desempenhos_MQO_tradicional)}")
print(f"Valor mínimo: {np.min(desempenhos_MQO_tradicional)}")
print("-------------------------------")


desempenhos_MQO_regularizado = np.array(desempenhos_MQO_regularizado)
print("MQO regularizado (0,25):")
print(f"Média: {np.mean(desempenhos_MQO_regularizado[:,1])}")
print(f"Desvio Padrão: {np.std(desempenhos_MQO_regularizado[:,1])}")
print(f"Valor máximo: {np.max(desempenhos_MQO_regularizado[:,1])}")
print(f"Valor mínimo: {np.min(desempenhos_MQO_regularizado[:,1])}")
print("-------------------------------")

print("MQO regularizado (0,5):")
print(f"Média: {np.mean(desempenhos_MQO_regularizado[:,2])}")
print(f"Desvio Padrão: {np.std(desempenhos_MQO_regularizado[:,2])}")
print(f"Valor máximo: {np.max(desempenhos_MQO_regularizado[:,2])}")
print(f"Valor mínimo: {np.min(desempenhos_MQO_regularizado[:,2])}")
print("-------------------------------")

print("MQO regularizado (0,75):")
print(f"Média: {np.mean(desempenhos_MQO_regularizado[:,3])}")
print(f"Desvio Padrão: {np.std(desempenhos_MQO_regularizado[:,3])}")
print(f"Valor máximo: {np.max(desempenhos_MQO_regularizado[:,3])}")
print(f"Valor mínimo: {np.min(desempenhos_MQO_regularizado[:,3])}")
print("-------------------------------")

print("MQO regularizado (1):")
print(f"Média: {np.mean(desempenhos_MQO_regularizado[:,4])}")
print(f"Desvio Padrão: {np.std(desempenhos_MQO_regularizado[:,4])}")
print(f"Valor máximo: {np.max(desempenhos_MQO_regularizado[:,4])}")
print(f"Valor mínimo: {np.min(desempenhos_MQO_regularizado[:,4])}")
print("-------------------------------")