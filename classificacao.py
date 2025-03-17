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

figure_index = 1

# 1.

c1, c2, c3, c4, c5 = 1, 2, 3, 4, 5

EMGs = np.loadtxt("datasets\\EMGsDataset.csv", delimiter=',')

X_MQO = np.vstack((
    EMGs[EMGs[:,-1] == c1,:2],
    EMGs[EMGs[:,-1] == c2,:2],
    EMGs[EMGs[:,-1] == c3,:2],
    EMGs[EMGs[:,-1] == c4,:2],
    EMGs[EMGs[:,-1] == c5,:2],
))

N, p = EMGs.shape[0], EMGs.shape[1] - 1 # Removendo y

X_MQO = np.hstack((
    np.ones((N, 1)), X_MQO
))

Y_MQO = np.vstack((
    np.tile(np.array([[1, -1, -1, -1, -1]]), (1000, 1)),
    np.tile(np.array([[-1, 1, -1, -1, -1]]), (1000, 1)),
    np.tile(np.array([[-1, -1, 1, -1, -1]]), (1000, 1)),
    np.tile(np.array([[-1, -1, -1, 1, -1]]), (1000, 1)),
    np.tile(np.array([[-1, -1, -1, -1, 1]]), (1000, 1)),
))

W = np.linalg.pinv(X_MQO.T@X_MQO)@X_MQO.T@Y_MQO