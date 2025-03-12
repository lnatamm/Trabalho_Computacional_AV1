import numpy as np
import matplotlib.pyplot as plt

# Tarefa de Regressão

# Configuração padrão dos plots
def get_plot_configuration(file, n_figure):
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
    plot.set_title("Atividade Enzimática")

    return plot

# 1.
atividade_enzimatica = np.loadtxt("atividade_enzimatica.csv", delimiter=",")

plot_1 = get_plot_configuration(atividade_enzimatica, 1)

# 2.
X = atividade_enzimatica[:,:2]

y = atividade_enzimatica[:,-1]

# 3.

# MQO Tradicional

plot_MQO_tradicional = get_plot_configuration(atividade_enzimatica, 2)

x_axis = np.linspace(-4, 6, 100)
y_axis = np.linspace(-9, 9, 100)
X3d, Y3d = np.meshgrid(x_axis, y_axis)

X_MQO_tradicional = np.hstack((
    np.ones((X.shape[0], 1)), X
))
B_MQO_tradicional = np.linalg.pinv(X_MQO_tradicional.T@X_MQO_tradicional)@X_MQO_tradicional.T@y

Z_MQO_tradicional = B_MQO_tradicional[0] + B_MQO_tradicional[1]*X3d + B_MQO_tradicional[2]* Y3d

plot_MQO_tradicional.plot_surface(X3d, Y3d, Z_MQO_tradicional, cmap='gray')

plt.show()