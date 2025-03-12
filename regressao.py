import numpy as np
import matplotlib.pyplot as plt

# Tarefa de Regressão

# 1.
atividade_enzimatica = np.loadtxt("atividade_enzimatica.csv", delimiter=",")

fig1 = plt.figure(1)
plot1 = fig1.add_subplot(projection='3d')

plot1.scatter(
    atividade_enzimatica[:,0], atividade_enzimatica[:,1], atividade_enzimatica[:,2],
    c="teal",
    edgecolors="k"
)

plot1.set_xlabel("Temperatura")
plot1.set_ylabel("pH da solução")
plot1.set_zlabel("Atividade Enzimática")
plot1.set_title("Nível de Atividade Enzimática")

# 2.
X = atividade_enzimatica[:,:2]

y = atividade_enzimatica[:,-1]

# 3.
plt.show()