import random
import numpy as np
import matplotlib.pyplot as plt

def gerar_instancia(n, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    p = np.random.randint(1, 20, n)
    d = np.random.randint(10, 100, n)
    w = np.random.randint(1, 10, n)
    return list(zip(p, d, w))

def calcular_atraso(seq):
    tempo = 0
    total = 0
    for p, d, w in seq:
        tempo += p
        atraso = max(0, tempo - d)
        total += w * atraso
    return total

def edd(tarefas):
    return sorted(tarefas, key=lambda x: x[1])

def wspt(tarefas):
    return sorted(tarefas, key=lambda x: x[0] / x[2])

def simulated_annealing(tarefas, T0=100, alpha=0.95, iter_max=1000):
    atual = wspt(tarefas)
    best = atual[:]
    f_atual = calcular_atraso(atual)
    f_best = f_atual
    T = T0
    custos = [f_atual]

    for i in range(iter_max):
        vizinho = atual[:]
        a, b = random.sample(range(len(tarefas)), 2)
        vizinho[a], vizinho[b] = vizinho[b], vizinho[a]

        f_vizinho = calcular_atraso(vizinho)
        delta = f_vizinho - f_atual

        if delta < 0 or random.random() < np.exp(-delta / T):
            atual = vizinho
            f_atual = f_vizinho
            if f_vizinho < f_best:
                best = vizinho
                f_best = f_vizinho

        T *= alpha
        custos.append(f_best)

    return best, f_best, custos

if __name__ == "__main__":
    tarefas = gerar_instancia(50)

    edd_seq = edd(tarefas)
    wspt_seq = wspt(tarefas)
    sa_seq, sa_valor, historico = simulated_annealing(tarefas)

    print("EDD:", calcular_atraso(edd_seq))
    print("WSPT:", calcular_atraso(wspt_seq))
    print("Simulated Annealing:", sa_valor)

    plt.plot(historico)
    plt.xlabel("Iterações")
    plt.ylabel("Melhor Custo")
    plt.title("Convergência - Simulated Annealing")
    plt.grid()
    plt.show()
