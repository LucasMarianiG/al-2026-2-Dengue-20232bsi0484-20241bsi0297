import numpy as np

def calcula_distribuicao_estacionaria_sistema(P):
    """
    Calcula a distribuição estacionária pi resolvendo o sistema linear (P^T - I)pi^T = 0
    com a condição de normalização sum(pi) = 1 (Caminho 1 da Etapa 3).
    """
    n = P.shape[0]
    
    # Monta a matriz A = (P^T - I)
    A = P.T - np.eye(n)
    
    # Substitui a última equação pela condição de normalização: pi_1 + pi_2 + ... + pi_n = 1
    A[-1, :] = 1.0
    
    # O lado direito da equação (b): zeros, com 1 na última posição
    b = np.zeros(n)
    b[-1] = 1.0
    
    # np.linalg.solve utiliza fatoração LU com pivotamento parcial (anotar isso no relatório)
    pi = np.linalg.solve(A, b)
    
    return pi