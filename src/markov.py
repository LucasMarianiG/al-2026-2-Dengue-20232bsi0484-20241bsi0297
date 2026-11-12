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

def calcula_distribuicao_estacionaria_potencias(P, tol=1e-12, max_iter=10000):
    """
    Calcula a distribuição estacionária pi iterando pi^(k+1) = pi^(k) * P
    até que a diferença (norma L1) seja menor que a tolerância (Caminho 2 da Etapa 3).
    """
    n = P.shape[0]
    
    # Começa com uma distribuição uniforme (poderia ser e_1 ou e_4 como pede o enunciado)
    pi = np.ones(n) / n 
    
    iteracoes = 0
    while iteracoes < max_iter:
        pi_novo = np.dot(pi, P)
        
        # Critério de parada: norma L1 da diferença (soma dos valores absolutos)
        erro = np.sum(np.abs(pi_novo - pi))
        if erro < tol:
            break
            
        pi = pi_novo
        iteracoes += 1
        
    return pi


def calcula_distribuicao_estacionaria_autovetor(P):
    """
    Calcula a distribuição estacionária pi encontrando o autovetor à esquerda 
    associado ao autovalor 1 (Caminho 3 da Etapa 3).
    """
    # Calcula autovalores e autovetores de P transposta
    autovalores, autovetores = np.linalg.eig(P.T)
    
    # Encontra o índice do autovalor mais próximo de 1
    indice_lambda_1 = np.argmin(np.abs(autovalores - 1.0))
    
    # Extrai o autovetor correspondente (e garante que pegamos apenas a parte real)
    pi_nao_normalizado = np.real(autovetores[:, indice_lambda_1])
    
    # Normaliza para que a soma seja 1
    pi = pi_nao_normalizado / np.sum(pi_nao_normalizado)
    
    return pi