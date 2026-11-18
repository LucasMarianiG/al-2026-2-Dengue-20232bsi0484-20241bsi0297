"""Núcleo matemático das cadeias de Markov (Etapa 3): distribuição estacionária por três caminhos,
primitividade (menor k com P^k > 0) e limite de Wielandt.

Convenção: P é estocástica por linhas (cada linha soma 1) e π é vetor-linha com πP = π.
"""
# [IA] Claude Sonnet 5.5 - 2026-11-18 - Reescrita de src/markov.py: validação, potências com diagnóstico, autovetor com checagens, resíduo, menor k e Wielandt (base original do autor B com Gemini)
import warnings
from typing import NamedTuple

import numpy as np


def validar_estocastica(P, tol=1e-12):
    """Confere que P é quadrada, sem entradas negativas e com linhas somando 1. Devolve P como float."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P deve ser quadrada; formato recebido: {P.shape}")
    if (P < 0).any():
        raise ValueError("P tem entradas negativas")
    if not np.allclose(P.sum(axis=1), 1.0, atol=tol, rtol=0):
        raise ValueError("as linhas de P devem somar 1")
    return P


def calcula_distribuicao_estacionaria_sistema(P):
    """Caminho (i): resolve (Pᵀ − I)πᵀ = 0 com a última equação trocada por Σπ = 1 (LU com pivotamento).

    Se a cadeia tiver mais de uma classe recorrente, o sistema fica singular e levanta LinAlgError.
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


class ResultadoPotencias(NamedTuple):
    pi: np.ndarray
    iteracoes: int
    convergiu: bool


def calcula_distribuicao_estacionaria_potencias(P, pi0=None, tol=1e-12, max_iter=100_000):
    """Caminho (ii): itera π(k+1) = π(k)P até ‖π(k+1) − π(k)‖₁ < tol.

    pi0 é a distribuição inicial (padrão: uniforme). Devolve (pi, iteracoes, convergiu);
    se não convergir em max_iter passos (ex.: cadeia periódica), avisa com RuntimeWarning.
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    pi = np.full(n, 1.0 / n) if pi0 is None else np.asarray(pi0, dtype=float).copy()
    if pi.shape != (n,) or (pi < 0).any() or abs(pi.sum() - 1.0) > 1e-9:
        raise ValueError("pi0 deve ser distribuição de probabilidade com n entradas")
    for k in range(1, max_iter + 1):
        novo = pi @ P
        erro = np.abs(novo - pi).sum()
        pi = novo
        if erro < tol:
            return ResultadoPotencias(pi, k, True)
    warnings.warn(f"método das potências não convergiu em {max_iter} iterações (último passo: {erro:.2e})",
                  RuntimeWarning, stacklevel=2)
    return ResultadoPotencias(pi, max_iter, False)


def calcula_distribuicao_estacionaria_autovetor(P):
    """Caminho (iii): autovetor à esquerda de P para λ = 1 (autovetor de Pᵀ), normalizado para somar 1.

    Exige λ = 1 simples (cadeia com uma só classe recorrente); senão levanta ValueError.
    """
    P = validar_estocastica(P)
    autovalores, autovetores = np.linalg.eig(P.T)
    perto_de_1 = np.abs(autovalores - 1.0) < 1e-9
    if perto_de_1.sum() != 1:
        raise ValueError(f"autovalor 1 não é simples (multiplicidade encontrada: {int(perto_de_1.sum())})")
    v = autovetores[:, np.argmax(perto_de_1)]
    if np.abs(v.imag).max() > 1e-9 * max(1.0, np.abs(v.real).max()):
        raise ValueError("autovetor de λ = 1 com parte imaginária relevante")
    v = v.real
    return v / v.sum()


def residuo_estacionario(pi, P):
    """‖πP − π‖₁: quanto π falha em ser estacionária (zero para o π verdadeiro)."""
    P = np.asarray(P, dtype=float)
    pi = np.asarray(pi, dtype=float)
    return float(np.abs(pi @ P - pi).sum())


def limite_wielandt(n):
    """Limite de Wielandt: toda matriz primitiva n×n satisfaz P^k > 0 para algum k ≤ (n−1)² + 1."""
    return (n - 1) ** 2 + 1


def menor_k_positivo(P, k_max=None):
    """Menor k ≥ 1 com todas as entradas de P^k positivas, ou None se não houver até k_max.

    Só importa o padrão de zeros: usa potências booleanas (sem erro de arredondamento).
    k_max padrão = limite de Wielandt; passando dele sem sucesso, a matriz não é primitiva.
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    if k_max is None:
        k_max = limite_wielandt(n)
    padrao = P > 0
    atual = padrao.copy()
    for k in range(1, k_max + 1):
        if atual.all():
            return k
        atual = (atual.astype(np.uint8) @ padrao.astype(np.uint8)) > 0
    return None
# [/IA]
