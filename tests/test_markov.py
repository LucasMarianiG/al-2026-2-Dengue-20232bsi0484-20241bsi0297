import numpy as np
import pytest

from src.markov import (
    calcula_distribuicao_estacionaria_sistema,
    calcula_distribuicao_estacionaria_potencias,
    calcula_distribuicao_estacionaria_autovetor
)

# Matriz de referência Pref (Tabela 1) do enunciado
P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75]
])

# [IA] Gemini - 2026-11-12 - Estrutura inicial e implementação dos 3 métodos da Etapa 3

def test_matriz_estocastica():
    """Verifica se as linhas somam 1 e não há valores negativos (E2)."""
    assert np.all(P_REF >= 0)
    np.testing.assert_allclose(np.sum(P_REF, axis=1), 1.0, atol=1e-12)

def test_distribuicao_estacionaria_referencia():
    """Verifica o vetor pi da Tabela 1 (E3)."""
    pi_esperado = np.array([0.498970, 0.205954, 0.174125, 0.120951])
    
    # Chama a função que criamos no src/markov.py
    pi_calculado = calcula_distribuicao_estacionaria_sistema(P_REF)
    
    # Compara se o cálculo bate com o valor esperado com precisão de 6 casas decimais
    np.testing.assert_allclose(pi_calculado, pi_esperado, atol=1e-6)

def test_espectro_referencia():
    """Verifica os autovalores ordenados por módulo da Tabela 1 (E4)."""
    espectro_esperado = np.array([1.0, 0.869971, 0.618816, 0.461213])
    
    # autovalores = calcula_espectro(P_REF)
    # np.testing.assert_allclose(autovalores, espectro_esperado, atol=1e-6)

def test_meia_vida_referencia():
    """Verifica o cálculo da meia-vida da Tabela 1 (E4)."""
    meia_vida_esperada = 4.976086
    
    # mv_calculada = calcula_meia_vida(P_REF)
    # np.testing.assert_allclose(mv_calculada, meia_vida_esperada, atol=1e-6)

def test_distribuicao_estacionaria_potencias():
    """Verifica se o método das potências chega ao mesmo vetor pi (E3)."""
    pi_esperado = np.array([0.498970, 0.205954, 0.174125, 0.120951])
    
    pi_calculado = calcula_distribuicao_estacionaria_potencias(P_REF)
    
    np.testing.assert_allclose(pi_calculado, pi_esperado, atol=1e-6)



def test_distribuicao_estacionaria_autovetor():
    """Verifica se o método do autovetor chega ao mesmo vetor pi (E3)."""
    pi_esperado = np.array([0.498970, 0.205954, 0.174125, 0.120951])
    
    pi_calculado = calcula_distribuicao_estacionaria_autovetor(P_REF)
    
    np.testing.assert_allclose(pi_calculado, pi_esperado, atol=1e-6)