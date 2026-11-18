# [IA] Claude Sonnet 5.5 - 2026-11-18 - tests/test_markov_e5.py: testes do nucleo da Etapa 5 (primeira passagem por 4 sistemas 3x3, matriz fundamental Z, Kac)
"""Testes de primeira passagem em src/markov.py: M por quatro sistemas 3x3 e por Z = (I - P + 1π)^-1."""
import numpy as np
import pytest

from src.markov import (
    calcula_distribuicao_estacionaria_sistema as pi_sistema,
    matriz_fundamental_Z,
    primeira_passagem_fundamental,
    primeira_passagem_sistemas,
)

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])
# Tabela 1 do enunciado
DIAG_TABELA1 = np.array([2.004128, 4.855455, 5.743011, 8.267802])
M14_TABELA1 = 46.873065
M41_TABELA1 = 17.035647


def _m_por_serie_de_potencias(P, j, termos=4000):
    """Terceiro caminho: m_ij = Σ_k Q^k 1 (série de Neumann do bloco sem a linha/coluna j), sem inverter nem resolver."""
    n = P.shape[0]
    idx = [k for k in range(n) if k != j]
    Q = P[np.ix_(idx, idx)]
    soma = np.zeros(n - 1)
    atual = np.ones(n - 1)
    for _ in range(termos):
        soma += atual
        atual = Q @ atual
    m = np.zeros(n)
    m[idx] = soma
    m[j] = 1.0 + P[j, idx] @ soma
    return m


# ---------------------------------------------------------------- Tabela 1
def test_M_por_sistemas_reproduz_tabela1():
    M = primeira_passagem_sistemas(P_REF)
    np.testing.assert_allclose(np.diag(M), DIAG_TABELA1, atol=1e-6)
    assert M[0, 3] == pytest.approx(M14_TABELA1, abs=1e-6)
    assert M[3, 0] == pytest.approx(M41_TABELA1, abs=1e-6)


def test_M_por_Z_reproduz_tabela1():
    M = primeira_passagem_fundamental(P_REF)
    np.testing.assert_allclose(np.diag(M), DIAG_TABELA1, atol=1e-6)
    assert M[0, 3] == pytest.approx(M14_TABELA1, abs=1e-6)
    assert M[3, 0] == pytest.approx(M41_TABELA1, abs=1e-6)


# ---------------------------------------------------------------- Kac e concordancia
def test_kac_diagonal_e_inverso_de_pi_ate_1e_10():
    pi = pi_sistema(P_REF)
    for M in (primeira_passagem_sistemas(P_REF), primeira_passagem_fundamental(P_REF)):
        np.testing.assert_allclose(np.diag(M), 1.0 / pi, rtol=0, atol=1e-10)


def test_sistemas_e_matriz_fundamental_concordam_ate_1e_10():
    a, b = primeira_passagem_sistemas(P_REF), primeira_passagem_fundamental(P_REF)
    assert np.abs(a - b).max() < 1e-10


def test_M_confere_com_serie_de_potencias_terceiro_caminho():
    M = primeira_passagem_sistemas(P_REF)
    for j in range(4):
        m = _m_por_serie_de_potencias(P_REF, j)
        np.testing.assert_allclose(M[:, j], m, atol=1e-8)


def test_equacao_2_do_enunciado_vale_para_todo_i_e_j():
    """m_ij = 1 + Σ_{k≠j} p_ik m_kj (condicionando no primeiro passo), inclusive i = j."""
    M = primeira_passagem_sistemas(P_REF)
    for i in range(4):
        for j in range(4):
            lado_dir = 1.0 + sum(P_REF[i, k] * M[k, j] for k in range(4) if k != j)
            assert M[i, j] == pytest.approx(lado_dir, abs=1e-10)


def test_M_positiva_e_diagonal_maior_ou_igual_a_um():
    M = primeira_passagem_sistemas(P_REF)
    assert (M > 0).all()
    assert (np.diag(M) >= 1.0).all()


# ---------------------------------------------------------------- casos a mao
def test_cadeia_de_dois_estados_a_mao():
    """P = [[1-a, a], [b, 1-b]]: m_12 = 1/a, m_21 = 1/b, π = (b, a)/(a+b), m_11 = (a+b)/b."""
    a, b = 0.2, 0.5
    P = np.array([[1 - a, a], [b, 1 - b]])
    for f in (primeira_passagem_sistemas, primeira_passagem_fundamental):
        M = f(P)
        assert M[0, 1] == pytest.approx(1 / a, abs=1e-12)
        assert M[1, 0] == pytest.approx(1 / b, abs=1e-12)
        assert M[0, 0] == pytest.approx((a + b) / b, abs=1e-12)
        assert M[1, 1] == pytest.approx((a + b) / a, abs=1e-12)


def test_matriz_uniforme_tem_primeira_passagem_igual_ao_numero_de_estados():
    n = 4
    P = np.full((n, n), 1.0 / n)
    for f in (primeira_passagem_sistemas, primeira_passagem_fundamental):
        np.testing.assert_allclose(f(P), np.full((n, n), float(n)), atol=1e-12)


def test_ciclo_deterministico_tres_estados_nao_e_primitivo_mas_M_existe():
    """1→2→3→1: m_12 = 1, m_13 = 2, m_jj = 3 (π uniforme). A fórmula de Z vale (cadeia irredutível)."""
    P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    M = primeira_passagem_sistemas(P)
    assert M[0, 1] == pytest.approx(1.0) and M[0, 2] == pytest.approx(2.0)
    np.testing.assert_allclose(np.diag(M), 3.0, atol=1e-12)


# ---------------------------------------------------------------- Z
def test_Z_satisfaz_Z1_igual_1_e_piZ_igual_pi():
    pi = pi_sistema(P_REF)
    Z = matriz_fundamental_Z(P_REF, pi)
    np.testing.assert_allclose(Z @ np.ones(4), np.ones(4), atol=1e-10)
    np.testing.assert_allclose(pi @ Z, pi, atol=1e-10)


def test_Z_inversa_confere_com_I_menos_P_mais_1pi():
    pi = pi_sistema(P_REF)
    Z = matriz_fundamental_Z(P_REF, pi)
    A = np.eye(4) - P_REF + np.outer(np.ones(4), pi)
    np.testing.assert_allclose(Z @ A, np.eye(4), atol=1e-10)


def test_Z_aceita_pi_omitido():
    np.testing.assert_allclose(matriz_fundamental_Z(P_REF), matriz_fundamental_Z(P_REF, pi_sistema(P_REF)), atol=1e-12)


# ---------------------------------------------------------------- simulacao (terceiro caminho de M)
def test_simulacao_com_seed_2026_confere_m14_e_m41():
    """Estimativa Monte Carlo (default_rng(2026)) de m14 e m41; tolerância de ~4 erros-padrão."""
    rng = np.random.default_rng(2026)
    acumulada = np.cumsum(P_REF, axis=1)

    def tempo(i, j):
        t, x = 0, i
        while True:
            x = int(np.searchsorted(acumulada[x], rng.random(), side="right"))
            t += 1
            if x == j:
                return t

    n = 20000
    for i, j, esperado in ((0, 3, M14_TABELA1), (3, 0, M41_TABELA1)):
        amostra = np.array([tempo(i, j) for _ in range(n)])
        ep = amostra.std(ddof=1) / np.sqrt(n)
        assert abs(amostra.mean() - esperado) < 4 * ep


# ---------------------------------------------------------------- validacao de entrada
@pytest.mark.parametrize("f", [primeira_passagem_sistemas, primeira_passagem_fundamental])
def test_rejeita_matriz_nao_estocastica(f):
    P = P_REF.copy()
    P[0, 0] += 0.1
    with pytest.raises(ValueError):
        f(P)


@pytest.mark.parametrize("f", [primeira_passagem_sistemas, primeira_passagem_fundamental])
def test_rejeita_matriz_nao_quadrada(f):
    with pytest.raises(ValueError):
        f(np.ones((3, 4)) / 4)


def test_sistemas_rejeita_cadeia_com_estado_inalcancavel():
    """Se j não é alcançável de algum i, (I − Q_j) é singular: erro claro, não inf/NaN silencioso."""
    P = np.array([[1.0, 0.0], [0.5, 0.5]])  # estado 2 não é alcançável de 1
    with pytest.raises(ValueError):
        primeira_passagem_sistemas(P)
# [/IA]
