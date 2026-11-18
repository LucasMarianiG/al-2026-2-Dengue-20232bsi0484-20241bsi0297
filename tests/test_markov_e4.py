# [IA] Claude Sonnet 5.5 - 2026-11-18 - tests/test_markov_e4.py: testes do nucleo da Etapa 4 (espectro por dois caminhos, meia-vida, quadrados sucessivos, diagonalizacao, kappa_1)
"""Testes do núcleo da Etapa 4 (src/markov.py).

Grupos: (1) Tabela 1 (espectro, meia-vida, e1P^4, ||e1P^20 - pi||_1);
(2) espectro por dois caminhos independentes (eig x polinômio característico de Faddeev-LeVerrier);
(3) potências por três caminhos (multiplicação repetida, quadrados sucessivos, diagonalização);
(4) número de condição κ1(V); (5) casos com autovalores complexos e cadeia não primitiva.
"""
import numpy as np
import pytest

from src.markov import (
    calcula_distribuicao_estacionaria_sistema,
    calcula_espectro,
    calcula_meia_vida,
    diagonaliza,
    espectro_por_polinomio,
    norma_1,
    numero_condicao_1,
    polinomio_caracteristico,
    potencia_por_diagonalizacao,
    potencia_por_multiplicacao,
    potencia_quadrados_sucessivos,
)

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])


def _aleatoria(rng, n=4):
    M = rng.random((n, n)) + 0.05
    return M / M.sum(axis=1, keepdims=True)


# ------------------------------------------------------------ 1) Tabela 1
def test_tabela1_espectro_ordenado_por_modulo():
    esp = calcula_espectro(P_REF)
    np.testing.assert_allclose(esp, [1.0, 0.869971, 0.618816, 0.461213], atol=1e-6)
    assert (np.diff(np.abs(esp)) <= 1e-15).all()          # módulos não crescentes


def test_tabela1_meia_vida():
    assert calcula_meia_vida(P_REF) == pytest.approx(4.976086, abs=1e-6)


def test_tabela1_e1_P4_e_distancia_apos_20_passos():
    pi = calcula_distribuicao_estacionaria_sistema(P_REF)
    e1 = np.array([1.0, 0, 0, 0])
    np.testing.assert_allclose(e1 @ potencia_quadrados_sucessivos(P_REF, 4)[0],
                               [0.728667, 0.172331, 0.077617, 0.021385], atol=1e-6)
    d = np.abs(e1 @ potencia_quadrados_sucessivos(P_REF, 20)[0] - pi).sum()
    assert d == pytest.approx(0.045556, abs=1e-6)


# --------------------------------------- 2) espectro por dois caminhos
def test_faddeev_leverrier_em_matriz_2x2_a_mao():
    # P = [[a, b], [c, d]]: p(x) = x^2 - (a+d) x + (ad - bc)
    c = polinomio_caracteristico(np.array([[0.7, 0.3], [0.4, 0.6]]))
    np.testing.assert_allclose(c, [1.0, -1.3, 0.7 * 0.6 - 0.3 * 0.4], atol=1e-14)


def test_faddeev_leverrier_da_traco_e_determinante():
    rng = np.random.default_rng(2026)
    for _ in range(10):
        P = _aleatoria(rng)
        c = polinomio_caracteristico(P)
        assert c[0] == 1.0
        assert -c[1] == pytest.approx(np.trace(P), abs=1e-13)
        assert (-1) ** 4 * c[4] == pytest.approx(np.linalg.det(P), abs=1e-13)


def test_espectro_por_polinomio_confere_com_eig_na_pref():
    a = calcula_espectro(P_REF)
    b = espectro_por_polinomio(P_REF)
    np.testing.assert_allclose(b, a, atol=1e-10)


def test_espectro_dois_caminhos_em_20_matrizes_aleatorias():
    rng = np.random.default_rng(2026)
    for _ in range(20):
        P = _aleatoria(rng)
        a, b = calcula_espectro(P), espectro_por_polinomio(P)
        assert np.abs(a - b).max() < 1e-10
        assert abs(a[0] - 1) < 1e-12 and (np.abs(a[1:]) < 1).all()   # Perron-Frobenius


def test_soma_dos_autovalores_e_traco_produto_e_determinante():
    esp = calcula_espectro(P_REF)
    assert abs(esp.sum() - np.trace(P_REF)) < 1e-10
    assert abs(np.prod(esp) - np.linalg.det(P_REF)) < 1e-10


# ------------------------------------------- 3) potências por três caminhos
def test_quadrados_sucessivos_52_conta_7_multiplicacoes():
    # 52 = 110100 em binário: P^2, P^4, P^8, P^16, P^32 (5 quadrados) e P^4·P^16·P^32 = P^52 (2 produtos) => 7
    M, mult = potencia_quadrados_sucessivos(P_REF, 52)
    assert mult == 7
    np.testing.assert_allclose(M, np.linalg.matrix_power(P_REF, 52), atol=1e-14, rtol=0)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 8, 20, 52, 100])
def test_tres_caminhos_para_P_elevado_a_n(n):
    q, _ = potencia_quadrados_sucessivos(P_REF, n)
    m = potencia_por_multiplicacao(P_REF, n)
    V, lam, Vinv = diagonaliza(P_REF)
    d = potencia_por_diagonalizacao(V, lam, Vinv, n)
    assert np.abs(q - m).max() < 1e-12
    assert np.abs(d - m).max() < 1e-12
    np.testing.assert_allclose(q.sum(axis=1), 1.0, atol=1e-12)      # Proposição 1: P^n é estocástica
    assert (q >= -1e-15).all()


def test_potencia_zero_e_identidade_e_n_negativo_falha():
    M, mult = potencia_quadrados_sucessivos(P_REF, 0)
    np.testing.assert_array_equal(M, np.eye(4))
    assert mult == 0
    with pytest.raises(ValueError):
        potencia_quadrados_sucessivos(P_REF, -1)
    with pytest.raises(ValueError):
        potencia_quadrados_sucessivos(np.ones((2, 3)), 2)
    with pytest.raises(ValueError):
        potencia_quadrados_sucessivos(P_REF, 2.5)       # expoente precisa ser inteiro


def test_diagonalizacao_reconstroi_P():
    rng = np.random.default_rng(2026)
    for P in [P_REF] + [_aleatoria(rng) for _ in range(10)]:
        V, lam, Vinv = diagonaliza(P)
        np.testing.assert_allclose(V @ np.diag(lam) @ Vinv, P, atol=1e-12)
        np.testing.assert_allclose(V @ Vinv, np.eye(4), atol=1e-12)
        assert (np.diff(np.abs(lam)) <= 1e-15).all()


def test_potencias_diagonalizacao_vs_quadrados_em_matrizes_aleatorias():
    rng = np.random.default_rng(2026)
    for _ in range(10):
        P = _aleatoria(rng)
        V, lam, Vinv = diagonaliza(P)
        d = potencia_por_diagonalizacao(V, lam, Vinv, 52)
        q, _ = potencia_quadrados_sucessivos(P, 52)
        assert np.abs(d - q).max() < 1e-12


# ------------------------------------------------------ 4) número de condição
def test_norma_1_a_mao_e_contra_numpy():
    M = np.array([[1.0, -2.0], [3.0, 4.0]])
    assert norma_1(M) == 6.0                      # colunas: |1|+|3| = 4 e |-2|+|4| = 6
    rng = np.random.default_rng(2026)
    for _ in range(5):
        A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        assert norma_1(A) == pytest.approx(np.linalg.norm(A, 1), rel=1e-14)


def test_kappa1_identidade_e_um_e_confere_com_numpy():
    assert numero_condicao_1(np.eye(4)) == pytest.approx(1.0, abs=1e-14)
    V, lam, Vinv = diagonaliza(P_REF)
    k = numero_condicao_1(V)
    assert k == pytest.approx(np.linalg.cond(V, 1), rel=1e-9)
    assert k >= 1.0
    assert k == pytest.approx(norma_1(V) * norma_1(Vinv), rel=1e-9)


def test_kappa1_a_mao_colunas_quase_paralelas():
    # V = [[1, 1], [0, e]]: ||V||_1 = 1 + e;  V^-1 = [[1, -1/e], [0, 1/e]]: ||V^-1||_1 = 2/e  =>  κ1 = 2(1+e)/e
    for e in (1e-1, 1e-3, 1e-6):
        V = np.array([[1.0, 1.0], [0.0, e]])
        assert numero_condicao_1(V) == pytest.approx(2 * (1 + e) / e, rel=1e-9)


# ----------------------------------------- 5) complexos e não primitiva
def test_autovalores_complexos_ordem_e_meia_vida_analitica():
    ciclo = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
    P = 0.5 * np.eye(3) + 0.5 * ciclo             # autovalores 1 e 0,5 + 0,5 ω, ω = e^{±2πi/3}
    esp = calcula_espectro(P)
    assert np.iscomplexobj(esp)
    assert esp[0] == pytest.approx(1.0)
    assert abs(esp[1]) == pytest.approx(0.5, abs=1e-12) == pytest.approx(abs(esp[2]), abs=1e-12)
    assert esp[1].imag > 0 > esp[2].imag          # par conjugado, positivo primeiro (ordem determinística)
    assert calcula_meia_vida(P) == pytest.approx(1.0, abs=1e-12)    # ln(1/2)/ln(1/2)
    V, lam, Vinv = diagonaliza(P)
    d = potencia_por_diagonalizacao(V, lam, Vinv, 10)
    assert np.abs(d.imag).max() < 1e-12            # resultado é real, a parte imaginária é ruído
    assert np.abs(d.real - np.linalg.matrix_power(P, 10)).max() < 1e-12


def test_cadeia_periodica_tem_meia_vida_infinita():
    assert calcula_meia_vida(np.array([[0.0, 1.0], [1.0, 0.0]])) == np.inf   # |λ2| = 1
    assert calcula_meia_vida(np.eye(3)) == np.inf


def test_dois_estados_autovalor_1_menos_a_menos_b():
    a, b = 0.3, 0.1
    P = np.array([[1 - a, a], [b, 1 - b]])
    esp = calcula_espectro(P)
    np.testing.assert_allclose(esp, [1.0, 1 - a - b], atol=1e-14)
    assert calcula_meia_vida(P) == pytest.approx(np.log(0.5) / np.log(0.6), rel=1e-12)
# [/IA]
