# [IA] Claude Sonnet 5.5 - 2026-11-18 - tests/test_markov_e3.py: testes do nucleo da Etapa 3 (3 metodos, potencias com diagnostico, primitividade, Wielandt, residuo)
"""Testes do núcleo da Etapa 3 (src/markov.py).

Quatro grupos:
  1) Tabela 1 do enunciado (P_ref), com erro absoluto < 1e-6;
  2) propriedades em matrizes aleatórias (semente 2026): os três métodos concordam até 1e-10;
  3) casos-limite que mostram por que a primitividade importa (cadeia periódica, reduzível);
  4) menor k com P^k > 0 e o limite de Wielandt (n-1)^2 + 1.
"""
import numpy as np
import pytest

from src.markov import (
    ResultadoPotencias,
    calcula_distribuicao_estacionaria_autovetor,
    calcula_distribuicao_estacionaria_potencias,
    calcula_distribuicao_estacionaria_sistema,
    limite_wielandt,
    menor_k_positivo,
    residuo_estacionario,
    validar_estocastica,
)

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])
PI_REF = np.array([0.498970, 0.205954, 0.174125, 0.120951])  # Tabela 1


def _e(i, n=4):
    v = np.zeros(n)
    v[i] = 1.0
    return v


# ------------------------------------------------------------ 1) Tabela 1
def test_tabela1_pi_pelos_tres_metodos():
    np.testing.assert_allclose(calcula_distribuicao_estacionaria_sistema(P_REF), PI_REF, atol=1e-6)
    np.testing.assert_allclose(calcula_distribuicao_estacionaria_potencias(P_REF).pi, PI_REF, atol=1e-6)
    np.testing.assert_allclose(calcula_distribuicao_estacionaria_autovetor(P_REF), PI_REF, atol=1e-6)


def test_tres_metodos_concordam_ate_1e10_na_pref():
    a = calcula_distribuicao_estacionaria_sistema(P_REF)
    b = calcula_distribuicao_estacionaria_potencias(P_REF).pi
    c = calcula_distribuicao_estacionaria_autovetor(P_REF)
    for x, y in [(a, b), (a, c), (b, c)]:
        assert np.abs(x - y).sum() < 1e-10


@pytest.mark.parametrize("inicio", ["e1", "e4", "uniforme"])
def test_potencias_a_partir_de_e1_e4_e_uniforme(inicio):
    pi0 = {"e1": _e(0), "e4": _e(3), "uniforme": np.full(4, 0.25)}[inicio]
    r = calcula_distribuicao_estacionaria_potencias(P_REF, pi0=pi0)
    assert isinstance(r, ResultadoPotencias)
    assert r.convergiu is True
    assert 0 < r.iteracoes < 10_000
    np.testing.assert_allclose(r.pi, PI_REF, atol=1e-6)
    assert residuo_estacionario(r.pi, P_REF) < 1e-10


def test_potencias_partindo_de_e1_precisa_de_mais_passos_que_do_uniforme():
    # a cadeia sai "longe" de pi quando parte de e1; o número de iterações reflete isso
    k_e1 = calcula_distribuicao_estacionaria_potencias(P_REF, pi0=_e(0)).iteracoes
    k_uni = calcula_distribuicao_estacionaria_potencias(P_REF, pi0=np.full(4, 0.25)).iteracoes
    assert k_e1 > k_uni


def test_iteracoes_coerentes_com_lambda2():
    # erro ~ |lambda2|^k; com |lambda2| = 0,869971 e tol = 1e-12, k fica perto de ln(1e-12)/ln(|lambda2|) = 198
    k = calcula_distribuicao_estacionaria_potencias(P_REF, pi0=_e(0)).iteracoes
    previsto = np.log(1e-12) / np.log(0.869971)
    assert 0.7 * previsto < k < 1.3 * previsto


def test_contagem_exata_de_iteracoes_confere_com_laco_independente():
    # recontagem à mão: k = primeiro passo em que ||pi(k) - pi(k-1)||_1 < 1e-12
    pi = _e(0)
    for k in range(1, 100_000):
        nxt = pi @ P_REF
        if np.abs(nxt - pi).sum() < 1e-12:
            break
        pi = nxt
    r = calcula_distribuicao_estacionaria_potencias(P_REF, pi0=_e(0))
    assert r.iteracoes == k
    np.testing.assert_allclose(r.pi, nxt, atol=0, rtol=0)


def test_residuo_zero_no_pi_certo_e_positivo_no_errado():
    assert residuo_estacionario(PI_REF, P_REF) < 1e-5
    pi_certo = calcula_distribuicao_estacionaria_sistema(P_REF)
    assert residuo_estacionario(pi_certo, P_REF) < 1e-14
    assert residuo_estacionario(np.full(4, 0.25), P_REF) > 0.05


# ------------------------------------------------ 2) propriedades aleatórias
def _aleatoria(rng, n=4):
    M = rng.random((n, n)) + 0.05  # estritamente positiva => primitiva
    return M / M.sum(axis=1, keepdims=True)


def test_propriedades_em_20_matrizes_aleatorias():
    rng = np.random.default_rng(2026)
    for _ in range(20):
        P = _aleatoria(rng)
        a = calcula_distribuicao_estacionaria_sistema(P)
        b = calcula_distribuicao_estacionaria_potencias(P, pi0=_e(0)).pi
        c = calcula_distribuicao_estacionaria_autovetor(P)
        for x in (a, b, c):
            assert abs(x.sum() - 1) < 1e-12
            assert (x >= 0).all()
            assert residuo_estacionario(x, P) < 1e-10
        assert np.abs(a - b).sum() < 1e-10 and np.abs(a - c).sum() < 1e-10


@pytest.mark.parametrize("a, b", [(0.3, 0.1), (0.5, 0.5), (0.01, 0.2)])
def test_cadeia_de_2_estados_tem_formula_fechada(a, b):
    P = np.array([[1 - a, a], [b, 1 - b]])
    esperado = np.array([b, a]) / (a + b)
    np.testing.assert_allclose(calcula_distribuicao_estacionaria_sistema(P), esperado, atol=1e-12)
    np.testing.assert_allclose(calcula_distribuicao_estacionaria_autovetor(P), esperado, atol=1e-12)
    np.testing.assert_allclose(calcula_distribuicao_estacionaria_potencias(P).pi, esperado, atol=1e-10)


# ------------------------------------------------------------ 3) casos-limite
PERIODICA = np.array([[0.0, 1.0], [1.0, 0.0]])  # período 2: não é primitiva


def test_cadeia_periodica_potencias_nao_convergem_e_avisam():
    with pytest.warns(RuntimeWarning, match="convergiu"):
        r = calcula_distribuicao_estacionaria_potencias(PERIODICA, pi0=_e(0, 2), max_iter=500)
    assert r.convergiu is False
    assert r.iteracoes == 500
    # já o sistema linear encontra o pi único (0,5; 0,5): a primitividade só afeta a convergência
    np.testing.assert_allclose(calcula_distribuicao_estacionaria_sistema(PERIODICA), [0.5, 0.5], atol=1e-12)


def test_cadeia_reduzivel_pi_nao_e_unico_e_o_codigo_acusa():
    identidade = np.eye(3)  # cada estado é absorvente: qualquer distribuição é estacionária
    with pytest.raises(np.linalg.LinAlgError):
        calcula_distribuicao_estacionaria_sistema(identidade)
    with pytest.raises(ValueError, match="simples"):
        calcula_distribuicao_estacionaria_autovetor(identidade)


def test_max_iter_pequeno_avisa_nao_convergencia():
    with pytest.warns(RuntimeWarning, match="convergiu"):
        r = calcula_distribuicao_estacionaria_potencias(P_REF, pi0=_e(0), max_iter=3)
    assert r.convergiu is False and r.iteracoes == 3


def test_validacao_de_entrada():
    with pytest.raises(ValueError):
        validar_estocastica(np.ones((2, 3)) / 3)                  # não quadrada
    with pytest.raises(ValueError):
        validar_estocastica(np.array([[1.2, -0.2], [0.5, 0.5]]))  # entrada negativa
    with pytest.raises(ValueError):
        validar_estocastica(np.array([[0.5, 0.4], [0.5, 0.5]]))   # linha soma 0,9
    for f in (calcula_distribuicao_estacionaria_sistema, calcula_distribuicao_estacionaria_autovetor):
        with pytest.raises(ValueError):
            f(np.array([[0.5, 0.4], [0.5, 0.5]]))
    with pytest.raises(ValueError):
        calcula_distribuicao_estacionaria_potencias(P_REF, pi0=np.array([0.5, 0.5, 0.5, 0.5]))  # soma 2


# --------------------------------------------- 4) menor k e limite de Wielandt
WIELANDT_4 = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.5, 0.5, 0, 0]])  # caso extremo


def test_limite_de_wielandt():
    assert [limite_wielandt(n) for n in (2, 3, 4)] == [2, 5, 10]


def test_menor_k_positivo_da_tabela1():
    assert menor_k_positivo(P_REF) == 2


def test_menor_k_atinge_exatamente_o_limite_de_wielandt():
    assert menor_k_positivo(WIELANDT_4) == 10 == limite_wielandt(4)


def test_menor_k_para_matriz_positiva_e_1():
    assert menor_k_positivo(np.full((4, 4), 0.25)) == 1


@pytest.mark.parametrize("P", [PERIODICA, np.eye(3), np.array([[1.0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5]])])
def test_menor_k_none_quando_nao_primitiva(P):
    assert menor_k_positivo(P) is None


def test_menor_k_nunca_passa_de_wielandt_em_matrizes_primitivas():
    rng = np.random.default_rng(2026)
    for _ in range(200):
        n = 4
        P = (rng.random((n, n)) < 0.4) * rng.random((n, n))
        P[np.arange(n), (np.arange(n) + 1) % n] += 0.5  # garante um ciclo; pode ou não ser primitiva
        P = P / P.sum(axis=1, keepdims=True)
        k = menor_k_positivo(P)
        if k is not None:
            assert 1 <= k <= limite_wielandt(n)
            Pk = np.linalg.matrix_power(P, k)
            assert (Pk > 0).all()                                  # k funciona...
            if k > 1:
                assert not (np.linalg.matrix_power(P, k - 1) > 0).all()  # ...e k-1 não
# [/IA]
