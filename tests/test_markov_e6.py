# [IA] Claude Sonnet 5.5 - 2026-11-23 - tests/test_markov_e6.py: testes do nucleo da Etapa 6 (cadeia absorvente: N, t, variancia, B, serie de Neumann, simulacao)
"""Testes da cadeia absorvente em src/markov.py: Tabela 1, casos à mão, propriedades e falhas esperadas."""
import numpy as np
import pytest

from src.markov import (
    matriz_fundamental_inversa,
    menor_m_norma_menor_que_1,
    neumann_ate_tolerancia,
    particao_absorvente,
    prob_absorcao,
    prob_absorcao_sistema,
    simular_absorcao,
    soma_parcial_neumann,
    tempo_ate_absorcao,
    tempo_ate_absorcao_sistema,
    trajetoria_neumann,
    variancia_tempo_absorcao,
)

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])
# Tabela 1 do enunciado (nível 4 absorvente, saindo de 1, 2 e 3)
T_TABELA1 = np.array([46.873065, 39.473684, 26.470588])
VAR_TABELA1 = np.array([1788.363542, 1732.320352, 1427.271899])
# Tabela 1: B com transientes {2, 3} e absorventes {1, 4}: linhas = saindo de 2 e de 3; colunas = (1, 4)
B_TABELA1 = np.array([[0.742718, 0.257282], [0.461165, 0.538835]])

# Passeio aleatório simétrico em {0,1,2,3} com 0 e 3 absorventes (ruína do jogador): respostas exatas
P_RUINA = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0],
    [0.0, 0.5, 0.0, 0.5],
    [0.0, 0.0, 0.0, 1.0],
])


def _vermelho_absorvente(P=P_REF):
    _, _, Q, R = particao_absorvente(P, [3])
    return Q, R


def _verde_e_vermelho(P=P_REF):
    _, _, Q, R = particao_absorvente(P, [0, 3])
    return Q, R


# ---------------------------------------------------------------- partição Q, R
def test_particao_blocos_de_pref():
    transientes, absorventes, Q, R = particao_absorvente(P_REF, [0, 3])
    assert transientes == [1, 2] and absorventes == [0, 3]
    np.testing.assert_array_equal(Q, [[0.65, 0.13], [0.15, 0.65]])
    np.testing.assert_array_equal(R, [[0.20, 0.02], [0.05, 0.15]])


def test_particao_ignora_a_linha_dos_absorventes_e_aceita_ordem_qualquer():
    adulterada = P_REF.copy()
    adulterada[3] = [0.25, 0.25, 0.25, 0.25]  # a linha do estado absorvente não pode influir
    _, _, Q1, R1 = particao_absorvente(P_REF, [3])
    _, _, Q2, R2 = particao_absorvente(adulterada, [3])
    np.testing.assert_array_equal(Q1, Q2)
    np.testing.assert_array_equal(R1, R2)
    assert particao_absorvente(P_REF, [3, 0, 3])[1] == [0, 3]


@pytest.mark.parametrize("absorventes", [[], [4], [-1], [0, 1, 2, 3]])
def test_particao_rejeita_conjuntos_invalidos(absorventes):
    with pytest.raises(ValueError):
        particao_absorvente(P_REF, absorventes)


def test_particao_rejeita_matriz_nao_estocastica():
    with pytest.raises(ValueError):
        particao_absorvente(P_REF * 0.9, [3])


# ---------------------------------------------------------------- Tabela 1
def test_t_reproduz_tabela1_pelos_dois_caminhos():
    Q, _ = _vermelho_absorvente()
    N = matriz_fundamental_inversa(Q)
    np.testing.assert_allclose(tempo_ate_absorcao(N), T_TABELA1, atol=1e-6)
    np.testing.assert_allclose(tempo_ate_absorcao_sistema(Q), T_TABELA1, atol=1e-6)


def test_variancia_reproduz_tabela1():
    Q, _ = _vermelho_absorvente()
    N = matriz_fundamental_inversa(Q)
    np.testing.assert_allclose(variancia_tempo_absorcao(N), VAR_TABELA1, atol=1e-6)
    np.testing.assert_allclose(variancia_tempo_absorcao(N, tempo_ate_absorcao(N)), VAR_TABELA1, atol=1e-6)


def test_B_reproduz_tabela1_pelos_dois_caminhos():
    Q, R = _verde_e_vermelho()
    N = matriz_fundamental_inversa(Q)
    np.testing.assert_allclose(prob_absorcao(N, R), B_TABELA1, atol=1e-6)
    np.testing.assert_allclose(prob_absorcao_sistema(Q, R), B_TABELA1, atol=1e-6)


def test_B_vezes_1_igual_a_1_ate_1e_12():
    Q, R = _verde_e_vermelho()
    N = matriz_fundamental_inversa(Q)
    B = prob_absorcao(N, R)
    np.testing.assert_allclose(B @ np.ones(2), np.ones(2), rtol=0, atol=1e-12)
    np.testing.assert_allclose(prob_absorcao_sistema(Q, R) @ np.ones(2), np.ones(2), rtol=0, atol=1e-12)


def test_caminhos_concordam_ate_1e_10_em_matriz_aleatoria():
    rng = np.random.default_rng(2026)
    for _ in range(20):
        P = rng.random((5, 5)) + 0.05
        P = P / P.sum(axis=1, keepdims=True)
        _, _, Q, R = particao_absorvente(P, [1, 4])
        N = matriz_fundamental_inversa(Q)
        np.testing.assert_allclose(tempo_ate_absorcao(N), tempo_ate_absorcao_sistema(Q), rtol=0, atol=1e-10)
        np.testing.assert_allclose(prob_absorcao(N, R), prob_absorcao_sistema(Q, R), rtol=0, atol=1e-10)
        np.testing.assert_allclose(prob_absorcao(N, R).sum(axis=1), 1.0, rtol=0, atol=1e-12)
        assert (variancia_tempo_absorcao(N) >= -1e-9).all()


# ---------------------------------------------------------------- casos à mão (respostas exatas)
def test_ruina_do_jogador_t_e_B_exatos():
    _, _, Q, R = particao_absorvente(P_RUINA, [0, 3])
    N = matriz_fundamental_inversa(Q)
    np.testing.assert_allclose(N, [[4 / 3, 2 / 3], [2 / 3, 4 / 3]], atol=1e-12)
    np.testing.assert_allclose(tempo_ate_absorcao(N), [2.0, 2.0], atol=1e-12)  # t_i = i (3 − i)
    np.testing.assert_allclose(prob_absorcao(N, R), [[2 / 3, 1 / 3], [1 / 3, 2 / 3]], atol=1e-12)
    np.testing.assert_allclose(variancia_tempo_absorcao(N), [2.0, 2.0], atol=1e-12)


def test_geometrica_um_estado_t_e_variancia_exatos():
    # Um transiente que permanece com prob. q: tempo até sair é geométrico, E = 1/(1−q), Var = q/(1−q)²
    q = 0.75
    P = np.array([[q, 1 - q], [0.0, 1.0]])
    _, _, Q, R = particao_absorvente(P, [1])
    N = matriz_fundamental_inversa(Q)
    assert N[0, 0] == pytest.approx(4.0, abs=1e-12)
    assert tempo_ate_absorcao(N)[0] == pytest.approx(4.0, abs=1e-12)
    assert variancia_tempo_absorcao(N)[0] == pytest.approx(12.0, abs=1e-12)
    assert prob_absorcao(N, R)[0, 0] == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- hipótese da proposição 4 e série de Neumann
def test_menor_m_da_norma_em_pref_e_2():
    Q, _ = _vermelho_absorvente()
    assert np.abs(Q).sum(axis=1).max() == pytest.approx(1.0)  # ∥Q∥∞ = 1: m = 1 não basta
    assert menor_m_norma_menor_que_1(Q) == 2


def test_menor_m_e_1_quando_a_norma_ja_e_menor_que_1():
    assert menor_m_norma_menor_que_1(np.array([[0.2, 0.3], [0.1, 0.5]])) == 1


def test_menor_m_e_none_quando_ha_classe_fechada_e_inversa_recusa():
    Q = np.array([[1.0, 0.0], [0.3, 0.2]])  # o estado 1 nunca sai: Q^m tem linha de soma 1 para todo m
    assert menor_m_norma_menor_que_1(Q, m_max=50) is None
    # a recusa vem da conferência da hipótese (e não só do LinAlgError, que também é um ValueError)
    with pytest.raises(ValueError, match="alcança"):
        matriz_fundamental_inversa(Q)


def test_matriz_fundamental_exige_quadrada():
    with pytest.raises(ValueError):
        matriz_fundamental_inversa(np.ones((2, 3)))


def test_produto_telescopico_I_menos_Q_vezes_soma_parcial():
    # Proposição 4: (I − Q)(I + Q + … + Q^K) = I − Q^(K+1)
    Q, _ = _vermelho_absorvente()
    for K in (0, 1, 5, 40):
        esquerda = (np.eye(3) - Q) @ soma_parcial_neumann(Q, K)
        direita = np.eye(3) - np.linalg.matrix_power(Q, K + 1)
        np.testing.assert_allclose(esquerda, direita, rtol=0, atol=1e-12)


def test_soma_parcial_a_mao_caso_1x1():
    Q = np.array([[0.5]])
    assert soma_parcial_neumann(Q, 0)[0, 0] == 1.0
    assert soma_parcial_neumann(Q, 3)[0, 0] == pytest.approx(1 + 0.5 + 0.25 + 0.125)
    assert soma_parcial_neumann(Q, 60)[0, 0] == pytest.approx(2.0, abs=1e-12)  # N = 1/(1 − 0.5)


@pytest.mark.parametrize("K", [-1, 2.5])
def test_soma_parcial_rejeita_K_invalido(K):
    with pytest.raises(ValueError):
        soma_parcial_neumann(np.array([[0.5]]), K)


def test_neumann_ate_tolerancia_coincide_com_a_inversa_em_pref():
    for Q, _ in (_vermelho_absorvente(), _verde_e_vermelho()):
        N = matriz_fundamental_inversa(Q)
        S, K = neumann_ate_tolerancia(Q, tol=1e-14)
        np.testing.assert_allclose(S, N, rtol=0, atol=1e-10)
        assert K > 1


def test_neumann_nao_converge_se_max_termos_curto():
    Q, _ = _vermelho_absorvente()
    with pytest.raises(ValueError):
        neumann_ate_tolerancia(Q, tol=1e-14, max_termos=5)


def test_trajetoria_neumann_decresce_e_comeca_em_norma_de_N_menos_I():
    Q, _ = _vermelho_absorvente()
    N = matriz_fundamental_inversa(Q)
    erros = trajetoria_neumann(Q, N, 1000)
    assert len(erros) == 1001
    assert erros[0] == pytest.approx(np.abs(N - np.eye(3)).sum(axis=1).max())
    assert (np.diff(erros[:500]) <= 1e-12).all()
    assert erros[-1] < 1e-6
    # o erro cai à taxa do raio espectral de Q (≈ 0,9766): razão de erros sucessivos → ρ(Q)
    rho = np.abs(np.linalg.eigvals(Q)).max()
    assert erros[800] / erros[799] == pytest.approx(rho, abs=1e-3)
    # coerência com a soma parcial explícita
    assert erros[7] == pytest.approx(np.abs(N - soma_parcial_neumann(Q, 7)).sum(axis=1).max(), abs=1e-12)


# ---------------------------------------------------------------- simulação (default_rng(2026))
def test_simulacao_reproduz_t_e_B_de_pref_dentro_do_erro_padrao():
    rng = np.random.default_rng(2026)
    tempos, _ = simular_absorcao(P_REF, [3], 0, 100_000, rng)
    assert tempos.mean() == pytest.approx(T_TABELA1[0], abs=0.6)  # erro-padrão ≈ 0,13
    assert tempos.var(ddof=1) == pytest.approx(VAR_TABELA1[0], rel=0.05)
    _, destinos = simular_absorcao(P_REF, [0, 3], 1, 100_000, rng)
    assert (destinos == 3).mean() == pytest.approx(B_TABELA1[0, 1], abs=0.01)  # erro-padrão ≈ 0,0014


def test_simulacao_e_reprodutivel_com_a_mesma_semente():
    a = simular_absorcao(P_REF, [3], 1, 500, np.random.default_rng(2026))
    b = simular_absorcao(P_REF, [3], 1, 500, np.random.default_rng(2026))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_simulacao_cadeia_deterministica():
    # 0 → 1 → 2 (absorvente): sempre 2 passos, sempre destino 2
    P = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    tempos, destinos = simular_absorcao(P, [2], 0, 50, np.random.default_rng(2026))
    assert (tempos == 2).all() and (destinos == 2).all()


def test_simulacao_destinos_so_em_absorventes_e_tempos_positivos():
    tempos, destinos = simular_absorcao(P_REF, [0, 3], 2, 2000, np.random.default_rng(2026))
    assert set(np.unique(destinos)) <= {0, 3}
    assert (tempos >= 1).all()


def test_simulacao_recusa_inicial_absorvente_e_nao_absorcao():
    with pytest.raises(ValueError):
        simular_absorcao(P_REF, [3], 3, 10, np.random.default_rng(2026))
    with pytest.raises(ValueError):
        simular_absorcao(P_REF, [3], 0, 100, np.random.default_rng(2026), max_passos=2)
# [/IA]
