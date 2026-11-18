# [IA] Claude Sonnet 5.5 - 2026-11-18 - tests/test_etapa3.py: testes do pipeline da Etapa 3 (P estimada, 3 metodos em Serra/Vitoria/ES, ocupacao, saidas)
"""Testes de src/etapa3.py: dos dados congelados até π por três caminhos."""
import numpy as np
import pandas as pd
import pytest

from src import etapa3
from src.dados import carregar_serie, contar_transicoes
from src.etapa3 import (
    analisar,
    contagens_e_ocupacao,
    geocodes_do_estado,
    matriz_transicao,
    trajetoria_erro,
)
from src.markov import menor_k_positivo

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])
PI_REF = np.array([0.498970, 0.205954, 0.174125, 0.120951])
SERRA, VITORIA = 3205002, 3205309


# ------------------------------------------------------------- matriz_transicao
def test_matriz_transicao_a_mao():
    P = matriz_transicao(np.array([[2, 2], [1, 3]]))
    np.testing.assert_allclose(P, [[0.5, 0.5], [0.25, 0.75]], atol=0, rtol=0)


def test_matriz_transicao_linha_sem_saidas_levanta_erro():
    with pytest.raises(ValueError, match="linha"):
        matriz_transicao(np.array([[2, 2], [0, 0]]))


def test_matriz_transicao_rejeita_contagem_negativa_ou_nao_quadrada():
    with pytest.raises(ValueError):
        matriz_transicao(np.array([[1, -1], [1, 1]]))
    with pytest.raises(ValueError):
        matriz_transicao(np.ones((2, 3)))


# ------------------------------------------------ contagens e ocupação (dados reais)
def test_geocodes_do_estado_sao_78_e_incluem_serra_e_vitoria():
    g = geocodes_do_estado()
    assert len(g) == 78 and SERRA in g and VITORIA in g


def test_contagens_serra_conferem_com_dados_e_com_ocupacao():
    C, ocup = contagens_e_ocupacao([SERRA])
    assert (C == contar_transicoes(carregar_serie(SERRA))).all()
    assert C.sum() == 573
    # ocupação = semanas por nível na série (segundo caminho: value_counts)
    vc = carregar_serie(SERRA)["nivel"].value_counts()
    assert ocup.tolist() == [int(vc.get(k, 0)) for k in (1, 2, 3, 4)]


def test_contagens_do_estado_somam_44694_transicoes():
    C, ocup = contagens_e_ocupacao(geocodes_do_estado())
    assert C.sum() == 44694              # total conferido na auditoria da E1
    assert (C >= 0).all() and ocup.sum() > C.sum()


# ------------------------------------------------------------------- analisar
def test_pipeline_reproduz_tabela1_do_enunciado():
    # contagens fabricadas 1000 por linha => P estimada = P_ref exatamente
    r = analisar(np.rint(P_REF * 1000).astype(int), np.array([1, 1, 1, 1]))
    np.testing.assert_allclose(r["pi_sistema"], PI_REF, atol=1e-6)
    assert r["menor_k"] == 2 and r["wielandt"] == 10


@pytest.mark.parametrize("geocodes", [[SERRA], [VITORIA], "estado"])
def test_tres_metodos_concordam_nos_dados_reais(geocodes):
    g = geocodes_do_estado() if geocodes == "estado" else geocodes
    C, ocup = contagens_e_ocupacao(g)
    r = analisar(C, ocup)
    P = r["P"]
    assert np.allclose(P.sum(axis=1), 1)
    assert r["discordancia_max"] < 1e-10                      # critério do enunciado
    for nome in ("sistema", "autovetor", "potencias_e1", "potencias_e4", "potencias_uniforme"):
        pi = r["pi_" + nome]
        assert abs(pi.sum() - 1) < 1e-12 and (pi >= 0).all()
        assert r["residuo_" + nome] < 1e-10
    for ini in ("e1", "e4", "uniforme"):
        assert r["convergiu_" + ini] and 0 < r["iteracoes_" + ini] < 100_000
    # quarto caminho, independente dos três: P^k para k grande tem todas as linhas ≈ π
    Pk = np.linalg.matrix_power(P, 4000)
    np.testing.assert_allclose(Pk, np.tile(r["pi_sistema"], (4, 1)), atol=1e-9)
    # primitividade: segundo cálculo do menor k e limite de Wielandt
    assert r["menor_k"] == menor_k_positivo(P) and 1 <= r["menor_k"] <= r["wielandt"] == 10
    assert (np.linalg.matrix_power(P, r["menor_k"]) > 0).all()


def test_ocupacao_e_pi_sao_distribuicoes_diferentes_porem_proximas_ordem():
    C, ocup = contagens_e_ocupacao([SERRA])
    r = analisar(C, ocup)
    assert abs(r["freq_empirica"].sum() - 1) < 1e-12
    assert r["l1_pi_vs_empirica"] == pytest.approx(np.abs(r["pi_sistema"] - r["freq_empirica"]).sum())
    assert r["l1_pi_vs_empirica"] > 0       # π ≠ frequência empírica (série não estacionária)


def test_iteracoes_seguem_lambda2():
    C, ocup = contagens_e_ocupacao([SERRA])
    r = analisar(C, ocup)
    assert 0 < r["lambda2"] < 1
    previsto = np.log(1e-12) / np.log(r["lambda2"])
    # partindo do pior caso (e1/e4) o nº de passos fica da ordem do previsto
    for ini in ("e1", "e4", "uniforme"):
        assert 0.3 * previsto < r["iteracoes_" + ini] < 1.5 * previsto


def test_trajetoria_erro_decresce_e_termina_perto_de_zero():
    from src.markov import calcula_distribuicao_estacionaria_sistema as sistema
    pi = sistema(P_REF)                       # π exato (não o arredondado da Tabela 1)
    e1 = np.array([1.0, 0, 0, 0])
    erros = trajetoria_erro(P_REF, pi, e1, 300)
    assert erros.shape == (301,) and erros[0] == pytest.approx(np.abs(e1 - pi).sum())
    assert erros[-1] < 1e-8 and erros[50] < erros[10]
    assert erros[20] == pytest.approx(0.045556, abs=1e-6)   # Tabela 1: ‖e1 P^20 − π‖₁


def _iteracoes_a_mao(P, pi0, tol=1e-12):
    pi = np.asarray(pi0, dtype=float)
    for k in range(1, 100_000):
        nxt = pi @ P
        if np.abs(nxt - pi).sum() < tol:
            return k, nxt
        pi = nxt


def test_iteracoes_e_pi_de_cada_inicio_conferem_com_laco_independente():
    C, ocup = contagens_e_ocupacao([SERRA])
    r = analisar(C, ocup)
    for nome, pi0 in (("e1", np.eye(4)[0]), ("e4", np.eye(4)[3]), ("uniforme", np.full(4, 0.25))):
        k, pi = _iteracoes_a_mao(r["P"], pi0)
        assert r["iteracoes_" + nome] == k
        np.testing.assert_allclose(r["pi_potencias_" + nome], pi, atol=0, rtol=0)
    assert r["iteracoes_e1"] != r["iteracoes_e4"]      # os inícios realmente diferem


@pytest.mark.parametrize("geocodes", [[SERRA], [VITORIA], "estado"])
def test_lambda2_confere_com_deflacao_independente(geocodes):
    # P − 1π tem os autovalores de P com o 1 trocado por 0; seu maior módulo é |λ2|
    # (em Serra λ2, λ3 são um par complexo de mesmo módulo; Vitória e ES têm módulos distintos)
    C, ocup = contagens_e_ocupacao(geocodes_do_estado() if geocodes == "estado" else geocodes)
    r = analisar(C, ocup)
    D = r["P"] - np.outer(np.ones(4), r["pi_sistema"])
    assert r["lambda2"] == pytest.approx(np.abs(np.linalg.eigvals(D)).max(), abs=1e-10)


def test_analisar_menor_k_atinge_wielandt_com_contagens_de_ciclo():
    C = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0]])
    r = analisar(C, np.array([1, 1, 1, 1]))
    assert r["menor_k"] == 10 == r["wielandt"]
    assert r["discordancia_max"] < 1e-10


def test_main_para_se_os_metodos_discordam(tmp_path, monkeypatch):
    real = etapa3.analisar
    monkeypatch.setattr(etapa3, "analisar", lambda C, o: {**real(C, o), "discordancia_max": 1.0})
    with pytest.raises(SystemExit, match="discordam"):
        etapa3.main(saida=tmp_path)


# ------------------------------------------------------------------------- main
def test_main_grava_tabelas_e_figura(tmp_path):
    tabela = etapa3.main(saida=tmp_path)
    assert (tmp_path / "tabelas" / "e3_pi_metodos.csv").exists()
    assert (tmp_path / "tabelas" / "e3_resumo.csv").exists()
    assert (tmp_path / "figuras" / "e3_convergencia_serra.png").stat().st_size > 5_000
    pi = pd.read_csv(tmp_path / "tabelas" / "e3_pi_metodos.csv")
    assert set(pi["territorio"]) == {"Serra", "Vitória", "ES"}
    assert set(pi["metodo"]) == {"sistema", "autovetor", "potencias_e1", "potencias_e4", "potencias_uniforme", "freq_empirica"}
    res = pd.read_csv(tmp_path / "tabelas" / "e3_resumo.csv")
    assert len(res) == 3 and (res["discordancia_max"] < 1e-10).all()
    assert res.set_index("territorio")["transicoes"].to_dict() == {"Serra": 573, "Vitória": 573, "ES": 44694}
    assert len(tabela) == len(pi)
# [/IA]
