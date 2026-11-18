# [IA] Claude Sonnet 5.5 - 2026-11-18 - tests/test_etapa4.py: testes do pipeline da Etapa 4 (espectro, meia-vida, P^52 por duas vias, kappa_1, curvas de decaimento)
"""Testes de src/etapa4.py: espectro, memória e diagonalização das matrizes de Serra, Vitória e ES."""
import numpy as np
import pandas as pd
import pytest

from src import etapa4
from src.etapa3 import SERRA, ESCOLHIDO, contagens_e_ocupacao, geocodes_do_estado, matriz_transicao
from src.etapa4 import ESTILOS, analisar_matriz, constante_C, curvas_decaimento
from src.markov import calcula_distribuicao_estacionaria_sistema as pi_sistema

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])


def _P(geocodes):
    C, _ = contagens_e_ocupacao(geocodes_do_estado() if geocodes == "estado" else geocodes)
    return matriz_transicao(C)


TERRITORIOS = [[SERRA], [ESCOLHIDO], "estado"]


# ------------------------------------------------------ Tabela 1 (P_ref)
def test_analisar_matriz_reproduz_tabela1():
    r = analisar_matriz(P_REF)
    np.testing.assert_allclose(r["espectro"], [1.0, 0.869971, 0.618816, 0.461213], atol=1e-6)
    assert r["lambda2"] == pytest.approx(0.869971, abs=1e-6)
    assert r["meia_vida"] == pytest.approx(4.976086, abs=1e-6)
    assert r["dif_traco"] < 1e-10 and r["dif_det"] < 1e-10
    assert r["dif_traco_poli"] < 1e-10 and r["dif_det_poli"] < 1e-10
    assert r["discrepancia_espectro"] < 1e-10
    assert r["produtos_quadrados"] == 7
    assert r["dif_P52_diagonalizacao"] < 1e-12
    assert r["dif_P52_matrix_power"] < 1e-12 and r["dif_P52_multiplicacao"] < 1e-12
    assert r["kappa1"] >= 1 and np.isfinite(r["kappa1"])
    assert r["distancia_minima_autovalores"] > 0.1            # autovalores distintos => diagonalizável
    assert r["residuo_diagonalizacao"] < 1e-12


def test_curvas_conferem_com_potencias_de_matriz_por_outro_caminho():
    pi = pi_sistema(P_REF)
    curvas = curvas_decaimento(P_REF, pi, 60)
    assert curvas.shape == (4, 61)
    for n in (0, 1, 4, 20, 50):
        Pn = np.linalg.matrix_power(P_REF, n)
        esperado = np.abs(Pn - pi).sum(axis=1)                # ||e_i P^n - pi||_1 = soma das |linha i de P^n - pi|
        np.testing.assert_allclose(curvas[:, n], esperado, atol=1e-13)
    assert curvas[0, 20] == pytest.approx(0.045556, abs=1e-6)   # Tabela 1
    assert curvas[0, 0] == pytest.approx(np.abs(np.eye(4)[0] - pi).sum(), abs=1e-15)


def test_decaimento_assintotico_tem_razao_lambda2_segundo_caminho_para_lambda2():
    # |λ2| também sai da razão de erros sucessivos (independente de qualquer decomposição espectral)
    r = analisar_matriz(P_REF)
    c = r["curvas"]
    for i in range(4):
        razao = c[i, 71] / c[i, 70]
        assert razao == pytest.approx(r["lambda2"], abs=2e-3)


def test_constante_C_majora_as_curvas_e_e_justa():
    r = analisar_matriz(P_REF)
    c, lam2, C = r["curvas"], r["lambda2"], r["C"]
    n = np.arange(c.shape[1])
    ref = C * lam2 ** n
    valido = c > 1e-10
    assert (c[valido] <= ref[np.broadcast_to(n, c.shape)[valido]] * (1 + 1e-12)).all()   # majora
    razoes = (c / lam2 ** n)[valido]
    assert razoes.max() == pytest.approx(C, rel=1e-12)                                  # e encosta em algum ponto
    assert C > 0


def test_constante_C_a_mao():
    curvas = np.array([[2.0, 1.0, 0.5], [1.0, 0.5, 0.25]])     # = 2·(1/2)^n e (1/2)^n
    assert constante_C(curvas, 0.5, piso=1e-10) == pytest.approx(2.0)


def test_estilos_sao_distintos_e_sem_cor():
    assert len(ESTILOS) == 4
    assert len({(e["linestyle"], e["marker"]) for e in ESTILOS}) == 4        # tipo de linha + marcador distintos
    assert all(e["color"] in ("black", "0.25", "0.4") for e in ESTILOS)       # tons de cinza apenas


# ------------------------------------------------------- dados reais
@pytest.mark.parametrize("geocodes", TERRITORIOS)
def test_verificacoes_obrigatorias_nos_dados_reais(geocodes):
    P = _P(geocodes)
    r = analisar_matriz(P)
    esp = r["espectro"]
    assert abs(esp[0] - 1) < 1e-12 and (np.abs(esp[1:]) < 1).all()
    assert r["dif_traco"] < 1e-10 and r["dif_det"] < 1e-10                  # critério do enunciado
    assert r["dif_traco_poli"] < 1e-10 and r["dif_det_poli"] < 1e-10
    assert r["discrepancia_espectro"] < 1e-10
    assert r["dif_P52_diagonalizacao"] < 1e-10
    assert r["dif_P52_matrix_power"] < 1e-12 and r["dif_P52_multiplicacao"] < 1e-12
    assert r["imag_max_P52"] < 1e-12
    assert r["produtos_quadrados"] == 7
    assert r["distancia_minima_autovalores"] > 1e-3 and np.isfinite(r["kappa1"]) and r["kappa1"] < 1e4
    assert r["meia_vida"] == pytest.approx(np.log(0.5) / np.log(r["lambda2"]), rel=1e-12)
    # segundo caminho para |λ2|: raio espectral de P − 1π (deflação)
    D = P - np.outer(np.ones(4), r["pi"])
    assert r["lambda2"] == pytest.approx(np.abs(np.linalg.eigvals(D)).max(), abs=1e-10)
    # P^52 já "esqueceu" a origem: cada linha está perto de π (relação com a meia-vida)
    assert r["erro_P52_max"] < 0.05


@pytest.mark.parametrize("geocodes", [[SERRA], "estado"])
def test_erro_P52_max_confere_com_matrix_power(geocodes):
    P = _P(geocodes)
    r = analisar_matriz(P)
    esperado = np.abs(np.linalg.matrix_power(P, 52) - r["pi"]).sum(axis=1).max()
    assert r["erro_P52_max"] == pytest.approx(esperado, abs=1e-13)
    assert r["erro_P52_max"] > 1e-4          # após 52 semanas ainda resta memória: não é zero


def test_distancia_minima_entre_autovalores_a_mao():
    # P diagonal estocástica não existe com autovalores distintos; usa 2 estados: λ = 1 e 1 − a − b
    P = np.array([[0.7, 0.3], [0.1, 0.9]])               # λ2 = 0,6
    assert analisar_matriz(P)["distancia_minima_autovalores"] == pytest.approx(0.4, abs=1e-12)


def test_serra_tem_par_complexo_e_vitoria_es_tem_espectro_real():
    rs, rv, re = (analisar_matriz(_P(g)) for g in TERRITORIOS)
    assert np.iscomplexobj(rs["espectro"]) and abs(rs["espectro"][1].imag) > 1e-3
    assert rs["espectro"][1] == pytest.approx(np.conj(rs["espectro"][2]), abs=1e-12)
    assert rs["arg_lambda2"] > 0
    for r in (rv, re):
        assert not np.iscomplexobj(r["espectro"]) and r["arg_lambda2"] == 0
    # ordem de memória observada: Vitória > ES > Serra (|λ2| = 0,9113; 0,8858; 0,8794)
    assert rv["lambda2"] > re["lambda2"] > rs["lambda2"]


def test_curvas_reais_decaem_e_ficam_sob_a_reta():
    for g in TERRITORIOS:
        r = analisar_matriz(_P(g))
        c = r["curvas"]
        assert c.shape[0] == 4 and c.shape[1] > 100
        assert c[:, -1].max() < 1e-9                         # a janela cobre a convergência
        n = np.arange(c.shape[1])
        valido = c > 1e-10
        assert (c[valido] <= (r["C"] * r["lambda2"] ** n * (1 + 1e-12))[np.broadcast_to(n, c.shape)[valido]]).all()


# ------------------------------------------------------------- main
def test_main_grava_tabelas_e_figuras(tmp_path):
    etapa4.main(saida=tmp_path)
    esp = pd.read_csv(tmp_path / "tabelas" / "e4_espectro.csv")
    res = pd.read_csv(tmp_path / "tabelas" / "e4_resumo.csv")
    assert set(esp["territorio"]) == {"Serra", "Vitória", "ES"} and len(esp) == 12
    assert list(esp.columns) == ["territorio", "k", "re", "im", "modulo"]
    assert len(res) == 3
    for col in ("lambda2", "meia_vida", "arg_lambda2", "kappa1", "dif_P52_diagonalizacao", "C",
                "dif_traco", "dif_det", "discrepancia_espectro"):
        assert col in res.columns
    assert (res["dif_traco"] < 1e-10).all() and (res["dif_det"] < 1e-10).all()
    for nome in ("serra", "vitoria", "es"):
        assert (tmp_path / "figuras" / f"e4_decaimento_{nome}.png").stat().st_size > 5_000


@pytest.mark.parametrize("campo, mensagem", [
    ("dif_traco", "traço"), ("dif_traco_poli", "traço"),
    ("dif_det", "determinante"), ("dif_det_poli", "determinante"),
    ("dif_P52_diagonalizacao", r"P\^52"),
])
def test_main_para_se_uma_verificacao_obrigatoria_falha(tmp_path, monkeypatch, campo, mensagem):
    real = etapa4.analisar_matriz
    monkeypatch.setattr(etapa4, "analisar_matriz", lambda P, pi=None: {**real(P, pi), campo: 1.0})
    with pytest.raises(SystemExit, match=mensagem):
        etapa4.main(saida=tmp_path)


def test_main_para_se_os_dois_caminhos_do_espectro_discordam(tmp_path, monkeypatch):
    real = etapa4.analisar_matriz
    monkeypatch.setattr(etapa4, "analisar_matriz", lambda P, pi=None: {**real(P, pi), "discrepancia_espectro": 1.0})
    with pytest.raises(SystemExit, match="espectro"):
        etapa4.main(saida=tmp_path)
# [/IA]
