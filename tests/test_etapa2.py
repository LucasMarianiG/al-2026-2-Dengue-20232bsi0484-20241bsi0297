# [IA] Claude Sonnet 5.5 - 2026-11-23 - tests/test_etapa2.py: testes da Etapa 2 (contagens por dois caminhos, P estimada, erros-padrao, Laplace, verificacoes, saidas)
"""Testes de src/etapa2.py: matriz de contagens, P̂, erros-padrão e suavização de Laplace.

Contrato que src/etapa2.py deve cumprir (escreva o código para fazer estes testes passarem):

  contagens_por_laco(serie)          -> C (4x4, int)  caminho 1: laço explícito sobre pares de semanas
  contagens_por_crosstab(serie)      -> C (4x4, int)  caminho 2: pandas.crosstab (independente do 1)
  erros_padrao(C)                    -> matriz sqrt(p̂_ij (1 - p̂_ij) / n_i)
  suavizar_laplace(C, alpha=0.5)     -> P^(alpha) = (n_ij + alpha) / (n_i + k*alpha), k = nº de níveis
  efeito_laplace(C, alpha=0.5)       -> ‖π - π^(alpha)‖₁
  verificar(C, P, transicoes_validas, tol=1e-12)
                                     -> levanta ValueError se P < 0, P1 != 1 ou soma(C) != transicoes_validas
  main(pasta_raw=PASTA_RAW, saida=Path("results"))
                                     -> grava saida/tabelas/e2_resumo.csv e saida/figuras/e2_mapas_calor.png
                                        e devolve o resumo (linhas: Serra, Vitória, ES;
                                        colunas: territorio, soma_C, transicoes_validas, delta_pi_laplace)

Os valores "a mão" estão comentados para você conferir com a calculadora.
"""
import numpy as np
import pandas as pd
import pytest

from src import etapa2
from src.dados import carregar_serie, contar_transicoes
from src.etapa2 import (
    contagens_por_crosstab,
    contagens_por_laco,
    efeito_laplace,
    erros_padrao,
    suavizar_laplace,
    verificar,
)
from src.etapa3 import ESCOLHIDO, SERRA, contagens_e_ocupacao, geocodes_do_estado, matriz_transicao

CAMINHOS = [contagens_por_laco, contagens_por_crosstab]
BASE = pd.Timestamp("2026-01-04")  # um domingo, como o início de uma SE


def serie(semanas, niveis):
    """Série pequena: semanas = deslocamentos (em semanas) a partir de BASE; niveis pode ter np.nan."""
    return pd.DataFrame({
        "data_iniSE": [BASE + pd.Timedelta(weeks=int(s)) for s in semanas],
        "nivel": np.array(niveis, dtype=float),
    })


# Série construída à mão, com UMA lacuna (falta a semana 4):
#   semana: 0  1  2  3  [4 ausente]  5  6  7
#   nível : 1  1  2  3               3  4  4
# Pares consecutivos válidos: 1→1, 1→2, 2→3, 3→4, 4→4. O par 3→3 atravessaria a lacuna e NÃO conta.
SERIE_LACUNA = serie([0, 1, 2, 3, 5, 6, 7], [1, 1, 2, 3, 3, 4, 4])
C_LACUNA = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
])


# ------------------------------------------------------------------ contagens a mão
@pytest.mark.parametrize("contagem", CAMINHOS)
def test_contagens_a_mao_com_lacuna(contagem):
    np.testing.assert_array_equal(contagem(SERIE_LACUNA), C_LACUNA)


@pytest.mark.parametrize("contagem", CAMINHOS)
def test_nenhuma_transicao_atravessa_lacuna(contagem):
    C = contagem(SERIE_LACUNA)
    assert C[2, 2] == 0      # 3→3 só existiria "pulando" a semana 4
    assert C.sum() == 5      # 7 semanas observadas, 6 pares, 1 deles atravessa a lacuna


@pytest.mark.parametrize("contagem", CAMINHOS)
def test_nivel_ausente_interrompe_a_sequencia(contagem):
    # níveis 1, 2, (ausente), 2, 3 em semanas consecutivas: só valem 1→2 e 2→3
    C = contagem(serie([0, 1, 2, 3, 4], [1, 2, np.nan, 2, 3]))
    esperado = np.zeros((4, 4), dtype=int)
    esperado[0, 1] = 1
    esperado[1, 2] = 1
    np.testing.assert_array_equal(C, esperado)


@pytest.mark.parametrize("contagem", CAMINHOS)
def test_serie_com_uma_semana_nao_gera_transicoes(contagem):
    C = contagem(serie([0], [2]))
    assert C.shape == (4, 4)
    assert C.sum() == 0


@pytest.mark.parametrize("contagem", CAMINHOS)
def test_contagens_sao_inteiras_e_nao_negativas(contagem):
    C = contagem(SERIE_LACUNA)
    assert np.issubdtype(C.dtype, np.integer)
    assert (C >= 0).all()


def test_caminhos_concordam_com_contar_transicoes_da_etapa1():
    for s in (SERIE_LACUNA, serie([0, 1, 2, 3, 4], [1, 2, np.nan, 2, 3])):
        ref = contar_transicoes(s)
        np.testing.assert_array_equal(contagens_por_laco(s), ref)
        np.testing.assert_array_equal(contagens_por_crosstab(s), ref)


@pytest.mark.parametrize("geocode", [SERRA, ESCOLHIDO])
def test_caminhos_concordam_nos_dados_reais(geocode):
    s = carregar_serie(geocode)
    ref = contar_transicoes(s)
    np.testing.assert_array_equal(contagens_por_laco(s), ref)
    np.testing.assert_array_equal(contagens_por_crosstab(s), ref)


# ------------------------------------------------------------------ erros-padrão
def test_erros_padrao_a_mao():
    # C = [[2,2],[1,3]] → P̂ = [[.5,.5],[.25,.75]], n_i = (4, 4)
    # linha 1: sqrt(.5*.5/4) = 0,25 ; linha 2: sqrt(.25*.75/4) = sqrt(3)/8 ≈ 0,216506
    ep = erros_padrao(np.array([[2, 2], [1, 3]]))
    esperado = np.array([[0.25, 0.25], [np.sqrt(3) / 8, np.sqrt(3) / 8]])
    np.testing.assert_allclose(ep, esperado, atol=1e-12)


def test_erros_padrao_zero_em_entradas_estruturais():
    ep = erros_padrao(C_LACUNA)
    assert (ep[C_LACUNA == 0] == 0).all()    # p = 0 → erro-padrão 0
    assert ep[1, 2] == 0                      # p = 1 → erro-padrão 0 também


def test_erros_padrao_decresce_com_mais_observacoes():
    # mesma proporção p = 0,5, o dobro de observações → erro-padrão cai por sqrt(2)
    pequeno = erros_padrao(np.array([[2, 2], [2, 2]]))
    grande = erros_padrao(np.array([[4, 4], [4, 4]]))
    np.testing.assert_allclose(pequeno / grande, np.sqrt(2), atol=1e-12)


def test_erros_padrao_linha_sem_saidas_levanta_erro():
    with pytest.raises(ValueError, match="linha"):
        erros_padrao(np.array([[2, 2], [0, 0]]))


# ------------------------------------------------------------------ Laplace
def test_laplace_a_mao():
    # C = [[2,2],[1,3]], alpha = 0,5, k = 2:
    #   linha 1: (2+.5)/(4+1) = 0,5 ; linha 2: (1+.5)/5 = 0,3 e (3+.5)/5 = 0,7
    P = suavizar_laplace(np.array([[2, 2], [1, 3]]), alpha=0.5)
    np.testing.assert_allclose(P, [[0.5, 0.5], [0.3, 0.7]], atol=1e-12)


def test_laplace_com_alpha_zero_e_o_estimador_de_maxima_verossimilhanca():
    C = np.array([[2, 2], [1, 3]])
    np.testing.assert_allclose(suavizar_laplace(C, alpha=0.0), matriz_transicao(C), atol=1e-12)


def test_laplace_e_estocastica_e_positiva_mesmo_com_zeros_estruturais():
    P = suavizar_laplace(C_LACUNA, alpha=0.5)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    assert (P > 0).all()    # os zeros viram α/(n_i + 4α) > 0: transições "inventadas"


def test_laplace_linha_sem_saidas_vira_uniforme():
    # com n_i = 0: (0 + α)/(0 + kα) = 1/k — a suavização "inventa" uma linha inteira
    C = np.array([[2, 2], [0, 0]])
    P = suavizar_laplace(C, alpha=0.5)
    np.testing.assert_allclose(P[1], [0.5, 0.5], atol=1e-12)


def test_laplace_rejeita_alpha_negativo():
    with pytest.raises(ValueError):
        suavizar_laplace(np.array([[2, 2], [1, 3]]), alpha=-0.5)


def test_efeito_laplace_a_mao():
    # P̂ = [[.5,.5],[.25,.75]] → π = (1/3, 2/3) ; P^(0,5) = [[.5,.5],[.3,.7]] → π^(α) = (3/8, 5/8)
    # ‖π - π^(α)‖₁ = |1/3 - 3/8| + |2/3 - 5/8| = 1/24 + 1/24 = 1/12 ≈ 0,083333
    e = efeito_laplace(np.array([[2, 2], [1, 3]]), alpha=0.5)
    assert e == pytest.approx(1 / 12, abs=1e-12)


def test_efeito_laplace_nulo_com_alpha_zero():
    assert efeito_laplace(np.array([[2, 2], [1, 3]]), alpha=0.0) == pytest.approx(0.0, abs=1e-12)


def test_efeito_laplace_em_serra_e_positivo_e_limitado():
    C = contar_transicoes(carregar_serie(SERRA))
    e = efeito_laplace(C, alpha=0.5)
    assert 0 < e <= 2    # a distância L1 entre duas distribuições nunca passa de 2


# ------------------------------------------------------------------ verificações obrigatórias
def test_verificar_aceita_matriz_correta():
    P = matriz_transicao(C_LACUNA)
    verificar(C_LACUNA, P, transicoes_validas=5)    # não levanta


def test_verificar_rejeita_entrada_negativa():
    P = matriz_transicao(C_LACUNA)
    P[0, 0], P[0, 1] = -0.5, 1.5    # linha ainda soma 1, mas há entrada negativa
    with pytest.raises(ValueError):
        verificar(C_LACUNA, P, transicoes_validas=5)


def test_verificar_rejeita_linha_que_nao_soma_um():
    P = matriz_transicao(C_LACUNA)
    P[0, 0] += 1e-9    # acima da tolerância de 1e-12
    with pytest.raises(ValueError):
        verificar(C_LACUNA, P, transicoes_validas=5)


def test_verificar_rejeita_soma_de_C_diferente_das_transicoes_validas():
    P = matriz_transicao(C_LACUNA)
    with pytest.raises(ValueError, match="transi"):
        verificar(C_LACUNA, P, transicoes_validas=6)


# ------------------------------------------------------------------ coerência com a Etapa 1
@pytest.fixture(scope="module")
def auditoria():
    return pd.read_csv("results/e1_auditoria.csv").set_index("geocode")


@pytest.mark.parametrize("geocode", [SERRA, ESCOLHIDO])
def test_soma_de_C_igual_as_transicoes_validas_da_auditoria(geocode, auditoria):
    C = contagens_por_laco(carregar_serie(geocode))
    assert C.sum() == auditoria.loc[geocode, "transicoes_validas"]


def test_soma_de_C_do_es_igual_a_soma_da_auditoria(auditoria):
    geocodes = geocodes_do_estado()
    C_es, _ = contagens_e_ocupacao(geocodes)
    assert len(geocodes) == 78
    assert C_es.sum() == auditoria.loc[geocodes, "transicoes_validas"].sum()


def test_P_estimada_de_serra_e_estocastica_com_tolerancia_1e_12():
    C = contagens_por_laco(carregar_serie(SERRA))
    P = matriz_transicao(C)
    assert (P >= 0).all()
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12, rtol=0)


# ------------------------------------------------------------------ pipeline completo
def test_main_grava_tabelas_e_figura(tmp_path):
    resumo = etapa2.main(saida=tmp_path)
    assert (tmp_path / "tabelas" / "e2_resumo.csv").exists()
    assert (tmp_path / "figuras" / "e2_mapas_calor.png").exists()
    assert list(resumo["territorio"]) == ["Serra", "Vitória", "ES"]
    assert (resumo["soma_C"] == resumo["transicoes_validas"]).all()
    assert (resumo["delta_pi_laplace"] > 0).all()


def test_main_para_se_a_verificacao_falha(tmp_path, monkeypatch):
    def reprova(*args, **kwargs):
        raise ValueError("soma de C diferente das transições válidas")

    monkeypatch.setattr(etapa2, "verificar", reprova)
    with pytest.raises(ValueError, match="transi"):
        etapa2.main(saida=tmp_path)
# [/IA]