# [IA] Claude Sonnet 5.5 - 2026-11-22 - tests/test_canal2.py: testes do sorteio reprodutivel das 10 semanas por municipio para a conferencia do Canal 2 (E1)
"""Testes de src/canal2.py: 10 semanas sorteadas por município (semente 2026), respondido pelo professor."""
import numpy as np
import pandas as pd
import pytest

from src import canal2
from src.canal2 import SEMANAS_POR_MUNICIPIO, SEMENTE, sortear_semanas
from src.dados import carregar_serie
from src.etapa1 import ESCOLHIDO, SERRA


def _serie(n=40, sem_nivel=()):
    datas = pd.date_range("2020-01-05", periods=n, freq="7D")
    niveis = [1 + (k % 4) for k in range(n)]
    for k in sem_nivel:
        niveis[k] = np.nan
    return pd.DataFrame({"data_iniSE": datas, "SE": [202001 + k for k in range(n)], "nivel": niveis,
                         "casos": range(n), "versao_modelo": "2026-11-10"})


def test_constantes_do_enunciado_e_da_resposta_do_professor():
    assert SEMENTE == 2026 and SEMANAS_POR_MUNICIPIO == 10  # "10 semanas para cada município"


def test_sorteio_tem_10_semanas_distintas_ordenadas_por_municipio():
    t = sortear_semanas({1: _serie(), 2: _serie()})
    assert len(t) == 20
    for g in (1, 2):
        d = t[t["geocode"] == g]["data_iniSE"]
        assert len(d) == 10 and d.is_unique and d.is_monotonic_increasing


def test_sorteio_e_reprodutivel_e_depende_da_semente():
    a = sortear_semanas({1: _serie(), 2: _serie()})
    b = sortear_semanas({1: _serie(), 2: _serie()})
    pd.testing.assert_frame_equal(a, b)
    c = sortear_semanas({1: _serie(), 2: _serie()}, semente=2027)
    assert not a["data_iniSE"].equals(c["data_iniSE"])


def test_so_sorteia_semanas_com_nivel_registrado():
    s = _serie(30, sem_nivel=range(0, 15))
    t = sortear_semanas({1: s}, k=10)
    assert t["nivel"].notna().all()
    assert set(t["data_iniSE"]) <= set(s.loc[s["nivel"].notna(), "data_iniSE"])


def test_linha_sorteada_carrega_o_nivel_e_os_casos_da_serie():
    s = _serie()
    t = sortear_semanas({1: s})
    juntos = t.merge(s, on="data_iniSE", suffixes=("", "_serie"))
    assert (juntos["nivel"] == juntos["nivel_serie"]).all() and (juntos["casos"] == juntos["casos_serie"]).all()


def test_poucas_semanas_validas_levanta_erro():
    with pytest.raises(ValueError):
        sortear_semanas({1: _serie(8)})


def test_protocolo_serra_primeiro_depois_vitoria_com_um_unico_gerador():
    """Caminho independente: um default_rng(2026), Serra e depois Vitória, choice sem reposição, ordenado por data."""
    rng = np.random.default_rng(2026)
    esperado = {}
    for g in (SERRA, ESCOLHIDO):
        s = carregar_serie(g)
        s = s[s["nivel"].notna()].reset_index(drop=True)
        idx = sorted(rng.choice(len(s), size=10, replace=False))
        esperado[g] = list(s.loc[idx, "data_iniSE"])
    t = sortear_semanas({g: carregar_serie(g) for g in (SERRA, ESCOLHIDO)})
    for g in (SERRA, ESCOLHIDO):
        assert list(t[t["geocode"] == g]["data_iniSE"]) == esperado[g]


def test_main_grava_tabela_pronta_para_conferir_no_site(tmp_path):
    t = canal2.main(saida=tmp_path)
    arq = tmp_path / "tabelas" / "e1_canal2_semanas.csv"
    lido = pd.read_csv(arq)
    assert len(lido) == 20 and set(lido["geocode"]) == {SERRA, ESCOLHIDO}
    assert lido["nivel_no_site"].isna().all()  # coluna a ser preenchida na conferência manual
    assert lido["url"].str.contains("info.dengue.mat.br/alerta/").all()
    assert len(t) == 20


def test_sem_reposicao_com_exatamente_k_semanas_sorteia_todas_distintas():
    t = sortear_semanas({1: _serie(10)}, k=10)
    assert t["data_iniSE"].is_unique and len(t) == 10


def test_erro_informa_quantas_semanas_faltam():
    with pytest.raises(ValueError, match="precisa de 10"):
        sortear_semanas({1: _serie(8)})


def test_main_sorteia_serra_primeiro_e_confere_com_o_protocolo(tmp_path):
    t = canal2.main(saida=tmp_path)
    esperado = sortear_semanas({g: carregar_serie(g) for g in (SERRA, ESCOLHIDO)})
    assert list(t["geocode"]) == [SERRA] * 10 + [ESCOLHIDO] * 10
    assert list(t["data_iniSE"]) == list(pd.to_datetime(esperado["data_iniSE"]).dt.strftime("%Y-%m-%d"))
# [/IA]
