# [IA] Claude Sonnet 5.5 - 2026-11-22 - tests/test_canal2.py: testes do sorteio reprodutivel das 10 semanas por municipio para a conferencia do Canal 2 (E1)
"""Testes de src/canal2.py: 10 semanas sorteadas por município (semente 2026), respondido pelo professor."""
import numpy as np
import pandas as pd
import pytest

from src import canal2
from src.canal2 import CORES, SEMANAS_POR_MUNICIPIO, SEMENTE, conferir, sortear_semanas
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
    # cada linha aponta para o relatório DAQUELA semana (a página /alerta/ só mostra a semana corrente)
    esperado = [f"https://info.dengue.mat.br/report/ES/{g}/{se}" for g, se in zip(lido["geocode"], lido["SE"])]
    assert list(lido["url"]) == esperado
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


# ------------------------------------------------------------------ conferência com o site
def _sorteio_pequeno():
    return pd.DataFrame({"geocode": [1, 1, 2], "data_iniSE": ["2020-01-05", "2020-01-12", "2020-01-05"],
                         "SE": [202001, 202002, 202001], "nivel": [1, 3, 4], "versao_modelo": "2026-11-10"})


def _manual(cores):
    return pd.DataFrame({"geocode": [1, 1, 2], "SE": [202001, 202002, 202001], "cor_site": cores})


def test_cores_mapeiam_para_niveis_do_infodengue():
    assert CORES == {"Verde": 1, "Amarelo": 2, "Laranja": 3, "Vermelho": 4}


def test_conferir_coincide_diverge_e_sem_conferencia():
    c = conferir(_sorteio_pequeno(), _manual(["Verde", "Amarelo", "?"]))
    assert list(c["nivel_site"].astype("Int64")) == [1, 2, pd.NA]
    assert list(c["situacao"]) == ["coincide", "diverge", "sem_conferencia"]


def test_conferir_exige_todas_as_semanas_sorteadas_e_so_elas():
    m = _manual(["Verde", "Verde", "Verde"])
    with pytest.raises(ValueError, match="faltam"):
        conferir(_sorteio_pequeno(), m.iloc[:2])
    extra = pd.concat([m, pd.DataFrame({"geocode": [9], "SE": [202001], "cor_site": ["Verde"]})])
    with pytest.raises(ValueError, match="sobram"):
        conferir(_sorteio_pequeno(), extra)


def test_conferir_rejeita_cor_desconhecida_e_duplicata():
    with pytest.raises(ValueError, match="cor"):
        conferir(_sorteio_pequeno(), _manual(["Verde", "Azul", "Verde"]))
    dup = pd.concat([_manual(["Verde", "Verde", "Verde"]), _manual(["Verde", "Verde", "Verde"]).iloc[:1]])
    with pytest.raises(ValueError, match="duplicad"):
        conferir(_sorteio_pequeno(), dup)


def test_leitura_manual_real_cobre_exatamente_as_20_semanas_sorteadas():
    sorteio = sortear_semanas({g: carregar_serie(g) for g in (SERRA, ESCOLHIDO)})
    manual = pd.read_csv("data/canal2_niveis_site.csv")
    assert set(zip(manual["geocode"], manual["SE"])) == set(zip(sorteio["geocode"], sorteio["SE"]))
    assert (manual["conferido_em"] == "2026-11-22").all()


def test_main_grava_conferencia_com_os_20_resultados_reais(tmp_path):
    canal2.main(saida=tmp_path)
    c = pd.read_csv(tmp_path / "tabelas" / "e1_canal2_conferencia.csv")
    assert len(c) == 20 and c["situacao"].notna().all()
    # fato observado em 22/11/2026: as 20 semanas coincidem com os dados congelados
    assert (c["situacao"] == "coincide").sum() == 20
    assert (c["nivel_dados"] == c["nivel_site"]).all()


def test_conferencia_mantem_a_ordem_do_sorteio(tmp_path):
    canal2.main(saida=tmp_path)
    sorteio = pd.read_csv(tmp_path / "tabelas" / "e1_canal2_semanas.csv")
    c = pd.read_csv(tmp_path / "tabelas" / "e1_canal2_conferencia.csv")
    assert list(zip(c["geocode"], c["SE"])) == list(zip(sorteio["geocode"], sorteio["SE"]))
# [/IA]

